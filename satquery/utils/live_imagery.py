"""Live satellite-imagery acquisition for a user-drawn region of interest.

Given a bounding box (west, south, east, north in EPSG:4326) and a date
window, this module searches a public STAC catalog (Microsoft Planetary
Computer) for Sentinel-2 L2A optical and/or Sentinel-1 GRD SAR scenes, reads
only the ROI window from the cloud-optimised GeoTIFFs, and returns standard
:class:`~satquery.utils.geospatial.RasterImage` objects that flow through the
unchanged validation -> intent -> specialist pipeline.

Everything degrades gracefully offline: if the catalog or network is
unavailable, a deterministic synthetic scene rendered to the exact requested
geographic extent is produced instead and clearly flagged in its metadata, so
the product remains fully demonstrable without connectivity.
"""

from __future__ import annotations

import hashlib
import io
import math
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from satquery.config import CHANGE_CLASS_ORDER, CHANGE_CLASS_PALETTES
from satquery.utils.geospatial import (
    HAS_RASTERIO,
    ImageMetadata,
    RasterImage,
    infer_modality,
)
from satquery.utils.logger import get_logger

logger = get_logger("live_imagery")

try:  # pragma: no cover - optional dependency
    import folium  # type: ignore

    HAS_FOLIUM = True
except Exception:  # pragma: no cover
    folium = None  # type: ignore
    HAS_FOLIUM = False

MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
S1_COLLECTION = "sentinel-1-grd"

#: hard wall-clock budget for the whole live acquisition attempt; when it is
#: exceeded (e.g. an offline machine silently dropping packets) we fall back
#: to the synthetic scene instead of hanging the UI.
LIVE_FETCH_TIMEOUT_S = 60.0
_HTTP_SOCKET_TIMEOUT_S = 15.0

#: Sentinel-2 asset ids assembled in the package's band order (R, G, B, NIR).
#: Real Sentinel-2 naming: B04 = red (665 nm), B03 = green (560 nm),
#: B02 = blue (490 nm), B08 = NIR (842 nm).
S2_BAND_ASSETS: Tuple[str, str, str, str] = ("B04", "B03", "B02", "B08")
S1_POLARISATIONS: Tuple[str, str] = ("vv", "vh")

DEFAULT_MAX_DIM = 512
SAR_DB_RANGE: Tuple[float, float] = (-25.0, 2.0)  # dB window stretched to 0..1


# ------------------------------------------------------------------ results ---
@dataclass
class LiveFetchResult:
    """Images acquired for an ROI plus acquisition bookkeeping."""

    images: List[RasterImage] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def synthetic(self) -> bool:
        return self.info.get("source") == "synthetic-fallback"


# --------------------------------------------------------------- geo helpers --
BBox = Tuple[float, float, float, float]


def validate_bbox(bbox: Sequence[float]) -> BBox:
    """Normalise and validate a (west, south, east, north) WGS84 bbox."""
    west, south, east, north = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError("ROI coordinates must be finite numbers.")
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValueError("Longitudes must be within [-180, 180].")
    if not (-85.0 <= south <= 85.0 and -85.0 <= north <= 85.0):
        raise ValueError("Latitudes must be within [-85, 85].")
    if east <= west or north <= south:
        raise ValueError("ROI extent is degenerate: need west < east and south < north.")
    if (east - west) > 2.0 or (north - south) > 2.0:
        raise ValueError("ROI too large (>2 degrees per side); draw a smaller region.")
    return (west, south, east, north)


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    west, south, east, north = bbox
    return ((west + east) / 2.0, (south + north) / 2.0)


def _window_dims(bbox: BBox, max_dim: int) -> Tuple[int, int]:
    """Pixel dims preserving the bbox aspect with the long side at ``max_dim``."""
    west, south, east, north = bbox
    aspect = (north - south) / max(1e-9, east - west)  # height / width
    if aspect >= 1.0:
        return max_dim, max(48, int(max_dim / aspect))
    return max(48, int(max_dim * aspect)), max_dim


def _extent_transform(bbox: BBox, width: int, height: int) -> Tuple[List[float], Tuple[float, float]]:
    """Linear WGS84 affine for an image spanning ``bbox`` at ``width x height``."""
    west, south, east, north = bbox
    px = (east - west) / width
    py = (north - south) / height
    transform = [px, 0.0, west, 0.0, -py, north]
    return transform, (px, py)


def _acquisition_extra(source: str, platform: str, when: str, sensor: str,
                       **extra: Any) -> Dict[str, Any]:
    payload = {
        "acquisition": {
            "source": source,
            "platform": platform,
            "datetime": when,
            "sensor": sensor,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    }
    payload["acquisition"].update(extra)
    return payload


def _build_raster(name: str, data: np.ndarray, bbox: BBox, modality_hint: str,
                  extra: Dict[str, Any], preview: Optional[np.ndarray]) -> RasterImage:
    h, w = data.shape[:2]
    transform, (px, py) = _extent_transform(bbox, w, h)
    meta = ImageMetadata(
        name=name, path=Path(name), fmt="COG (remote)" if "live" in source_key(extra) else "GeoTIFF",
        width=w, height=h, bands=data.shape[2] if data.ndim == 3 else 1,
        crs="EPSG:4326", crs_valid=True, transform=transform,
        resolution=(px, py), bounds=bbox,
        size_mb=round(data.nbytes / (1024 * 1024), 2),
        extra=extra,
    )
    modality, conf = infer_modality(data, meta)
    if modality_hint in ("optical", "sar"):
        # trust the declared sensor over heuristics for live acquisitions
        conf = max(conf, 0.9) if modality == modality_hint else 0.85
        modality = modality_hint
    meta.modality, meta.modality_confidence = modality, conf
    return RasterImage(metadata=meta, data=data.astype(np.float32), preview=preview)


def source_key(extra: Dict[str, Any]) -> str:
    acq = extra.get("acquisition", {})
    return str(acq.get("source", ""))


# ------------------------------------------------------------ pixel scaling ---
def _resample(band: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    rows = np.linspace(0, band.shape[0] - 1, out_h).astype(int)
    cols = np.linspace(0, band.shape[1] - 1, out_w).astype(int)
    return band[np.ix_(rows, cols)].astype(np.float32)


def _stretch(band: np.ndarray) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(finite, (2.0, 98.0))
    if hi <= lo + 1e-9:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo + 1e-9:
            return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _true_colour_preview(bands_rgb: List[np.ndarray], out_h: int, out_w: int) -> np.ndarray:
    """True-colour uint8 preview from bands already in R/G/B order."""
    red, green, blue = (_stretch(b) for b in bands_rgb[:3])
    rgb = np.dstack([red, green, blue])
    h, w = rgb.shape[:2]
    rows = np.linspace(0, h - 1, out_h).astype(int)
    cols = np.linspace(0, w - 1, out_w).astype(int)
    return (rgb[np.ix_(rows, cols)] * 255).astype(np.uint8)


# --------------------------------------------------------------- STAC access ---
def _mpc_catalog() -> Any:
    from planetary_computer import sign_inplace  # type: ignore
    from pystac_client import Client  # type: ignore

    return Client.open(MPC_STAC_URL, modifier=sign_inplace)


def _run_with_timeout(fn: Any, *args: Any) -> Any:
    """Run ``fn`` on a worker thread with a hard wall-clock budget.

    The executor is deliberately NOT used as a context manager: its ``__exit__``
    joins the worker, which would block until a hung network call finishes and
    defeat the deadline. On timeout the pool is abandoned (``wait=False``) and
    control returns immediately; the leaked worker dies once its sockets time
    out on their own.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="satquery-live")
    try:
        future = pool.submit(fn, *args)
        return future.result(timeout=LIVE_FETCH_TIMEOUT_S)
    except FuturesTimeoutError as exc:
        raise TimeoutError(
            f"live acquisition exceeded {LIVE_FETCH_TIMEOUT_S:.0f}s "
            "(network unreachable?)"
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _datetime_range(start: date, end: date) -> str:
    return f"{start.isoformat()}/{end.isoformat()}"


def _read_asset_window(href: str, bbox: BBox, out_h: int, out_w: int) -> np.ndarray:
    """Read the ROI window of a remote COG, resampled to ``out_h x out_w``."""
    import rasterio  # type: ignore
    from rasterio.warp import transform_bounds  # type: ignore
    from rasterio.windows import from_bounds  # type: ignore

    env_kwargs = {
        "GDAL_HTTP_CONNECT_TIMEOUT": "10",
        "GDAL_HTTP_TIMEOUT": "45",
        "GDAL_HTTP_MAX_RETRY": "1",
        "GDAL_HTTP_VERSION": "2",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.tif.aux.xml,.msl",
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
        "VSICURL_CACHE_SIZE": "64",
    }
    with rasterio.Env(**env_kwargs):
        with rasterio.open(href) as src:
            dst_bbox = bbox
            if src.crs is not None and "4326" not in str(src.crs).upper():
                dst_bbox = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            window = from_bounds(*dst_bbox, transform=src.transform)
            window = window.round_offsets().round_lengths()
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            data = src.read(1, window=window).astype(np.float32)
    return _resample(data, out_h, out_w)


def _pick_item(catalog: Any, collection: str, bbox: BBox, dt_range: str,
               cloud_max: Optional[float]) -> Tuple[Any, Dict[str, Any]]:
    search_kwargs: Dict[str, Any] = {
        "collections": [collection], "bbox": list(bbox), "datetime": dt_range,
        "max_items": 12,
    }
    if cloud_max is not None and collection == S2_COLLECTION:
        search_kwargs["query"] = {"eo:cloud_cover": {"lt": cloud_max}}
    items = list(catalog.search(**search_kwargs).items())
    if not items:
        raise RuntimeError(f"no {collection} scenes cover the ROI in this window")
    if collection == S2_COLLECTION:
        item = min(items, key=lambda it: it.properties.get("eo:cloud_cover", 0.0))
    else:
        item = items[-1]
    props = {
        "datetime": str(item.properties.get("datetime", "")),
        "platform": str(item.properties.get("platform", collection)),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "item_id": str(item.id),
    }
    return item, props


def _fetch_optical_live(catalog: Any, bbox: BBox, start: date, end: date,
                        max_dim: int, cloud_max: Optional[float]) -> Tuple[RasterImage, Dict[str, Any]]:
    item, props = _pick_item(catalog, S2_COLLECTION, bbox, _datetime_range(start, end), cloud_max)
    out_h, out_w = _window_dims(bbox, max_dim)

    bands: List[np.ndarray] = []
    for asset_id in S2_BAND_ASSETS:
        if asset_id not in item.assets:
            raise RuntimeError(f"asset {asset_id} missing from scene {props['item_id']}")
        bands.append(_read_asset_window(item.assets[asset_id].href, bbox, out_h, out_w))
    data = np.dstack([_stretch(b) for b in bands]).astype(np.float32)
    preview = _true_colour_preview(bands, min(out_h, 720), min(out_w, 720))
    extra = _acquisition_extra(
        "planetary-computer (live)", props["platform"], props["datetime"], "optical",
        cloud_cover=props["cloud_cover"], scene_id=props["item_id"],
        bands="B04,B03,B02,B08 (red,green,blue,nir)",
    )
    img = _build_raster(f"s2_optical_{props['datetime'][:10]}.tif", data, bbox, "optical", extra, preview)
    return img, props


def _fetch_sar_live(catalog: Any, bbox: BBox, start: date, end: date,
                    max_dim: int) -> Tuple[RasterImage, Dict[str, Any]]:
    item, props = _pick_item(catalog, S1_COLLECTION, bbox, _datetime_range(start, end), None)
    out_h, out_w = _window_dims(bbox, max_dim)

    pols: List[np.ndarray] = []
    for pol in S1_POLARISATIONS:
        if pol not in item.assets:
            if pol == "vv":
                raise RuntimeError(f"VV asset missing from scene {props['item_id']}")
            continue
        amp = _read_asset_window(item.assets[pol].href, bbox, out_h, out_w)
        amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
        db = 10.0 * np.log10(np.maximum(amp, 1e-6) ** 2)
        lo, hi = SAR_DB_RANGE
        pols.append(np.clip((db - lo) / (hi - lo), 0.0, 1.0).astype(np.float32))
    data = np.dstack(pols).astype(np.float32)

    from satquery.utils.geospatial import _rgb_preview, _resize_preview

    preview = _resize_preview(_rgb_preview(data), 720)
    extra = _acquisition_extra(
        "planetary-computer (live)", props["platform"], props["datetime"], "sar",
        polarisations="+".join(S1_POLARISATIONS[: len(pols)]),
        scene_id=props["item_id"], note="dB-normalised backscatter, VV/VH",
    )
    img = _build_raster(f"s1_sar_{props['datetime'][:10]}.tif", data, bbox, "sar", extra, preview)
    return img, props


# ------------------------------------------------------ synthetic fallback ----
def _seed_from_bbox(bbox: BBox) -> int:
    payload = ",".join(f"{v:.5f}" for v in bbox).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16) % (2 ** 31)


def _synthetic_pair(bbox: BBox, start: date, end: date, include_sar: bool,
                    max_dim: int) -> List[RasterImage]:
    """Deterministic optical/SAR/bi-temporal scenes rendered onto the exact ROI."""
    from satquery.utils.demo_data import _render_optical, _render_sar, build_layout

    seed = _seed_from_bbox(bbox)
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(np.clip(max_dim, 128, 512))
    layout = build_layout(seed=seed % 9973, size=size)

    west, south, east, north = bbox
    images: List[RasterImage] = []

    def finish(name: str, arr: np.ndarray, when: date, sensor: str) -> RasterImage:
        from satquery.utils.geospatial import _rgb_preview, _resize_preview

        extra = _acquisition_extra(
            "synthetic-fallback", "demo-renderer", f"{when.isoformat()}T00:00:00Z", sensor,
            note="offline fallback - catalog unreachable; scene rendered to the requested extent",
        )
        preview = _rgb_preview(arr)
        if preview is not None:
            preview = _resize_preview(preview, 720)
        return _build_raster(name, arr, bbox, sensor, extra, preview)

    t1 = finish(f"synthetic_optical_{start.isoformat()}.tif", _render_optical(layout, seed), start, "optical")
    images.append(t1)
    changed = build_layout(seed=seed % 9973, size=size, water_scale=0.88,
                           urban_growth=0.055, veg_loss=0.14)
    t2 = finish(f"synthetic_optical_{end.isoformat()}.tif", _render_optical(changed, seed + 3), end, "optical")
    images.append(t2)
    if include_sar:
        images.append(finish(f"synthetic_sar_{end.isoformat()}.tif",
                             _render_sar(layout, seed, two_band=True), end, "sar"))
    return images


# ----------------------------------------------------------------- main API ---
def _live_attempt(box: BBox, start: date, end: date, include_sar: bool,
                  max_dim: int, cloud_max: Optional[float]) -> Tuple[List[RasterImage], Dict[str, Any], Optional[Dict[str, Any]]]:
    """One live-acquisition attempt; raises on any network/catalog failure."""
    catalog = _mpc_catalog()
    images: List[RasterImage] = []
    t1, p1 = _fetch_optical_live(catalog, box, start, start + timedelta(days=14), max_dim, cloud_max)
    images.append(t1)
    t2, p2 = _fetch_optical_live(catalog, box, end - timedelta(days=14), end, max_dim, cloud_max)
    images.append(t2)
    scenes_info = {"scenes": {"t1": p1, "t2": p2}}
    sar_info: Optional[Dict[str, Any]] = None
    if include_sar:
        sar, ps = _fetch_sar_live(catalog, box, end - timedelta(days=14), end, max_dim)
        images.append(sar)
        sar_info = ps
    return images, scenes_info, sar_info


def fetch_roi_imagery(bbox: Sequence[float], start: date, end: date,
                      include_sar: bool = False, max_dim: int = DEFAULT_MAX_DIM,
                      cloud_max: Optional[float] = 30.0,
                      allow_synthetic_fallback: bool = True) -> LiveFetchResult:
    """Acquire imagery for an ROI; falls back to a synthetic scene offline.

    Returns up to three images: optical at ``start`` (T1), optical at ``end``
    (T2, enabling bi-temporal change), and optionally SAR near ``end`` for
    cross-modal fusion.
    """
    box = validate_bbox(bbox)
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    result = LiveFetchResult(info={
        "bbox": list(box), "start": start.isoformat(), "end": end.isoformat(),
        "include_sar": include_sar, "source": "planetary-computer (live)",
    })
    try:
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(_HTTP_SOCKET_TIMEOUT_S)
        try:
            catalog, scenes_info, sar_info = _run_with_timeout(_live_attempt, box, start, end,
                                                               include_sar, max_dim, cloud_max)
        finally:
            socket.setdefaulttimeout(old_timeout)
        result.images = list(catalog)
        result.info.update(scenes_info)
        if sar_info:
            result.info["sar_scene"] = sar_info
    except Exception as exc:  # noqa: BLE001 - network/catalog failures are expected
        logger.warning("live acquisition failed: %s", exc)
        result.warnings.append(f"Live catalog unavailable ({type(exc).__name__}: {str(exc)[:140]}).")
        if not allow_synthetic_fallback:
            raise
        result.info["source"] = "synthetic-fallback"
        result.info.pop("scenes", None)
        result.info.pop("sar_scene", None)
        result.images = _synthetic_pair(box, start, end, include_sar, max_dim)
        result.warnings.append(
            "Serving a deterministic synthetic scene rendered to the requested ROI extent; "
            "all analyses run identically on it."
        )
    return result


# --------------------------------------------------------------- geo exports ---
def image_to_geotiff(img: RasterImage) -> bytes:
    """Serialize a RasterImage (any band count) to in-memory GeoTIFF bytes."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required for GeoTIFF export.")
    import rasterio  # type: ignore
    from rasterio.transform import Affine  # type: ignore
    from rasterio.io import MemoryFile  # type: ignore

    data = img.data
    h, w, c = data.shape
    transform = Affine(*img.metadata.transform[:6]) if img.metadata.transform else Affine.identity()
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff", height=h, width=w, count=c, dtype="float32",
            crs=img.metadata.crs or "EPSG:4326", transform=transform,
        ) as dst:
            dst.write(np.moveaxis(data, -1, 0))
        return mem.read()


def class_map_to_geotiff(index_map: np.ndarray, palette: Dict[int, Tuple[int, int, int]],
                         bounds: Optional[Sequence[float]] = None) -> bytes:
    """Serialize a labelled class/change map (H,W ints) to GeoTIFF with a colormap.

    Pass the source image's ``(west, south, east, north)`` bounds to receive a
    correctly geotagged raster (openable at the right location in QGIS).
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required for GeoTIFF export.")
    import rasterio  # type: ignore
    from rasterio.transform import Affine  # type: ignore
    from rasterio.io import MemoryFile  # type: ignore

    h, w = index_map.shape
    if bounds and len(bounds) == 4:
        transform, _ = _extent_transform(tuple(bounds), w, h)
        transform = Affine(*transform[:6])  # rasterio requires an Affine instance
    else:
        transform = Affine.identity()
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff", height=h, width=w, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(index_map.astype(np.uint8), 1)
            dst.write_colormap(1, {idx: (*rgb, 255) for idx, rgb in palette.items()})
        return mem.read()


def change_labels_to_indices(labels: np.ndarray) -> np.ndarray:
    """Map string change labels to the canonical CHANGE_CLASS_ORDER indices."""
    order = {name: idx for idx, name in enumerate(CHANGE_CLASS_ORDER)}
    out = np.zeros(labels.shape, dtype=np.uint8)
    for name, idx in order.items():
        out[labels == name] = idx
    return out


def _polygon_for_bbox(bbox: BBox) -> List[List[List[float]]]:
    west, south, east, north = bbox
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return [[ring]]


def grounding_geojson(img: RasterImage, regions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grounding regions (pixel bboxes) as exact WGS84 polygon features."""
    if not img.metadata.bounds or not img.metadata.transform:
        raise ValueError("image is not georeferenced; cannot export GeoJSON.")
    west0, south0, _, north0 = img.metadata.bounds
    px, py = img.metadata.transform[0], abs(img.metadata.transform[4])
    h, w = img.height, img.width
    features: List[Dict[str, Any]] = []
    for region in regions:
        x0, y0, x1, y1 = region["bbox"]
        gx0 = x0 / w * (px * w)
        gx1 = (x1 + 1) / w * (px * w)
        gy0 = north0 - (y0 / h) * (py * h)
        gy1 = north0 - ((y1 + 1) / h) * (py * h)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [[[west0 + gx0, gy1], [west0 + gx1, gy1],
                                          [west0 + gx1, gy0], [west0 + gx0, gy0],
                                          [west0 + gx0, gy1]]]},
            "properties": {
                "kind": "grounding_region",
                "term": img.metadata.extra.get("grounding_term", "target"),
                "region_id": region.get("region_id"),
                "quadrant": region.get("quadrant"),
                "coverage_pct": region.get("coverage_pct"),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def region_polygons_geojson(img: RasterImage, labels: np.ndarray,
                            name_for_index: Optional[Dict[int, str]] = None,
                            palette: Optional[Dict[int, Tuple[int, int, int]]] = None) -> Dict[str, Any]:
    """Vectorise a labelled H,W map into per-region polygon features.

    Each 4-connected component becomes one polygon feature carrying its class
    label and pixel area. Boundaries are the component's axis-aligned hull -
    compact, lossless for class identity, and directly usable in QGIS/geojson.io.
    """
    if not img.metadata.bounds:
        raise ValueError("image is not georeferenced; cannot export GeoJSON.")
    from scipy import ndimage  # type: ignore

    west0, south0, _, north0 = img.metadata.bounds
    px, py = img.metadata.transform[0], abs(img.metadata.transform[4])
    h, w = labels.shape
    features: List[Dict[str, Any]] = []
    unique = [int(v) for v in np.unique(labels) if int(v) != 0]
    for value in unique:
        mask = labels == value
        labelled, n = ndimage.label(mask)
        slices = ndimage.find_objects(labelled)
        for i, sl in enumerate(slices, start=1):
            rows, cols = sl
            area = int((labelled[sl] == i).sum())
            if area < 8:
                continue
            x0, x1 = cols.start, cols.stop
            y0, y1 = rows.start, rows.stop
            gx0, gx1 = west0 + x0 / w * (px * w), west0 + x1 / w * (px * w)
            gy1, gy0 = north0 - y0 / h * (py * h), north0 - y1 / h * (py * h)
            ring = [[gx0, gy1], [gx1, gy1], [gx1, gy0], [gx0, gy0], [gx0, gy1]]
            props: Dict[str, Any] = {
                "kind": "classified_region",
                "class_index": value,
                "area_px": area,
            }
            if name_for_index and value in name_for_index:
                props["class"] = name_for_index[value]
            if palette and value in palette:
                props["color"] = "#%02x%02x%02x" % palette[value]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            })
    return {"type": "FeatureCollection", "features": features}


# ----------------------------------------------------------------- folium ----
def build_result_map(img: RasterImage, overlay_rgb: Optional[np.ndarray] = None,
                     overlay_name: str = "analysis overlay",
                     rectangles: Optional[List[Dict[str, Any]]] = None) -> Any:
    """Folium map centred on the image with an optional image overlay + boxes.

    ``rectangles`` items carry pixel ``bbox`` plus ``label``/``color`` and are
    converted to WGS84 leaflet rectangles.
    """
    if not HAS_FOLIUM:
        raise RuntimeError("folium is required for interactive maps.")
    if not img.metadata.bounds:
        raise ValueError("image is not georeferenced; cannot render a map.")
    import base64 as _b64

    west, south, east, north = img.metadata.bounds
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2],
                      zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    folium.Rectangle([[south, west], [north, east]], color="#222", weight=1,
                     fill=False).add_to(fmap)
    if overlay_rgb is not None:
        rgba = np.asarray(overlay_rgb)
        if rgba.ndim == 2:  # single-band -> greyscale RGBA
            rgba = np.dstack([rgba] * 3 + [np.full_like(rgba, 255)])
        elif rgba.shape[2] == 3:  # RGB -> RGBA (alpha 255 keeps overlay visible)
            alpha = np.full(rgba.shape[:2], 255, dtype=rgba.dtype)
            rgba = np.dstack([rgba, alpha])
        folium.raster_layers.ImageOverlay(
            image=rgba,  # array input lets folium do the mercator projection itself
            bounds=[[south, west], [north, east]], opacity=0.75,
            name=overlay_name, mercator_project=True,
        ).add_to(fmap)
    if rectangles:
        h, w = img.height, img.width
        px, py = img.metadata.transform[0], abs(img.metadata.transform[4])
        for rect in rectangles:
            x0, y0, x1, y1 = rect["bbox"]
            lat_n = north - (y0 / h) * (py * h)
            lat_s = north - ((y1 + 1) / h) * (py * h)
            lon_w = west + (x0 / w) * (px * w)
            lon_e = west + ((x1 + 1) / w) * (px * w)
            folium.Rectangle(
                [[lat_s, lon_w], [lat_n, lon_e]],
                color=rect.get("color", "#d7301f"), weight=2, fill=True,
                fill_opacity=0.15,
                tooltip=str(rect.get("label", "region")),
            ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap
