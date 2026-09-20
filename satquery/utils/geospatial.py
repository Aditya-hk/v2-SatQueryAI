"""Geospatial utilities: GeoTIFF/TIFF/PNG/JPEG I/O, CRS validation, pair
compatibility checks, sensor-modality inference, co-registration, spectral-index
analysis, and land-cover reasoning shared by the specialist models.

Design notes
------------
* Images are read into ``float32`` arrays in HWC (height, width, bands) order,
  min-max stretched per band to the 0..1 range. GeoTIFF metadata (CRS, affine
  transform, resolution, bounds) is preserved on :class:`ImageMetadata`.
* Validation never raises for user-facing failures; it records problems and
  returns structured results so the agent can emit precise audit messages.
* Optional heavy dependencies (rasterio, pyproj) degrade gracefully: when a
  GeoTIFF is readable without them (or the file is PNG/JPEG) the code still
  functions, and CRS checks report "not available" rather than crashing.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from satquery.config import (
    BENCHMARK_EXTENSIONS,
    BARESOIL_BRIGHTNESS_THRESHOLD,
    BUILTUP_NDBI_THRESHOLD,
    CLOUD_BRIGHTNESS_THRESHOLD,
    CLOUD_SATURATION_THRESHOLD,
    CLOUD_WARNING_FRACTION,
    DEFAULT_TARGET_CRS,
    EXTENT_OVERLAP_ERROR,
    EXTENT_OVERLAP_WARN,
    GEOTIFF_EXTENSIONS,
    LANDCOVER_CLASS_NAMES,
    MIN_IMAGE_SIDE_PX,
    MODALITY_GRAY_OPTICAL_SATURATION,
    MODALITY_HSV_SAMPLE_STRIDE,
    MODALITY_OPTICAL_SATURATION,
    MODALITY_SAR_MAX_VARIANCE,
    PAIR_ALIGNMENT_TOLERANCE_PX,
    PREVIEW_MAX_SIDE,
    SAR_BUILTUP_BACKSCATTER_MIN,
    SAR_BUILTUP_TEXTURE_MIN,
    SAR_WATER_BACKSCATTER_MAX,
    SUPPORTED_EXTENSIONS,
    VEGETATION_NDVI_THRESHOLD,
    WATER_NDWI_THRESHOLD,
)

_EPSG_CACHE: Dict[str, str] = {
    "EPSG:4326": "EPSG:4326",
    "EPSG:32643": "EPSG:32643",
    "EPSG:3857": "EPSG:3857",
}

try:  # pragma: no cover - exercised via integration tests when available
    import rasterio  # type: ignore

    HAS_RASTERIO = True
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    HAS_RASTERIO = False

try:  # pragma: no cover
    from pyproj import CRS as _PyprojCRS  # type: ignore

    HAS_PYPROJ = True
except Exception:  # pragma: no cover
    _PyprojCRS = None  # type: ignore
    HAS_PYPROJ = False

try:  # pragma: no cover
    from PIL import Image  # type: ignore

    HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    HAS_PIL = False


# --------------------------------------------------------------------- data --
@dataclass
class ImageMetadata:
    """Metadata describing one uploaded remote-sensing image."""

    name: str
    path: Path
    fmt: str
    width: int
    height: int
    bands: int
    crs: Optional[str] = None
    crs_valid: Optional[bool] = None
    transform: Optional[List[float]] = None
    resolution: Optional[Tuple[float, float]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    nodata: Optional[float] = None
    dtype: str = "float32"
    modality: str = "optical"
    modality_confidence: float = 0.5
    size_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "format": self.fmt,
            "width": self.width,
            "height": self.height,
            "bands": self.bands,
            "crs": self.crs,
            "crs_valid": self.crs_valid,
            "transform": self.transform,
            "resolution": self.resolution,
            "bounds": self.bounds,
            "nodata": self.nodata,
            "dtype": self.dtype,
            "modality": self.modality,
            "modality_confidence": self.modality_confidence,
            "size_mb": self.size_mb,
            "extra": self.extra,
        }

    @property
    def is_georeferenced(self) -> bool:
        return bool(self.crs and self.transform)

    @property
    def is_sar(self) -> bool:
        return self.modality == "sar"

    def resolution_m(self) -> Optional[float]:
        """Average ground resolution in metres (None for geographic CRS in degrees)."""
        if not self.resolution:
            return None
        avg = (abs(self.resolution[0]) + abs(self.resolution[1])) / 2.0
        if self.crs and "4326" in self.crs.upper():
            return avg * 111_320.0 * np.cos(np.deg2rad(20.0))
        return avg


@dataclass
class RasterImage:
    """An in-memory remote-sensing image plus its metadata and previews."""

    metadata: ImageMetadata
    data: np.ndarray  # HWC float32, roughly 0..1
    preview: Optional[np.ndarray] = None  # HWC uint8 RGB
    issues: List[str] = field(default_factory=list)
    reprojected_from: Optional[str] = None

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def bands(self) -> int:
        return int(self.data.shape[2]) if self.data.ndim == 3 else 1

    @property
    def modality(self) -> str:
        return self.metadata.modality

    def to_dict(self) -> Dict[str, Any]:
        payload = self.metadata.to_dict()
        payload["issues"] = list(self.issues)
        payload["reprojected_from"] = self.reprojected_from
        return payload


# ------------------------------------------------------------------ helpers --
def _file_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in GEOTIFF_EXTENSIONS:
        return "GeoTIFF" if HAS_RASTERIO else "TIFF"
    if suffix in BENCHMARK_EXTENSIONS:
        return "PNG" if suffix == ".png" else "JPEG"
    return suffix.lstrip(".").upper()


def _normalise_channel(channel: np.ndarray) -> np.ndarray:
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=np.float32)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo + 1e-9:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo + 1e-9:
            return np.zeros_like(channel, dtype=np.float32)
    out = (channel.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


_RGB_NAME_ALIASES: Dict[str, Tuple[str, ...]] = {
    "red": ("b04", "b4", "band4", "sr_b4", "red"),
    "green": ("b03", "b3", "band3", "sr_b3", "green"),
    "blue": ("b02", "b2", "band2", "sr_b2", "blue"),
}


def _rgb_band_indices(data: np.ndarray,
                      band_names: Optional[Sequence[str]] = None) -> Tuple[int, int, int]:
    """Choose which channels form the R/G/B preview.

    Package band-order convention for optical rasters is **(R, G, B, NIR, ...)**
    (channels 0..2 display as RGB directly, channel 3 is NIR). For stacks whose
    native order differs we disambiguate with, in priority order:

    1. explicit band names (Sentinel-2 ``B04/B03/B02``, Landsat ``sr_b4..b2``,
       or generic ``red/green/blue`` descriptions) when available;
    2. band-count heuristics - 3/4-band rasters are taken as-is (RGB / RGB+NIR),
       5-band ``B,G,R,RE,N``-style stacks use (2, 1, 0), and >=6-band
       Sentinel-2 ``B01..B12`` stacks use (3, 2, 1) = B04, B03, B02.
    """
    c = data.shape[2] if data.ndim == 3 else 1
    if band_names and data.ndim == 3:
        picked: Dict[str, int] = {}
        for idx, raw in enumerate(band_names[:c]):
            name = str(raw).strip().lower()
            for role, aliases in _RGB_NAME_ALIASES.items():
                if name in aliases and role not in picked:
                    picked[role] = idx
        if len(picked) == 3:
            return picked["red"], picked["green"], picked["blue"]
    if c <= 4:
        return 0, 1, min(2, c - 1)      # RGB or RGB+NIR (package order)
    if c == 5:
        return 2, 1, 0                  # B,G,R,RE,NIR-style stacks
    return 3, 2, 1                      # Sentinel-2 B01..B12 stacks -> B04,B03,B02


def _rgb_preview(data: np.ndarray,
                 band_names: Optional[Sequence[str]] = None) -> Optional[np.ndarray]:
    """Build an RGB uint8 preview; false-colour for SAR (VV/VH/texture)."""
    if data.ndim != 3:
        return None
    h, w, c = data.shape
    if c >= 3:
        r, g, b = _rgb_band_indices(data, band_names)
        rgb = np.dstack([
            _normalise_channel(data[:, :, r]),
            _normalise_channel(data[:, :, g]),
            _normalise_channel(data[:, :, b]),
        ])
    elif c == 2:
        vv, vh = _normalise_channel(data[:, :, 0]), _normalise_channel(data[:, :, 1])
        texture = _normalise_channel(
            np.abs(np.diff(vv, axis=0, prepend=vv[:1, :]))
        )
        rgb = np.dstack([vv, vh, texture])
    else:
        band = _normalise_channel(data[:, :, 0])
        rgb = np.dstack([band, band, band])
    return (rgb * 255).astype(np.uint8)


def _resize_preview(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return rgb
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    if HAS_PIL:
        return np.asarray(Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR))
    ys = np.linspace(0, h - 1, new_h).astype(int)
    xs = np.linspace(0, w - 1, new_w).astype(int)
    return rgb[np.ix_(ys, xs)]


# ------------------------------------------------------------------- reading --
def read_image(path: Path, preview_max_side: int = PREVIEW_MAX_SIDE) -> RasterImage:
    """Read GeoTIFF/TIFF/PNG/JPEG into a :class:`RasterImage` (never raises)."""
    path = Path(path)
    issues: List[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: GeoTIFF/TIFF for geospatial data; "
            f"PNG/JPEG only for benchmark datasets."
        )
    size_mb = path.stat().st_size / (1024 * 1024)

    meta = ImageMetadata(
        name=path.name, path=path, fmt=_file_format(path),
        width=0, height=0, bands=0, size_mb=round(size_mb, 2),
    )
    data: Optional[np.ndarray] = None

    if HAS_RASTERIO and suffix in GEOTIFF_EXTENSIONS:
        try:
            with rasterio.open(path) as src:
                raw = src.read().astype(np.float32)
                if raw.ndim == 3:
                    data = np.moveaxis(raw, 0, -1)
                else:
                    data = raw[np.newaxis].transpose(1, 2, 0)
                meta.width, meta.height, meta.bands = int(src.width), int(src.height), int(src.count)
                meta.dtype = str(src.dtypes[0]) if src.dtypes else "float32"
                meta.nodata = float(src.nodata) if src.nodata is not None else None
                if src.crs is not None:
                    meta.crs = str(src.crs)
                    meta.crs_valid = True
                else:
                    issues.append("No CRS defined in GeoTIFF metadata.")
                if src.transform is not None:
                    t = src.transform
                    meta.transform = [t.a, t.b, t.c, t.d, t.e, t.f]
                    meta.resolution = (float(t.a), float(abs(t.e)))
                    meta.bounds = (
                        float(src.bounds.left), float(src.bounds.bottom),
                        float(src.bounds.right), float(src.bounds.top),
                    )
                try:
                    descriptions = [str(d) for d in (src.descriptions or ())]
                except Exception:
                    descriptions = []
                if any(d for d in descriptions):
                    meta.extra["band_names"] = descriptions
                meta.extra["driver"] = str(src.driver)
        except Exception as exc:
            issues.append(f"rasterio failed to open file ({type(exc).__name__}: {exc}); trying fallback readers.")

    if data is None:
        arr = _read_via_pil_or_tifffile(path, issues)
        if arr is None:
            raise ValueError(f"Could not decode image '{path.name}'. Corrupted or unsupported file?")
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        data = arr.astype(np.float32)
        if data.ndim == 3 and data.shape[2] > 3:
            data = data[:, :, :3]
        data = _normalise_channel(data) if data.ndim == 2 else np.dstack(
            [_normalise_channel(data[:, :, i]) for i in range(data.shape[2])]
        )
        meta.height, meta.width = int(data.shape[0]), int(data.shape[1])
        meta.bands = int(data.shape[2]) if data.ndim == 3 else 1
        meta.dtype = "uint8"

    if meta.height < MIN_IMAGE_SIDE_PX or meta.width < MIN_IMAGE_SIDE_PX:
        issues.append(
            f"Image is very small ({meta.width}x{meta.height}); minimum supported side is {MIN_IMAGE_SIDE_PX}px."
        )

    stretched = np.dstack(
        [_normalise_channel(data[:, :, i]) for i in range(data.shape[2])]
    ) if data.ndim == 3 else data[:, :, np.newaxis]
    stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    modality, modality_conf = infer_modality(stretched, meta)
    meta.modality, meta.modality_confidence = modality, modality_conf

    preview = _rgb_preview(stretched, meta.extra.get("band_names"))
    if preview is not None:
        preview = _resize_preview(preview, preview_max_side)

    return RasterImage(metadata=meta, data=stretched, preview=preview, issues=issues)


def _read_via_pil_or_tifffile(path: Path, issues: List[str]) -> Optional[np.ndarray]:
    if HAS_PIL:
        try:
            img = Image.open(path)
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
            return np.asarray(img)
        except Exception as exc:
            issues.append(f"PIL failed to decode ({type(exc).__name__}: {exc}).")
    try:  # lightweight TIFF fallback
        import tifffile  # type: ignore

        return np.asarray(tifffile.imread(path))
    except Exception:
        return None


# ------------------------------------------------------------------- modality --
def infer_modality(data: np.ndarray, meta: Optional[ImageMetadata] = None) -> Tuple[str, float]:
    """Heuristic optical/multispectral vs SAR classifier.

    SAR imagery is grayscale (single-band or polarimetric channels) with low
    overall saturation, plus very low per-channel variance compared with the
    scene-level variance that multispectral bands exhibit.
    """
    hint = _sar_name_hint(meta)
    if meta is not None and meta.bands == 1:
        if hint:
            return "sar", 0.90
        return _gray_modality(data)
    if meta is not None and meta.bands == 2:
        return ("sar", 0.92) if hint else ("sar", 0.75)
    if data.ndim == 3 and data.shape[2] >= 3:
        if hint:
            return "sar", 0.85
        return _color_modality(data[:, :, :3])
    if data.ndim == 3 and data.shape[2] == 1:
        return _gray_modality(data)
    return "optical", 0.5


def _gray_modality(data: np.ndarray) -> Tuple[str, float]:
    """Single-band decision: SAR speckle is spatially decorrelated, so the mean
    gradient magnitude approaches the pixel std-dev, whereas photographic /
    optical scenes are spatially correlated (gradient << std)."""
    band = data[:, :, 0]
    std = float(np.std(band))
    ptp = float(np.percentile(band, 98) - np.percentile(band, 2))
    gy, gx = np.gradient(band)
    speckle_ratio = float(np.mean(np.sqrt(gx * gx + gy * gy))) / (std + 1e-6)
    if std < MODALITY_SAR_MAX_VARIANCE and ptp > 0.30:
        return "sar", 0.70
    if speckle_ratio > 0.90 and ptp > 0.20:
        return "sar", 0.68
    return "optical", 0.55


def _color_modality(rgb: np.ndarray) -> Tuple[str, float]:
    stride = max(1, MODALITY_HSV_SAMPLE_STRIDE)
    r = np.clip(rgb[::stride, ::stride, 0], 0, 1)
    g = np.clip(rgb[::stride, ::stride, 1], 0, 1)
    b = np.clip(rgb[::stride, ::stride, 2], 0, 1)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    mean_sat = float(np.mean(sat))
    if mean_sat >= MODALITY_OPTICAL_SATURATION:
        return "optical", min(0.95, 0.55 + mean_sat)
    if mean_sat < MODALITY_GRAY_OPTICAL_SATURATION:
        per_ch = [float(np.std(rgb[:, :, i])) for i in range(3)]
        spread = float(np.mean(per_ch))
        if spread < MODALITY_SAR_MAX_VARIANCE:
            return "sar", 0.72
        return "optical", 0.62
    return "optical", 0.58


# ----------------------------------------------------------------------- CRS --
def validate_crs(crs_str: Optional[str]) -> Tuple[bool, str]:
    """Validate a CRS identifier; returns (is_valid, human-readable message)."""
    if not crs_str:
        return False, "No CRS metadata found. Upload a properly georeferenced GeoTIFF."
    if HAS_PYPROJ:
        try:
            crs = _PyprojCRS.from_user_input(crs_str)
            return True, f"CRS {crs.to_string()} is valid ({crs.name})."
        except Exception as exc:
            return False, f"Invalid CRS '{crs_str}': {exc}"
    known = _EPSG_CACHE.get(str(crs_str).upper())
    if known:
        return True, f"CRS {known} is valid (known code)."
    if str(crs_str).upper().startswith("EPSG:"):
        return True, f"CRS {crs_str} assumed valid (pyproj unavailable)."
    return False, f"Could not validate CRS '{crs_str}'."


def _bounds_overlap(a: Tuple[float, float, float, float],
                    b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    a_area = max(1e-12, (ax1 - ax0) * (ay1 - ay0))
    b_area = max(1e-12, (bx1 - bx0) * (by1 - by0))
    return inter / min(a_area, b_area)


def check_pair_compatibility(primary: ImageMetadata, secondary: ImageMetadata,
                             require_same_crs: bool = True,
                             tolerance_px: float = PAIR_ALIGNMENT_TOLERANCE_PX) -> Dict[str, Any]:
    """Validate a paired submission (bi-temporal or cross-modal)."""
    problems: List[str] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {"crs_match": None, "dimensions_match": None,
                            "resolution_match": None, "co_registration": None,
                            "extent_overlap": None, "reprojection_advised": False,
                            "resampling_advised": False}

    if primary.crs and secondary.crs:
        crs_match = str(primary.crs).upper() == str(secondary.crs).upper()
        info["crs_match"] = crs_match
        if crs_match:
            info.setdefault("crs", primary.crs)
        elif require_same_crs:
            problems.append(
                f"CRS mismatch: '{primary.name}' is {primary.crs}, "
                f"'{secondary.name}' is {secondary.crs}. Reproject both images to a common CRS and retry."
            )
            info["reprojection_advised"] = True
        else:
            warnings.append(
                f"CRS differs between images ({primary.crs} vs {secondary.crs}); "
                "proceeding without reprojection because the pair is cross-modal."
            )
            info["reprojection_advised"] = True
    else:
        missing = [m.name for m in (primary, secondary) if not m.crs]
        problems.append(
            "Missing CRS metadata on " + ", ".join(missing) +
            ". Pair analysis requires georeferenced GeoTIFF inputs."
        )

    if (primary.width, primary.height) != (secondary.width, secondary.height):
        problems.append(
            f"Image dimensions mismatch: {primary.width}x{primary.height} vs "
            f"{secondary.width}x{secondary.height}. Cannot perform pair analysis."
        )
        info["dimensions_match"] = False
    else:
        info["dimensions_match"] = True

    if primary.resolution and secondary.resolution:
        if primary.crs and secondary.crs and str(primary.crs) == str(secondary.crs):
            ratio = abs(primary.resolution[0]) / max(1e-12, abs(secondary.resolution[0]))
            res_match = 0.5 <= ratio <= 2.0
            info["resolution_match"] = res_match
            if not res_match:
                warnings.append(
                    f"Resolution differs significantly ({primary.resolution[0]:.4g} vs "
                    f"{secondary.resolution[0]:.4g}). Resampling to the coarser resolution is advised."
                )
                info["resampling_advised"] = True
            elif abs(ratio - 1.0) > 0.01:
                warnings.append(
                    f"Minor resolution difference ({primary.resolution[0]:.4g} vs "
                    f"{secondary.resolution[0]:.4g}); auto-resampling applied."
                )

    if primary.bounds and secondary.bounds and str(primary.crs) == str(secondary.crs):
        overlap = _bounds_overlap(primary.bounds, secondary.bounds)
        info["extent_overlap"] = round(overlap, 4)
        if overlap < EXTENT_OVERLAP_ERROR:
            problems.append(
                f"Spatial extents barely overlap ({overlap:.0%}). Verify the two images cover the same area."
            )
        elif overlap < EXTENT_OVERLAP_WARN:
            warnings.append(f"Partial extent overlap ({overlap:.0%}); change statistics may be biased.")

    if primary.transform and secondary.transform and str(primary.crs) == str(secondary.crs) \
            and primary.resolution and secondary.resolution:
        try:
            dx = abs(primary.transform[2] - secondary.transform[2])
            dy = abs(primary.transform[5] - secondary.transform[5])
            px = max(abs(primary.resolution[0]), abs(primary.resolution[1]), 1e-9)
            offset_px = max(dx, dy) / px
            info["co_registration"] = {"offset_px": round(offset_px, 3),
                                       "tolerance_px": tolerance_px}
            if offset_px > tolerance_px:
                warnings.append(
                    f"Geotransform offset {offset_px:.2f}px exceeds {tolerance_px}px tolerance; "
                    "images may not be perfectly co-registered."
                )
        except Exception as exc:
            warnings.append(f"Could not compare geotransforms ({exc}).")

    return {"problems": problems, "warnings": warnings, "info": info,
            "valid": len(problems) == 0}


def reproject_to(img: RasterImage, target_crs: str = DEFAULT_TARGET_CRS) -> RasterImage:
    """Reproject a georeferenced image to ``target_crs`` (requires rasterio)."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is not installed; reprojection is unavailable.")
    if not img.metadata.crs or not img.metadata.transform:
        raise ValueError(f"Image '{img.metadata.name}' is not georeferenced; cannot reproject.")
    from rasterio.warp import Resampling, calculate_default_transform, reproject as rio_reproject  # type: ignore
    from rasterio.transform import Affine  # type: ignore

    src_t = Affine(*img.metadata.transform[:6])
    h, w = img.height, img.width
    left, top = src_t.c, src_t.f
    right = left + src_t.a * w
    bottom = top + src_t.e * h
    transform, out_w, out_h = calculate_default_transform(
        img.metadata.crs, target_crs, w, h, left, bottom, right, top
    )
    bands = img.bands
    dst = np.zeros((bands, out_h, out_w), dtype=np.float32)
    src = np.moveaxis(img.data, -1, 0)
    for i in range(bands):
        rio_reproject(
            source=src[i], destination=dst[i],
            src_transform=src_t, src_crs=img.metadata.crs,
            dst_transform=transform, dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    data = np.moveaxis(dst, 0, -1)
    data = np.dstack([_normalise_channel(data[:, :, i]) for i in range(data.shape[2])])
    t = transform
    meta = ImageMetadata(
        name=img.metadata.name, path=img.metadata.path, fmt=img.metadata.fmt,
        width=out_w, height=out_h, bands=bands, crs=target_crs, crs_valid=True,
        transform=[t.a, t.b, t.c, t.d, t.e, t.f],
        resolution=(float(t.a), float(abs(t.e))),
        bounds=(float(t.c), float(t.f + t.e * out_h), float(t.c + t.a * out_w), float(t.f)),
        modality=img.metadata.modality,
        modality_confidence=img.metadata.modality_confidence,
        size_mb=img.metadata.size_mb,
    )
    preview = _resize_preview(_rgb_preview(data), PREVIEW_MAX_SIDE)
    return RasterImage(metadata=meta, data=data, preview=preview,
                       reprojected_from=img.metadata.crs)


# ------------------------------------------------------------ spectral tools --
def compute_ndvi(img: RasterImage) -> np.ndarray:
    """NDVI proxy.

    ``>=4 bands`` (package order R,G,B,NIR): true (NIR-Red)/(NIR+Red).
    ``3 bands`` (true colour): (Green-Blue)/ as a vegetation-vs-water proxy.
    ``<=2 bands``: (B0-B1)/ difference of the first two bands.
    """
    d = img.data
    if d.shape[2] >= 4:
        red, nir = d[:, :, 0], d[:, :, 3]
        den = nir + red
        return np.where(den > 1e-6, (nir - red) / np.maximum(den, 1e-6), 0.0)
    if d.shape[2] == 3:
        green, blue = d[:, :, 1], d[:, :, 2]
        den = green + blue
        return np.where(den > 1e-6, (green - blue) / np.maximum(den, 1e-6), 0.0)
    a = d[:, :, 0]
    b = d[:, :, 1] if d.shape[2] > 1 else a * 0.5
    den = a + b
    return np.where(den > 1e-6, (a - b) / np.maximum(den, 1e-6), 0.0)


def compute_ndwi(img: RasterImage) -> np.ndarray:
    """NDWI proxy.

    ``>=4 bands`` (package order R,G,B,NIR): (Green-NIR)/ (McFeeters).
    ``3 bands``: (Blue-Green)/ since
    water is brighter in blue than vegetation in true colour. ``<=2 bands``:
    negative of the NDVI proxy so water remains the positive class.
    """
    d = img.data
    if d.shape[2] >= 4:
        green, nir = d[:, :, 1], d[:, :, 3]
        den = green + nir
        return np.where(den > 1e-6, (green - nir) / np.maximum(den, 1e-6), 0.0)
    if d.shape[2] == 3:
        blue, green = d[:, :, 2], d[:, :, 1]
        den = blue + green
        return np.where(den > 1e-6, (blue - green) / np.maximum(den, 1e-6), 0.0)
    return -compute_ndvi(img)


def compute_ndbi(img: RasterImage) -> np.ndarray:
    """NDBI proxy for built-up surfaces.

    Built-up spectra are flat/bright: reflectance in the redder band exceeds
    the NIR (or green) band, so (Red-NIR)/ ``>=4 bands`` (package order
    R,G,B,NIR) and (Red-Green)/ for true colour; vegetation and water go negative.
    """
    d = img.data
    if d.shape[2] >= 4:
        red, nir = d[:, :, 0], d[:, :, 3]
    elif d.shape[2] == 3:
        red, nir = d[:, :, 0], d[:, :, 1]
    else:
        red, nir = d[:, :, 0], d[:, :, 0] * 0.9
    den = red + nir
    return np.where(den > 1e-6, (red - nir) / np.maximum(den, 1e-6), 0.0)


def _smooth(mask: np.ndarray, kernel: int = 3) -> np.ndarray:
    if kernel <= 1 or mask.size == 0:
        return mask
    k = np.ones((kernel, kernel), dtype=np.float32) / (kernel * kernel)
    from numpy.lib.stride_tricks import sliding_window_view

    pad = kernel // 2
    padded = np.pad(mask, pad, mode="edge")
    windows = sliding_window_view(padded, (kernel, kernel))
    return np.einsum("ijkl,kl->ij", windows, k)


def cloud_fraction(img: RasterImage) -> float:
    """Estimate cloudy fraction from bright, low-saturation pixels."""
    if img.bands < 3:
        return 0.0
    rgb = img.data[:, :, :3]
    bright = rgb.mean(axis=2)
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    sat = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    cloudy = (bright > CLOUD_BRIGHTNESS_THRESHOLD) & (sat < CLOUD_SATURATION_THRESHOLD)
    return float(cloudy.mean())


# ------------------------------------------------- land-cover / SAR analysis --
def classify_landcover(img: RasterImage, water_thr: float = WATER_NDWI_THRESHOLD,
                       veg_thr: float = VEGETATION_NDVI_THRESHOLD,
                       built_thr: float = BUILTUP_NDBI_THRESHOLD,
                       soil_thr: float = BARESOIL_BRIGHTNESS_THRESHOLD) -> np.ndarray:
    """Rule-based land-cover segmentation on spectral indices (optical)."""
    if img.bands < 3:
        raise ValueError("Land-cover classification needs an optical/multispectral image (>=3 bands).")
    ndwi = compute_ndwi(img)
    ndvi = compute_ndvi(img)
    ndbi = compute_ndbi(img)
    brightness = img.data[:, :, :3].mean(axis=2)
    h, w = ndvi.shape
    out = np.zeros((h, w), dtype=np.uint8)
    classes = {name: i for i, name in enumerate(LANDCOVER_CLASS_NAMES)}
    out[(ndwi > water_thr) & (ndvi < veg_thr)] = classes["water"]
    veg = (ndvi > veg_thr) & (out == 0)
    out[veg] = classes["vegetation"]
    built = (ndbi > built_thr) & (ndvi <= veg_thr) & (out == 0)
    out[built] = classes["built_up"]
    soil = (brightness > soil_thr) & (ndvi < veg_thr) & (out == 0)
    out[soil] = classes["bare_soil"]
    return _smooth(out.astype(np.float32), 3).round().astype(np.uint8)


def classify_sar_structure(img: RasterImage) -> np.ndarray:
    """Segment SAR imagery into water / built-up / other using backscatter + texture."""
    band = img.data[:, :, 0]
    texture = np.abs(np.diff(band, axis=0, prepend=band[:1, :]))
    water = band < SAR_WATER_BACKSCATTER_MAX
    built = (band > SAR_BUILTUP_BACKSCATTER_MIN) & (texture > SAR_BUILTUP_TEXTURE_MIN)
    out = np.zeros(band.shape, dtype=np.uint8)
    out[built] = 3
    veg = (~water) & (~built)
    out[veg] = 2
    out[water] = 1
    return _smooth(out.astype(np.float32), 3).round().astype(np.uint8)


def sar_texture(img: RasterImage) -> np.ndarray:
    """Local SAR texture proxy (gradient magnitude of the first band)."""
    band = img.data[:, :, 0]
    gy, gx = np.gradient(band)
    return np.sqrt(gx * gx + gy * gy)


def sar_backscatter_stats(img: RasterImage) -> Dict[str, float]:
    band = img.data[:, :, 0]
    return {
        "mean": round(float(band.mean()), 4),
        "std": round(float(band.std()), 4),
        "p98": round(float(np.percentile(band, 98)), 4),
    }


def hls_to_rgb(h: float, l: float, s: float) -> Tuple[float, float, float]:
    return colorsys.hls_to_rgb(h, l, s)


def colorize_classes(classes: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((*classes.shape, 3), dtype=np.uint8)
    for cls, color in palette.items():
        out[classes == cls] = color
    return out


def colorize_change(change_map: np.ndarray, palette: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    h, w = change_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in palette.items():
        out[change_map == label] = color
    return out


def connected_regions(mask: np.ndarray, min_pixels: int, top_k: int = 8) -> List[Dict[str, Any]]:
    """Simple 4-connected component labelling returning bounding boxes per region."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and labels[r, c] == 0:
                current += 1
                stack = [(r, c)]
                labels[r, c] = current
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    regions: List[Dict[str, Any]] = []
    for lab in range(1, current + 1):
        ys, xs = np.where(labels == lab)
        if ys.size < min_pixels:
            continue
        regions.append({
            "label": int(lab),
            "pixels": int(ys.size),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "centroid": [float(xs.mean()), float(ys.mean())],
        })
    regions.sort(key=lambda rg: rg["pixels"], reverse=True)
    return regions[:top_k]


def fraction_of_classes(classes: np.ndarray) -> Dict[str, float]:
    total = classes.size
    return {
        name: round(float(np.sum(classes == idx)) / total, 4)
        for idx, name in enumerate(LANDCOVER_CLASS_NAMES)
    }


_SAR_NAME_HINTS = ("sar", "vv", "vh", "s1a", "s1b", "sentinel-1", "sentinel_1",
                   "risat", "ers", "asarg", "slc", "grd")


def _sar_name_hint(meta: Optional[ImageMetadata]) -> bool:
    """Sensor hints from file names (e.g. RISAT/Sentinel-1/S1) - a soft prior
    that is combined with pixel statistics by :func:`infer_modality`."""
    if meta is None:
        return False
    name = (meta.name or "").lower()
    if "optical" in name or "_s2" in name or "sentinel-2" in name:
        return False
    return any(hint in name for hint in _SAR_NAME_HINTS)


def quadrant_name(centroid: Tuple[float, float], width: float, height: float) -> str:
    """Human-readable quadrant (e.g. 'north-west') for a pixel centroid."""
    cx, cy = centroid
    ns = "north" if cy < height / 2.0 else "south"
    ew = "west" if cx < width / 2.0 else "east"
    return f"{ns}-{ew}"
