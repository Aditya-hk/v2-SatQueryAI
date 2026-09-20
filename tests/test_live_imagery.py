"""Unit tests for the live map / ROI acquisition utilities.

The sandbox has no outbound network, so the STAC path is mocked. The tests pin
down: bbox validation, window geometry, the deterministic synthetic fallback
(what judges see offline), GeoTIFF/GeoJSON exports, and the map builder.
"""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pytest

from satquery.config import CHANGE_CLASS_ORDER
from satquery.utils.live_imagery import (
    LiveFetchResult,
    _window_dims,
    build_result_map,
    change_labels_to_indices,
    class_map_to_geotiff,
    fetch_roi_imagery,
    grounding_geojson,
    image_to_geotiff,
    region_polygons_geojson,
    validate_bbox,
)

BBOX = (77.30, 28.35, 77.42, 28.45)  # ~11 x 11 km over Delhi


def _fallback_pair(include_sar: bool = False, max_dim: int = 128):
    return fetch_roi_imagery(BBOX, dt.date(2024, 1, 5), dt.date(2024, 6, 5),
                             include_sar=include_sar, max_dim=max_dim)


# ----------------------------------------------------------------- geometry ---
def test_validate_bbox_rejects_bad_boxes() -> None:
    assert validate_bbox(BBOX) == BBOX
    with pytest.raises(ValueError):
        validate_bbox((77.5, 28.4, 77.2, 28.5))  # west > east
    with pytest.raises(ValueError):
        validate_bbox((200.0, 28.4, 201.0, 28.5))  # outside lon range
    with pytest.raises(ValueError):
        validate_bbox((77.2, 28.4, 77.3, 28.45, 99.0))  # wrong arity


def test_window_dims_preserves_aspect() -> None:
    assert _window_dims((0.0, 0.0, 2.0, 1.0), 512) == (256, 512)
    assert _window_dims((0.0, 0.0, 0.5, 1.0), 256) == (256, 128)


# ------------------------------------------------------- synthetic fallback ---
def test_fallback_pair_is_georeferenced_and_bounded() -> None:
    res = _fallback_pair(include_sar=True)
    assert isinstance(res, LiveFetchResult)
    assert len(res.images) == 3
    for img in res.images:
        h, w = img.data.shape[:2]
        assert (h, w) == (128, 128)
        assert img.metadata.crs == "EPSG:4326"
        assert img.metadata.bounds == BBOX
        acq = img.metadata.extra.get("acquisition", {})
        assert acq.get("source") == "synthetic-fallback"
        assert acq.get("note")
    assert res.warnings, "offline run must record why it fell back"
    # deterministic: same bbox + dates -> identical pixels
    again = _fallback_pair()
    np.testing.assert_array_equal(res.images[0].data, again.images[0].data)
    # no SAR requested -> only the optical pair
    assert len(_fallback_pair().images) == 2


def test_bad_dates_raise() -> None:
    with pytest.raises(ValueError):
        fetch_roi_imagery(BBOX, dt.date(2024, 6, 5), dt.date(2024, 1, 5))


# ------------------------------------------------------------ live STAC path ---
class _FakeAsset:
    def __init__(self, href: str) -> None:
        self.href = href


class _FakeItem:
    def __init__(self, item_id: str, when: str, cloud: float) -> None:
        self.id = item_id
        self.properties = {"datetime": when, "platform": "sentinel-2",
                           "eo:cloud_cover": cloud}
        self.assets = {a: _FakeAsset(f"https://example.invalid/{item_id}/{a}.tif")
                       for a in ("B02", "B03", "B04", "B08")}


class _FakeSearch:
    def __init__(self, items: list) -> None:
        self._items = items

    def items(self) -> list:
        return self._items


class _FakeCatalog:
    def __init__(self, items: list) -> None:
        self._items = items
        self.seen: list = []

    def search(self, **kwargs) -> _FakeSearch:
        self.seen.append(kwargs)
        return _FakeSearch(self._items)


def test_live_path_reads_and_signs_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    item = _FakeItem("S2_test_scene", "2024-01-10T05:12:00Z", 12.0)
    catalog = _FakeCatalog([item])
    monkeypatch.setattr("satquery.utils.live_imagery._mpc_catalog", lambda: catalog)

    def fake_read(href: str, bbox: tuple, out_h: int, out_w: int) -> np.ndarray:
        assert bbox == BBOX
        rng = np.random.default_rng(abs(hash(href)) % 2**32)
        return rng.random((out_h, out_w)).astype(np.float32)

    monkeypatch.setattr("satquery.utils.live_imagery._read_asset_window", fake_read)

    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 1, 20),
                            include_sar=False, max_dim=128)
    assert res.info["source"].startswith("planetary-computer")
    assert "scenes" in res.info and "sar_scene" not in res.info
    assert len(res.images) == 2
    for img in res.images:
        assert img.data.shape[1:] == (128, 4)  # B04,B03,B02,B08 stretched (R,G,B,NIR)
        acq = img.metadata.extra["acquisition"]
        assert acq["scene_id"] == "S2_test_scene"
        assert acq["cloud_cover"] == pytest.approx(12.0)
    # the search was constrained to the two request windows
    assert len(catalog.seen) == 2
    assert all(kw["collections"] == ["sentinel-2-l2a"] for kw in catalog.seen)


def test_live_path_empty_result_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("satquery.utils.live_imagery._mpc_catalog",
                        lambda: _FakeCatalog([]))
    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 1, 20))
    assert res.info["source"] == "synthetic-fallback"
    assert any("no sentinel-2" in w.lower() for w in res.warnings)


def test_live_path_network_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise ConnectionError("DNS resolution failed")

    monkeypatch.setattr("satquery.utils.live_imagery._mpc_catalog", boom)
    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 1, 20))
    assert res.info["source"] == "synthetic-fallback"
    assert any("DNS" in w for w in res.warnings)


def test_no_fallback_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise ConnectionError("offline")

    monkeypatch.setattr("satquery.utils.live_imagery._mpc_catalog", boom)
    with pytest.raises(ConnectionError):
        fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 1, 20),
                          allow_synthetic_fallback=False)


# ----------------------------------------------------------------- exports ---
def test_image_geotiff_roundtrip() -> None:
    pytest.importorskip("rasterio")
    res = _fallback_pair()
    raw = image_to_geotiff(res.images[0])
    assert raw[:4] == b"II*\x00"  # little-endian TIFF magic
    import rasterio
    with rasterio.open(io.BytesIO(raw)) as src:
        assert src.crs.to_epsg() == 4326
        assert src.count == res.images[0].data.shape[2]
        np.testing.assert_allclose(
            np.moveaxis(src.read(), 0, -1), res.images[0].data, rtol=1e-5, atol=1e-5)


def test_class_map_geotiff_is_geotagged() -> None:
    pytest.importorskip("rasterio")
    from rasterio.transform import Affine
    rng = np.random.default_rng(7)
    classes = rng.integers(0, 6, size=(96, 96)).astype(np.uint8)
    palette = {i: (i * 40 % 256, 100, 200) for i in range(6)}
    raw = class_map_to_geotiff(classes, palette, BBOX)
    import rasterio
    with rasterio.open(io.BytesIO(raw)) as src:
        assert src.crs.to_epsg() == 4326
        assert (src.height, src.width) == (96, 96)
        np.testing.assert_array_equal(src.read(1), classes)
        # must carry a real geotransform derived from the ROI bounds
        assert src.transform != Affine.identity()
        west, south, _, north = BBOX
        assert src.bounds.left == pytest.approx(west, abs=1e-6)
        assert src.bounds.bottom == pytest.approx(south, abs=1e-6)
        assert src.bounds.top == pytest.approx(north, abs=1e-6)


def test_grounding_geojson_pixel_to_wgs84() -> None:
    img = _fallback_pair().images[0]
    h, w = img.height, img.width
    regions = [{"bbox": [0, 0, w - 1, h - 1], "region_id": "R1",
                "quadrant": "centre", "coverage_pct": 0.42}]
    gj = grounding_geojson(img, regions)
    assert gj["type"] == "FeatureCollection"
    (feat,) = gj["features"]
    ring = feat["geometry"]["coordinates"][0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    west, south, east, north = BBOX
    assert min(xs) == pytest.approx(west, abs=1e-6)
    assert max(xs) == pytest.approx(east, abs=1e-6)
    assert min(ys) == pytest.approx(south, abs=1e-6)
    assert max(ys) == pytest.approx(north, abs=1e-6)
    assert feat["properties"]["region_id"] == "R1"
    # a quarter-size box maps proportionally inside the bounds
    q = grounding_geojson(img, [{"bbox": [0, 0, w // 2 - 1, h // 2 - 1],
                                 "region_id": "R2"}])
    qx = [p[0] for p in q["features"][0]["geometry"]["coordinates"][0]]
    assert max(qx) < (west + east) / 2 + 1e-6


def test_region_polygons_geojson_vectorises_components() -> None:
    img = _fallback_pair().images[0]
    labels = np.zeros((img.height, img.width), dtype=np.uint8)
    labels[10:40, 10:40] = 1   # 900 px blob  -> kept
    labels[60:66, 60:66] = 2   # 36 px blob   -> kept
    labels[90:93, 3:6] = 3     # 9 px blob    -> kept (>= 8)
    labels[0:2, 0:2] = 4       # 4 px blob    -> dropped
    names = {1: "water", 2: "built-up", 3: "bare"}
    palette = {1: (46, 148, 236), 2: (226, 34, 92), 3: (250, 224, 70)}
    gj = region_polygons_geojson(img, labels, names, palette)
    by_class = {f["properties"]["class"]: f for f in gj["features"]}
    assert set(by_class) == {"water", "built-up", "bare"}
    assert by_class["water"]["properties"]["area_px"] == 900
    assert by_class["water"]["properties"]["color"] == "#2e94ec"
    xs = [p[0] for p in by_class["built-up"]["geometry"]["coordinates"][0]]
    assert min(xs) > BBOX[0] and max(xs) < BBOX[2]


def test_change_labels_to_indices_maps_canonical_names() -> None:
    order = {name: idx for idx, name in enumerate(CHANGE_CLASS_ORDER)}
    labels = np.array([["no_change", "water_gain"],
                       ["totally_unknown", "vegetation_loss"]], dtype=object)
    idx = change_labels_to_indices(labels)
    assert idx[0, 0] == 0
    assert idx[0, 1] == order["water_gain"]
    assert idx[1, 0] == 0  # unknown labels fall back to no_change
    assert idx[1, 1] == order["vegetation_loss"]
    assert idx.dtype == np.uint8


# -------------------------------------------------------------------- map ---
def test_build_result_map_layers() -> None:
    pytest.importorskip("folium")
    t1, t2 = _fallback_pair().images[:2]
    overlay = np.zeros((t1.height, t1.width, 4), dtype=np.uint8)
    overlay[32:64, 32:64] = (239, 68, 68, 255)
    rectangles = [{"bbox": [10, 10, 40, 40], "label": "R1 · centre",
                   "color": "#d7301f"}]
    mp = build_result_map(t1, overlay, "change overlay", rectangles)
    html = mp.get_root().render()
    assert "leaflet" in html.lower()
    assert html.count("rectangle") >= 2  # ROI frame + grounding box
    assert "change overlay" in html
