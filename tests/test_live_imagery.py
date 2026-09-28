"""Unit tests for the live map / ROI acquisition utilities.

The suite is hermetic: an autouse fixture makes the STAC catalog unreachable,
and tests that exercise the live path install their own window-aware fake
catalog (patching after the fixture runs). The tests pin down: bbox validation,
window geometry, search escalation (widen / relax cloud / force a bi-temporal
pair), the deterministic synthetic fallback, GeoTIFF/GeoJSON exports, and the
map builder.
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


@pytest.fixture(autouse=True)
def _offline_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite hermetic — no real STAC traffic.

    Tests that want the live path patch the catalog themselves; their
    ``monkeypatch.setattr`` runs after this fixture and therefore wins.
    """
    def offline() -> None:
        raise ConnectionError("offline test sandbox")

    monkeypatch.setattr("satquery.utils.live_imagery._mpc_catalog", offline)


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
    """STAC stand-in that honours bbox/cloud filters and the datetime window."""

    def __init__(self, items: list) -> None:
        self._items = items
        self.seen: list = []

    def search(self, **kwargs) -> _FakeSearch:
        self.seen.append(kwargs)
        start_s, _, end_s = str(kwargs.get("datetime", "")).partition("/")
        lo = dt.date.fromisoformat(start_s) if start_s else dt.date.min
        hi = dt.date.fromisoformat(end_s) if end_s else dt.date.max
        query = kwargs.get("query") or {}
        cap = None
        if "eo:cloud_cover" in query:
            cap = float(query["eo:cloud_cover"].get("lt", 100.0))
        hits = []
        for item in self._items:
            day = dt.date.fromisoformat(str(item.properties["datetime"])[:10])
            if not (lo <= day <= hi):
                continue
            clouds = item.properties.get("eo:cloud_cover")
            if cap is not None and clouds is not None and float(clouds) >= cap:
                continue
            hits.append(item)
        return _FakeSearch(hits)


def _fake_read(href: str, bbox: tuple, out_h: int, out_w: int) -> np.ndarray:
    assert bbox == BBOX
    rng = np.random.default_rng(abs(hash(href)) % 2**32)
    return rng.random((out_h, out_w)).astype(np.float32)


def _install(monkeypatch: pytest.MonkeyPatch, items: list) -> _FakeCatalog:
    catalog = _FakeCatalog(items)
    monkeypatch.setattr("satquery.utils.live_imagery._mpc_catalog", lambda: catalog)
    monkeypatch.setattr("satquery.utils.live_imagery._read_asset_window", _fake_read)
    return catalog


def _acq(img) -> dict:
    return img.metadata.extra["acquisition"]


def test_live_path_reads_and_signs_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _install(monkeypatch, [
        _FakeItem("S2_jan_scene", "2024-01-03T05:12:00Z", 12.0),
        _FakeItem("S2_sep_scene", "2024-09-18T05:12:00Z", 4.0),
    ])

    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 9, 20),
                            include_sar=False, max_dim=128)
    assert res.info["source"].startswith("planetary-computer")
    assert "scenes" in res.info and "sar_scene" not in res.info
    assert len(res.images) == 2
    # one search per epoch: the requested ±7-day windows both had a scene, so
    # nothing had to escalate
    assert len(catalog.seen) == 2
    assert [kw["collections"] for kw in catalog.seen] == [["sentinel-2-l2a"]] * 2
    assert [str(_acq(img)["scene_id"]) for img in res.images] == ["S2_jan_scene", "S2_sep_scene"]
    for img in res.images:
        assert img.data.shape[1:] == (128, 4)  # B04,B03,B02,B08 stretched (R,G,B,NIR)
        assert _acq(img)["widened"] is False
        assert _acq(img)["cloud_filter"] == pytest.approx(30.0)
    assert _acq(res.images[0])["cloud_cover"] == pytest.approx(12.0)


def test_search_widens_when_the_requested_window_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old code gave up here and served a synthetic scene."""
    catalog = _install(monkeypatch, [
        _FakeItem("S2_jan_scene", "2024-01-03T05:12:00Z", 5.0),
        _FakeItem("S2_distant", "2024-03-01T05:12:00Z", 3.0),
    ])
    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 4, 25),
                            include_sar=False, max_dim=128)
    assert res.info["source"].startswith("planetary-computer")
    t2 = _acq(res.images[1])
    assert t2["scene_id"] == "S2_distant"
    assert t2["widened"] is True
    assert t2["days_from_target"] > 7
    assert any("no usable scene within" in w for w in res.warnings), res.warnings
    assert catalog.seen, "the search must have been attempted"


def test_cloud_threshold_is_relaxed_before_widening(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _install(monkeypatch, [
        _FakeItem("S2_jan_scene", "2024-01-03T05:12:00Z", 9.0),
        _FakeItem("S2_cloudy", "2024-09-18T05:12:00Z", 88.0),
    ])
    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 9, 20),
                            include_sar=False, max_dim=128)
    assert res.info["source"].startswith("planetary-computer")
    t2 = _acq(res.images[1])
    assert t2["scene_id"] == "S2_cloudy"
    assert t2["widened"] is False                      # same ±7-day window
    assert t2["cloud_filter"] is None                   # cloud cap dropped
    assert t2["cloud_cover"] == pytest.approx(88.0)
    assert any("cloud threshold relaxed" in w for w in res.warnings), res.warnings
    assert any("accepted 88% cloud cover" in w for w in res.warnings), res.warnings


def test_bi_temporal_pair_is_forced_onto_distinct_scenes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both epochs score the same scene highest; T2 must be re-picked.

    ``S2_clear`` is clear enough to outrank the nearer but 88%-cloudy
    ``S2_cloudy`` for *both* target dates, so the pair would collapse onto one
    acquisition without the distinct-scene guard.
    """
    _install(monkeypatch, [
        _FakeItem("S2_clear", "2024-03-15T05:12:00Z", 5.0),
        _FakeItem("S2_cloudy", "2024-03-20T05:12:00Z", 88.0),
    ])
    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 6, 1),
                            include_sar=False, max_dim=128)
    assert res.info["source"].startswith("planetary-computer")
    ids = [str(_acq(img)["scene_id"]) for img in res.images]
    assert len(set(ids)) == 2, ids
    assert any("re-picked" in w for w in res.warnings), res.warnings


def test_single_scene_roi_warns_but_still_returns_a_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, [_FakeItem("S2_only", "2024-04-01T05:12:00Z", 6.0)])
    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 6, 1),
                            include_sar=False, max_dim=128)
    assert res.info["source"].startswith("planetary-computer")
    assert len(res.images) == 2
    assert any("same scene" in w for w in res.warnings), res.warnings


def test_sar_failure_degrades_to_the_optical_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, [
        _FakeItem("S2_jan_scene", "2024-01-03T05:12:00Z", 9.0),
        _FakeItem("S2_sep_scene", "2024-09-18T05:12:00Z", 4.0),
    ])

    def no_sar(*args, **kwargs):
        raise RuntimeError("VV asset missing from scene S1_test")

    monkeypatch.setattr("satquery.utils.live_imagery._fetch_sar_live", no_sar)
    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 9, 20),
                            include_sar=True, max_dim=128)
    assert res.info["source"].startswith("planetary-computer"), res.warnings
    assert len(res.images) == 2, "the optical pair must survive a SAR failure"
    assert any("SAR acquisition failed" in w for w in res.warnings), res.warnings


def test_future_window_is_rejected_with_a_clear_message() -> None:
    start = dt.date.today() + dt.timedelta(days=5)
    with pytest.raises(ValueError, match="future"):
        fetch_roi_imagery(BBOX, start, start + dt.timedelta(days=10))


def test_future_end_date_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, [
        _FakeItem("S2_jan_scene", "2024-01-03T05:12:00Z", 9.0),
        _FakeItem("S2_recent", f"{dt.date.today().isoformat()}T05:12:00Z", 4.0),
    ])
    today = dt.date.today()
    res = fetch_roi_imagery(BBOX, today - dt.timedelta(days=60),
                            today + dt.timedelta(days=30), include_sar=False,
                            max_dim=128)
    assert any("clamped" in w for w in res.warnings), res.warnings
    assert res.info["end"] == today.isoformat()


def test_empty_catalog_escalates_fully_then_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _install(monkeypatch, [])
    res = fetch_roi_imagery(BBOX, dt.date(2024, 1, 1), dt.date(2024, 1, 20))
    assert res.info["source"] == "synthetic-fallback"
    # both epochs escalate through every step before the synthetic fallback
    assert len(catalog.seen) == 12, [kw["datetime"] for kw in catalog.seen]
    widths = [str(kw["datetime"]) for kw in catalog.seen]
    assert widths[0] != widths[-1], "windows must progressively widen"
    assert any("query" not in kw for kw in catalog.seen), \
        "the cloud threshold must be dropped at some point"
    assert res.info.get("fallback_reason"), "the reason must reach the audit trail"
    assert any("±365" in w for w in res.warnings), res.warnings


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
