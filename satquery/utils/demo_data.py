"""Synthetic, georeferenced demo scenes for SatQuery AI.

Generates deterministic optical/multispectral, SAR, bi-temporal and
cross-modal demo imagery (with real EPSG:4326 GeoTIFF metadata when rasterio
is available) so the app can be demonstrated end-to-end without downloading
any dataset. The scene layout (a coastal city with farms, a lake and bare
soil) is rendered from analytic masks, and the bi-temporal pair applies
controlled change: urban growth, vegetation loss and shoreline retreat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from satquery.config import CACHE_DIR, PREVIEW_MAX_SIDE
from satquery.utils.geospatial import (
    HAS_PIL,
    HAS_RASTERIO,
    ImageMetadata,
    RasterImage,
    _normalise_channel,
    _resize_preview,
    infer_modality,
)

DEMO_DIR = Path(CACHE_DIR) / "demo"
SCENE_SIZE = 256
PIXEL_SIZE_DEG = 0.00012  # ~13 m at the reference latitude
ORIGIN_LON, ORIGIN_LAT = 77.10, 20.10

_CLASS_COLORS: Dict[int, Tuple[float, float, float]] = {
    0: (0.55, 0.58, 0.50),   # other / mixed
    1: (0.16, 0.30, 0.52),   # water
    2: (0.24, 0.50, 0.26),   # vegetation
    3: (0.60, 0.60, 0.62),   # built-up
    4: (0.70, 0.60, 0.42),   # bare soil
}
_SAR_BACKSCATTER: Dict[int, float] = {0: 0.30, 1: 0.04, 2: 0.22, 3: 0.68, 4: 0.38}


@dataclass
class SceneLayout:
    """Binary class masks composing one synthetic scene."""

    water: np.ndarray
    vegetation: np.ndarray
    built_up: np.ndarray
    bare_soil: np.ndarray
    other: np.ndarray


def _smooth_noise(rng: np.random.Generator, h: int, w: int, seed_shift: float = 0.0) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    xn, yn = x / max(1, w - 1), y / max(1, h - 1)
    acc = np.zeros((h, w), dtype=np.float32)
    for octave in range(1, 4):
        for _ in range(2):
            fx = rng.uniform(0.5, 1.5) * octave
            fy = rng.uniform(0.5, 1.5) * octave
            phase = rng.uniform(0, 2 * np.pi) + seed_shift
            acc += np.sin(2 * np.pi * (fx * xn + fy * yn) + phase).astype(np.float32) / octave
    return acc


def build_layout(seed: int = 7, size: int = SCENE_SIZE, water_scale: float = 1.0,
                 urban_growth: float = 0.0, veg_loss: float = 0.0) -> SceneLayout:
    """Render class masks; ``urban_growth`` expands the city, ``veg_loss`` thins
    vegetation and ``water_scale`` shrinks the lake (bi-temporal change)."""
    rng = np.random.default_rng(seed)
    h = w = size
    y, x = np.mgrid[0:h, 0:w]
    xn, yn = x / max(1, w - 1), y / max(1, h - 1)

    water = (((xn - 0.26) / (0.21 * water_scale)) ** 2 + ((yn - 0.72) / (0.17 * water_scale)) ** 2) < 1.0

    noise = _smooth_noise(rng, h, w)
    vegetation = noise > (0.10 + veg_loss)

    left_edge = 0.58 - urban_growth
    city_region = (xn > left_edge) & (xn < 0.97) & (yn > 0.16) & (yn < 0.88)
    ix = np.floor(xn * 36).astype(int)
    iy = np.floor(yn * 36).astype(int)
    built_up = city_region & ((ix + iy) % 2 == 0)
    arterial = (np.abs(yn - 0.5) < 0.012) | (np.abs(xn - (left_edge - 0.03)) < 0.006)
    built_up = built_up | (city_region & arterial)

    soil_noise = _smooth_noise(rng, h, w, seed_shift=1.7)
    bare_soil = (yn < 0.07) | (soil_noise > 0.62)

    classes = np.zeros((h, w), dtype=np.uint8)
    classes[vegetation] = 2
    classes[bare_soil & (classes == 0)] = 4
    classes[water] = 1
    classes[built_up] = 3
    other = classes == 0
    return SceneLayout(water=classes == 1, vegetation=classes == 2,
                       built_up=classes == 3, bare_soil=classes == 4, other=other)


def _render_optical(layout: SceneLayout, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 101)
    h, w = layout.water.shape
    classes = np.zeros((h, w), dtype=np.uint8)
    classes[layout.bare_soil] = 4
    classes[layout.vegetation] = 2
    classes[layout.water] = 1
    classes[layout.built_up] = 3

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in _CLASS_COLORS.items():
        mask = classes == cls
        shade = 0.92 + 0.16 * _smooth_noise(rng, h, w, seed_shift=cls)
        for channel in range(3):
            rgb[:, :, channel][mask] = color[channel] * shade[mask]
    rgb += rng.normal(0.0, 0.02, size=rgb.shape).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)


def _render_sar(layout: SceneLayout, seed: int, two_band: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed + 202)
    h, w = layout.water.shape
    classes = np.zeros((h, w), dtype=np.uint8)
    classes[layout.bare_soil] = 4
    classes[layout.vegetation] = 2
    classes[layout.water] = 1
    classes[layout.built_up] = 3

    base = np.vectorize(_SAR_BACKSCATTER.get)(classes).astype(np.float32)
    speckle = rng.gamma(shape=3.5, scale=0.30, size=(h, w)).astype(np.float32)
    amp = np.clip(base * speckle, 0.0, 1.0)
    glint = (classes == 3) & (rng.random((h, w)) < 0.06)
    amp[glint] = np.clip(amp[glint] + 0.30, 0.0, 1.0)
    vv = amp[:, :, np.newaxis]
    if two_band:
        vh = np.clip(0.55 * amp, 0.0, 1.0)[:, :, np.newaxis]
        return np.concatenate([vv, vh], axis=2).astype(np.float32)
    return vv.astype(np.float32)


def _build_metadata(name: str, data: np.ndarray) -> ImageMetadata:
    h, w = data.shape[:2]
    bands = data.shape[2] if data.ndim == 3 else 1
    px = PIXEL_SIZE_DEG
    transform = [px, 0.0, ORIGIN_LON, 0.0, -px, ORIGIN_LAT]
    meta = ImageMetadata(
        name=name, path=DEMO_DIR / name, fmt="GeoTIFF" if HAS_RASTERIO else "TIFF",
        width=w, height=h, bands=bands, crs="EPSG:4326", crs_valid=True,
        transform=transform, resolution=(px, px),
        bounds=(ORIGIN_LON, ORIGIN_LAT - h * px, ORIGIN_LON + w * px, ORIGIN_LAT),
        size_mb=round(data.nbytes / (1024 * 1024), 2),
    )
    modality, conf = infer_modality(data, meta)
    meta.modality, meta.modality_confidence = modality, conf
    return meta


def make_raster(name: str, data: np.ndarray) -> RasterImage:
    meta = _build_metadata(name, data)
    from satquery.utils.geospatial import _rgb_preview

    preview = _rgb_preview(data)
    if preview is not None:
        preview = _resize_preview(preview, PREVIEW_MAX_SIDE)
    return RasterImage(metadata=meta, data=data.astype(np.float32), preview=preview)


def save_raster(img: RasterImage, path: Path) -> Path:
    """Persist a demo raster as GeoTIFF (rasterio) or plain TIFF (PIL fallback)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = img.data
    if HAS_RASTERIO:
        import rasterio
        from rasterio.transform import Affine

        transform = Affine(*img.metadata.transform[:6])
        h, w, c = data.shape
        with rasterio.open(
            path, "w", driver="GTiff", height=h, width=w, count=c,
            dtype="float32", crs=img.metadata.crs, transform=transform,
        ) as dst:
            dst.write(np.moveaxis(data, -1, 0))
    elif HAS_PIL:
        from PIL import Image

        from satquery.utils.geospatial import _rgb_preview

        Image.fromarray(_rgb_preview(data)).save(path, format="TIFF")
    return path


def ensure_demo_images(force: bool = False) -> Dict[str, RasterImage]:
    """Create (or load cached) demo rasters; returns in-memory RasterImages.

    Files are written to the cache directory so the demo exercises the exact
    GeoTIFF ingestion path used for user uploads (including CRS metadata).
    """
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "single_optical": DEMO_DIR / "demo_optical_s2.tif",
        "single_sar": DEMO_DIR / "demo_sar_s1.tif",
        "t1": DEMO_DIR / "demo_t1_2023.tif",
        "t2": DEMO_DIR / "demo_t2_2024.tif",
        "fusion_optical": DEMO_DIR / "demo_fusion_optical.tif",
        "fusion_sar": DEMO_DIR / "demo_fusion_sar.tif",
    }
    if force:
        for path in files.values():
            if path.exists():
                path.unlink()

    missing = any(not p.exists() for p in files.values())
    if missing:
        layout = build_layout(seed=7)
        save_raster(make_raster(files["single_optical"].name, _render_optical(layout, seed=7)),
                    files["single_optical"])
        save_raster(make_raster(files["single_sar"].name, _render_sar(layout, seed=7)),
                    files["single_sar"])

        t1_layout = build_layout(seed=11)
        save_raster(make_raster(files["t1"].name, _render_optical(t1_layout, seed=11)), files["t1"])
        t2_layout = build_layout(seed=11, water_scale=0.88, urban_growth=0.055, veg_loss=0.14)
        save_raster(make_raster(files["t2"].name, _render_optical(t2_layout, seed=11)), files["t2"])

        fusion_layout = build_layout(seed=23)
        save_raster(make_raster(files["fusion_optical"].name, _render_optical(fusion_layout, seed=23)),
                    files["fusion_optical"])
        save_raster(make_raster(files["fusion_sar"].name, _render_sar(fusion_layout, seed=23, two_band=True)),
                    files["fusion_sar"])

    from satquery.utils.geospatial import read_image

    return {key: read_image(path) for key, path in files.items()}


def demo_sets(images: Optional[Dict[str, RasterImage]] = None) -> Dict[str, List[RasterImage]]:
    """Named demo bundles wired to sidebar buttons in the GUI."""
    images = images or ensure_demo_images()
    return {
        "Single optical (Sentinel-2 style)": [images["single_optical"]],
        "Single SAR (Sentinel-1 style)": [images["single_sar"]],
        "Bi-temporal pair (2023 vs 2024)": [images["t1"], images["t2"]],
        "Cross-modal pair (optical + SAR)": [images["fusion_optical"], images["fusion_sar"]],
    }
