"""Central configuration and default parameters for SatQuery AI (SIH-167).

All tunable thresholds used across validation, modality inference, spectral-index
analysis, change detection, fusion, and confidence estimation live here so that
they can be adjusted from a single place (and surfaced in the audit reports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

# ------------------------------------------------------------------ identity --
APP_NAME: str = "SatQuery AI"
APP_VERSION: str = "1.0.0"
PROBLEM_STATEMENT_ID: str = "SIH-167"

PACKAGE_ROOT: Path = Path(__file__).resolve().parent          # .../satquery
PROJECT_ROOT: Path = PACKAGE_ROOT.parent                      # repository root
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
CACHE_DIR: Path = PROJECT_ROOT / ".cache"
CHECKPOINT_DIR: Path = PACKAGE_ROOT / "models" / "checkpoints"

# -------------------------------------------------------------------- inputs --
GEOTIFF_EXTENSIONS: tuple = (".tif", ".tiff")
BENCHMARK_EXTENSIONS: tuple = (".png", ".jpg", ".jpeg")
SUPPORTED_EXTENSIONS: tuple = GEOTIFF_EXTENSIONS + BENCHMARK_EXTENSIONS
BENCHMARK_FORMAT_NOTE: str = (
    "PNG/JPEG inputs are accepted only for prescribed benchmark datasets "
    "(VRSBench / RSVQA / CDVQA); geospatial submissions should use GeoTIFF/TIFF."
)

MAX_FILE_SIZE_MB: float = 2048.0
PREVIEW_MAX_SIDE: int = 720
MIN_IMAGE_SIDE_PX: int = 32

# --------------------------------------------------------------- geospatial ---
PAIR_ALIGNMENT_TOLERANCE_PX: float = 1.0
DEFAULT_TARGET_CRS: str = "EPSG:4326"
EXTENT_OVERLAP_WARN: float = 0.95
EXTENT_OVERLAP_ERROR: float = 0.50

# ------------------------------------------------------------ modality rules --
MODALITY_SAR_MAX_VARIANCE: float = 0.045
MODALITY_GRAY_OPTICAL_SATURATION: float = 0.085
MODALITY_OPTICAL_SATURATION: float = 0.11
MODALITY_HSV_SAMPLE_STRIDE: int = 4

CLOUD_BRIGHTNESS_THRESHOLD: float = 0.88
CLOUD_SATURATION_THRESHOLD: float = 0.16
CLOUD_WARNING_FRACTION: float = 0.08

# --------------------------------------------------------- spectral indices ---
WATER_NDWI_THRESHOLD: float = 0.10
VEGETATION_NDVI_THRESHOLD: float = 0.30
BUILTUP_NDBI_THRESHOLD: float = 0.00
BARESOIL_BRIGHTNESS_THRESHOLD: float = 0.45

# SAR structural thresholds (stretch-normalised amplitude, 0..1)
SAR_WATER_BACKSCATTER_MAX: float = 0.16
SAR_BUILTUP_BACKSCATTER_MIN: float = 0.42
SAR_BUILTUP_TEXTURE_MIN: float = 0.14

# --------------------------------------------------------------- land cover ---
LANDCOVER_CLASS_NAMES: tuple = ("other", "water", "vegetation", "built_up", "bare_soil")
LANDCOVER_PALETTES: Dict[int, Tuple[int, int, int]] = {
    0: (124, 124, 124),
    1: (46, 108, 196),
    2: (56, 168, 82),
    3: (222, 72, 58),
    4: (198, 166, 110),
}
CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "other": "other / mixed surface",
    "water": "water",
    "vegetation": "vegetation",
    "built_up": "built-up",
    "bare_soil": "bare soil",
}

# ------------------------------------------------------------ change detection -
DEFAULT_CHANGE_THRESHOLD: float = 0.12
CHANGE_MIN_REGION_PIXELS: int = 24
CHANGE_INDEX_DELTA: float = 0.15
CHANGE_CLASS_PALETTES: Dict[str, Tuple[int, int, int]] = {
    "vegetation_loss": (232, 88, 46),
    "vegetation_gain": (120, 208, 96),
    "new_built_up": (226, 34, 92),
    "demolition": (168, 84, 255),
    "water_gain": (46, 148, 236),
    "water_loss": (24, 96, 190),
    "increased_backscatter": (250, 160, 40),
    "decreased_backscatter": (120, 120, 220),
    "surface_disturbance": (250, 224, 70),
}
CHANGE_CLASS_ORDER: tuple = tuple(CHANGE_CLASS_PALETTES.keys())
CHANGE_CLASS_DISPLAY: Dict[str, str] = {
    "vegetation_loss": "vegetation loss",
    "vegetation_gain": "vegetation gain / regrowth",
    "new_built_up": "new built-up growth",
    "demolition": "built-up removal / demolition",
    "water_gain": "water expansion (e.g. flooding)",
    "water_loss": "water recession / drying",
    "increased_backscatter": "SAR backscatter increase (structural growth)",
    "decreased_backscatter": "SAR backscatter decrease (surface loss)",
    "surface_disturbance": "generic surface disturbance",
}

# ------------------------------------------------------------------ captions --
CAPTION_CLASS_TEMPLATES: Dict[str, str] = {
    "vegetation": "vegetation covering {fraction} of the scene",
    "built_up": "built-up structures across {fraction} of the scene",
    "water": "a water body occupying {fraction} of the scene",
    "bare_soil": "bare soil / exposed ground over {fraction} of the scene",
    "other": "mixed transitional surfaces over {fraction} of the scene",
}
CAPTION_OPENERS: tuple = (
    "High-resolution remote-sensing scene showing",
    "Satellite view of",
    "Orthorectified imagery depicting",
)

# -------------------------------------------------------------------- agent ---
CONFIDENCE_WEIGHTS: Dict[str, float] = {"data_quality": 0.30, "model": 0.45, "agreement": 0.25}
LOW_CONFIDENCE_THRESHOLD: float = 0.30
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.60
REGION_TOP_K: int = 8

# ------------------------------------------------------------- live imagery ---
#: Public STAC catalog used for on-demand ROI acquisition (Microsoft Planetary
#: Computer; Sentinel-2 L2A optical + Sentinel-1 GRD SAR, free, no API key).
MPC_STAC_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
ROI_MAX_SIDE_DEG: float = 2.0          # guardrail: largest drawable ROI per side
ROI_MAX_DIM_PX: int = 512              # longest pixel side of a fetched window
ROI_CLOUD_COVER_MAX: float = 30.0      # % - scene filter for optical acquisition
ROI_RESULT_MAP_HEIGHT_PX: int = 420    # folium result-overlay height in the GUI

# ------------------------------------------------------------------ AppConfig -
@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration passed through the agent and specialist tools."""

    preview_max_side: int = PREVIEW_MAX_SIDE
    change_threshold: float = DEFAULT_CHANGE_THRESHOLD
    change_min_region_px: int = CHANGE_MIN_REGION_PIXELS
    grounding_top_k: int = REGION_TOP_K
    target_crs: str = DEFAULT_TARGET_CRS
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    medium_confidence_threshold: float = MEDIUM_CONFIDENCE_THRESHOLD
    max_file_size_mb: float = MAX_FILE_SIZE_MB
    device: str = "auto"
    enable_lora_adapter: bool = True
    adapter_dir: Path = field(default_factory=lambda: CHECKPOINT_DIR)

    def resolved_device(self) -> str:
        """Resolve 'auto' to the best available torch device (cuda > mps > cpu)."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover - torch optional
            pass
        return "cpu"

    def as_dict(self) -> Dict[str, Any]:
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in self.__dict__.items()
        }


DEFAULT_CONFIG = AppConfig()
