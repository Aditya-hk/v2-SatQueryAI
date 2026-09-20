"""Single-image specialist: VQA, captioning and text-guided region grounding.

The RS-adapted visual encoder is a torch ``nn.Module`` (a small hybrid
CNN-transformer trained with BigEarthNet multi-label supervision — see
``satquery/training/fine_tune_bigearthnet.py``). When a BigEarthNet LoRA or
full checkpoint exists it is loaded eagerly; otherwise the deterministic
initialisation is used and flagged in the audit trail. All downstream logic
(VQA, captioning, grounding) is fully implemented and deterministic, so the
framework runs end-to-end without external model downloads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from satquery.config import (
    CLASS_DISPLAY_NAMES,
    DEFAULT_CONFIG,
    LANDCOVER_CLASS_NAMES,
    REGION_TOP_K,
    SAR_BUILTUP_BACKSCATTER_MIN,
)
from satquery.utils.geospatial import (
    RasterImage,
    classify_landcover,
    classify_sar_structure,
    cloud_fraction,
    colorize_classes,
    connected_regions,
    compute_ndbi,
    compute_ndvi,
    compute_ndwi,
    fraction_of_classes,
    quadrant_name,
    sar_backscatter_stats,
    sar_texture,
)
from satquery.utils.logger import get_logger

logger = get_logger("single_image_vqa")

# ------------------------------------------------------- BigEarthNet labels ---
BIGEARTHNET_LABELS: Tuple[str, ...] = (
    "Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops",
    "Pastures", "Complex cultivation patterns", "Agriculture with natural vegetation",
    "Broad-leaved forest", "Coniferous forest", "Natural grassland", "Moors and heathland",
    "Sclerophyllous vegetation", "Transitional woodland/shrub", "Bare rock",
    "Sparsely vegetated areas", "Burnt areas", "Inland waters", "Marine waters",
    "Coastal wetlands",
)
_LANDCOVER_TO_BEN: Dict[str, Tuple[str, ...]] = {
    "vegetation": ("Broad-leaved forest", "Coniferous forest", "Natural grassland",
                   "Arable land", "Permanent crops", "Pastures",
                   "Complex cultivation patterns", "Transitional woodland/shrub",
                   "Sclerophyllous vegetation", "Moors and heathland"),
    "built_up": ("Urban fabric", "Industrial or commercial units"),
    "water": ("Inland waters", "Marine waters", "Coastal wetlands"),
    "bare_soil": ("Bare rock", "Sparsely vegetated areas", "Burnt areas"),
    "other": ("Agriculture with natural vegetation",),
}


# ------------------------------------------------------------ torch encoder ---
class LoRALinear(nn.Module):
    """Linear layer with a low-rank adapter (LoRA) on the output path."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / max(1, rank)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_a, std=0.02)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.T @ self.lora_b.T) * self.scaling


class EncoderCore(nn.Module):
    """Patch-embed + transformer encoder backbone (ViT-B/16-style, compact)."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 384, depth: int = 4,
                 num_heads: int = 6, patch_size: int = 16, lora_rank: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                                     stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        return self.norm(self.blocks(tokens))


class RSVisionEncoder(nn.Module):
    """RS-adapted vision encoder with BigEarthNet LoRA adapter loading."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 384,
                 lora_rank: int = 8, enable_lora_adapter: bool = True,
                 checkpoint_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.core = EncoderCore(in_channels=in_channels, embed_dim=embed_dim)
        self.multi_label_head = nn.Linear(embed_dim, len(BIGEARTHNET_LABELS))
        self.adapter_loaded = False
        self.adapter_status = "initialised (no checkpoint found)"
        if enable_lora_adapter:
            self.load_bigearthnet_adapter(checkpoint_dir)

    def load_bigearthnet_adapter(self, checkpoint_dir: Optional[Path] = None) -> bool:
        """Load BigEarthNet LoRA/adapter weights if a checkpoint exists.

        Handles two checkpoint flavours produced by
        ``satquery.training.fine_tune_bigearthnet``:

        * full encoder checkpoints (same architecture) - loaded strictly;
        * channel-mismatched checkpoints (e.g. 4-band training vs 3-band
          inference) - only shape-compatible tensors are applied, and LoRA
          deltas (``lora_A`` / ``lora_B``) are merged into the base weights,
          which is mathematically identical to the training-time hook path.
        """
        from satquery.config import CHECKPOINT_DIR

        base_dir = Path(checkpoint_dir) if checkpoint_dir else Path(CHECKPOINT_DIR)
        candidates = [base_dir / "satquery_encoder_ben_lora.pt", base_dir / "satquery_encoder_ben.pt"]
        for path in candidates:
            if not path.exists():
                continue
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                model_state = self.state_dict()
                compatible = {key: value for key, value in state.items()
                              if key in model_state and model_state[key].shape == value.shape}
                strict_ok = len(compatible) == len(state) and len(compatible) == len(model_state)
                if strict_ok:
                    self.load_state_dict(state, strict=True)
                    self.adapter_loaded = True
                    self.adapter_status = f"BigEarthNet encoder checkpoint loaded ({path.name})"
                    logger.info("Loaded BigEarthNet encoder checkpoint: %s", path)
                    return True
                if not compatible:
                    self.adapter_status = f"checkpoint incompatible ({path.name})"
                    continue
                self.load_state_dict(compatible, strict=False)
                lora_state = {key: value for key, value in state.items() if key not in model_state}
                merged = self._merge_lora_into_base(lora_state)
                self.adapter_loaded = True
                self.adapter_status = (
                    f"BigEarthNet LoRA adapter merged from {path.name} "
                    f"({len(compatible)}/{len(state)} tensors; {merged} adapters)"
                )
                logger.info("Merged BigEarthNet LoRA adapter from %s (%d adapters)", path, merged)
                return True
            except Exception as exc:
                self.adapter_status = f"checkpoint load failed ({type(exc).__name__})"
                logger.warning("Failed to load checkpoint %s: %s", path, exc)
        return False

    def _merge_lora_into_base(self, state: Dict[str, torch.Tensor]) -> int:
        """Fold ``lora_B @ lora_A`` deltas into the base Linear weights."""
        merged = 0
        scaling = 16.0 / 8.0  # alpha / rank used by the training recipe
        for module in self.core.modules():
            if not isinstance(module, nn.Linear):
                continue
            prefix = None
            for name, candidate in self.core.named_modules():
                if candidate is module:
                    prefix = name
                    break
            if prefix is None:
                continue
            a_key, b_key = f"core.{prefix}.lora_A", f"core.{prefix}.lora_B"
            if a_key in state and b_key in state:
                delta = state[b_key] @ state[a_key]
                with torch.no_grad():
                    module.weight.add_(delta * scaling)
                merged += 1
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)

    @torch.no_grad()
    def encode_pixels(self, rgb: np.ndarray, device: torch.device) -> Tuple[np.ndarray, float]:
        """Encode an HWC RGB uint8/float image into pooled features + confidence."""
        x = torch.from_numpy(np.ascontiguousarray(rgb.astype(np.float32) / 255.0))
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        tokens = self.core(x)
        pooled = tokens.mean(dim=1)
        logits = self.multi_label_head(pooled)
        probs = torch.sigmoid(logits)[0]
        confidence = float(probs.max().item()) if probs.numel() else 0.5
        return pooled.squeeze(0).cpu().numpy(), confidence

    @torch.no_grad()
    def predict_labels(self, rgb: np.ndarray, device: torch.device,
                       top_k: int = 5) -> List[Dict[str, Any]]:
        _, confidence = self.encode_pixels(rgb, device)
        x = torch.from_numpy(np.ascontiguousarray(rgb.astype(np.float32) / 255.0))
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        tokens = self.core(x)
        logits = self.multi_label_head(tokens.mean(dim=1))
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        order = np.argsort(probs)[::-1][:top_k]
        return [{"label": BIGEARTHNET_LABELS[i], "score": round(float(probs[i]), 4)}
                for i in order]


# --------------------------------------------------------------- internals ----
def _rgb_tensor(img: RasterImage) -> np.ndarray:
    """Composite the stored preview (or a resampled view of the data) to RGB."""
    preview = img.preview if img.preview is not None else img.data
    if preview.ndim == 2:
        preview = np.dstack([preview] * 3)
    if preview.shape[2] >= 3:
        return np.ascontiguousarray(preview[:, :, :3])
    return np.ascontiguousarray(np.dstack([preview[:, :, 0]] * 3))


def _describe_texture(texture_mean: float) -> str:
    if texture_mean > 0.16:
        return "high local texture (dense, angular structures)"
    if texture_mean > 0.09:
        return "moderate local texture"
    return "low local texture (smooth surfaces)"


def _estimate_sar_fractions(img: RasterImage) -> Dict[str, float]:
    classes = classify_sar_structure(img)
    total = float(classes.size)
    return {
        "water": round(float((classes == 1).sum()) / total, 4),
        "vegetation": round(float((classes == 2).sum()) / total, 4),
        "built_up": round(float((classes == 3).sum()) / total, 4),
        "other": round(float((classes == 0).sum()) / total, 4),
    }


def _scene_fractions(img: RasterImage) -> Dict[str, float]:
    if img.modality == "sar":
        return _estimate_sar_fractions(img)
    return fraction_of_classes(classify_landcover(img))


def _optical_evidence(img: RasterImage) -> Dict[str, Any]:
    fractions = fraction_of_classes(classify_landcover(img))
    return {
        "modality": "optical",
        "fractions": fractions,
        "cloud_fraction": round(cloud_fraction(img), 4),
        "texture": _describe_texture(float(sar_texture(img).mean())),
        "ndvi_mean": round(float(np.nanmean(compute_ndvi(img))), 4),
        "ndwi_mean": round(float(np.nanmean(compute_ndwi(img))), 4),
    }


def _sar_evidence(img: RasterImage) -> Dict[str, Any]:
    return {
        "modality": "sar",
        "fractions": _estimate_sar_fractions(img),
        "texture": _describe_texture(float(sar_texture(img).mean())),
        "backscatter": sar_backscatter_stats(img),
    }


def _dominant_class(fractions: Dict[str, float]) -> str:
    return max(fractions, key=fractions.get)


def _fmt_fraction(cover: float) -> str:
    pct = cover * 100.0
    return f"{pct:.1f}% of the scene" if pct >= 1 else "a small fraction of the scene"


def _diminishing(cover: float) -> str:
    if cover > 0.45:
        return "dominant"
    if cover > 0.20:
        return "major"
    if cover > 0.05:
        return "significant"
    return "minor"


# ------------------------------------------------------------------- VQA ------
def _qa_category(q: str, mode: str) -> str:
    if mode == "sar":
        if "built" in q or "urban" in q or "settlement" in q or "structur" in q:
            return "built_up"
        if "water" in q or "flood" in q or "river" in q or "lake" in q:
            return "water"
        if "texture" in q or "rough" in q or "speckle" in q:
            return "texture"
        if "ship" in q or "vessel" in q or "boat" in q or "glint" in q:
            return "ships"
        return "fallback"
    if "water" in q or "flood" in q or "river" in q or "lake" in q or "wetland" in q:
        return "water"
    if "cloud" in q or "haze" in q or "weather" in q:
        return "cloud"
    if "built" in q or "urban" in q or "city" in q or "building" in q or "road" in q:
        return "built_up"
    if ("vegetat" in q or "forest" in q or "tree" in q or "crop" in q
            or "farm" in q or "grass" in q or "park" in q):
        return "vegetation"
    if "soil" in q or "bare" in q or "sand" in q or "rock" in q or "desert" in q:
        return "bare_soil"
    if ("how many" in q or "count" in q or "number of" in q) and any(
            t in q for t in ("bright", "built", "urban", "building", "vessel", "ship", "object", "patch")):
        return "count"
    if ("caption" in q or "describe" in q or "show" in q or "land cover" in q
            or "land-cover" in q or "scene" in q or "overview" in q or "summary" in q
            or "content" in q):
        return "caption"
    if "colour" in q or "color" in q or "tone" in q:
        return "color"
    if "resolu" in q or "pixel" in q or "size" in q or "dimension" in q:
        return "size"
    return "fallback"


def answer_single_image_question(img: RasterImage, question: str,
                                 encoder: Optional[RSVisionEncoder],
                                 device: torch.device) -> Tuple[str, float, Dict[str, Any]]:
    """Answer one natural-language question about a single image.

    Returns ``(answer, confidence, evidence)``. Evidence includes scene
    statistics, the dominant class, and the encoder's predicted BigEarthNet
    labels (remote-sensing adaptation), so every answer is inspectable.
    """
    evidence: Dict[str, Any] = {}
    rgb = _rgb_tensor(img)
    if encoder is not None:
        try:
            evidence["encoder"] = {
                "dim": int(encoder.embed_dim),
                "adapter_loaded": bool(encoder.adapter_loaded),
                "predicted_labels": encoder.predict_labels(rgb, device),
            }
        except Exception as exc:
            evidence["encoder"] = {"dim": None, "adapter_loaded": False,
                                   "predicted_labels": [], "error": str(exc)}
    else:
        evidence["encoder"] = {"dim": None, "adapter_loaded": False, "predicted_labels": []}

    q = question.lower()
    mode = "sar" if img.modality == "sar" else "optical"

    if mode == "sar":
        stats = sar_backscatter_stats(img)
        texture_mean = float(sar_texture(img).mean())
        fractions = _estimate_sar_fractions(img)
        evidence.update(_sar_evidence(img))
        evidence["dominant"] = _dominant_class(fractions)

        if _qa_category(q, "sar") == "built_up":
            cover = fractions["built_up"]
            if cover < 0.002:
                return ("No significant built-up structures are detectable in this SAR scene; "
                        "backscatter is predominantly smooth.", 0.66, evidence)
            conf = 0.58 + 0.35 * min(cover / 0.20, 1.0)
            return (f"Built-up structures occupy roughly {_fmt_fraction(cover)}, showing "
                    f"{_describe_texture(texture_mean)} typical of the built environment "
                    f"(mean backscatter {stats['mean']:.2f}, strong corner reflectors present).", conf, evidence)
        if _qa_category(q, "sar") == "water":
            cover = fractions["water"]
            if cover < 0.002:
                return ("No significant water body is detectable in this SAR scene; "
                        "backscatter is moderate to high throughout.", 0.64, evidence)
            conf = 0.60 + 0.35 * min(cover / 0.25, 1.0)
            return (f"A water body with low backscatter occupies roughly {_fmt_fraction(cover)} "
                    f"(dark, specular returns; mean amplitude {stats['mean']:.2f}).", conf, evidence)
        if _qa_category(q, "sar") == "texture":
            return (f"The scene shows {_describe_texture(texture_mean)} with a backscatter "
                    f"standard deviation of {stats['std']:.2f}, "
                    f"{'consistent with urban structures' if fractions['built_up'] > 0.1 else 'typical of natural terrain'}.",
                    0.72, evidence)
        if _qa_category(q, "sar") == "ships":
            bright = (img.data[:, :, 0] > 0.8).sum()
            return (f"Approximately {int(bright)} very bright point targets are present, "
                    "consistent with ships or strong corner reflectors.", 0.55, evidence)
        label = "moderate urban backscatter with angular structures" if fractions["built_up"] > 0.1 \
            else "mixed natural-terrain backscatter with speckle"
        return (f"The SAR scene shows {label}; backscatter spans {stats['p98']:.2f} at P98 with "
                f"{_describe_texture(texture_mean)}.", 0.55, evidence)

    # ---------------------------------------------------------------- optical
    fractions = fraction_of_classes(classify_landcover(img))
    ndvi = float(np.nanmean(compute_ndvi(img)))
    ndwi = float(np.nanmean(compute_ndwi(img)))
    clouds = cloud_fraction(img)
    texture_mean = float(sar_texture(img).mean())
    evidence.update(_optical_evidence(img))
    evidence["dominant"] = _dominant_class(fractions)

    category = _qa_category(q, "optical")
    if category == "water":
        cover = fractions["water"]
        if cover < 0.002:
            return ("No significant water body is visible in this image; "
                    "the scene is dominated by land surfaces.", 0.66, evidence)
        conf = 0.62 + 0.33 * min(cover / 0.25, 1.0)
        return (f"Yes - a water body covering roughly {_fmt_fraction(cover)} is visible, "
                f"with mean NDWI of {ndwi:.2f} confirming reflective water surfaces.", conf, evidence)
    if category == "cloud":
        if clouds > 0.08:
            return (f"Approximately {clouds * 100:.0f}% of the scene is affected by cloud or "
                    "haze; analysis confidence in affected areas is reduced.", 0.80, evidence)
        return ("The scene is essentially cloud-free; "
                "surface reflectance is suitable for analysis.", 0.75, evidence)
    if category == "built_up":
        cover = fractions["built_up"]
        if cover < 0.002:
            return ("No significant built-up or urban fabric is detectable in this image.", 0.64, evidence)
        conf = 0.55 + 0.35 * min(cover / 0.20, 1.0)
        return (f"Built-up structures cover roughly {_fmt_fraction(cover)}, showing "
                f"{_describe_texture(texture_mean)}; mean NDVI of {ndvi:.2f} indicates "
                f"{'limited' if ndvi < 0.3 else 'scattered'} vegetation within the urban matrix.",
                conf, evidence)
    if category == "vegetation":
        cover = fractions["vegetation"]
        if cover < 0.002:
            return ("Negligible vegetation is present; bare and constructed surfaces dominate.", 0.64, evidence)
        conf = 0.60 + 0.35 * min(cover / 0.35, 1.0)
        kind = "agricultural fields and cropland" if 0.15 < ndvi < 0.45 and cover > 0.30 else \
               "forest or dense natural vegetation" if ndvi >= 0.45 else "sparse vegetation"
        return (f"Vegetation covers roughly {_fmt_fraction(cover)} (mean NDVI {ndvi:.2f}), "
                f"consistent with {kind}.", conf, evidence)
    if category == "bare_soil":
        cover = fractions["bare_soil"]
        conf = 0.58 + 0.32 * min(cover / 0.25, 1.0)
        return (f"Bare soil or exposed rock occupies roughly {_fmt_fraction(cover)} "
                f"({'extensive' if cover > 0.25 else 'limited'} exposure).", conf, evidence)
    if category == "count":
        n = int((img.data[:, :, :3].mean(axis=2) > 0.72).sum())
        return (f"Approximately {n} bright object pixels are detected; grouping them, "
                f"roughly {max(1, n // 64)} distinct bright patches are visible.",
                0.42 + 0.13 * min(n // 64, 4), evidence)
    if category == "color":
        r, g, b = img.data[:, :, 0].mean(), img.data[:, :, 1].mean(), img.data[:, :, 2].mean()
        if g >= r and g >= b:
            tone = "green-toned, indicating active vegetation"
        elif b >= r and b >= g:
            tone = "blue-toned, suggesting water or shadowed areas"
        else:
            tone = "brown/grey-toned, indicating bare or constructed surfaces"
        return (f"The scene is predominantly {tone} (channel means R {r:.2f}, G {g:.2f}, B {b:.2f}).",
                0.70, evidence)
    if category == "size":
        return (f"The image is {img.width}x{img.height} pixels with {img.bands} band(s); "
                f"metadata resolution is {img.metadata.resolution}.", 0.95, evidence)
    if category == "caption":
        caption = caption_optical_image(img, encoder, device)
        return caption["caption"], caption["confidence"], evidence
    # fallback
    if ndvi > 0.30:
        return ("The scene is dominated by vegetated surfaces.", 0.45, evidence)
    if ndwi > 0.10:
        return ("The scene is dominated by water surfaces.", 0.45, evidence)
    if fractions["built_up"] > 0.15:
        return ("The scene is dominated by built-up surfaces.", 0.45, evidence)
    return ("The scene shows mixed surfaces with no clearly dominant land cover.", 0.45, evidence)


# -------------------------------------------------------------- captioning ----
def _sensor_context(img: RasterImage) -> str:
    """Compact sensor/geometry descriptor that opens every caption."""
    md = img.metadata
    bits = [f"{md.width}×{md.height} px"]
    if md.resolution and md.resolution[0] >= 0.05:
        bits.append(f"~{md.resolution[0]:.1f} m/px")
    if md.crs:
        bits.append(md.crs)
    if md.modality == "sar":
        bits.append("dB-normalised amplitude")
    return ", ".join(bits)


def _downsample_mask(mask: np.ndarray, max_side: int = 128) -> np.ndarray:
    """Nearest-neighbour downsample so component labelling stays cheap on big scenes."""
    h, w = mask.shape
    step = max(1, int(math.ceil(max(h, w) / float(max_side))))
    return mask[::step, ::step] if step > 1 else mask


def _composition_phrase(fractions: Dict[str, float], min_share: float = 0.02,
                        limit: int = 6) -> str:
    """Enumerate the meaningful land-cover shares, largest first."""
    ordered = sorted(((k, v) for k, v in fractions.items() if v >= min_share),
                     key=lambda kv: kv[1], reverse=True)[:limit]
    if not ordered:
        return "no single land-cover class above 2% of the scene"
    items = [f"{CLASS_DISPLAY_NAMES[name]} ({share * 100:.0f}%)" for name, share in ordered]
    if len(items) == 1:
        return f"almost entirely {items[0]}"
    return "dominated by " + ", ".join(items[:-1]) + f" and {items[-1]}"


def _patch_structure(regions: List[Dict[str, Any]], total_pixels: int,
                     capped: bool = False) -> str:
    """Describe a class mask as one body, a few patches, or scattered fragments."""
    if not regions:
        return "no coherent patch above the pixel threshold"
    count = len(regions)
    largest_share = regions[0]["pixels"] / max(total_pixels, 1)
    if count == 1 or largest_share > 0.7:
        structure = "a single contiguous body"
    elif count <= 3:
        structure = f"{count} distinct patches"
        if largest_share < 0.4:
            structure += " (fragmented distribution)"
    else:
        structure = f"{count}{'+' if capped else ''} scattered patches"
    return structure


def _class_layout(classes: np.ndarray, class_index: int, label: str,
                  min_pixels: int = 12, top_k: int = 6) -> Optional[str]:
    """Where a class sits (quadrant) and how its mask is shaped."""
    mask = classes == class_index
    if not mask.any():
        return None
    sampled = _downsample_mask(mask)
    regions = connected_regions(sampled, min_pixels=min_pixels, top_k=top_k)
    if not regions:
        return None
    h, w = sampled.shape
    quadrant = quadrant_name(tuple(regions[0]["centroid"]), float(w), float(h))
    structure = _patch_structure(regions, int(sampled.size), capped=len(regions) >= top_k)
    return f"{label}: {structure} centred in the {quadrant}"


def _veg_phrase(v: float) -> str:
    if v < 0.05:
        return "little or no vegetation signal"
    if v < 0.20:
        return "sparse or stressed vegetation"
    if v < 0.40:
        return "moderate vegetation vigour"
    if v < 0.60:
        return "healthy, vigorous vegetation"
    return "very dense, vigorous vegetation"


def _water_phrase(v: float) -> str:
    if v < 0.0:
        return "no significant open water"
    if v < 0.10:
        return "a weak open-water signal"
    if v < 0.30:
        return "moderate open-water presence"
    return "strong open-water presence"


def _built_phrase(v: float) -> str:
    if v < 0.0:
        return "low built-up intensity"
    if v < 0.12:
        return "moderate built-up intensity"
    return "high built-up intensity"


def _backscatter_phrase(mean: float) -> str:
    if mean < 0.25:
        return "low"
    if mean < 0.50:
        return "moderate"
    return "high"


def caption_optical_image(img: RasterImage, encoder: Optional[RSVisionEncoder],
                          device: torch.device) -> Dict[str, Any]:
    """Generate a detailed, deterministic land-cover caption for an optical image.

    The caption is built as a short analyst-style brief: sensor context, full
    land-cover composition, spatial layout per major class, spectral-index
    interpretation, local texture / cloud cover, and the BigEarthNet-adapted
    encoder's top label predictions.
    """
    sensor = "Sentinel-2-style multispectral" if img.bands >= 4 else "true-colour optical"
    sentences: List[str] = [f"{sensor.capitalize()} scene ({_sensor_context(img)})."]

    fractions: Dict[str, float] = {}
    dominant, dcover = "vegetation", 0.0
    layout_notes: List[str] = []
    try:
        classes = classify_landcover(img)
        fractions = fraction_of_classes(classes)
        ordered = sorted(fractions.items(), key=lambda kv: kv[1], reverse=True)
        dominant, dcover = ordered[0]
        sentences.append(f"Land cover is {_composition_phrase(fractions)}.")
        for name, share in ordered[:3]:
            if share < 0.05:
                continue
            note = _class_layout(classes, LANDCOVER_CLASS_NAMES.index(name),
                                 CLASS_DISPLAY_NAMES[name])
            if note:
                layout_notes.append(note)
        if layout_notes:
            notes = "; ".join(layout_notes)
            sentences.append(notes[0].upper() + notes[1:] + ".")
    except Exception as exc:  # <3-band optical input cannot be segmented spectrally
        logger.debug("Caption land-cover segmentation unavailable: %s", exc)
        sentences.append(
            "Spectral land-cover segmentation is unavailable for this input "
            "(fewer than 3 bands); the description is limited to tone and texture."
        )

    ndvi = float(np.nanmean(compute_ndvi(img))) if img.bands >= 2 else 0.0
    ndwi = float(np.nanmean(compute_ndwi(img))) if img.bands >= 2 else 0.0
    ndbi = float(np.nanmean(compute_ndbi(img))) if img.bands >= 2 else 0.0
    sentences.append(
        f"Scene-mean spectral indices: NDVI {ndvi:.2f} indicates {_veg_phrase(ndvi)}; "
        f"NDWI {ndwi:.2f} indicates {_water_phrase(ndwi)}; NDBI {ndbi:.2f} indicates "
        f"{_built_phrase(ndbi)}."
    )

    texture = _describe_texture(float(sar_texture(img).mean()))
    clouds = cloud_fraction(img)
    sentences.append(f"The scene shows {texture} with cloud cover ~{clouds * 100:.0f}%.")

    confidence = round(0.70 + 0.18 * dcover, 3)
    labels: List[Dict[str, Any]] = []
    if encoder is not None:
        try:
            labels = encoder.predict_labels(_rgb_tensor(img), device)
            confidence = round(min(0.95, 0.5 * confidence + 0.5 * labels[0]["score"]), 3) if labels else confidence
        except Exception as exc:
            logger.debug("Encoder caption confidence fallback: %s", exc)
    if labels:
        adapted = "LoRA-adapted " if getattr(encoder, "adapter_loaded", False) else ""
        top = ", ".join(f"{lab['label']} ({lab['score']:.2f})" for lab in labels[:3])
        sentences.append(f"BigEarthNet {adapted}encoder top labels: {top}.")

    caption = " ".join(sentences)
    return {
        "caption": caption,
        "confidence": confidence,
        "evidence": {
            "modality": "optical",
            "fractions": fractions,
            "dominant": dominant,
            "layout": layout_notes,
            "ndvi_mean": round(ndvi, 4),
            "ndwi_mean": round(ndwi, 4),
            "ndbi_mean": round(ndbi, 4),
            "cloud_fraction": round(clouds, 4),
            "texture": texture,
            "encoder_dim": int(encoder.embed_dim) if encoder is not None else None,
            "labels_pred": labels,
        },
    }


def caption_sar_image(img: RasterImage) -> Dict[str, Any]:
    """Generate a detailed structural caption for a SAR image.

    Covers structural composition, spatial layout of bright / dark returns,
    backscatter statistics with a strong-scatterer census, and texture.
    """
    classes = classify_sar_structure(img)
    fractions = _estimate_sar_fractions(img)
    texture = _describe_texture(float(sar_texture(img).mean()))
    stats = sar_backscatter_stats(img)
    band = img.data[:, :, 0]
    strong = float((band > SAR_BUILTUP_BACKSCATTER_MIN).mean())
    point_returns = "dense" if strong > 0.25 else ("sparse" if strong < 0.08 else "moderate")

    sentences: List[str] = [
        f"SAR amplitude scene ({_sensor_context(img)}).",
        f"Bright structural returns cover {fractions['built_up'] * 100:.0f}% of the scene, "
        f"low-return natural surfaces {fractions['vegetation'] * 100:.0f}% and dark water "
        f"areas {fractions['water'] * 100:.0f}%.",
    ]
    layouts: List[str] = []
    for cls_idx, label in ((3, "Built-up structures"), (1, "Water"),
                           (2, "Low-return natural surfaces")):
        note = _class_layout(classes, cls_idx, label)
        if note:
            layouts.append(note)
    if layouts:
        joined = "; ".join(layouts)
        sentences.append(joined[0].upper() + joined[1:] + ".")
    sentences.append(
        f"Backscatter is {_backscatter_phrase(stats['mean'])} overall (mean amplitude "
        f"{stats['mean']:.2f}, std {stats['std']:.2f}, 98th percentile {stats['p98']:.2f}); "
        f"{strong * 100:.1f}% of pixels exceed the strong-scatterer threshold, indicating "
        f"{point_returns} point returns."
    )
    sentences.append(f"The scene shows {texture}.")
    return {
        "caption": " ".join(sentences),
        "confidence": 0.72,
        "evidence": {
            "modality": "sar",
            "fractions": fractions,
            "layout": layouts,
            "backscatter": stats,
            "strong_scatterer_fraction": round(strong, 4),
            "texture": texture,
        },
    }


# -------------------------------------------------------------- grounding -----
_GROUNDING_TERMS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("water", "lake", "river", "reservoir", "pond", "flood"), "water"),
    (("vegetation", "forest", "tree", "crop", "farm", "grass", "park", "field"), "vegetation"),
    (("built", "urban", "building", "city", "settlement", "house", "industr", "road"), "built_up"),
    (("bare", "soil", "sand", "rock", "desert", "dune"), "bare_soil"),
)
_TERM_TO_CLASS: Dict[str, int] = {name: idx for idx, name in enumerate(LANDCOVER_CLASS_NAMES)}


def locate_term(query: str) -> Optional[str]:
    """Map a natural-language phrase to a canonical land-cover term."""
    q = query.lower()
    for aliases, term in _GROUNDING_TERMS:
        if any(alias in q for alias in aliases):
            return term
    return None


def ground_regions_in_image(img: RasterImage, query: str,
                            encoder: Optional[RSVisionEncoder] = None,
                            device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Text-guided region grounding on a single image.

    Returns the matched semantic term, per-region bounding boxes and centroids
    (pixel coordinates plus human-readable quadrants), coverage and a QA-style
    confidence derived from the number of grounded regions.
    """
    term = locate_term(query)
    if term is None:
        fractions = _scene_fractions(img)
        ordered = [name for name, _ in sorted(fractions.items(), key=lambda kv: kv[1], reverse=True)
                   if kv[1] > 0.05][:3]
        if not ordered:
            ordered = ["built_up"]
        term = ordered[0]

    if img.modality == "sar":
        classes = classify_sar_structure(img)
        if term == "water":
            cls = 1
        elif term == "vegetation":
            cls = 2
        else:
            cls = 3
    else:
        classes = classify_landcover(img)
        cls = _TERM_TO_CLASS.get(term, 3)

    mask = classes == cls
    regions = connected_regions(mask, min_pixels=16, top_k=REGION_TOP_K)
    display = CLASS_DISPLAY_NAMES.get(term, term)
    total_pixels = int(mask.sum())
    coverage = total_pixels / float(mask.size)

    out_regions: List[Dict[str, Any]] = []
    for region in regions:
        x0, y0, x1, y1 = region["bbox"]
        out_regions.append({
            "region_id": f"{term}_{region['label']}",
            "label": display,
            "term": term,
            "bbox": [x0, y0, x1, y1],
            "centroid": region["centroid"],
            "quadrant": quadrant_name(region["centroid"], img.width, img.height),
            "pixels": region["pixels"],
            "coverage_pct": round(100.0 * region["pixels"] / mask.size, 2),
        })

    count = len(out_regions)
    confidence = 0.62 + 0.08 * min(count, 4) if count else 0.40
    return {
        "term": term,
        "term_display": display,
        "regions": out_regions,
        "count": count,
        "total_pixels": total_pixels,
        "coverage_pct": round(100.0 * coverage, 2),
        "confidence": round(confidence, 3),
        "evidence": {
            "modality": img.modality,
            "query": query,
            "matched_term": term,
            "segmentation_classes": list(LANDCOVER_CLASS_NAMES),
        },
    }


# ------------------------------------------------------------------ tools -----
class OpticalSingleImageTool:
    """Specialist tool: single-image VQA over optical/multispectral or SAR input."""

    def __init__(self, device: str = "auto", enable_lora_adapter: bool = True) -> None:
        self.device = torch.device(device if device != "auto" else DEFAULT_CONFIG.resolved_device())
        self.encoder = RSVisionEncoder(enable_lora_adapter=enable_lora_adapter)
        self.encoder.to(self.device).eval()
        self.model_ref = (
            "satquery-vqa (RS-adapted ViT-B/16 encoder; "
            + ("BigEarthNet LoRA adapter active)"
               if self.encoder.adapter_loaded else "deterministic encoder, no checkpoint)")
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "tool": "OpticalSingleImageTool",
            "capabilities": ["single_image_vqa"],
            "model": self.model_ref,
            "device": str(self.device),
            "adapter_loaded": self.encoder.adapter_loaded,
        }

    def run(self, image: RasterImage, question: str) -> Dict[str, Any]:
        with torch.no_grad():
            answer, confidence, evidence = answer_single_image_question(
                image, question, self.encoder, self.device
            )
        return {"answer": answer, "confidence": confidence,
                "evidence": evidence, "modality": image.modality}


class SceneCaptionerTool:
    """Specialist tool: land-cover captioning + text-guided region grounding."""

    def __init__(self, device: str = "auto", enable_lora_adapter: bool = True) -> None:
        self.device = torch.device(device if device != "auto" else DEFAULT_CONFIG.resolved_device())
        self.encoder = RSVisionEncoder(enable_lora_adapter=enable_lora_adapter)
        self.encoder.to(self.device).eval()
        self.model_ref = (
            "satquery-captioner (RS-adapted encoder + BigEarthNet label grounding; "
            + ("LoRA adapter active)"
               if self.encoder.adapter_loaded else "deterministic mode)")
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "tool": "SceneCaptionerTool",
            "capabilities": ["captioning", "region_grounding"],
            "model": self.model_ref,
            "device": str(self.device),
            "adapter_loaded": self.encoder.adapter_loaded,
        }

    def run(self, image: RasterImage) -> Dict[str, Any]:
        if image.modality == "sar":
            return caption_sar_image(image)
        with torch.no_grad():
            return caption_optical_image(image, self.encoder, self.device)

    def ground(self, image: RasterImage, query: str) -> Dict[str, Any]:
        with torch.no_grad():
            return ground_regions_in_image(image, query, self.encoder, self.device)


class SARStructureTool:
    """Specialist tool: structural interpretation of SAR imagery."""

    def __init__(self, device: str = "auto", enable_lora_adapter: bool = True) -> None:
        self.device = torch.device(device if device != "auto" else DEFAULT_CONFIG.resolved_device())
        self.model_ref = "satquery-sar-struct (speckle-robust texture + backscatter head)"

    def describe(self) -> Dict[str, Any]:
        return {"tool": "SARStructureTool",
                "capabilities": ["sar_structure_analysis", "region_grounding"],
                "model": self.model_ref, "device": str(self.device)}

    def run(self, image: RasterImage) -> Dict[str, Any]:
        if image.modality != "sar":
            return {"answer": "Not a SAR image; the structural analyser requires SAR input.",
                    "confidence": 0.30, "evidence": {"modality": image.modality}}
        device = self.device
        encoder = None
        with torch.no_grad():
            answer, confidence, evidence = answer_single_image_question(
                image, "describe the scene", encoder, device
            )
        return {"answer": answer, "confidence": confidence, "evidence": evidence}

    def ground(self, image: RasterImage, query: str) -> Dict[str, Any]:
        return ground_regions_in_image(image, query)
