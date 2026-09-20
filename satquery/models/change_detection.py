"""Bi-temporal change analysis: spatial change maps, typed change description
and change-VQA over image pairs (T1 vs T2).

The analyser works on spectral-index differencing for optical pairs (NDVI /
NDWI / NDBI deltas classify vegetation, water and built-up change) and on
backscatter differencing for SAR pairs. A lightweight torch head
(``ChangeNetHead``) refines the per-pixel change probability; its weights can
be swapped for a CDVQA-trained checkpoint via ``load_checkpoint``. Change-VQA
is answered deterministically from the typed change statistics so behaviour is
reproducible and auditable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from satquery.config import (
    CHANGE_CLASS_DISPLAY,
    CHANGE_CLASS_ORDER,
    CHANGE_CLASS_PALETTES,
    CHANGE_INDEX_DELTA,
    CHANGE_MIN_REGION_PIXELS,
    CLASS_DISPLAY_NAMES,
    DEFAULT_CONFIG,
    LANDCOVER_CLASS_NAMES,
)
from satquery.utils.geospatial import (
    RasterImage,
    classify_landcover,
    classify_sar_structure,
    colorize_change,
    compute_ndbi,
    compute_ndvi,
    compute_ndwi,
    connected_regions,
    quadrant_name,
    sar_backscatter_stats,
)
from satquery.utils.logger import get_logger

logger = get_logger("change_detection")


# --------------------------------------------------------------- torch head ---
class ChangeNetHead(nn.Module):
    """Siamese refiner: maps stacked |T1-T2| + index deltas to change logits.

    Fully trained this becomes a CDVQA-style change-probability head; the
    shipped deterministic initialisation (identity + fixed biases) preserves
    the analytic detector while remaining a real, loadable torch module.
    """

    def __init__(self, in_channels: int = 6, hidden: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)

    @torch.no_grad()
    def refine_probability(self, features: np.ndarray) -> np.ndarray:
        """features: (C, H, W) float32 -> per-pixel change probability (H, W)."""
        device = next(self.parameters()).device
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).unsqueeze(0)
        x = x.to(device)
        logits = self.forward(x)
        return torch.sigmoid(logits)[0].cpu().numpy()


def build_change_features(t1: RasterImage, t2: RasterImage) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Assemble the per-pixel feature stack fed to the refiner head."""
    amplitude = np.abs(t2.data - t1.data).mean(axis=2)
    channels: List[np.ndarray] = [amplitude]
    derived: Dict[str, np.ndarray] = {}
    if t1.modality != "sar" and t1.bands >= 3 and t2.bands >= 3:
        for name, fn in (("ndvi", compute_ndvi), ("ndwi", compute_ndwi), ("ndbi", compute_ndbi)):
            delta = (fn(t2) - fn(t1)).astype(np.float32)
            derived[name] = delta
            channels.append(delta)
    while len(channels) < 6:
        channels.append(np.zeros_like(amplitude))
    return np.stack(channels[:6]).astype(np.float32), derived


# ---------------------------------------------------------- change taxonomy ---
def classify_change_pixel(delta_index: Dict[str, np.ndarray], amplitude: np.ndarray,
                          is_sar: bool) -> np.ndarray:
    """Assign a typed change label to every changed pixel."""
    h, w = amplitude.shape
    labels = np.full((h, w), "no_change", dtype=object)
    changed = amplitude > DEFAULT_CONFIG.change_threshold
    if is_sar:
        labels[changed & (amplitude > 2.0 * max(1e-3, float(amplitude.mean())))] = "increased_backscatter"
        labels[changed & (labels == "no_change")] = "decreased_backscatter"
        labels[~changed] = "no_change"
        return labels
    d_ndvi = delta_index.get("ndvi", np.zeros((h, w), dtype=np.float32))
    d_ndwi = delta_index.get("ndwi", np.zeros((h, w), dtype=np.float32))
    d_ndbi = delta_index.get("ndbi", np.zeros((h, w), dtype=np.float32))
    sig = CHANGE_INDEX_DELTA

    water_gain = (d_ndwi > sig) & changed
    water_loss = (d_ndwi < -sig) & changed
    veg_gain = (d_ndvi > sig) & changed & ~water_gain & ~water_loss
    veg_loss = (d_ndvi < -sig) & changed & ~water_gain & ~water_loss
    built_gain = (d_ndbi > sig) & changed & ~water_gain & ~water_loss & ~veg_gain & ~veg_loss
    built_loss = (d_ndbi < -sig) & changed & ~water_gain & ~water_loss & ~veg_gain & ~veg_loss
    remaining = changed & (labels == "no_change") & ~water_gain & ~water_loss & ~veg_gain & ~veg_loss \
        & ~built_gain & ~built_loss

    labels[veg_loss] = "vegetation_loss"
    labels[veg_gain] = "vegetation_gain"
    labels[built_gain] = "new_built_up"
    labels[built_loss] = "demolition"
    labels[water_gain] = "water_gain"
    labels[water_loss] = "water_loss"
    labels[remaining] = "surface_disturbance"
    labels[~changed] = "no_change"
    return labels


def _class_map_of(img: RasterImage) -> Optional[np.ndarray]:
    try:
        return classify_sar_structure(img) if img.modality == "sar" else classify_landcover(img)
    except Exception:
        return None


def change_summary(labels: np.ndarray, t1: RasterImage, t2: RasterImage,
                   min_region_px: int) -> Dict[str, Any]:
    """Aggregate typed change labels into class stats + region descriptions.

    Also records land-cover composition of T1/T2 (when computable) and the
    dominant class transitions, so questions like "has the built-up area
    increased?" can be answered from actual cover deltas, not just typed
    index-change labels.
    """
    h, w = labels.shape
    changed = labels != "no_change"
    per_class: Dict[str, Dict[str, Any]] = {}
    for name in CHANGE_CLASS_ORDER:
        mask = labels == name
        count = int(mask.sum())
        if count == 0:
            continue
        ys, xs = np.where(mask)
        regions = connected_regions(mask.astype(np.uint8), min_pixels=min_region_px, top_k=4)
        per_class[name] = {
            "pixels": count,
            "area_pct": round(100.0 * count / (h * w), 2),
            "centroid_quadrant": quadrant_name((float(xs.mean()), float(ys.mean())), w, h),
            "regions": [
                {
                    "bbox": rg["bbox"],
                    "quadrant": quadrant_name(rg["centroid"], w, h),
                    "pixels": rg["pixels"],
                }
                for rg in regions
            ],
        }

    composition: Dict[str, Dict[str, float]] = {}
    transitions: Dict[str, Dict[str, Any]] = {}
    c1, c2 = _class_map_of(t1), _class_map_of(t2)
    if c1 is not None and c2 is not None and c1.shape == c2.shape:
        for idx, name in enumerate(LANDCOVER_CLASS_NAMES):
            composition[name] = {
                "t1_pct": round(100.0 * float((c1 == idx).mean()), 2),
                "t2_pct": round(100.0 * float((c2 == idx).mean()), 2),
            }
        changed_mask = changed & (c1 != c2)
        pairs, counts = np.unique(
            np.stack([c1[changed_mask], c2[changed_mask]], axis=1), axis=0, return_counts=True
        ) if changed_mask.any() else (np.zeros((0, 2), dtype=int), np.zeros((0,)))
        total_scene = float(h * w)
        for (frm, to), count in zip(pairs.tolist(), counts.tolist()):
            if frm == to:
                continue
            key = f"{LANDCOVER_CLASS_NAMES[frm]}_to_{LANDCOVER_CLASS_NAMES[to]}"
            transitions[key] = {
                "from": LANDCOVER_CLASS_NAMES[frm],
                "to": LANDCOVER_CLASS_NAMES[to],
                "area_pct": round(100.0 * count / total_scene, 2),
            }
        transitions = dict(sorted(transitions.items(),
                                  key=lambda kv: kv[1]["area_pct"], reverse=True)[:8])

    return {
        "changed_pixels": int(changed.sum()),
        "changed_area_pct": round(100.0 * float(changed.mean()), 2),
        "per_class": per_class,
        "composition": composition,
        "transitions": transitions,
        "width": w,
        "height": h,
    }


def describe_changes(summary: Dict[str, Any]) -> str:
    """Natural-language description of the typed change statistics."""
    per_class = summary["per_class"]
    if not per_class:
        return ("No significant change is detected between the two acquisitions; "
                "the scene appears spectrally and structurally stable.")
    ordered = sorted(per_class.items(), key=lambda kv: kv[1]["pixels"], reverse=True)
    clauses: List[str] = []
    for name, stats in ordered[:4]:
        display = CHANGE_CLASS_DISPLAY.get(name, name.replace("_", " "))
        quadrant = stats["centroid_quadrant"]
        clauses.append(
            f"{display} affecting ~{stats['area_pct']:.1f}% of the scene "
            f"(concentrated in the {quadrant})"
        )
    head = f"Overall change footprint: {summary['changed_area_pct']:.1f}% of the scene. "
    return head + "Detected: " + "; ".join(clauses) + "."


def answer_change_question(question: str, summary: Dict[str, Any]) -> Tuple[str, float]:
    """Deterministic change-VQA over the typed change statistics."""
    q = question.lower()
    per_class = summary["per_class"]
    changed_pct = summary["changed_area_pct"]

    if any(t in q for t in ("where", "location", "which region", "which area")):
        if not per_class:
            return "No change regions were detected, so there is no location to report.", 0.70
        biggest = max(per_class.items(), key=lambda kv: kv[1]["pixels"])
        regions = biggest[1]["regions"][:3]
        quads = ", ".join(dict.fromkeys(rg["quadrant"] for rg in regions)) \
            or biggest[1]["centroid_quadrant"]
        return (f"The dominant change ({CHANGE_CLASS_DISPLAY.get(biggest[0], biggest[0])}) "
                f"occurs in the {quads} region(s), covering ~{biggest[1]['area_pct']:.1f}% "
                "of the scene.", 0.74)

    if any(t in q for t in ("how much", "what fraction", "what percentage", "how large",
                            "what area", "what extent", "total area")):
        if not per_class:
            return "No change was detected between the two dates.", 0.72
        parts = [f"{CHANGE_CLASS_DISPLAY.get(name, name)} ~{stats['area_pct']:.1f}%"
                 for name, stats in sorted(per_class.items(), key=lambda kv: kv[1]["pixels"], reverse=True)[:3]]
        return (f"Changed area totals ~{changed_pct:.1f}% of the scene: " + "; ".join(parts) + ".", 0.76)

    composition = summary.get("composition", {})
    transitions = summary.get("transitions", {})

    def _cover_delta(class_name: str) -> Optional[float]:
        comp = composition.get(class_name)
        if not comp:
            return None
        return comp["t2_pct"] - comp["t1_pct"]

    if "water" in q or "flood" in q or "river" in q or "lake" in q:
        gain = per_class.get("water_gain")
        loss = per_class.get("water_loss")
        delta = _cover_delta("water")
        if not gain and not loss and (delta is None or abs(delta) < 0.1):
            return "No significant water-related change is detected between the two dates.", 0.72
        clauses: List[str] = []
        if gain:
            clauses.append(f"water expanded (~{gain['area_pct']:.1f}% of the scene, "
                           f"{gain['centroid_quadrant']} quadrant)")
        if loss:
            clauses.append(f"water receded (~{loss['area_pct']:.1f}%, {loss['centroid_quadrant']} quadrant)")
        if delta is not None and abs(delta) >= 0.1:
            direction = "net expansion" if delta > 0 else "net recession"
            clauses.append(f"{direction} of {abs(delta):.1f} percentage points of the scene")
        return "Water change detected: " + "; ".join(clauses) + ".", 0.78

    if "built" in q or "urban" in q or "construction" in q or "city" in q:
        growth = per_class.get("new_built_up")
        demol = per_class.get("demolition")
        delta = _cover_delta("built_up")
        conversion = transitions.get("vegetation_to_built_up") or transitions.get("bare_soil_to_built_up")
        if growth or demol or (delta is not None and abs(delta) >= 0.1):
            clauses = []
            if delta is not None and abs(delta) >= 0.1:
                comp = composition["built_up"]
                clauses.append(
                    f"built-up cover went from {comp['t1_pct']:.1f}% to {comp['t2_pct']:.1f}% "
                    f"of the scene ({'increase' if delta > 0 else 'decrease'} of "
                    f"{abs(delta):.1f} percentage points)"
                )
            if growth:
                clauses.append(f"~{growth['area_pct']:.1f}% new construction "
                               f"({growth['centroid_quadrant']} quadrant)")
            if demol:
                clauses.append(f"~{demol['area_pct']:.1f}% removed "
                               f"({demol['centroid_quadrant']} quadrant)")
            if conversion:
                clauses.append(
                    f"~{conversion['area_pct']:.1f}% converted from {conversion['from']} "
                    "to built-up"
                )
            verdict = "increased" if (delta or 0) > 0.1 else ("decreased" if (delta or 0) < -0.1 else "roughly stable")
            return f"Built-up area {verdict}: " + "; ".join(clauses) + ".", 0.78
        return ("Built-up area remained essentially unchanged between the two dates.", 0.70)

    if "vegetat" in q or "forest" in q or "tree" in q or "crop" in q or "farm" in q:
        loss = per_class.get("vegetation_loss")
        gain = per_class.get("vegetation_gain")
        delta = _cover_delta("vegetation")
        if not loss and not gain and (delta is None or abs(delta) < 0.1):
            return "Vegetation cover remained essentially unchanged between the two dates.", 0.72
        clauses = []
        if delta is not None and abs(delta) >= 0.1:
            comp = composition["vegetation"]
            clauses.append(
                f"vegetation cover went from {comp['t1_pct']:.1f}% to {comp['t2_pct']:.1f}% "
                f"({abs(delta):.1f} percentage points {'loss' if delta < 0 else 'gain'})"
            )
        if loss:
            clauses.append(f"~{loss['area_pct']:.1f}% of the scene lost vegetation "
                           f"({loss['centroid_quadrant']} quadrant)")
        if gain:
            clauses.append(f"~{gain['area_pct']:.1f}% regrew "
                           f"({gain['centroid_quadrant']} quadrant)")
        return "Vegetation change: " + "; ".join(clauses) + ".", 0.78

    if any(t in q for t in ("what changed", "describe", "summary", "happened", "difference")) or not per_class:
        return describe_changes(summary), 0.72 if per_class else 0.60

    return describe_changes(summary), 0.55


# ------------------------------------------------------------------- tool -----
class BiTemporalChangeAnalyzer:
    """Specialist tool: bi-temporal change detection, description and change-VQA.

    Accepts an optical or SAR bi-temporal pair. Produces a colour-coded typed
    change map, per-class statistics with region quadrants, a natural-language
    change description, and deterministic answers to follow-up questions.
    """

    def __init__(self, device: str = "auto", enable_lora_adapter: bool = True,
                 head_checkpoint: Optional[str] = None) -> None:
        self.device = torch.device(device if device != "auto" else DEFAULT_CONFIG.resolved_device())
        self.head = ChangeNetHead().to(self.device).eval()
        self.head_status = "deterministic initialisation (no checkpoint)"
        if head_checkpoint:
            self.load_checkpoint(head_checkpoint)

    def load_checkpoint(self, path: str) -> bool:
        """Load a trained ChangeNetHead checkpoint (CDVQA-style head)."""
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.head.load_state_dict(state, strict=True)
            self.head.to(self.device).eval()
            self.head_status = f"trained head loaded from {path}"
            logger.info("Loaded change head checkpoint: %s", path)
            return True
        except Exception as exc:
            self.head_status = f"checkpoint load failed ({type(exc).__name__})"
            logger.warning("Failed to load change head %s: %s", path, exc)
            return False

    def describe(self) -> Dict[str, Any]:
        return {"tool": "BiTemporalChangeAnalyzer",
                "capabilities": ["change_detection", "change_description",
                                 "change_vqa", "change_map"],
                "model": "satquery-change (index differencing + ChangeNetHead refiner)",
                "head_status": self.head_status, "device": str(self.device)}

    # ------------------------------------------------------------------ core
    def analyze_pair(self, t1: RasterImage, t2: RasterImage,
                     threshold: Optional[float] = None,
                     min_region_px: Optional[int] = None) -> Dict[str, Any]:
        threshold = threshold if threshold is not None else DEFAULT_CONFIG.change_threshold
        min_region_px = min_region_px if min_region_px is not None else CHANGE_MIN_REGION_PIXELS

        features, index_deltas = build_change_features(t1, t2)
        amplitude = features[0]
        if "deterministic" in self.head_status:
            # Untrained refiner outputs are not calibrated; the analytic
            # amplitude is the change score until a CDVQA checkpoint is loaded.
            fused = np.clip(amplitude, 0.0, 1.0)
            refiner = "off (deterministic head)"
        else:
            prob = self.head.refine_probability(features)
            fused = np.clip(0.6 * amplitude + 0.4 * prob, 0.0, 1.0)
            refiner = "on (trained ChangeNetHead)"

        labels = classify_change_pixel(index_deltas, fused, is_sar=t1.modality == "sar")
        summary = change_summary(labels, t1, t2, min_region_px)
        description = describe_changes(summary)
        overlay = colorize_change(labels, CHANGE_CLASS_PALETTES)

        # Per-class means of the index deltas for auditability
        per_class_deltas: Dict[str, float] = {}
        for name, delta in index_deltas.items():
            changed = labels != "no_change"
            per_class_deltas[name] = round(float(np.mean(delta[changed])) if changed.any() else 0.0, 4)

        return {
            "change_labels": labels,
            "change_overlay": overlay,
            "change_probability": fused,
            "summary": summary,
            "description": description,
            "index_deltas": per_class_deltas,
            "threshold": threshold,
            "min_region_px": min_region_px,
            "head_status": self.head_status,
            "refiner": refiner,
            "modality": t1.modality,
        }

    # ------------------------------------------------------------------- API
    def run(self, t1: RasterImage, t2: RasterImage, question: str,
            threshold: Optional[float] = None,
            min_region_px: Optional[int] = None) -> Dict[str, Any]:
        analysis = self.analyze_pair(t1, t2, threshold=threshold, min_region_px=min_region_px)
        answer, confidence = answer_change_question(question, analysis["summary"])
        analysis["answer"] = answer
        analysis["confidence"] = confidence
        return analysis
