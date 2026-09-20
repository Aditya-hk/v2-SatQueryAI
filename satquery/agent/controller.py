"""Agentic controller for SatQuery AI.

Orchestrates one full query cycle:

1. input validation (format, CRS, dimensions, modality, pair compatibility);
2. natural-language intent classification with graceful fallback;
3. tool selection from the Specialist Model Registry and lazy instantiation;
4. execution with permitted parameters only;
5. confidence estimation (data quality, model, cross-modal agreement);
6. auditable execution summary + evidence-grounded result.

Only observable behaviour (task, tools, parameters, outputs, timings) is
recorded in the execution trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from satquery.agent.intent_classifier import IntentResult, classify_intent, fallback_task_for
from satquery.config import (
    CONFIDENCE_WEIGHTS,
    CLOUD_WARNING_FRACTION,
    DEFAULT_CONFIG,
    AppConfig,
)
from satquery.models.change_detection import BiTemporalChangeAnalyzer
from satquery.models.cross_modal_fusion import OpticalSARFusionTool
from satquery.models.registry import (
    TASK_CHANGE_ANALYSIS,
    TASK_CROSS_MODAL,
    TASK_SINGLE_IMAGE,
    SpecialistModelRegistry,
    default_registry,
)
from satquery.models.single_image_vqa import OpticalSingleImageTool
from satquery.utils.geospatial import (
    ImageMetadata,
    RasterImage,
    check_pair_compatibility,
    read_image,
    reproject_to,
    validate_crs,
)
from satquery.utils.logger import ExecutionSummary, get_logger

logger = get_logger("controller")


# ------------------------------------------------------------------ result ----
@dataclass
class AgentResult:
    """End-to-end outcome of one agentic query cycle."""

    answer: str
    confidence: float
    confidence_breakdown: Dict[str, Any]
    task: str
    sub_capability: str
    intent: Dict[str, Any]
    validation: Dict[str, Any]
    tools_used: List[str]
    images: List[RasterImage]
    metrics: Dict[str, Any] = field(default_factory=dict)
    visual_evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    summary: Optional[ExecutionSummary] = None
    execution_summary: Optional[Dict[str, Any]] = None
    status: str = "completed"

    @property
    def primary_image(self) -> Optional[RasterImage]:
        return self.images[0] if self.images else None


# ---------------------------------------------------------- confidence model --
def estimate_confidence(validation: Dict[str, Any], images: List[RasterImage],
                        model_confidences: List[float],
                        agreement: Optional[float]) -> Tuple[float, Dict[str, Any]]:
    """Composite confidence in [0, 1] from three interpretable components.

    * ``data_quality`` - CRS validity, pair compatibility, modality certainty,
      cloud cover (weighted per ``config.CONFIDENCE_WEIGHTS``).
    * ``model``        - mean specialist-tool confidence.
    * ``agreement``    - cross-view consistency (optical vs SAR class
      agreement, or typed-change consistency); neutral when not applicable.
    """
    n_images = len(images)
    dq = 1.0
    reasons: List[str] = []
    for img in images:
        if img.metadata.crs is None:
            dq -= 0.15
            reasons.append(f"{img.metadata.name}: no CRS")
        elif img.metadata.crs_valid is False:
            dq -= 0.10
            reasons.append(f"{img.metadata.name}: invalid CRS")
        if img.metadata.modality_confidence < 0.70:
            dq -= 0.05
            reasons.append(f"{img.metadata.name}: uncertain modality")
    for problem in validation.get("problems", []):
        dq -= 0.20
        reasons.append(f"pair problem: {problem[:80]}")
    n_warn = len(validation.get("warnings", []))
    if n_warn:
        dq -= 0.05 * min(n_warn, 3)
        reasons.append(f"{n_warn} validation warning(s)")
    for img in images:
        if img.modality != "sar":
            clouds = img.metadata.extra.get("cloud_fraction", 0.0)
            if clouds > CLOUD_WARNING_FRACTION:
                dq -= 0.08
                reasons.append(f"{img.metadata.name}: cloud cover {clouds:.0%}")
    dq = float(np.clip(dq, 0.20, 1.0))

    model = float(np.clip(np.mean(model_confidences) if model_confidences else 0.4, 0.0, 1.0))
    if agreement is None:
        # neutral prior when no second view exists; dominant-class clarity modulates it
        if n_images == 1:
            fractions = images[0].metadata.extra.get("fractions", {})
            dominant = max(fractions.values()) if fractions else 0.4
            agreement = float(np.clip(0.55 + 0.35 * dominant, 0.4, 0.95))
        else:
            agreement = 0.75

    weights = CONFIDENCE_WEIGHTS
    overall = weights["data_quality"] * dq + weights["model"] * model + weights["agreement"] * agreement
    breakdown = {
        "data_quality": round(dq, 4),
        "model": round(model, 4),
        "agreement": round(float(agreement), 4),
        "weighted_overall": round(float(np.clip(overall, 0.05, 0.99)), 4),
        "weights": dict(weights),
        "factors": reasons[:8],
    }
    return breakdown["weighted_overall"], breakdown


# ---------------------------------------------------------------- controller --
class AgentController:
    """Agentic controller: validate -> classify intent -> route -> execute ->
    estimate confidence -> emit auditable summary."""

    def __init__(self, config: Optional[AppConfig] = None,
                 registry: Optional[SpecialistModelRegistry] = None,
                 auto_reproject_pairs: bool = True) -> None:
        self.config = config or DEFAULT_CONFIG
        self.registry = registry or default_registry()
        self.auto_reproject_pairs = auto_reproject_pairs

    # ------------------------------------------------------------ validation
    def validate_images(self, images: List[RasterImage]) -> Dict[str, Any]:
        """Validate one or more images; returns a structured report."""
        problems: List[str] = []
        warnings: List[str] = []
        per_image: List[Dict[str, Any]] = []

        for img in images:
            meta = img.metadata
            ok, message = validate_crs(meta.crs)
            if meta.crs is None:
                warnings.append(f"{meta.name}: {message}")
            elif not ok:
                problems.append(f"{meta.name}: {message}")
            for issue in img.issues:
                warnings.append(f"{meta.name}: {issue}")
            if meta.modality_confidence < 0.60:
                warnings.append(
                    f"{meta.name}: modality inferred as '{meta.modality}' with low confidence "
                    f"({meta.modality_confidence:.2f}); verify sensor metadata."
                )
            clouds = img.metadata.extra.get("cloud_fraction", 0.0)
            if img.modality != "sar" and clouds > CLOUD_WARNING_FRACTION:
                warnings.append(
                    f"{meta.name}: ~{clouds * 100:.0f}% cloud cover detected; "
                    "confidence may be reduced in affected areas."
                )
            per_image.append({
                "name": meta.name,
                "format": meta.fmt,
                "width": meta.width,
                "height": meta.height,
                "bands": meta.bands,
                "crs": meta.crs,
                "crs_valid": meta.crs_valid,
                "modality": meta.modality,
                "modality_confidence": meta.modality_confidence,
                "size_mb": meta.size_mb,
            })

        pair_report: Dict[str, Any] = {}
        if len(images) == 2:
            primary, secondary = images
            cross_modal = primary.modality != secondary.modality
            pair_report = check_pair_compatibility(primary.metadata, secondary.metadata,
                                                   require_same_crs=not cross_modal)
            problems.extend(pair_report.get("problems", []))
            warnings.extend(pair_report.get("warnings", []))

        return {
            "valid": len(problems) == 0,
            "problems": problems,
            "warnings": warnings,
            "per_image": per_image,
            "pair": pair_report.get("info", {}),
            "n_images": len(images),
            "modalities": [img.modality for img in images],
        }

    def load_images(self, paths: List[Path]) -> List[RasterImage]:
        """Read image files through the geospatial ingestion layer."""
        images: List[RasterImage] = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = read_image(path, preview_max_side=self.config.preview_max_side)
            img.metadata.extra["cloud_fraction"] = round(
                _safe_cloud_fraction(img), 4)
            images.append(img)
        return images

    # ------------------------------------------------------------- main flow
    def process(self, query: str, images: List[RasterImage],
                auto_reproject: Optional[bool] = None) -> AgentResult:
        """Run one full agentic cycle. Never raises for analysis failures:
        problems are captured in ``AgentResult.errors`` / ``warnings``."""
        auto_reproject = self.auto_reproject_pairs if auto_reproject is None else auto_reproject
        summary = ExecutionSummary(
            query=query,
            inputs=[img.to_dict() for img in images],
            config=self.config.as_dict(),
        )
        warnings: List[str] = []
        errors: List[str] = []
        validation: Dict[str, Any] = {}
        task_for_routing: str = TASK_SINGLE_IMAGE

        try:
            # ------------------------------------------------ 1. validation ---
            with summary.timed_step("input_validation", "geospatial",
                                    "Validating formats, CRS, dimensions and modality"):
                validation = self.validate_images(images)
                if not validation["valid"]:
                    errors.extend(validation["problems"])
                warnings.extend(validation["warnings"])
                for img in images:
                    summary.step(
                        "metadata", "geospatial",
                        f"{img.metadata.name}: {img.metadata.fmt}, {img.metadata.width}x"
                        f"{img.metadata.height}, {img.metadata.bands} band(s), CRS "
                        f"{img.metadata.crs or 'MISSING'}, modality {img.metadata.modality} "
                        f"({img.metadata.modality_confidence:.2f})",
                    )
                if not images:
                    raise ValueError("No images provided; nothing to analyse.")

            # ------------------------------------- 1b. pair co-registration ---
            if len(images) == 2 and auto_reproject:
                images = self._maybe_reproject_pair(images, summary, warnings)
                validation = self.validate_images(images)
                warnings.extend(validation["warnings"])

            # --------------------------------------------- 2. intent parsing ---
            with summary.timed_step("intent_classification", "agent",
                                    "Parsing natural-language query"):
                intent = classify_intent(query)
                task_for_routing = intent.task
                fallback = fallback_task_for(intent.task, images)
                if fallback is not None:
                    fallback_task, reason = fallback
                    if fallback_task is None:
                        raise ValueError(reason)
                    summary.step("intent_fallback", "agent", reason, status="warning",
                                 requested_task=intent.task, fallback_task=fallback_task)
                    warnings.append(reason)
                    intent = IntentResult(
                        task=fallback_task,
                        sub_capability="vqa" if fallback_task == TASK_SINGLE_IMAGE
                        else intent.sub_capability,
                        confidence=max(0.30, intent.confidence - 0.25),
                        matched_keywords=intent.matched_keywords,
                        grounding_term=intent.grounding_term,
                        parameters=intent.parameters,
                        fallback_used=True,
                        fallback_reason=reason,
                        explanation=reason,
                    )
                    task_for_routing = fallback_task
                summary.step("intent_resolved", "agent", intent.explanation,
                             task=intent.task, sub_capability=intent.sub_capability,
                             confidence=intent.confidence,
                             keywords=intent.matched_keywords,
                             parameters=intent.parameters)
                if not self.registry.supports(task_for_routing, images):
                    raise ValueError(
                        f"Task '{task_for_routing}' is not supported for "
                        f"{len(images)} image(s) with modalities {validation['modalities']}."
                    )

            # ------------------------------------ 3. registry tool selection ---
            with summary.timed_step("tool_selection", "registry",
                                    "Selecting specialist tools from the registry"):
                tool_ids = self.registry.select_tools(task_for_routing, images)
                summary.selected_task = task_for_routing
                summary.step("tools_selected", "registry",
                             f"Registry selected {tool_ids} for task '{task_for_routing}'",
                             tools=tool_ids)

            # -------------------------------------------- 4. tool execution ----
            model_confidences: List[float] = []
            agreement: Optional[float] = None
            answer = ""
            metrics: Dict[str, Any] = {}
            visual_evidence: Dict[str, Any] = {}

            if task_for_routing == TASK_SINGLE_IMAGE:
                answer, model_confidences, metrics, visual_evidence = self._run_single_image(
                    summary, intent, images, tool_ids
                )
            elif task_for_routing == TASK_CHANGE_ANALYSIS:
                answer, model_confidences, agreement, metrics, visual_evidence = \
                    self._run_change_analysis(summary, intent, images, tool_ids)
            elif task_for_routing == TASK_CROSS_MODAL:
                answer, model_confidences, agreement, metrics, visual_evidence = \
                    self._run_cross_modal(summary, intent, images, tool_ids)
            else:  # defensive; registry.supports should have caught this
                raise ValueError(f"Unhandled task '{task_for_routing}'.")

            # ------------------------------------- 5. confidence estimation ----
            with summary.timed_step("confidence_estimation", "agent",
                                    "Estimating composite confidence"):
                images[0].metadata.extra["fractions"] = metrics.get("fractions", {})
                confidence, breakdown = estimate_confidence(
                    validation, images, model_confidences, agreement
                )
                summary.set_outcome(
                    confidence, breakdown,
                    result_digest={
                        "answer_preview": answer[:180],
                        "task": task_for_routing,
                        "tools": tool_ids,
                        "metrics_keys": sorted(metrics.keys()),
                    },
                )
                summary.step("confidence_computed", "agent",
                             f"Confidence {confidence:.3f} from {breakdown['weights']}",
                             **{k: v for k, v in breakdown.items() if k != "weights"})

            summary.finalize("completed")
            return AgentResult(
                answer=answer,
                confidence=confidence,
                confidence_breakdown=breakdown,
                task=task_for_routing,
                sub_capability=intent.sub_capability,
                intent=intent.to_dict(),
                validation=validation,
                tools_used=tool_ids,
                images=images,
                metrics=metrics,
                visual_evidence=visual_evidence,
                warnings=warnings,
                errors=errors,
                summary=summary,
                execution_summary=summary.to_dict(),
            )

        except Exception as exc:
            logger.exception("Agentic cycle failed")
            errors.append(f"{type(exc).__name__}: {exc}")
            summary.step("fatal", "controller", str(exc), status="error")
            summary.finalize("error")
            return AgentResult(
                answer=f"Analysis failed: {exc}",
                confidence=0.0,
                confidence_breakdown={"error": str(exc)},
                task=task_for_routing,
                sub_capability="n/a",
                intent={"task": task_for_routing, "error": str(exc)},
                validation=validation,
                tools_used=[],
                images=images,
                metrics={},
                visual_evidence={},
                warnings=warnings,
                errors=errors,
                summary=summary,
                execution_summary=summary.to_dict(),
                status="error",
            )

    # -------------------------------------------------------- task runners --
    def _run_single_image(self, summary: ExecutionSummary, intent: IntentResult,
                          images: List[RasterImage], tool_ids: List[str]
                          ) -> Tuple[str, List[float], Dict[str, Any], Dict[str, Any]]:
        image = images[0]
        confidences: List[float] = []
        answer = ""
        metrics: Dict[str, Any] = {"fractions": {}}
        visual_evidence: Dict[str, Any] = {}

        for tool_id in tool_ids:
            tool = self.registry.get_tool(tool_id, images)
            params: Dict[str, Any] = {"image": image.metadata.name}
            start = time.perf_counter()
            try:
                if tool_id == "single_image_vqa":
                    params["question"] = summary.query
                    output = tool.run(image, summary.query)
                    answer = output["answer"]
                    metrics.update({
                        "vqa_confidence": output["confidence"],
                        "dominant_class": output["evidence"].get("dominant"),
                        "fractions": output["evidence"].get("fractions", {}),
                    })
                    if output["evidence"].get("fractions"):
                        image.metadata.extra["fractions"] = output["evidence"]["fractions"]
                    visual_evidence.setdefault("encoder", output["evidence"].get("encoder"))
                elif tool_id == "scene_captioner":
                    if intent.sub_capability == "grounding":
                        params["grounding_query"] = summary.query
                        output = tool.ground(image, summary.query)
                        answer = _format_grounding_answer(output, summary.query)
                        metrics.update({
                            "grounding_count": output["count"],
                            "grounding_coverage_pct": output["coverage_pct"],
                            "grounding_term": output["term"],
                        })
                        visual_evidence["grounding"] = {
                            k: v for k, v in output.items()
                            if k in ("term", "term_display", "regions", "count",
                                     "coverage_pct", "total_pixels")
                        }
                        visual_evidence["grounding_regions"] = output["regions"]
                        visual_evidence["grounding_term"] = output["term"]
                    else:
                        output = tool.run(image)
                        caption = output["caption"]
                        # avoid duplicating the caption when the VQA answer is
                        # already a scene description for the same image
                        if caption[:48] not in answer:
                            answer = f"{caption} {answer}".strip()
                        metrics.update({
                            "caption_confidence": output["confidence"],
                            "caption_fractions": output["evidence"].get("fractions", {}),
                        })
                        visual_evidence["caption"] = {k: v for k, v in output["evidence"].items()
                                                      if k != "labels_pred"}
                        visual_evidence["caption_labels"] = output["evidence"].get("labels_pred", [])
                    confidences.append(output["confidence"])
                elif tool_id == "sar_structure_analyser":
                    if intent.sub_capability == "grounding":
                        params["grounding_query"] = summary.query
                        output = tool.ground(image, summary.query)
                        answer = _format_grounding_answer(output, summary.query)
                        metrics.update({
                            "grounding_count": output["count"],
                            "grounding_term": output["term"],
                        })
                        visual_evidence["grounding"] = {
                            k: v for k, v in output.items()
                            if k in ("term", "term_display", "regions", "count",
                                     "coverage_pct", "total_pixels")
                        }
                        visual_evidence["grounding_regions"] = output["regions"]
                        visual_evidence["grounding_term"] = output["term"]
                    else:
                        output = tool.run(image)
                        answer = output["answer"] if not answer else f"{answer} {output['answer']}"
                        metrics["sar_structure"] = output["evidence"]
                    confidences.append(output["confidence"])
                else:
                    raise ValueError(f"Unknown single-image tool '{tool_id}'.")
            finally:
                summary.record_tool_call(
                    tool_id, type(tool).__name__, getattr(tool, "model_ref", tool_id),
                    TASK_SINGLE_IMAGE, params,
                    duration_ms=(time.perf_counter() - start) * 1000.0,
                    status="ok", summary=_truncate(answer, 160),
                )

            if tool_id == "single_image_vqa":
                confidences.append(output.get("confidence", 0.5))

        return answer, confidences, metrics, visual_evidence

    def _run_change_analysis(self, summary: ExecutionSummary, intent: IntentResult,
                             images: List[RasterImage], tool_ids: List[str]
                             ) -> Tuple[str, List[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
        t1, t2 = images[0], images[1]
        tool = self.registry.get_tool("change_analyser", images)
        params = {
            "t1": t1.metadata.name,
            "t2": t2.metadata.name,
            "threshold": intent.parameters.get("change_threshold", self.config.change_threshold),
            "min_region_px": self.config.change_min_region_px,
        }
        start = time.perf_counter()
        try:
            output = tool.run(t1, t2, summary.query,
                              threshold=params["threshold"],
                              min_region_px=params["min_region_px"])
            summary_step = output["summary"]
            changed_pct = summary_step.get("changed_area_pct", 0.0)
            disturbance = summary_step.get("per_class", {}).get("surface_disturbance", {}).get("pixels", 0)
            changed_px = max(1, summary_step.get("changed_pixels", 1))
            agreement = float(np.clip(1.0 - 0.5 * (disturbance / changed_px), 0.45, 1.0))
            metrics = {
                "changed_area_pct": changed_pct,
                "changed_pixels": summary_step.get("changed_pixels"),
                "change_classes": {name: stats["area_pct"] for name, stats
                                   in summary_step.get("per_class", {}).items()},
                "threshold_used": params["threshold"],
                "head_status": output.get("head_status"),
            }
            visual_evidence = {
                "change_overlay": output["change_overlay"],
                "change_labels": output["change_labels"],
                "change_probability": output["change_probability"],
                "t1_preview": t1.preview,
                "t2_preview": t2.preview,
            }
            confidence = output["confidence"]
            answer = output["answer"]
        finally:
            summary.record_tool_call(
                "change_analyser", type(tool).__name__, getattr(tool, "model_ref", "satquery-change"),
                TASK_CHANGE_ANALYSIS, params,
                duration_ms=(time.perf_counter() - start) * 1000.0,
                status="ok", summary=_truncate(answer if 'answer' in dir() else "", 160),
            )
        return answer, [confidence], agreement, metrics, visual_evidence

    def _run_cross_modal(self, summary: ExecutionSummary, intent: IntentResult,
                         images: List[RasterImage], tool_ids: List[str]
                         ) -> Tuple[str, List[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
        optical, sar = (images[0], images[1]) if images[0].modality != "sar" else (images[1], images[0])
        tool = self.registry.get_tool("optical_sar_fusion", images)
        params = {"optical": optical.metadata.name, "sar": sar.metadata.name,
                  "query": summary.query}
        start = time.perf_counter()
        try:
            if intent.sub_capability == "grounding":
                output = tool.ground(optical, sar, summary.query)
                answer = _format_grounding_answer(output, summary.query)
                metrics = {
                    "grounding_term": output["term"],
                    "grounding_count": output["count"],
                    "fractions": {},
                }
                visual_evidence = {
                    "grounding_regions": output["regions"],
                    "grounding_term": output["term"],
                    "fused_overlay": None,
                }
                # fused overlay needs the class map; regenerate cheaply
                from satquery.models.cross_modal_fusion import (
                    _CLASS_COLORS,
                    CLASS_TO_IDX,
                    complementarity_analysis,
                )
                from satquery.utils.geospatial import colorize_classes

                fused, _ = complementarity_analysis(optical, sar, tool.encoder, tool.device)
                visual_evidence["fused_overlay"] = colorize_classes(
                    fused, {CLASS_TO_IDX[k]: v for k, v in _CLASS_COLORS.items()})
                visual_evidence["fused_classes"] = fused
                confidence = output["confidence"]
                agreement = _mean_agreement(output.get("evidence", {}).get("class_agreement_jaccard", {}))
            else:
                output = tool.run(optical, sar, summary.query)
                answer = output["answer"]
                metrics = {
                    "fractions": output.get("fractions", {}),
                    "class_agreement_jaccard": output["evidence"].get("class_agreement_jaccard"),
                    "complementarity_notes": output["evidence"].get("complementarity_notes", []),
                }
                visual_evidence = {
                    "fused_overlay": output["fused_overlay"],
                    "fused_classes": output["fused_classes"],
                    "optical_preview": optical.preview,
                    "sar_preview": sar.preview,
                }
                confidence = output["confidence"]
                agreement = _mean_agreement(output["evidence"].get("class_agreement_jaccard", {}))
        finally:
            summary.record_tool_call(
                "optical_sar_fusion", type(tool).__name__,
                getattr(tool, "model_ref", "satquery-fusion"),
                TASK_CROSS_MODAL, params,
                duration_ms=(time.perf_counter() - start) * 1000.0,
                status="ok", summary=_truncate(answer if 'answer' in dir() else "", 160),
            )
        return answer, [confidence], agreement, metrics, visual_evidence

    # ---------------------------------------------------------- pair repair --
    def _maybe_reproject_pair(self, images: List[RasterImage], summary: ExecutionSummary,
                              warnings: List[str]) -> List[RasterImage]:
        """Reproject the second image of a bi-temporal pair onto the first's CRS."""
        first, second = images
        if not (first.metadata.crs and second.metadata.crs):
            return images
        if str(first.metadata.crs) == str(second.metadata.crs):
            return images
        try:
            summary.step("reprojection", "geospatial",
                         f"CRS mismatch ({first.metadata.crs} vs {second.metadata.crs}); "
                         f"reprojecting '{second.metadata.name}' to {first.metadata.crs}.",
                         status="warning")
            warnings.append(
                f"Reprojected '{second.metadata.name}' from {second.metadata.crs} "
                f"to {first.metadata.crs} for pair compatibility."
            )
            fixed = reproject_to(second, first.metadata.crs)
            return [first, fixed]
        except Exception as exc:
            summary.step("reprojection", "geospatial",
                         f"Automatic reprojection failed: {exc}", status="warning")
            warnings.append(f"Automatic reprojection failed: {exc}")
            return images


# ------------------------------------------------------------------ helpers ---
def _mean_agreement(agreement: Dict[str, Any]) -> Optional[float]:
    values = [v for v in agreement.values() if isinstance(v, (int, float))]
    return float(np.mean(values)) if values else None


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_grounding_answer(output: Dict[str, Any], query: str) -> str:
    term = output.get("term_display", output.get("term", "target"))
    count = output.get("count", 0)
    coverage = output.get("coverage_pct", 0.0)
    regions = output.get("regions", [])
    if count == 0:
        return (f"No significant '{term}' regions were located in the image for the query "
                f"\"{_truncate(query, 80)}\".")
    listed = ", ".join(f"one in the {rg['quadrant']}" for rg in regions[:3])
    more = f" (+{count - 3} more)" if count > 3 else ""
    return (f"Grounded {count} '{term}' region(s) covering ~{coverage:.1f}% of the scene: "
            f"{listed}{more}.")


def _safe_cloud_fraction(img: RasterImage) -> float:
    try:
        from satquery.utils.geospatial import cloud_fraction

        return cloud_fraction(img)
    except Exception:
        return 0.0
