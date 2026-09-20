"""End-to-end test suite for SatQuery AI (SIH-167).

Covers: demo data generation, intent routing, input validation (CRS,
dimensions, pairs), all four mandatory specialist workflows, the agentic
controller, confidence estimation, reports (JSON + PDF), the registry, and
the LoRA training/checkpoint loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from satquery.agent.controller import AgentController, estimate_confidence
from satquery.agent.intent_classifier import classify_intent, fallback_task_for
from satquery.config import CHANGE_CLASS_ORDER, AppConfig
from satquery.models.change_detection import BiTemporalChangeAnalyzer
from satquery.models.cross_modal_fusion import OpticalSARFusionTool
from satquery.models.registry import default_registry
from satquery.models.single_image_vqa import (
    OpticalSingleImageTool,
    RSVisionEncoder,
    SceneCaptionerTool,
    ground_regions_in_image,
    locate_term,
)
from satquery.utils.demo_data import build_layout, ensure_demo_images, make_raster
from satquery.utils.geospatial import (
    ImageMetadata,
    RasterImage,
    _rgb_band_indices,
    _rgb_preview,
    check_pair_compatibility,
    classify_landcover,
    compute_ndvi,
    read_image,
    validate_crs,
)
from satquery.utils.logger import ExecutionSummary
from satquery.utils.report import build_report_payload, render_pdf_report, save_json_report
from satquery.training.fine_tune_bigearthnet import TrainConfig, train_lora


# ---------------------------------------------------------------- fixtures ----
@pytest.fixture(scope="session")
def demo() -> dict:
    return ensure_demo_images()


@pytest.fixture(scope="session")
def controller() -> AgentController:
    return AgentController()


# ---------------------------------------------------------------- demo data ---
def test_demo_images_are_georeferenced(demo):
    for name, img in demo.items():
        assert img.metadata.crs == "EPSG:4326", name
        assert img.metadata.transform is not None
        assert img.data.dtype == np.float32
        assert 0.0 <= float(img.data.min()) and float(img.data.max()) <= 1.0 + 1e-6


def test_demo_modalities(demo):
    assert demo["single_optical"].modality == "optical"
    assert demo["single_sar"].modality == "sar"
    assert demo["fusion_sar"].modality == "sar"


def test_bitemporal_demo_contains_change(demo):
    """The T2 scene must contain new built-up relative to T1."""
    from satquery.utils.geospatial import fraction_of_classes

    f1 = fraction_of_classes(classify_landcover(demo["t1"]))
    f2 = fraction_of_classes(classify_landcover(demo["t2"]))
    assert f2["built_up"] > f1["built_up"]
    assert f2["water"] < f1["water"]


# ------------------------------------------------------------------- intent ---
@pytest.mark.parametrize(
    "query,expected_task,expected_sub",
    [
        ("Describe the land cover and major objects visible in this image.",
         "single_image_analysis", "captioning"),
        ("What changed between these two dates, and where did the change occur?",
         "bi_temporal_change", "change_description"),
        ("Use the optical and SAR images together to identify built-up and water-covered regions.",
         "cross_modal_fusion", "vqa"),
        ("Highlight the water body referred to in the query.",
         "single_image_analysis", "grounding"),
        ("Has the built-up area increased, decreased, or remained unchanged?",
         "bi_temporal_change", "change_vqa"),
        ("Is there any vegetation visible?", "single_image_analysis", "vqa"),
    ],
)
def test_intent_routing(query, expected_task, expected_sub):
    intent = classify_intent(query)
    assert intent.task == expected_task
    assert intent.sub_capability == expected_sub
    assert 0.0 < intent.confidence <= 1.0


def test_intent_fallback_requires_pair():
    fallback = fallback_task_for("bi_temporal_change", images=[object()])
    assert fallback is not None
    task, reason = fallback
    assert task is None
    assert "exactly 2" in reason


def test_cross_modal_fallback_on_wrong_modality():
    fallback = fallback_task_for("cross_modal_fusion", images=[object(), object()])
    assert fallback is not None and fallback[0] == "single_image_analysis"


# --------------------------------------------------------------- validation ---
def test_validate_crs():
    ok, msg = validate_crs("EPSG:4326")
    assert ok
    ok, msg = validate_crs(None)
    assert not ok


def test_pair_validation_detects_dimension_mismatch(demo):
    t1 = demo["t1"]
    shrunk = make_raster("tiny.tif", demo["t1"].data[: 64, : 64, :])
    report = check_pair_compatibility(t1.metadata, shrunk.metadata)
    assert not report["valid"]
    assert any("dimensions" in p.lower() for p in report["problems"])


def test_pair_validation_accepts_bitemporal_pair(demo):
    report = check_pair_compatibility(demo["t1"].metadata, demo["t2"].metadata)
    assert report["valid"], report["problems"]


# -------------------------------------------------- single-image specialist ---
def test_vqa_optical(demo, controller):
    img = demo["single_optical"]
    result = controller.process("Is there any water visible in this image?", [img])
    assert result.task == "single_image_analysis"
    assert result.status == "completed"
    assert 0.0 < result.confidence <= 1.0
    assert "water" in result.answer.lower()


def test_captioning(demo):
    tool = SceneCaptionerTool()
    output = tool.run(demo["single_optical"])
    caption = output["caption"]
    assert caption
    assert 0.0 < output["confidence"] <= 1.0
    # detailed caption: composition + spatial layout + indices + texture/cloud
    assert caption.count(".") >= 4
    assert "optical scene" in caption.lower()
    assert "land cover" in caption.lower()
    assert "scene-mean spectral indices" in caption.lower()
    assert "ndvi" in caption.lower() and "ndwi" in caption.lower() and "ndbi" in caption.lower()
    assert "cloud cover" in caption.lower()
    assert "%" in caption
    evidence = output["evidence"]
    assert evidence["fractions"] and evidence["dominant"]
    assert {"ndvi_mean", "ndwi_mean", "ndbi_mean"} <= set(evidence)
    assert isinstance(evidence["layout"], list)


def test_sar_captioning_is_detailed(demo):
    tool = SceneCaptionerTool()
    output = tool.run(demo["single_sar"])
    caption = output["caption"]
    assert "sar" in caption.lower()
    assert "built-up" in caption.lower() and "water" in caption.lower()
    assert "backscatter" in caption.lower()
    assert "strong-scatterer" in caption.lower()
    assert caption.count(".") >= 3


def test_grounding_locates_water(demo):
    assert locate_term("Highlight the water body") == "water"
    output = ground_regions_in_image(demo["single_optical"], "Highlight the water body")
    assert output["count"] >= 1
    assert output["regions"][0]["bbox"] and len(output["regions"][0]["bbox"]) == 4


def test_vqa_sar(demo):
    tool = OpticalSingleImageTool()
    output = tool.run(demo["single_sar"], "Describe the land cover.")
    assert output["modality"] == "sar"
    assert output["answer"]
    assert 0.0 < output["confidence"] <= 1.0


# ------------------------------------------------------ change specialist ----
def test_change_description_and_map(demo):
    tool = BiTemporalChangeAnalyzer()
    output = tool.run(demo["t1"], demo["t2"],
                      "What changed between these two dates?")
    assert output["answer"]
    assert output["summary"]["changed_area_pct"] > 0.0
    assert output["change_labels"].shape == demo["t1"].data.shape[:2]
    assert set(str(x) for x in np.unique(output["change_labels"])) <= \
        set(CHANGE_CLASS_ORDER) | {"no_change"}


def test_change_builtup_question(demo, controller):
    result = controller.process(
        "Has the built-up area increased, decreased, or remained unchanged?",
        [demo["t1"], demo["t2"]],
    )
    assert result.task == "bi_temporal_change"
    assert "built-up" in result.answer.lower()
    assert "composition" in result.metrics or "change_classes" in result.metrics


def test_change_rejects_single_image(demo, controller):
    result = controller.process("What changed between these dates?", [demo["t1"]])
    assert result.status == "error"
    assert result.errors


# ------------------------------------------------------ fusion specialist ----
def test_fusion_joint_analysis(demo, controller):
    result = controller.process(
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        [demo["fusion_optical"], demo["fusion_sar"]],
    )
    assert result.task == "cross_modal_fusion"
    assert result.metrics.get("class_agreement_jaccard")
    notes = result.metrics.get("complementarity_notes", [])
    assert notes and "Bare soil" in notes[-1]
    assert result.visual_evidence.get("fused_overlay") is not None


def test_fusion_grounding(demo):
    tool = OpticalSARFusionTool()
    output = tool.ground(demo["fusion_optical"], demo["fusion_sar"],
                         "Highlight the water body")
    assert output["term"] == "water"
    assert output["count"] >= 1


# -------------------------------------------------- agentic orchestration -----
def test_execution_summary_auditability(demo, controller):
    result = controller.process(
        "Describe the land cover.", [demo["single_optical"]]
    )
    payload = result.summary.to_dict()
    for key in ("selected_task", "selected_tools", "execution_trace", "tool_calls",
                "confidence", "query"):
        assert key in payload, key
    assert payload["selected_task"] == result.task
    assert set(payload["selected_tools"]) == set(result.tools_used)
    for call in payload["tool_calls"]:
        assert {"tool_id", "tool_name", "model_ref", "parameters", "duration_ms"} <= set(call)


def test_confidence_estimation_components(demo, controller):
    result = controller.process("Describe the land cover.", [demo["single_optical"]])
    breakdown = result.confidence_breakdown
    assert {"data_quality", "model", "agreement", "weights"} <= set(breakdown)
    conf, _ = estimate_confidence(
        {"problems": [], "warnings": []}, [demo["single_optical"]], [0.9], 0.8
    )
    assert 0.6 < conf <= 0.99


def test_registry_selection_logic(demo):
    registry = default_registry()
    assert registry.select_tools("single_image_analysis", [demo["single_optical"]]) == \
        ["single_image_vqa", "scene_captioner"]
    assert registry.select_tools("single_image_analysis", [demo["single_sar"]]) == \
        ["single_image_vqa", "sar_structure_analyser"]
    assert registry.select_tools("bi_temporal_change", [demo["t1"], demo["t2"]]) == \
        ["change_analyser"]
    assert registry.select_tools("cross_modal_fusion",
                                 [demo["fusion_optical"], demo["fusion_sar"]]) == \
        ["optical_sar_fusion"]
    assert registry.describe()["tools"]


# ----------------------------------------------------------------- reports ----
def test_reports_json_and_pdf(demo, controller, tmp_path):
    result = controller.process("Describe the land cover.", [demo["single_optical"]])
    payload = build_report_payload(result.summary, result)
    json_path = save_json_report(payload, tmp_path / "report.json")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["selected_task"] == result.task
    assert parsed["answer"] == result.answer

    pdf_bytes = render_pdf_report(payload)
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf_bytes
    (tmp_path / "report.pdf").write_bytes(pdf_bytes)


# ------------------------------------------------------- rasterio ingestion ---
@pytest.mark.skipif(
    not __import__("satquery.utils.geospatial", fromlist=["HAS_RASTERIO"]).HAS_RASTERIO,
    reason="rasterio not installed",
)
def test_geotiff_roundtrip_metadata(tmp_path, demo):
    """A saved demo GeoTIFF must round-trip CRS + transform through read_image."""
    import rasterio
    from rasterio.transform import Affine

    from satquery.utils.demo_data import save_raster

    path = tmp_path / "roundtrip.tif"
    img = demo["single_optical"]
    save_raster(img, path)
    with rasterio.open(path) as src:
        assert src.crs is not None
        assert src.transform != Affine.identity()
    reloaded = read_image(path)
    assert reloaded.metadata.crs == "EPSG:4326"
    assert reloaded.metadata.width == img.metadata.width


# ---------------------------------------------------------- LoRA adaptation ---
def test_lora_training_and_checkpoint_roundtrip(tmp_path):
    config = TrainConfig(out_dir=tmp_path / "ckpt", epochs=1, batch_size=8,
                         log_every=10, synthetic_samples=64)
    report = train_lora(config, device="cpu")
    ckpt = Path(report["checkpoint"])
    assert ckpt.exists()
    assert report["injected_modules"], "LoRA adapters must be injected"
    assert report["history"] and "train_loss" in report["history"][0]

    encoder = RSVisionEncoder(in_channels=3, checkpoint_dir=tmp_path / "ckpt")
    assert encoder.adapter_loaded
    assert "merged" in encoder.adapter_status or "loaded" in encoder.adapter_status


def test_encoder_forward_shapes():
    encoder = RSVisionEncoder(enable_lora_adapter=False)
    x = torch.randn(2, 3, 64, 64)
    tokens = encoder.core(x)
    assert tokens.shape == (2, 16, encoder.embed_dim)


# --------------------------------------------------- band order & previews ---
def test_rgb_band_selection_uses_b04_b03_b02_for_multispectral_stacks():
    """Sentinel-2 B01..B12 stacks must preview as true colour, not B01/B02/B03."""
    stack = np.zeros((16, 16, 12), np.float32)
    stack[:, :, 3] = 0.8                      # B04 (red) is channel 3
    assert _rgb_band_indices(stack) == (3, 2, 1)
    preview = _rgb_preview(stack)
    assert preview.shape == (16, 16, 3)


def test_rgb_band_selection_honours_band_names():
    stack = np.zeros((8, 8, 5), np.float32)
    names = ["B01", "B02", "B03", "B04", "B08"]
    assert _rgb_band_indices(stack, names) == (3, 2, 1)
    assert _rgb_band_indices(stack, ["red", "green", "blue", "nir", "x"]) == (0, 1, 2)


def test_rgb_preview_excludes_nir_for_4band_rgbn():
    """NIR energy must not leak into the visible preview (classic false-colour bug)."""
    img_arr = np.zeros((8, 8, 4), np.float32)
    img_arr[:4, :4, 0] = 0.9                  # red top-left
    img_arr[4:, 4:, 3] = 1.0                  # NIR bottom-right
    preview = _rgb_preview(img_arr)
    assert preview[:4, :4, 0].mean() > 200    # red visible where red band bright
    assert preview[4:, 4:, 0].mean() < 10     # NIR quadrant stays dark in R
    assert preview[:, :, 1].max() < 10        # G/B channels empty


def test_ndvi_uses_rgn_package_order():
    """Vegetation (low red ch0, high NIR ch3) must yield positive NDVI."""
    veg = np.zeros((8, 8, 4), np.float32)
    veg[:, :, 0] = 0.1                        # red
    veg[:, :, 3] = 0.8                        # NIR
    meta = ImageMetadata(name="veg.tif", path=Path("veg.tif"), fmt="TIFF",
                         width=8, height=8, bands=4)
    ndvi = compute_ndvi(RasterImage(metadata=meta, data=veg))
    assert float(ndvi.mean()) > 0.7

    waterish = np.zeros((8, 8, 4), np.float32)
    waterish[:, :, 0] = 0.6                   # bright red, no NIR
    ndvi_neg = compute_ndvi(RasterImage(metadata=meta, data=waterish))
    assert float(ndvi_neg.mean()) < -0.7


# ------------------------------------------------------------ edge handling ---
def test_unsupported_format_rejected(tmp_path):
    bad = tmp_path / "scan.pdf"
    bad.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(ValueError):
        read_image(bad)


def test_low_quality_input_lowers_confidence(demo):
    controller = AgentController()
    noisy = make_raster("noisy.tif", np.clip(
        demo["single_optical"].data + np.random.default_rng(0).normal(
            0, 0.25, demo["single_optical"].data.shape).astype(np.float32), 0, 1))
    result = controller.process("Describe the land cover.", [noisy])
    assert result.status == "completed"
    assert result.confidence < 0.95


def test_app_config_defaults():
    config = AppConfig()
    assert config.resolved_device() in {"cpu", "cuda", "mps"}
    assert config.as_dict()["change_threshold"] == config.change_threshold
