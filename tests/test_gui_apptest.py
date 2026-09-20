"""Headless GUI integration tests using Streamlit's official AppTest harness.

Verifies the full user journey inside the real app script: demo-scene
selection, query submission, agentic execution, and the rendered result
surface (answer, confidence, visual evidence, execution summary, reports).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP_SCRIPT = str(Path(__file__).resolve().parents[1] / "satquery" / "app" / "main_gui.py")

BI_TEMPORAL = "Bi-temporal pair (2023 vs 2024)"
CROSS_MODAL = "Cross-modal pair (optical + SAR)"
SINGLE_OPTICAL = "Single optical (Sentinel-2 style)"

CHANGE_QUERY = "What changed between these two dates, and where did the change occur?"
FUSION_QUERY = "Use the optical and SAR images together to identify built-up and water-covered regions."
CAPTION_QUERY = "Describe the land cover and major objects visible in this image."
GROUNDING_QUERY = "Highlight the water body referred to in the query."


def _run_app(demo_scene: str, query: str) -> AppTest:
    at = AppTest.from_file(APP_SCRIPT, default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    # select the demo scene
    scene_select = at.sidebar.selectbox[0]
    scene_select.set_value(demo_scene).run()
    assert not at.exception, at.exception

    # choose / type the query, then click run
    query_input = at.text_input[0]
    query_input.set_value(query).run()
    run_buttons = [b for b in at.button if "Run agentic analysis" in b.label]
    assert run_buttons, "Run button missing"
    run_buttons[0].click().run()
    assert not at.exception, at.exception
    return at


def _rendered_text(at: AppTest) -> str:
    """Aggregate every text-bearing rendered element into one blob."""
    parts = [block.value for block in at.markdown]
    parts += [block.value for block in at.caption]
    parts += [block.value for block in getattr(at, "text", [])]
    parts += [block.label for block in at.metric]
    parts += [block.label for block in at.button]
    parts += [block.label for block in at.expander]
    return "\n".join(str(part) for part in parts)


def test_app_boots_to_empty_state():
    at = AppTest.from_file(APP_SCRIPT, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.selectbox[0].value == "-"


@pytest.mark.parametrize(
    "scene,query,expected_task,expect_marker",
    [
        (SINGLE_OPTICAL, CAPTION_QUERY, "single_image_analysis", "Scene-mean spectral indices"),
        (SINGLE_OPTICAL, GROUNDING_QUERY, "single_image_analysis", "Grounded"),
        (BI_TEMPORAL, CHANGE_QUERY, "bi_temporal_change", "dominant change"),
        (CROSS_MODAL, FUSION_QUERY, "cross_modal_fusion", "optical-SAR agreement"),
    ],
)
def test_full_user_journeys(scene, query, expected_task, expect_marker):
    at = _run_app(scene, query)
    blob = _rendered_text(at)

    assert f"Task: `{expected_task}`" in blob
    assert expect_marker.lower() in blob.lower()

    # confidence metrics are rendered
    metric_labels = [m.label for m in at.metric]
    assert "Confidence" in metric_labels
    assert "Data quality" in metric_labels

    # auditable execution summary expander exists with tool-call content
    expander_labels = [e.label for e in at.expander]
    assert any("Auditable execution summary" in label for label in expander_labels)
    assert any("Agentic routing" in label for label in expander_labels)

    # download buttons for JSON + PDF reports (rendered as download_button elements)
    dl_labels = [b.label for b in at.get("download_button")]
    assert any("Download JSON report" in label for label in dl_labels)
    assert any("Download PDF report" in label for label in dl_labels)


def test_change_journey_renders_change_map():
    at = _run_app(BI_TEMPORAL, CHANGE_QUERY)
    blob = _rendered_text(at)
    assert "Spatial change map" in blob
    # T1 + T2 originals plus the typed change map are rendered as images
    assert len(at.image) >= 3, f"expected >=3 rendered images, got {len(at.image)}"


def test_fusion_journey_renders_class_maps_and_notes():
    at = _run_app(CROSS_MODAL, FUSION_QUERY)
    blob = _rendered_text(at)
    assert "Joint optical-SAR classification" in blob
    assert "Per-class maps" in blob
    assert "Complementary contributions" in blob or "Bare soil" in blob


def test_grounding_journey_renders_region_table():
    at = _run_app(SINGLE_OPTICAL, GROUNDING_QUERY)
    frames = at.dataframe
    assert frames, "grounding region table missing"
    assert "Grounding overlay" in _rendered_text(at)


def test_validation_expander_reports_pair_ok():
    at = _run_app(BI_TEMPORAL, CHANGE_QUERY)
    validation = [e for e in at.expander if "Input validation" in e.label]
    assert validation
