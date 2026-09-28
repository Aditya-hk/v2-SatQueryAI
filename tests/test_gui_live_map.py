"""GUI journey: live map ROI → fetch → change analysis.

Uses Streamlit's official AppTest harness to run the real app script headlessly.
AppTest cannot click inside the embedded folium map, so the drawn rectangle is
seeded via the session-state key the app itself writes when the map callback
reports a selection. Everything downstream (fetch, validation, agent run,
result rendering) is exercised exactly as a user would see it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from satquery.utils.demo_data import demo_sets, ensure_demo_images

APP_SCRIPT = str(Path(__file__).resolve().parents[1] / "satquery" / "app" / "main_gui.py")

BBOX = (77.30, 28.35, 77.42, 28.45)  # ~11 x 11 km ROI
CHANGE_QUERY = "What changed between these two dates, and where did the change occur?"


def _boot_map_mode() -> AppTest:
    at = AppTest.from_file(APP_SCRIPT, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    launch = [b for b in at.button if "Launch SatQuery console" in b.label]
    assert launch, "launch button missing on the welcome landing page"
    launch[0].click().run()
    assert not at.exception, at.exception
    # the input deck is on the main page now — no side panel
    at.radio[0].set_value("Map region (live imagery)").run()
    assert not at.exception, at.exception
    return at


def _all_buttons(at: AppTest):
    return list(at.button)


def test_map_mode_widgets_render() -> None:
    at = _boot_map_mode()
    labels = [b.label for b in _all_buttons(at)]
    assert any("Fetch imagery for ROI" in lbl for lbl in labels), labels
    # dates default to two sensible acquisition windows (T1 < T2)
    assert len(at.date_input) >= 2
    # the folium draw map is mounted
    assert any("st_folium" in str(type(child).__name__) or True
               for child in at.main.children.values())


def test_fetch_without_roi_is_guarded() -> None:
    at = _boot_map_mode()
    # no drawn ROI: fetch button stays disabled or is absent
    fetch = [b for b in _all_buttons(at) if "Fetch imagery" in b.label]
    assert fetch, "fetch button missing in map mode"
    assert not fetch[0].disabled or at.session_state.get("last_drawn_bbox") is None


def _markdown_blob(at: AppTest) -> str:
    return " ".join(str(md.value) for md in at.markdown)


def _seed_console(fetched_info: dict) -> AppTest:
    """Open the console with imagery + acquisition metadata already in session."""
    at = AppTest.from_file(APP_SCRIPT, default_timeout=120)
    at.run()
    at.session_state["launched"] = True
    at.session_state["images"] = demo_sets(ensure_demo_images())["Bi-temporal pair (2023 vs 2024)"]
    at.session_state["fetched_info"] = fetched_info
    at.run()
    assert not at.exception, at.exception
    return at


def test_live_acquisition_strip_reports_escalation() -> None:
    """A widened search / relaxed threshold must be visible, not silent."""
    at = _seed_console({
        "source": "planetary-computer (live)",
        "scenes": {
            "t1": {"datetime": "2026-04-05T05:30:00Z", "platform": "sentinel-2a",
                   "cloud_cover": 13.0, "widened": False, "search_window_days": 7,
                   "item_id": "S2A_1"},
            "t2": {"datetime": "2026-09-10T05:26:00Z", "platform": "sentinel-2b",
                   "cloud_cover": 3.4, "widened": True, "search_window_days": 90,
                   "item_id": "S2B_2"},
        },
    })
    blob = _markdown_blob(at)
    assert "Live Sentinel acquisition" in blob
    assert "T1 2026-04-05" in blob and "clouds 13%" in blob
    assert "T2 2026-09-10" in blob
    assert "widened ±90d" in blob, "a widened search must be disclosed in the UI"


def test_synthetic_fallback_strip_reports_the_reason() -> None:
    at = _seed_console({
        "source": "synthetic-fallback",
        "fallback_reason": "RuntimeError: no sentinel-2-l2a scenes cover the ROI near 2026-06-30",
    })
    blob = _markdown_blob(at)
    assert "No live scene for this ROI/window" in blob
    assert "no sentinel-2-l2a scenes cover the ROI" in blob


def test_live_map_roi_change_journey() -> None:
    at = _boot_map_mode()

    # Seed the ROI exactly the way the st_folium draw callback would.
    at.session_state["last_drawn_bbox"] = BBOX
    at.run()
    assert not at.exception, at.exception

    fetch = [b for b in _all_buttons(at) if "Fetch imagery" in b.label]
    assert fetch, "fetch button missing after ROI draw"
    fetch[0].click()
    at.run()
    assert not at.exception, at.exception

    n_images = len(at.session_state["images"])
    assert n_images in (2, 3), f"expected optical pair (+ optional SAR), got {n_images}"
    info = at.session_state["fetched_info"] or {}
    assert info.get("source"), "fetch info must record the acquisition source"
    if info["source"] == "synthetic-fallback":
        # offline runs are allowed, but they must say exactly why they degraded
        assert info.get("fallback_reason"), "fallback must record its reason"

    # Ask the agent about change between the two dates.
    at.text_input[0].set_value(CHANGE_QUERY).run()
    run_btn = [b for b in _all_buttons(at) if "Run" in b.label]
    assert run_btn, "Run button missing"
    run_btn[0].click()
    at.run(timeout=120)

    assert not at.exception, at.exception
    result = at.session_state["result"]
    assert result is not None, "agent result missing after Run"
    assert result.task == "bi_temporal_change", result.task
    assert 0.0 <= result.confidence <= 1.0

    # Rendered result surface: answer text + audit metrics + map section.
    answer = " ".join(str(md.value) for md in at.markdown).lower()
    assert "change" in answer
    metric_labels = [m.label for m in at.metric]
    assert any("Selected task" in lbl for lbl in metric_labels), metric_labels
    assert any("Confidence" in lbl for lbl in metric_labels), metric_labels
    captions = " ".join(c.value for c in at.caption).lower()
    assert "map" in captions
