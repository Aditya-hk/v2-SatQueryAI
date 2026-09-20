"""SatQuery AI - interactive Streamlit GUI (SIH Problem Statement 167).

Run with:  streamlit run satquery/app/main_gui.py

Features
--------
* draw a region of interest directly on an interactive map and fetch live
  Sentinel-2 optical (and optionally Sentinel-1 SAR) imagery for it;
* multi-input uploads (GeoTIFF/TIFF/PNG/JPEG) with demo scenes;
* input validation panel (CRS, dimensions, modality, pair compatibility);
* side-by-side visual comparison: original(s) vs change map / grounding /
  fused optical-SAR overlay, plus per-class maps;
* confidence gauge with component breakdown;
* auditable execution summary drawer (task, tools, parameters, trace);
* downloadable JSON and PDF reports, plus geospatial GeoTIFF/GeoJSON exports;
* results rendered on an interactive map (change / classification overlays
  and grounded region rectangles in true geographic coordinates).
"""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datetime import date, timedelta

import numpy as np
import streamlit as st

# Make the package importable when launched via `streamlit run` from any CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:  # pragma: no cover - optional dependency
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium

    HAS_FOLIUM_UI = True
except Exception:  # pragma: no cover
    folium = None  # type: ignore
    Draw = None  # type: ignore
    st_folium = None  # type: ignore
    HAS_FOLIUM_UI = False

st.set_page_config(
    page_title="SatQuery AI - Agentic Remote Sensing Framework",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="auto",
)

from satquery import APP_NAME, APP_VERSION, PROBLEM_STATEMENT_ID
from satquery.agent.controller import AgentController
from satquery.config import (
    CHANGE_CLASS_DISPLAY,
    CHANGE_CLASS_ORDER,
    CHANGE_CLASS_PALETTES,
    CLASS_DISPLAY_NAMES,
    LANDCOVER_CLASS_NAMES,
    LOW_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)
from satquery.models.registry import default_registry
from satquery.utils.demo_data import demo_sets, ensure_demo_images
from satquery.utils.live_imagery import (
    build_result_map,
    change_labels_to_indices,
    class_map_to_geotiff,
    fetch_roi_imagery,
    grounding_geojson,
    image_to_geotiff,
    region_polygons_geojson,
    validate_bbox,
)
from satquery.utils.geospatial import (
    RasterImage,
    SUPPORTED_EXTENSIONS,
    colorize_classes,
    colorize_change,
)
from satquery.utils.logger import get_logger
from satquery.utils.report import build_report_payload, render_pdf_report, save_json_report

logger = get_logger("gui")

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

# Map origin when nothing has been acquired yet (matches the demo scenes).
DEFAULT_MAP_CENTER = (20.10, 77.10)








def inject_theme_css() -> None:
    """Design-system CSS: hero, cards, pills, legends, chips, empty state.

    Uses translucent surfaces so it composes with both the dark ground-station
    theme (.streamlit/config.toml) and the default light theme.
    """
    st.markdown(
        """
        <style>
        :root {
          --sq-accent:#4f8cff; --sq-accent-soft:rgba(79,140,255,.14);
          --sq-good:#2fbf71; --sq-warn:#f5a524; --sq-bad:#f04438;
          --sq-card:rgba(127,140,170,.10); --sq-border:rgba(127,140,170,.24);
          --sq-muted:rgba(127,140,170,.9);
        }
        h1, h2, h3 { letter-spacing:.2px; }
        .stImage img, [data-testid="stImage"] img {
            border-radius:10px; border:1px solid var(--sq-border);
            box-shadow:0 2px 10px rgba(0,0,0,.18);
        }
        .stExpander, [data-testid="stExpander"] {
            border:1px solid var(--sq-border) !important;
            border-radius:12px !important; overflow:hidden;
        }
        button[kind="primary"] { box-shadow:0 2px 14px rgba(79,140,255,.35); }
        .sq-hero { display:flex; align-items:center; gap:16px;
                   background:linear-gradient(135deg, var(--sq-accent-soft), rgba(47,191,113,.08));
                   border:1px solid var(--sq-border); border-radius:16px;
                   padding:16px 22px; margin-bottom:14px; }
        .sq-hero .sq-logo { font-size:2.4rem; line-height:1; }
        .sq-hero h1 { margin:0; font-size:1.65rem; }
        .sq-hero p { margin:2px 0 0 0; color:var(--sq-muted); font-size:.95rem; }
        .sq-answer { background:var(--sq-card); border:1px solid var(--sq-border);
                     border-left:4px solid var(--sq-accent); border-radius:12px;
                     padding:14px 16px; font-size:1.03rem; line-height:1.5; }
        .sq-pill { display:inline-block; padding:2px 12px; border-radius:999px;
                   color:#fff; font-size:.82rem; font-weight:700; letter-spacing:.4px;
                   vertical-align:middle; margin-left:8px; }
        .sq-legend { display:flex; flex-wrap:wrap; gap:6px 14px; font-size:.82rem;
                     margin:4px 0 10px 0; color:var(--sq-muted); }
        .sq-legend .sw { width:12px; height:12px; border-radius:3px; display:inline-block;
                         margin-right:5px; vertical-align:-1px; }
        .sq-chip { background:var(--sq-card); border:1px solid var(--sq-border);
                   border-radius:10px; padding:8px 10px; font-size:.84rem;
                   line-height:1.45; margin-bottom:6px; }
        .sq-chip .dim { color:var(--sq-muted); }
        .sq-srcstrip { display:flex; flex-wrap:wrap; gap:8px; margin:2px 0 8px 0; }
        .sq-srcstrip .tag { background:var(--sq-accent-soft);
                            border:1px solid var(--sq-border); border-radius:999px;
                            padding:3px 10px; font-size:.78rem; }
        .sq-toolcard { background:var(--sq-card); border:1px solid var(--sq-border);
                       border-radius:12px; padding:10px 14px; margin-bottom:8px; }
        .sq-toolcard .meta { color:var(--sq-muted); font-size:.8rem; }
        .sq-empty { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
                    gap:12px; margin:10px 0 16px 0; }
        .sq-empty .cell { background:var(--sq-card); border:1px solid var(--sq-border);
                          border-radius:14px; padding:16px; }
        .sq-empty .cell .ico { font-size:1.6rem; }
        .sq-empty .cell h4 { margin:6px 0 4px 0; }
        .sq-empty .cell p { margin:0; color:var(--sq-muted); font-size:.88rem; line-height:1.45; }
        .sq-example { background:var(--sq-card); border:1px dashed var(--sq-border);
                      border-radius:10px; padding:8px 12px; font-size:.9rem; margin-top:8px; }

        /* ============ motion design system (SatQuery UI v2) ============ */
        :root{
          --sq-accent-2:#8b5cf6; --sq-cyan:#2dd4bf; --sq-border-2:rgba(127,140,170,.40);
          --sq-mono:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace;
          --sq-ease:cubic-bezier(.22,.68,.24,1);
        }
        @keyframes sq-fade-up{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
        @keyframes sq-fade-in{from{opacity:0}to{opacity:1}}
        @keyframes sq-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
        @keyframes sq-pulse{0%,100%{opacity:.5;transform:scale(.9)}50%{opacity:1;transform:scale(1.08)}}
        @keyframes sq-spin{to{transform:rotate(360deg)}}
        @keyframes sq-aurora{0%,100%{transform:translate3d(-5%,-3%,0) scale(1.08)}
                             50%{transform:translate3d(5%,4%,0) scale(1.16)}}
        @keyframes sq-shimmer{0%{background-position:-420px 0}100%{background-position:420px 0}}
        @keyframes sq-scan{0%{transform:translateX(-120%)}100%{transform:translateX(220%)}}
        @keyframes sq-grow{from{width:0}}
        @keyframes sq-blink{0%,49%{opacity:1}50%,100%{opacity:0}}
        @keyframes sq-dash{from{stroke-dashoffset:100}}
        @keyframes sq-slide-in{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:none}}

        /* ---------- chrome: scrollbars, focus, type rhythm ---------- */
        *::-webkit-scrollbar{width:10px;height:10px}
        *::-webkit-scrollbar-track{background:transparent}
        *::-webkit-scrollbar-thumb{background:var(--sq-border-2);border-radius:8px;
            border:2px solid transparent;background-clip:content-box}
        *::-webkit-scrollbar-thumb:hover{background:var(--sq-accent);background-clip:content-box}
        ::selection{background:rgba(79,140,255,.35)}
        h1,h2,h3{letter-spacing:-.2px;font-weight:700}
        [data-testid="stMetricValue"]{font-variant-numeric:tabular-nums;letter-spacing:-.5px}
        button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{
            outline:2px solid var(--sq-accent)!important;outline-offset:2px}
        button[kind="primary"]{
            background:linear-gradient(120deg,var(--sq-accent),var(--sq-accent-2))!important;
            border:0!important;box-shadow:0 8px 26px rgba(79,140,255,.36);
            position:relative;overflow:hidden;
            transition:transform .22s var(--sq-ease),box-shadow .22s var(--sq-ease)}
        button[kind="primary"]:hover{transform:translateY(-2px);
            box-shadow:0 14px 34px rgba(79,140,255,.48)}
        button[kind="primary"]::after{content:'';position:absolute;top:0;bottom:0;left:0;width:36%;
            background:linear-gradient(100deg,transparent,rgba(255,255,255,.38),transparent);
            animation:sq-scan 3.6s var(--sq-ease) infinite}
        .stImage img,[data-testid="stImage"] img{
            border-radius:12px;border:1px solid var(--sq-border);
            box-shadow:0 10px 30px rgba(0,0,0,.26);
            transition:transform .35s var(--sq-ease),box-shadow .35s var(--sq-ease)}
        .stImage img:hover,[data-testid="stImage"] img:hover{transform:translateY(-3px) scale(1.004);
            box-shadow:0 18px 44px rgba(0,0,0,.36)}
        .stExpander,[data-testid="stExpander"]{background:linear-gradient(180deg,rgba(127,140,170,.05),transparent);
            transition:border-color .3s var(--sq-ease)}
        .stExpander:hover,[data-testid="stExpander"]:hover{border-color:var(--sq-border-2)!important}

        /* ---------- landing ---------- */
        .sq-land{position:relative;overflow:hidden;border-radius:22px;margin:4px 0 18px 0;
            padding:36px 34px 30px 34px;border:1px solid var(--sq-border);
            background:radial-gradient(1200px 460px at 10% -12%,rgba(79,140,255,.22),transparent 62%),
                       radial-gradient(900px 420px at 94% 4%,rgba(139,92,246,.20),transparent 60%),
                       linear-gradient(180deg,#0a1120,#0c1426);
            animation:sq-fade-in .8s var(--sq-ease) both}
        .sq-land .aurora{position:absolute;left:-10%;right:-10%;top:-32%;height:126%;pointer-events:none;
            filter:blur(28px);animation:sq-aurora 19s ease-in-out infinite;
            background:radial-gradient(520px 260px at 22% 42%,rgba(79,140,255,.32),transparent 70%),
                       radial-gradient(560px 300px at 74% 24%,rgba(139,92,246,.26),transparent 72%),
                       radial-gradient(440px 240px at 54% 78%,rgba(45,212,191,.20),transparent 70%)}
        .sq-land .mesh{position:absolute;inset:0;pointer-events:none;opacity:.5;
            background-image:linear-gradient(rgba(127,140,170,.10) 1px,transparent 1px),
                             linear-gradient(90deg,rgba(127,140,170,.10) 1px,transparent 1px);
            background-size:46px 46px;
            -webkit-mask-image:radial-gradient(78% 70% at 28% 28%,#000 28%,transparent 78%);
            mask-image:radial-gradient(78% 70% at 28% 28%,#000 28%,transparent 78%)}
        .sq-land .inner{position:relative;z-index:3;max-width:640px}
        .sq-badge{display:inline-flex;align-items:center;gap:9px;border-radius:999px;
            padding:5px 14px;font-size:.72rem;letter-spacing:1.1px;text-transform:uppercase;
            color:#cfe0ff;background:rgba(79,140,255,.14);border:1px solid var(--sq-border-2);
            animation:sq-fade-in .9s var(--sq-ease) both}
        .sq-badge .dot{width:7px;height:7px;border-radius:50%;background:var(--sq-good);
            animation:sq-pulse 2.2s ease-in-out infinite}
        .sq-h1{font-size:2.45rem;line-height:1.08;margin:18px 0 12px 0;font-weight:800;letter-spacing:-1.1px;
            background:linear-gradient(92deg,#eaf1ff 18%,#9fc4ff 56%,#c9b6ff 88%);
            -webkit-background-clip:text;background-clip:text;color:transparent;
            animation:sq-fade-up .9s var(--sq-ease) .06s both}
        .sq-lead{color:var(--sq-muted);font-size:1rem;line-height:1.62;max-width:596px;
            animation:sq-fade-up .9s var(--sq-ease) .16s both}
        .sq-cursor{display:inline-block;font-family:var(--sq-mono);color:var(--sq-cyan);
            animation:sq-fade-up .9s var(--sq-ease) .26s both}
        .sq-cursor i{display:inline-block;width:9px;height:15px;background:var(--sq-cyan);
            vertical-align:-2px;margin-left:6px;animation:sq-blink 1.1s step-end infinite}

        /* ---------- self-typing query console ---------- */
        .sq-type{margin:16px 0 4px 0;padding:12px 14px;border-radius:12px;
            background:rgba(6,11,20,.55);border:1px solid var(--sq-border);
            font-family:var(--sq-mono);font-size:.86rem;line-height:1.85;
            animation:sq-fade-up .9s var(--sq-ease) .22s both}
        .sq-type .pfx{color:var(--sq-cyan);margin-right:8px}
        .sq-tline{display:inline-block;vertical-align:bottom;overflow:hidden;white-space:nowrap;
            width:0;max-width:100%;color:#d6e2f7;
            border-right:2px solid transparent;
            animation:sq-typing 1.05s steps(44,end) var(--d,0s) forwards,
                      sq-caret .8s step-end var(--d,0s) 4}
        .sq-tline.last{animation:sq-typing 1.05s steps(44,end) var(--d,0s) forwards,
                      sq-caret .8s step-end var(--d,0s) infinite}
        @keyframes sq-typing{from{width:0}to{width:var(--w,30ch)}}
        @keyframes sq-caret{0%,49%{border-right-color:var(--sq-cyan)}
                            50%,100%{border-right-color:transparent}}
        .sq-ribbon{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 0 0}
        .sq-ribbon .rf{display:inline-flex;align-items:center;gap:7px;border-radius:10px;
            padding:7px 12px;font-size:.79rem;color:#cfe0ff;background:rgba(79,140,255,.10);
            border:1px solid var(--sq-border);animation:sq-fade-up .7s var(--sq-ease) both}
        .sq-ribbon .rf b{font-weight:650;color:#eaf1ff}
        .sq-ribbon .rf .ic{font-size:.95rem}

        /* ---------- console footer ---------- */
        .sq-foot{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:26px 0 6px 0;
            padding-top:14px;border-top:1px solid var(--sq-border);color:var(--sq-muted);
            font-size:.78rem;animation:sq-fade-in .8s var(--sq-ease) both}
        .sq-foot .b{color:#e8edf7;font-weight:650}
        .sq-foot .sep{opacity:.45}
        .sq-foot .status{margin-left:auto;display:inline-flex;align-items:center;gap:7px;
            background:rgba(47,191,113,.12);border:1px solid var(--sq-border);
            border-radius:999px;padding:3px 11px;color:#bdf0d2}
        .sq-foot .status .dot{width:7px;height:7px;border-radius:50%;background:var(--sq-good);
            animation:sq-pulse 2.4s ease-in-out infinite}
        .sq-orbit{position:absolute;right:-36px;top:-34px;width:344px;height:344px;pointer-events:none;
            opacity:.95;animation:sq-fade-in 1.4s var(--sq-ease) .2s both}
        .sq-orbit .globe{position:absolute;inset:98px;border-radius:50%;
            background:radial-gradient(circle at 34% 30%,rgba(79,140,255,.55),rgba(18,28,50,.92) 64%);
            border:1px solid var(--sq-border-2);animation:sq-float 7s ease-in-out infinite;
            box-shadow:inset 0 0 60px rgba(79,140,255,.30),0 0 44px rgba(79,140,255,.16)}
        .sq-orbit .ring{position:absolute;inset:0;border-radius:50%;
            border:1px dashed rgba(127,140,170,.34);animation:sq-spin 26s linear infinite}
        .sq-orbit .ring.b{inset:36px;animation-duration:17s;animation-direction:reverse;
            border-style:solid;border-color:rgba(127,140,170,.20)}
        .sq-orbit .sat{position:absolute;top:-7px;left:50%;width:16px;height:16px;margin-left:-8px;
            border-radius:4px;background:linear-gradient(135deg,#cfe0ff,#4f8cff);
            box-shadow:0 0 18px rgba(79,140,255,.85)}
        .sq-orbit .sweep{position:absolute;inset:98px;border-radius:50%;
            background:conic-gradient(from 0deg,rgba(45,212,191,.40),transparent 32%);
            animation:sq-spin 4.8s linear infinite;mix-blend-mode:screen}
        .sq-stat{display:inline-flex;flex-direction:column;gap:2px;margin:16px 26px 0 0;
            animation:sq-fade-up .8s var(--sq-ease) both}
        .sq-stat .v{font-size:1.5rem;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:-.6px}
        .sq-stat .k{font-size:.72rem;letter-spacing:1px;text-transform:uppercase;color:var(--sq-muted)}

        /* ---------- agent pipeline ---------- */
        .sq-flow{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 6px 0}
        .sq-flow .node{flex:1 1 148px;position:relative;overflow:hidden;border-radius:12px;
            padding:11px 12px 13px 12px;background:var(--sq-card);border:1px solid var(--sq-border);
            animation:sq-fade-up .5s var(--sq-ease) both;
            transition:transform .25s var(--sq-ease),border-color .25s var(--sq-ease)}
        .sq-flow .node:hover{transform:translateY(-3px)}
        .sq-flow .node .k{font-size:.66rem;letter-spacing:1.1px;text-transform:uppercase;color:var(--sq-muted)}
        .sq-flow .node .v{font-weight:700;font-size:.93rem;margin-top:3px}
        .sq-flow .node .ms{font-family:var(--sq-mono);font-size:.71rem;color:var(--sq-muted)}
        .sq-flow .node .idx{position:absolute;right:9px;top:7px;font-family:var(--sq-mono);
            font-size:.66rem;color:var(--sq-muted)}
        .sq-flow .node .bar{position:absolute;left:0;bottom:0;height:2px;width:100%;
            background:linear-gradient(90deg,var(--sq-accent),var(--sq-cyan));
            animation:sq-grow 1s var(--sq-ease) both}
        .sq-flow .node.ok{border-color:rgba(47,191,113,.42)}
        .sq-flow .node.ok .bar{background:linear-gradient(90deg,var(--sq-good),var(--sq-cyan))}
        .sq-flow .node.idle{opacity:.55}
        .sq-flow .node.idle .bar{background:rgba(127,140,170,.4)}

        /* ---------- confidence gauge ---------- */
        .sq-gauge{display:flex;align-items:center;gap:16px;background:var(--sq-card);
            border:1px solid var(--sq-border);border-radius:14px;padding:12px 16px;
            animation:sq-fade-up .5s var(--sq-ease) both}
        .sq-gauge svg{width:104px;height:104px;flex:0 0 auto;transform:rotate(-90deg)}
        .sq-gauge .track{fill:none;stroke:rgba(127,140,170,.20);stroke-width:9}
        .sq-gauge .arc{fill:none;stroke-width:9;stroke-linecap:round;stroke-dasharray:100;
            animation:sq-dash 1.2s var(--sq-ease) both}
        .sq-gauge .lbl{font-size:.9rem;font-weight:700}
        .sq-gauge .sub{font-size:.76rem;color:var(--sq-muted)}
        .sq-bars{flex:1;display:flex;flex-direction:column;gap:8px;min-width:170px}
        .sq-bars .row{display:flex;align-items:center;gap:9px;font-size:.77rem;color:var(--sq-muted)}
        .sq-bars .row .lbl{width:98px}
        .sq-bars .row .track{flex:1;height:7px;border-radius:6px;background:rgba(127,140,170,.20);overflow:hidden}
        .sq-bars .row .fill{height:100%;border-radius:6px;
            background:linear-gradient(90deg,var(--sq-accent),var(--sq-cyan));
            animation:sq-grow 1.1s var(--sq-ease) both}
        .sq-bars .row .val{width:40px;text-align:right;font-family:var(--sq-mono);color:#dbe5f7}

        /* ---------- audit terminal ---------- */
        .sq-term{background:#070b14;border:1px solid var(--sq-border);border-radius:12px;
            padding:12px 14px;font-family:var(--sq-mono);font-size:.75rem;line-height:1.75;
            max-height:330px;overflow:auto;animation:sq-fade-in .5s var(--sq-ease) both}
        .sq-term .ln{display:flex;gap:10px;white-space:pre-wrap;
            animation:sq-slide-in .34s var(--sq-ease) both}
        .sq-term .t{color:#7887a5;flex:0 0 62px}
        .sq-term .c{color:#9fc4ff;flex:0 0 108px}
        .sq-term .ok{color:var(--sq-good)} .sq-term .warn{color:var(--sq-warn)} .sq-term .err{color:var(--sq-bad)}
        .sq-term .m{color:#c6d2e8}
        .sq-term .cur{display:inline-block;width:8px;height:13px;background:var(--sq-cyan);
            vertical-align:-2px;animation:sq-blink 1.1s step-end infinite}

        /* ---------- KPIs, loading, frames ---------- */
        .sq-kpi{background:var(--sq-card);border:1px solid var(--sq-border);border-radius:14px;
            padding:12px 14px;animation:sq-fade-up .5s var(--sq-ease) both;
            transition:transform .25s var(--sq-ease),border-color .25s var(--sq-ease)}
        .sq-kpi:hover{transform:translateY(-3px);border-color:var(--sq-border-2)}
        .sq-kpi .k{font-size:.68rem;letter-spacing:1.1px;text-transform:uppercase;color:var(--sq-muted)}
        .sq-kpi .v{font-size:1.35rem;font-weight:750;letter-spacing:-.5px;font-variant-numeric:tabular-nums}
        .sq-kpi .d{font-size:.74rem;color:var(--sq-muted)}
        .sq-load{position:relative;overflow:hidden;border:1px solid var(--sq-border);
            border-radius:14px;padding:15px 17px;background:var(--sq-card);
            animation:sq-fade-in .3s var(--sq-ease) both}
        .sq-load .hd{display:flex;align-items:center;gap:10px;margin-bottom:12px;font-size:.86rem}
        .sq-load .dot{width:9px;height:9px;border-radius:50%;background:var(--sq-accent);
            animation:sq-pulse 1.1s ease-in-out infinite}
        .sq-load .l1{height:11px;border-radius:6px;margin:0 0 9px 0;background-size:420px 100%;
            background-image:linear-gradient(90deg,rgba(127,140,170,.16),rgba(127,140,170,.34),rgba(127,140,170,.16));
            animation:sq-shimmer 1.25s linear infinite}
        .sq-sec{display:flex;align-items:center;gap:10px;margin:20px 0 8px 0}
        .sq-sec .n{font-family:var(--sq-mono);font-size:.74rem;color:#0b1220;background:var(--sq-accent);
            border-radius:6px;padding:2px 7px}
        .sq-sec .t{font-weight:700;letter-spacing:-.2px}
        .sq-sec .h{color:var(--sq-muted);font-size:.85rem}

        /* ---------- workspace hero ---------- */
        .sq-hero{position:relative;overflow:hidden}
        .sq-hero::after{content:'';position:absolute;left:0;right:0;top:0;height:1px;
            background:linear-gradient(90deg,transparent,rgba(79,140,255,.75),rgba(45,212,191,.6),transparent);
            animation:sq-fade-in 1.2s var(--sq-ease) both}
        .sq-hero .sq-logo{animation:sq-float 6.5s ease-in-out infinite;filter:drop-shadow(0 6px 16px rgba(79,140,255,.35))}
        .sq-hero-txt{flex:1 1 auto}
        .sq-hero-status{margin-left:auto;display:inline-flex;align-items:center;gap:8px;
            background:rgba(79,140,255,.13);border:1px solid var(--sq-border-2);border-radius:999px;
            padding:5px 13px;font-size:.75rem;letter-spacing:.5px;color:#cfe0ff;white-space:nowrap}
        .sq-hero-status .dot{width:7px;height:7px;border-radius:50%;background:var(--sq-good);
            animation:sq-pulse 2.2s ease-in-out infinite}
        @media (max-width:900px){.sq-orbit{display:none}.sq-h1{font-size:1.85rem}
            .sq-hero-status{display:none}}
        @media (prefers-reduced-motion:reduce){
          *,*::before,*::after{animation-duration:.001ms!important;animation-iteration-count:1!important;
              transition-duration:.001ms!important}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_bbox(value: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse 'west, south, east, north' degrees; returns None when invalid."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        return None
    try:
        return validate_bbox([float(p) for p in parts])
    except (TypeError, ValueError):
        return None

# ----------------------------------------------------------------- palettes ---
_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (124, 124, 124), 1: (46, 108, 196), 2: (56, 168, 82),
    3: (222, 72, 58), 4: (198, 166, 110),
}
_CLASS_LEGEND: Dict[str, Tuple[int, int, int]] = {
    name: _CLASS_COLORS[idx] for idx, name in enumerate(LANDCOVER_CLASS_NAMES)
}
CLASS_LEGEND: Dict[str, Tuple[int, int, int]] = {
    CLASS_DISPLAY_NAMES[k]: v for k, v in _CLASS_LEGEND.items()
}
CHANGE_LEGEND: Dict[str, Tuple[int, int, int]] = {
    CHANGE_CLASS_DISPLAY.get(k, k): v for k, v in CHANGE_CLASS_PALETTES.items()
}

TERM_TO_CLASS: Dict[str, int] = {name: idx for idx, name in enumerate(LANDCOVER_CLASS_NAMES)}


# ----------------------------------------------------------------- utilities --
def rgb_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def overlay_regions(rgb: np.ndarray, regions: List[Dict[str, Any]],
                    color: Tuple[int, int, int] = (255, 210, 40)) -> np.ndarray:
    out = rgb.copy()
    for region in regions:
        x0, y0, x1, y1 = region["bbox"]
        out[max(0, y0):y0 + 3, max(0, x0):min(out.shape[1], x1 + 1)] = color
        out[max(0, y1 - 2):y1 + 1, max(0, x0):min(out.shape[1], x1 + 1)] = color
        out[max(0, y0):y1 + 1, max(0, x0):x0 + 3] = color
        out[max(0, y0):y1 + 1, max(0, y1 - 2):y1 + 1] = color
    return out


def legend_html(items: Dict[str, Tuple[int, int, int]]) -> str:
    chips = "".join(
        f"<span><span class='sw' style='background:rgb({r},{g},{b});'></span>{name}</span>"
        for name, (r, g, b) in items.items()
    )
    return f"<div class='sq-legend'>{chips}</div>"


def confidence_badge(confidence: float) -> str:
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        color, label = "var(--sq-good)", "HIGH"
    elif confidence >= LOW_CONFIDENCE_THRESHOLD:
        color, label = "var(--sq-warn)", "MEDIUM"
    else:
        color, label = "var(--sq-bad)", "LOW"
    return f"<span class='sq-pill' style='background:{color};'>{label} {confidence:.2f}</span>"


def show_image(img: RasterImage, caption: str = "") -> None:
    if img.preview is not None:
        st.image(img.preview, caption=caption, width="stretch")
    else:
        st.image(img.data, caption=caption, width="stretch")


# ------------------------------------------------------- UI v2 components ---
def _esc(text: Any) -> str:
    """Minimal HTML escape for values interpolated into styled blocks."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clip(text: Any, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _sec(number: str, title: str, hint: str = "") -> None:
    """Numbered console-style section header."""
    tail = f"<span class='h'>{_esc(hint)}</span>" if hint else ""
    st.markdown(
        f"<div class='sq-sec'><span class='n'>{_esc(number)}</span>"
        f"<span class='t'>{_esc(title)}</span>{tail}</div>",
        unsafe_allow_html=True,
    )


#: Controller stage -> audit-trace step name (None = derived from tool calls).
_STAGE_MAP: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("1", "Validate inputs", "input_validation"),
    ("2", "Classify intent", "intent_classification"),
    ("3", "Route specialists", "tool_selection"),
    ("4", "Execute specialists", None),
    ("5", "Score confidence", "confidence_estimation"),
)


def _stage_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Bind the five controller stages to real timings from the audit trace."""
    trace = summary.get("execution_trace") or []
    calls = summary.get("tool_calls") or []
    rows: List[Dict[str, Any]] = []
    for index, label, step in _STAGE_MAP:
        if step is None:
            ms = sum(float(call.get("duration_ms") or 0.0) for call in calls)
            detail = f"{len(calls)} tool(s) executed"
            done = bool(calls)
        else:
            events = [ev for ev in trace if ev.get("step") == step]
            ms = sum(float(ev.get("duration_ms") or 0.0) for ev in events)
            detail = str(events[0].get("message") or "") if events else "not reached"
            done = bool(events)
        rows.append({"index": index, "label": label, "ms": ms,
                     "detail": detail, "done": done})
    return rows


def agent_pipeline_html(summary: Dict[str, Any]) -> str:
    """Animated five-stage agent pipeline with per-stage timings."""
    nodes: List[str] = []
    for position, row in enumerate(_stage_rows(summary)):
        css = "ok" if row["done"] else "idle"
        nodes.append(
            f"<div class='node {css}' style='animation-delay:{position * 0.07:.2f}s'>"
            f"<span class='idx'>{_esc(row['index'])}/5</span>"
            f"<div class='k'>{_esc(row['label'])}</div>"
            f"<div class='v'>{row['ms']:.0f} ms</div>"
            f"<div class='ms'>{_esc(_clip(row['detail'], 48))}</div>"
            f"<div class='bar'></div></div>"
        )
    return "<div class='sq-flow'>" + "".join(nodes) + "</div>"


def confidence_gauge_html(value: float, breakdown: Dict[str, Any]) -> str:
    """Animated donut gauge plus the confidence-breakdown bars."""
    pct = max(0.0, min(1.0, float(value)))
    if pct >= MEDIUM_CONFIDENCE_THRESHOLD:
        colour, band = "#2fbf71", "high"
    elif pct >= LOW_CONFIDENCE_THRESHOLD:
        colour, band = "#f5a524", "medium"
    else:
        colour, band = "#f04438", "low"
    bars = []
    for position, (label, key) in enumerate(
        (("Data quality", "data_quality"), ("Model", "model"), ("Agreement", "agreement"))
    ):
        raw = breakdown.get(key)
        share = float(raw) if raw is not None else 0.0
        bars.append(
            "<div class='row'>"
            f"<span class='lbl'>{_esc(label)}</span>"
            "<span class='track'>"
            f"<span class='fill' style='width:{max(0.0, min(1.0, share)) * 100:.1f}%;"
            f"animation-delay:{0.15 + position * 0.09:.2f}s'></span></span>"
            f"<span class='val'>{share:.2f}</span></div>"
        )
    offsets = f"--sq-off:{100.0 - pct * 100.0:.1f}"
    return (
        "<div class='sq-gauge'>"
        "<svg viewBox='0 0 40 40' aria-hidden='true'>"
        "<circle class='track' cx='20' cy='20' r='16' pathLength='100'></circle>"
        f"<circle class='arc' cx='20' cy='20' r='16' pathLength='100' "
        f"style='stroke:{colour};{offsets}'></circle>"
        "</svg>"
        "<div>"
        f"<div class='lbl'>{pct:.2f}</div>"
        f"<div class='sub'>{band} confidence</div>"
        f"<div class='sub'>{len(_all_factor_text(breakdown))} factor(s) weighted</div>"
        "</div>"
        f"<div class='sq-bars'>{''.join(bars)}</div>"
        "</div>"
    )


def _all_factor_text(breakdown: Dict[str, Any]) -> List[str]:
    factors = breakdown.get("factors") or []
    return [str(f) for f in factors] if isinstance(factors, (list, tuple)) else [str(factors)]


def trace_terminal_html(events: List[Dict[str, Any]], limit: int = 60) -> str:
    """Mission-console rendering of the auditable execution trace."""
    lines: List[str] = []
    for position, event in enumerate(events[:limit]):
        status = str(event.get("status") or "ok")
        css = "ok" if status == "ok" else ("warn" if status == "warning" else "err")
        ms = float(event.get("duration_ms") or 0.0)
        lines.append(
            f"<div class='ln' style='animation-delay:{min(position * 0.025, 0.7):.2f}s'>"
            f"<span class='t'>{ms:8.1f}ms</span>"
            f"<span class='c'>{_esc(_clip(event.get('component') or '-', 16))}</span>"
            f"<span class='{css}'>{_esc(_clip(event.get('step') or '-', 24))}</span>"
            f"<span class='m'>{_esc(_clip(event.get('message') or '', 96))}</span></div>"
        )
    if len(events) > limit:
        lines.append(
            f"<div class='ln'><span class='t'></span><span class='m'>"
            f"… {len(events) - limit} earlier event(s)</span></div>"
        )
    lines.append(
        "<div class='ln'><span class='t'></span><span class='ok'>trace complete</span>"
        "<span class='m'>&gt; <span class='cur'></span></span></div>"
    )
    return "<div class='sq-term'>" + "".join(lines) + "</div>"


def loading_html(message: str = "Agentic pipeline running") -> str:
    """Animated skeleton shown while the agent executes."""
    widths = (96, 74, 88, 58)
    bars = "".join(
        f"<div class='l1' style='width:{w}%;animation-delay:{i * 0.08:.2f}s'></div>"
        for i, w in enumerate(widths)
    )
    return (
        "<div class='sq-load'>"
        f"<div class='hd'><span class='dot'></span><b>{_esc(message)}</b>"
        "<span style='color:var(--sq-muted);font-size:.78rem'>"
        "validate → classify intent → route → execute → score</span></div>"
        f"{bars}</div>"
    )


def _cta_demo() -> None:
    """Landing CTA: preselect the bi-temporal demo pair (sets widget keys before
    their widgets are instantiated on the next run — callbacks are the safe place)."""
    st.session_state["input_mode"] = "Demo scenes"
    st.session_state["demo_select"] = "Bi-temporal pair (2023 vs 2024)"


def _cta_map() -> None:
    """Landing CTA: switch the sidebar to live map acquisition."""
    st.session_state["input_mode"] = "Map region (live imagery)"


def _cta_upload() -> None:
    """Landing CTA: switch the sidebar to file upload."""
    st.session_state["input_mode"] = "Upload images"


def render_landing() -> None:
    """Animated landing stage for the true empty state (no imagery loaded)."""
    try:
        tool_count = len(default_registry().describe().get("tools", []))
    except Exception:  # never let a cosmetic stat break the page
        tool_count = 5
    typed = (
        "describe --scene land-cover",
        "change --t1 2023 --t2 2024",
        "fuse --optical --sar urban",
    )
    typer = "".join(
        "<div><span class='pfx'>&gt;</span>"
        f"<span class='sq-tline{' last' if i == len(typed) - 1 else ''}' "
        f"style='--w:{len(text) + 2}ch;--d:{0.45 + i * 1.25:.2f}s'>{_esc(text)}</span></div>"
        for i, text in enumerate(typed)
    )
    ribbon = "".join(
        f"<span class='rf' style='animation-delay:{0.62 + i * 0.08:.2f}s'>"
        f"<span class='ic'>{icon}</span><b>{_esc(name)}</b><span>{_esc(detail)}</span></span>"
        for i, (icon, name, detail) in enumerate(
            (("📝", "Single-image VQA", "captioning + region grounding"),
             ("🛰️", "Optical + SAR fusion", "spectral + structural features"),
             ("🕓", "Bi-temporal change", "description + spatial change map"))
        )
    )
    st.markdown(
        f"""
        <div class='sq-land'>
          <div class='aurora'></div>
          <div class='mesh'></div>
          <div class='sq-orbit'>
            <div class='ring'></div><div class='ring b'></div>
            <div class='globe'></div><div class='sweep'></div><div class='sat'></div>
          </div>
          <div class='inner'>
            <div class='sq-badge'><span class='dot'></span>
              {PROBLEM_STATEMENT_ID} · agentic multi-modal remote sensing
            </div>
            <div class='sq-h1'>Ask your satellite imagery<br>anything.</div>
            <p class='sq-lead'>Optical, multispectral, SAR and bi-temporal pairs — one agentic
            console that validates the inputs, classifies the intent, routes the question to
            specialist models, and answers with visual evidence, a confidence score and a
            complete audit trail.</p>
            <div>
              <span class='sq-stat' style='animation-delay:.30s'>
                <span class='v'>3</span><span class='k'>workflows</span></span>
              <span class='sq-stat' style='animation-delay:.38s'>
                <span class='v'>{tool_count}</span><span class='k'>specialist tools</span></span>
              <span class='sq-stat' style='animation-delay:.46s'>
                <span class='v'>19</span><span class='k'>class taxonomy</span></span>
              <span class='sq-stat' style='animation-delay:.54s'>
                <span class='v'>2</span><span class='k'>sensor modalities</span></span>
            </div>
            <div class='sq-type'>{typer}</div>
            <div class='sq-ribbon'>{ribbon}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 👈 Pick an input source to begin")
    st.markdown(
        """
        <div class='sq-empty'>
          <div class='cell'><div class='ico'>🗺️</div><h4>Map region</h4>
            <p>Draw a rectangle anywhere on Earth and fetch live Sentinel-2 /
            Sentinel-1 imagery for exactly that footprint.</p></div>
          <div class='cell'><div class='ico'>🧪</div><h4>Demo scenes</h4>
            <p>Offline georeferenced samples for every workflow — no network
            needed for the full demo.</p></div>
          <div class='cell'><div class='ico'>📂</div><h4>Upload images</h4>
            <p>Your own GeoTIFF / TIFF (geospatial) or PNG / JPEG (benchmark)
            files, up to two at a time.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cta = st.columns(3)
    cta[0].button("🧪 Load a demo scene", width="stretch",
                  on_click=_cta_demo, help="Instantly load the bi-temporal demo pair.")
    cta[1].button("🗺️ Draw a region on the map", width="stretch",
                  on_click=_cta_map, help="Switch to live Sentinel imagery for a drawn ROI.")
    cta[2].button("📂 Upload imagery", width="stretch",
                  on_click=_cta_upload, help="Analyse your own GeoTIFF / PNG files.")
    st.markdown(
        "<div class='sq-example'>Then ask: <i>“Describe the land cover and major objects "
        "visible in this image.”</i> · <i>“What changed between these two dates?”</i> · "
        "<i>“Use the optical and SAR images together to identify built-up and "
        "water-covered regions.”</i></div>",
        unsafe_allow_html=True,
    )


def load_uploads(files: List[Any], controller: AgentController) -> Tuple[List[RasterImage], List[str]]:
    images: List[RasterImage] = []
    errors: List[str] = []
    for f in files[:2]:
        try:
            suffix = Path(f.name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                errors.append(
                    f"{f.name}: unsupported format '{suffix}'. "
                    "Use GeoTIFF/TIFF (geospatial) or PNG/JPEG (benchmark data only)."
                )
                continue
            data = f.getvalue()
            if len(data) > controller.config.max_file_size_mb * 1024 * 1024:
                errors.append(f"{f.name}: file exceeds {controller.config.max_file_size_mb:.0f} MB limit.")
                continue
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(data)
            tmp.close()
            images.append(controller.load_images([Path(tmp.name)])[0])
        except Exception as exc:
            errors.append(f"{f.name}: {type(exc).__name__}: {exc}")
    return images, errors


def grounding_overlay_for(image: RasterImage, regions: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    if image.preview is None or not regions:
        return None
    scale = image.height / image.preview.shape[0]
    scaled = [
        {**region, "bbox": [int(v / scale) for v in region["bbox"]]}
        for region in regions
    ]
    return overlay_regions(image.preview, scaled)


def fused_overlay_preview(fused: np.ndarray, like: np.ndarray) -> np.ndarray:
    overlay = colorize_classes(fused, _CLASS_COLORS)
    h, w = like.shape[:2]
    rows = np.linspace(0, overlay.shape[0] - 1, h).astype(int)
    cols = np.linspace(0, overlay.shape[1] - 1, w).astype(int)
    return overlay[np.ix_(rows, cols)]


def per_class_maps(fused: np.ndarray, like: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    h, w = like.shape[:2]
    maps = []
    for idx, name in enumerate(LANDCOVER_CLASS_NAMES[1:], start=1):
        mask = (fused == idx).astype(np.uint8) * 255
        rows = np.linspace(0, mask.shape[0] - 1, h).astype(int)
        cols = np.linspace(0, mask.shape[1] - 1, w).astype(int)
        maps.append((CLASS_DISPLAY_NAMES[name], mask[np.ix_(rows, cols)]))
    return maps


def download_button(payload: bytes, label: str, filename: str, mime: str) -> None:
    st.download_button(label, data=payload, file_name=filename, mime=mime,
                       width="stretch")


# ---------------------------------------------------------------- session ----
def init_session() -> None:
    ss = st.session_state
    ss.setdefault("controller", AgentController())
    ss.setdefault("images", [])
    ss.setdefault("upload_errors", [])
    ss.setdefault("result", None)
    ss.setdefault("history", [])
    ss.setdefault("query", "")
    ss.setdefault("registry", default_registry())
    ss.setdefault("input_mode", "Demo scenes")
    ss.setdefault("demo_select", "-")              # sidebar demo-scene picker
    ss.setdefault("last_drawn_bbox", None)          # (w, s, e, n) from map draw
    ss.setdefault("fetched_info", None)            # acquisition metadata dict
    ss.setdefault("fetch_warnings", [])
    ss.setdefault("result_map", None)              # folium map for result overlay
    ss.setdefault("result_map_geo", None)          # bbox of result map images


def _acquire_from_map(bbox: Tuple[float, float, float, float], start: date, end: date,
                      include_sar: bool) -> None:
    """Fetch live imagery for an ROI and place it into the active session."""
    try:
        fetched = fetch_roi_imagery(bbox, start, end, include_sar=include_sar)
    except Exception as exc:  # fallback disabled or hard validation error
        st.error(f"Imagery acquisition failed: {type(exc).__name__}: {exc}")
        return
    order = {"optical": 0, "sar": 1}
    imgs = sorted(fetched.images, key=lambda im: (order.get(im.modality, 2), im.metadata.name))
    st.session_state.images = imgs
    st.session_state.upload_errors = []
    st.session_state.fetched_info = fetched.info
    st.session_state.fetch_warnings = list(fetched.warnings)
    st.session_state.result = None
    st.session_state.result_map = None


init_session()
inject_theme_css()


# ------------------------------------------------------------------ sidebar ---
with st.sidebar:
    st.markdown(f"## 🛰️ {APP_NAME}")
    st.caption(f"SIH Problem Statement {PROBLEM_STATEMENT_ID} · v{APP_VERSION}")

    _sec("01", "Inputs", "select the source")
    input_mode = st.radio(
        "Input source",
        ["Demo scenes", "Map region (live imagery)", "Upload images"],
        key="input_mode",
        help="Draw a region on the map to pull real Sentinel-2/Sentinel-1 imagery, "
             "load offline demo scenes, or upload your own GeoTIFF/PNG/JPEG files.",
    )

    if input_mode == "Demo scenes":
        demo_option = st.selectbox(
            "Load demo scene",
            ["-"] + list(demo_sets().keys()),
            index=0,
            key="demo_select",
            help="Synthetic georeferenced scenes (EPSG:4326 GeoTIFF) for offline demonstration.",
        )
        if demo_option != "-":
            sets = demo_sets(ensure_demo_images())
            st.session_state.images = sets[demo_option]
            st.session_state.upload_errors = []
            st.session_state.fetched_info = None
            st.session_state.fetch_warnings = []
        elif not st.session_state.images:
            st.session_state.images = []

    elif input_mode == "Map region (live imagery)":
        if not HAS_FOLIUM_UI:
            st.error("folium / streamlit-folium are not installed; map input is unavailable.")
        if HAS_FOLIUM_UI:
            st.caption("Draw a rectangle ⬚ on the map, pick dates, then fetch.")
            roi_center = DEFAULT_MAP_CENTER
            fmap = folium.Map(location=roi_center, zoom_start=5, tiles="OpenStreetMap",
                              control_scale=True)
            Draw(
                export=False, position="topleft",
                draw_options={"polyline": False, "polygon": False, "circle": False,
                              "circlemarker": False, "marker": False, "rectangle": True},
                edit_options={"edit": False, "remove": False},
            ).add_to(fmap)
            map_event = st_folium(
                fmap, height=340, width=None, use_container_width=True,
                returned_objects=["last_active_drawing"], key="roi_map",
            )
            drawing = (map_event or {}).get("last_active_drawing") or {}
            coords = ((drawing.get("geometry") or {}).get("coordinates") or [[]])
            if coords and len(coords[0]) >= 4:
                lons = [pt[0] for pt in coords[0]]
                lats = [pt[1] for pt in coords[0]]
                st.session_state.last_drawn_bbox = (
                    min(lons), min(lats), max(lons), max(lats)
                )
            bbox = st.session_state.get("last_drawn_bbox")
            if bbox:
                st.success(
                    f"ROI drawn: {bbox[0]:.4f}°, {bbox[1]:.4f}° → {bbox[2]:.4f}°, {bbox[3]:.4f}°"
                )
            else:
                st.info("No ROI drawn yet — use the ⬚ rectangle tool on the map.")

            c1, c2 = st.columns(2)
            roi_start = c1.date_input("T1 (from)", value=date.today() - timedelta(days=180),
                                      key="roi_start")
            roi_end = c2.date_input("T2 (to)", value=date.today() - timedelta(days=15),
                                    key="roi_end")
            include_sar = st.toggle("Also fetch SAR (Sentinel-1)", value=False, key="roi_sar")
            fetch_clicked = st.button("🛰️ Fetch imagery for ROI", type="primary", width="stretch",
                                      disabled=not bbox, key="roi_fetch")
            if fetch_clicked and bbox:
                _acquire_from_map(bbox, roi_start, roi_end, include_sar)
            elif fetch_clicked and not bbox:
                st.error("Draw a rectangle on the map first.")
            if st.session_state.fetch_warnings:
                for warn in st.session_state.fetch_warnings:
                    st.warning(warn)

    else:  # Upload images
        uploads = st.file_uploader(
            "Or upload images (max 2)",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if uploads:
            images_loaded, upload_errors = load_uploads(uploads, st.session_state.controller)
            st.session_state.images = images_loaded
            st.session_state.upload_errors = upload_errors
            st.session_state.fetched_info = None
            st.session_state.fetch_warnings = []
        else:
            st.session_state.images = []

    images = st.session_state.images
    st.markdown(f"**Active inputs:** {len(images)}")
    for img in images:
        st.markdown(
            f"<div class='sq-chip'>📄 <b>{img.metadata.name}</b><br>"
            f"<span class='dim'>{img.metadata.fmt} · {img.metadata.width}×{img.metadata.height} · "
            f"{img.metadata.bands} band(s)<br>"
            f"CRS: {img.metadata.crs or 'missing'} · {img.metadata.modality}</span></div>",
            unsafe_allow_html=True,
        )
    for error in st.session_state.upload_errors:
        st.error(error)
    fetched_info = st.session_state.get("fetched_info")
    if fetched_info:
        src = fetched_info.get("source", "")
        if "live" in src:
            scenes = fetched_info.get("scenes", {})
            bits = ["🛰️ Live scenes"]
            for key in ("t1", "t2"):
                sc = scenes.get(key) or {}
                if sc.get("datetime"):
                    bits.append(f"{key.upper()} {str(sc['datetime'])[:10]} ({sc.get('platform', 'S2')}, "
                                f"clouds {sc.get('cloud_cover', 0):.0f}%)")
            sar_scene = fetched_info.get("sar_scene") or {}
            if sar_scene.get("datetime"):
                bits.append(f"SAR {str(sar_scene['datetime'])[:10]} ({sar_scene.get('platform', 'S1')})")
            st.markdown(
                "<div class='sq-srcstrip'>"
                + "".join(f"<span class='tag'>{b}</span>" for b in bits)
                + "</div>",
                unsafe_allow_html=True,
            )
        elif src == "synthetic-fallback":
            st.markdown("<div class='sq-srcstrip'><span class='tag'>🧪 Offline fallback scene "
                        "(catalog unreachable)</span></div>", unsafe_allow_html=True)

    _sec("02", "Run", "agent + session")
    auto_reproject = st.toggle("Auto-reproject CRS-mismatched pairs", value=True)
    if st.button("🧹 Reset session", width="stretch"):
        for key in ("result", "history", "images", "upload_errors"):
            st.session_state[key] = None if key == "result" else []
        st.rerun()

    st.markdown("---")
    st.caption(
        "Single image → VQA + captioning/grounding\n\n"
        "Bi-temporal pair → change analysis\n\n"
        "Optical + SAR pair → joint fusion"
    )

# ------------------------------------------------------------------- header ---
_HERO_MODES = {
    "Demo scenes": "offline demo bundle",
    "Map region (live imagery)": "live acquisition",
    "Upload images": "user upload",
}
_hero_status = _HERO_MODES.get(st.session_state.get("input_mode", "Demo scenes"), "ready")
st.markdown(
    f"""
    <div class='sq-hero'>
      <div class='sq-logo'>🛰️</div>
      <div class='sq-hero-txt'>
        <h1>SatQuery AI</h1>
        <p>Agentic multi-modal remote-sensing assistant · ask questions about optical,
        SAR and bi-temporal satellite imagery · {PROBLEM_STATEMENT_ID}</p>
      </div>
      <div class='sq-hero-status'><span class='dot'></span>{_esc(_hero_status)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not images:
    render_landing()
    st.stop()

# ------------------------------------------------------------- validation ----
with st.expander("🧪 Input validation", expanded=False):
    validation = st.session_state.controller.validate_images(images)
    if validation["valid"]:
        st.success("All inputs valid - CRS, dimensions, modality and pair compatibility OK.")
    else:
        for problem in validation["problems"]:
            st.error(problem)
    for warning in validation["warnings"]:
        st.warning(warning)
    cols = st.columns(len(images))
    for col, img in zip(cols, images):
        with col:
            st.markdown(f"**{img.metadata.name}**")
            st.json(img.to_dict(), expanded=False)

# ----------------------------------------------------------------- query bar --
EXAMPLES_BY_N: Dict[int, List[str]] = {
    1: [
        "Describe the land cover and major objects visible in this image.",
        "Highlight the water body referred to in the query.",
        "Where are the built-up regions in this image?",
        "Is there any vegetation? How much?",
    ],
    2: [
        "What changed between these two dates, and where did the change occur?",
        "Has the built-up area increased, decreased, or remained unchanged?",
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
    ],
}
example = st.selectbox(
    "Example queries",
    ["Custom query…"] + EXAMPLES_BY_N.get(len(images), EXAMPLES_BY_N[1]),
    key="example_select",
)
if example != "Custom query…" and st.session_state.get("last_example") != example:
    # must be set BEFORE the text_input widget is instantiated in this run
    st.session_state.query_input = example
    st.session_state.last_example = example
    st.session_state.query = example
query = st.text_input(
    "Your query",
    key="query_input",
    placeholder="Ask anything about the loaded imagery…",
)
if not query.strip():
    st.markdown(
        "<div class='sq-example'>💡 Try: <i>“Describe the land cover”</i> · "
        "<i>“Highlight the water body”</i> · <i>“What changed between these dates?”</i> · "
        "<i>“Identify built-up regions using optical and SAR”</i></div>",
        unsafe_allow_html=True,
    )

col_run, col_clear = st.columns([1, 4])
run_clicked = col_run.button("🚀 Run agentic analysis", type="primary",
                             width="stretch")
if col_clear.button("Clear results", width="stretch"):
    st.session_state.result = None
    st.rerun()

if run_clicked:
    if not query.strip():
        st.error("Enter a natural-language query first.")
    else:
        t0 = time.perf_counter()
        run_slot = st.empty()
        run_slot.markdown(loading_html("Routing query to specialist models"),
                          unsafe_allow_html=True)
        try:
            result = st.session_state.controller.process(
                query.strip(), images, auto_reproject=auto_reproject
            )
        finally:
            run_slot.empty()
        result.duration_hint_ms = (time.perf_counter() - t0) * 1000.0  # type: ignore[attr-defined]
        st.session_state.result = result
        st.session_state.history.insert(0, {"query": query.strip(), "result": result})

result: Optional[Any] = st.session_state.result

if result is not None:
    st.markdown("---")
    st.markdown(
        "<div class='sq-srcstrip'>"
        + "".join(
            f"<span class='tag'>{label}</span>"
            for label in (f"🧭 intent: {result.task}", f"⚡ {result.sub_capability}")
        )
        + "".join(f"<span class='tag'>🧰 {t}</span>" for t in result.tools_used)
        + "</div>",
        unsafe_allow_html=True,
    )
    if result.status == "error":
        st.error(f"**Analysis failed** - {result.answer}")
        if result.errors:
            for err in result.errors:
                st.caption(err)
    else:
        # ------------------------------------------------- agentic pipeline --
        exec_summary: Dict[str, Any] = result.execution_summary or {}
        _sec("EXEC", "Agentic pipeline",
             f"5 stages · {exec_summary.get('duration_ms', 0.0):.0f} ms wall clock · "
             f"{len(exec_summary.get('tool_calls', []))} specialist call(s)")
        st.markdown(agent_pipeline_html(exec_summary), unsafe_allow_html=True)

        # ------------------------------------------------------------ answer --
        _sec("OUT", "Answer", "visual evidence + confidence below")
        head_l, head_r = st.columns([3, 1])
        with head_l:
            st.markdown(f"### 🧠 Answer  {confidence_badge(result.confidence)}",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div class='sq-answer'>{result.answer}</div>",
                unsafe_allow_html=True,
            )
        with head_r:
            st.metric("Confidence", f"{result.confidence:.2f}")
            st.caption(f"Task: `{result.task}` · sub-capability: `{result.sub_capability}`")
            st.caption(f"Tools: {', '.join(f'`{t}`' for t in result.tools_used) or 'none'}")

        breakdown = result.confidence_breakdown
        st.markdown(confidence_gauge_html(result.confidence, breakdown),
                    unsafe_allow_html=True)
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Data quality", f"{breakdown.get('data_quality', 0):.2f}")
        bc2.metric("Model", f"{breakdown.get('model', 0):.2f}")
        bc3.metric("Agreement", f"{breakdown.get('agreement', 0):.2f}")
        if breakdown.get("factors"):
            st.caption("Factors: " + "; ".join(breakdown["factors"]))

        # ------------------------------------------------------ visual maps --
        st.markdown("### 🗺️ Visual evidence")
        ve = result.visual_evidence
        if result.task == "bi_temporal_change":
            lcol, rcol = st.columns(2)
            with lcol:
                show_image(result.images[0], "Original · T1")
            with rcol:
                show_image(result.images[1], "Original · T2")
            if ve.get("change_overlay") is not None:
                st.markdown("**Spatial change map** (typed, relative to T2)")
                map_col, side_col = st.columns([3, 1])
                with map_col:
                    st.image(ve["change_overlay"], width="stretch")
                classes = ve.get("change_labels")
                with side_col:
                    if classes is not None:
                        total = classes.size
                        changed = int((classes != "no_change").sum())
                        present = [c for c in np.unique(classes) if c != "no_change"]
                        dominant = max(present, key=lambda c: int((classes == c).sum())) \
                            if present else "—"
                        side_col.metric("Changed area",
                                        f"{100.0 * changed / max(total, 1):.1f}%",
                                        f"{changed} px")
                        side_col.metric("Dominant change",
                                        CHANGE_CLASS_DISPLAY.get(dominant, str(dominant)))
                        side_col.metric("Analysis grid",
                                        f"{classes.shape[0]}×{classes.shape[1]}",
                                        "blocks")
                st.markdown(legend_html(CHANGE_LEGEND), unsafe_allow_html=True)
                if classes is not None:
                    present = [c for c in np.unique(classes) if c != "no_change"]
                    chips = ", ".join(
                        f"**{CHANGE_CLASS_DISPLAY.get(c, c)}** "
                        f"({int((classes == c).sum())} px)" for c in present
                    )
                    if chips:
                        st.caption("Change classes present: " + chips)
        elif result.task == "cross_modal_fusion":
            optical, sar = (result.images[0], result.images[1]) \
                if result.images[0].modality != "sar" else (result.images[1], result.images[0])
            lcol, rcol = st.columns(2)
            with lcol:
                show_image(optical, "Original · Optical")
            with rcol:
                show_image(sar, "Original · SAR")
            if ve.get("fused_overlay") is not None:
                st.markdown("**Joint optical-SAR classification** (fused spectral + structural)")
                st.image(ve["fused_overlay"], width="stretch")
                st.markdown(legend_html(CLASS_LEGEND), unsafe_allow_html=True)
                primary_view = optical.preview if optical.preview is not None else optical.data
                cls_maps = per_class_maps(ve["fused_classes"], primary_view)
                if cls_maps:
                    st.markdown("**Per-class maps**")
                    n_cols = min(4, len(cls_maps))
                    rows = [cls_maps[i:i + n_cols] for i in range(0, len(cls_maps), n_cols)]
                    for row in rows:
                        cols = st.columns(len(row))
                        for col, (name, mask) in zip(cols, row):
                            col.image(mask, caption=name, width="stretch")
                notes = result.metrics.get("complementarity_notes", [])
                for note in notes:
                    st.markdown(f"- {note}")
        else:
            primary = result.images[0]
            show_image(primary, "Original")
            regions = ve.get("grounding_regions") or []
            if regions:
                overlay = grounding_overlay_for(primary, regions)
                if overlay is not None:
                    term = ve.get("grounding_term", "target")
                    st.markdown(f"**Grounding overlay · term “{term}”**")
                    st.image(overlay, width="stretch")
                    st.markdown(legend_html({ve.get("grounding_term", "target"): (255, 210, 40)}),
                                unsafe_allow_html=True)
                    rows = [{"region": rg["region_id"], "quadrant": rg["quadrant"],
                             "bbox (px)": rg["bbox"], "coverage %": rg.get("coverage_pct")}
                            for rg in regions]
                    st.dataframe(rows, width="stretch", hide_index=True)

        # ------------------------------------------------- interactive map --
        if HAS_FOLIUM_UI:
            try:
                if result.task == "bi_temporal_change":
                    base_img = result.images[1] if len(result.images) > 1 else result.images[0]
                    overlay_rgb = ve.get("change_overlay")
                    rects_map: List[Dict[str, Any]] = []
                elif result.task == "cross_modal_fusion":
                    base_img = result.images[0] if result.images[0].modality != "sar" \
                        else result.images[1]
                    overlay_rgb = ve.get("fused_overlay")
                    rects_map = []
                else:
                    base_img = result.images[0]
                    ground_regions = ve.get("grounding_regions") or []
                    overlay_rgb = grounding_overlay_for(base_img, ground_regions) \
                        if ground_regions else None
                    rects_map = [
                        {"bbox": rg["bbox"],
                         "label": f"{rg.get('region_id')} · {rg.get('quadrant')}",
                         "color": "#d7301f"}
                        for rg in ground_regions
                    ]
                if base_img.metadata.bounds:
                    fmap_result = build_result_map(base_img, overlay_rgb,
                                                   "analysis overlay", rects_map)
                    with st.expander("🗺️ Results on interactive map", expanded=False):
                        st_folium(fmap_result, height=420, use_container_width=True,
                                  key="result_map")
            except Exception as exc:  # map is a bonus; never block the report
                logger.debug("interactive map render failed: %s", exc)

        # ------------------------------------------------------ intent card --
        with st.expander("🧭 Agentic routing (intent classification)", expanded=False):
            st.json(result.intent)

        # --------------------------------------------- execution summary -----
        with st.expander("📋 Auditable execution summary", expanded=False):
            summary = result.execution_summary or {}
            meta = st.columns(4)
            meta[0].metric("Selected task", summary.get("selected_task", result.task))
            meta[1].metric("Tools executed", len(summary.get("selected_tools", [])))
            meta[2].metric("Confidence", f"{result.confidence:.2f}")
            duration = summary.get("duration_ms", 0.0)
            meta[3].metric("Duration", f"{duration:.0f} ms")

            st.markdown("**Tool calls**")
            for call in summary.get("tool_calls", []):
                params_json = json.dumps(call.get("parameters", {}), default=str)
                call_summary = call.get("summary") or ""
                st.markdown(
                    f"<div class='sq-toolcard'>"
                    f"<b>{call['tool_name']}</b> &nbsp;<code>{call['tool_id']}</code>"
                    f" &nbsp;<span class='meta'>model: <code>{call['model_ref']}</code></span><br>"
                    f"<span class='meta'>{call.get('duration_ms', 0):.0f} ms · "
                    f"{call['status']} · params: {params_json}" 
                    f"{(' · ' + call_summary) if call_summary else ''}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**Execution trace**")
            st.markdown(trace_terminal_html(summary.get("execution_trace", [])),
                        unsafe_allow_html=True)
            if summary.get("warnings"):
                st.caption("⚠️ Warnings: " + " · ".join(summary["warnings"]))
            if summary.get("errors"):
                st.caption("❌ Errors: " + " · ".join(summary["errors"]))

        # ------------------------------------------------------------ reports -
        st.markdown("### 📦 Download reports")
        payload = build_report_payload(result.summary, result)
        d1, d2 = st.columns(2)
        json_bytes = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        d1.download_button(
            "⬇️ Download JSON report",
            data=json_bytes,
            file_name=f"satquery_report_{result.summary.session_id}.json",
            mime="application/json",
            width="stretch",
        )
        try:
            pdf_bytes = render_pdf_report(payload)
            d2.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name=f"satquery_report_{result.summary.session_id}.pdf",
                mime="application/pdf",
                width="stretch",
            )
        except Exception as exc:
            d2.error(f"PDF export failed: {exc}")

        # ------------------------------------------- geospatial data exports --
        geo_exports: List[Tuple[str, str, str, Any]] = []  # (label, mime, filename, bytes)
        sid = result.summary.session_id

        def _nearest_resize(index_map: np.ndarray, h: int, w: int) -> np.ndarray:
            rows = np.linspace(0, index_map.shape[0] - 1, h).astype(int)
            cols = np.linspace(0, index_map.shape[1] - 1, w).astype(int)
            return index_map[np.ix_(rows, cols)]

        for img in result.images:
            if not img.metadata.is_georeferenced:
                continue
            try:
                geo_exports.append((
                    f"⬇️ GeoTIFF · {img.metadata.name}", "image/tiff",
                    f"satquery_{sid}_{img.metadata.name}", image_to_geotiff(img),
                ))
            except Exception as exc:
                logger.debug("geotiff export failed for %s: %s", img.metadata.name, exc)

        try:
            if result.task == "bi_temporal_change" and ve.get("change_labels") is not None \
                    and result.images and result.images[-1].metadata.is_georeferenced:
                base = result.images[-1]
                idx_map = _nearest_resize(change_labels_to_indices(ve["change_labels"]),
                                          base.height, base.width)
                change_palette = {idx: CHANGE_CLASS_PALETTES[name]
                                  for idx, name in enumerate(CHANGE_CLASS_ORDER)}
                geo_exports.append((
                    "⬇️ GeoTIFF · change map (geotagged)", "image/tiff",
                    f"satquery_{sid}_change_map.tif",
                    class_map_to_geotiff(idx_map, change_palette, base.metadata.bounds),
                ))
                names = {idx: CHANGE_CLASS_DISPLAY.get(name, name)
                         for idx, name in enumerate(CHANGE_CLASS_ORDER)}
                geo_exports.append((
                    "⬇️ GeoJSON · change regions", "application/geo+json",
                    f"satquery_{sid}_change_regions.geojson",
                    json.dumps(region_polygons_geojson(base, idx_map, names, change_palette),
                               ensure_ascii=False).encode("utf-8"),
                ))
            elif result.task == "cross_modal_fusion" and ve.get("fused_classes") is not None \
                    and result.images:
                optical_img = result.images[0] if result.images[0].modality != "sar" \
                    else result.images[1]
                if optical_img.metadata.is_georeferenced:
                    idx_map = _nearest_resize(ve["fused_classes"],
                                              optical_img.height, optical_img.width)
                    fused_palette = {idx: _CLASS_COLORS[idx] for idx in sorted(set(np.unique(idx_map)))}
                    geo_exports.append((
                        "⬇️ GeoTIFF · joint classification (geotagged)", "image/tiff",
                        f"satquery_{sid}_fusion_classes.tif",
                        class_map_to_geotiff(idx_map, fused_palette, optical_img.metadata.bounds),
                    ))
                    names = {idx: CLASS_DISPLAY_NAMES.get(name, name)
                             for idx, name in enumerate(LANDCOVER_CLASS_NAMES)}
                    geo_exports.append((
                        "⬇️ GeoJSON · classified regions", "application/geo+json",
                        f"satquery_{sid}_fusion_regions.geojson",
                        json.dumps(region_polygons_geojson(optical_img, idx_map, names,
                                                           fused_palette),
                                   ensure_ascii=False).encode("utf-8"),
                    ))
            elif result.task == "single_image_analysis" and ve.get("grounding_regions") \
                    and result.images[0].metadata.is_georeferenced:
                geo_exports.append((
                    "⬇️ GeoJSON · grounded regions", "application/geo+json",
                    f"satquery_{sid}_grounding.geojson",
                    json.dumps(grounding_geojson(result.images[0], ve["grounding_regions"]),
                               ensure_ascii=False).encode("utf-8"),
                ))
        except Exception as exc:
            logger.debug("geo export failed: %s", exc)

        if geo_exports:
            st.markdown("**Geospatial exports** (open directly in QGIS / geojson.io)")
            export_cols = st.columns(min(3, len(geo_exports)))
            for i, (label, mime, fname, data_bytes) in enumerate(geo_exports):
                with export_cols[i % len(export_cols)]:
                    st.download_button(label, data=data_bytes, file_name=fname,
                                       mime=mime, width="stretch", key=f"geo_dl_{i}")

        # ----------------------------------------------------------- history --
        if len(st.session_state.history) > 1:
            with st.expander("🕘 Session history", expanded=False):
                for entry in st.session_state.history:
                    st.markdown(
                        f"- **{entry['query']}** → `{entry['result'].task}` "
                        f"(confidence {entry['result'].confidence:.2f})"
                    )

    # registry snapshot (always available)
    with st.expander("🧰 Specialist Model Registry", expanded=False):
        st.json(st.session_state.registry.describe())

# ------------------------------------------------------------------- footer ---
st.markdown(
    f"""
    <div class='sq-foot'>
      <span class='b'>🛰️ {APP_NAME}</span>
      <span class='sep'>·</span>
      <span>v{APP_VERSION}</span>
      <span class='sep'>·</span>
      <span>{PROBLEM_STATEMENT_ID} · agentic multi-modal remote sensing</span>
      <span class='sep'>·</span>
      <span>auditable execution trace on every query</span>
      <span class='status'><span class='dot'></span>engine ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)
