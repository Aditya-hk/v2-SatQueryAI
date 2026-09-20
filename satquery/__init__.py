"""SatQuery AI — Agentic multi-modal remote-sensing framework (SIH Problem Statement 167).

The package provides:

- ``satquery.utils``      : geospatial I/O + validation, audit logging, reporting, demo data.
- ``satquery.models``     : specialist models (single-image VQA, change detection, optical-SAR fusion)
                            and the Specialist Model Registry.
- ``satquery.agent``      : natural-language intent classification and the agentic controller.
- ``satquery.app``        : the interactive Streamlit GUI.
- ``satquery.training``   : BigEarthNet LoRA fine-tuning for the remote-sensing encoder.
"""

from __future__ import annotations

from satquery.config import APP_NAME, APP_VERSION, PROBLEM_STATEMENT_ID, DEFAULT_CONFIG

__all__ = ["APP_NAME", "APP_VERSION", "PROBLEM_STATEMENT_ID", "DEFAULT_CONFIG"]
__version__ = APP_VERSION
