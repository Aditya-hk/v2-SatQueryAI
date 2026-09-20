"""Auditable execution-summary and trace logging for SatQuery AI.

Every agent run produces a machine-readable :class:`ExecutionSummary` containing
the selected task, the specialist tools invoked, their parameters, timings,
warnings, errors, and the final confidence. This is the auditable execution
trace required by SIH Problem Statement 167.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from satquery.config import APP_NAME, APP_VERSION, PROBLEM_STATEMENT_ID, RUNS_DIR

_LOGGER_CONFIGURED = False


def configure_logging(level: int = logging.INFO, log_dir: Optional[Path] = None) -> None:
    """Configure a module-wide console + file logger (idempotent)."""
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return
    log_dir = Path(log_dir) if log_dir else Path(RUNS_DIR) / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / "satquery.log", encoding="utf-8"),
            ],
            force=True,
        )
    except Exception:  # pragma: no cover - read-only filesystems etc.
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(message)s")
    _LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring handlers on first use."""
    configure_logging()
    return logging.getLogger(f"satquery.{name}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class TraceEvent:
    """One observable step in the execution trace."""

    step: str
    component: str
    status: str = "ok"  # ok | warning | error
    message: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)
    duration_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    """Record of a specialist tool resolved from the registry and executed."""

    tool_id: str
    tool_name: str
    model_ref: str
    task: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    status: str = "ok"
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionSummary:
    """Thread-safe, serialisable audit trail for one agent session.

    The summary captures the observable execution trace only (selected task,
    tool names, permitted parameters, outputs, timings) as required by the
    problem statement; internal chain-of-thought is never recorded.
    """

    def __init__(self, query: str, inputs: Optional[List[Dict[str, Any]]] = None,
                 config: Optional[Dict[str, Any]] = None) -> None:
        self.session_id: str = f"satquery-{uuid.uuid4().hex[:12]}"
        self.started_at: str = _utc_now_iso()
        self.query: str = query
        self.inputs: List[Dict[str, Any]] = list(inputs or [])
        self.config: Dict[str, Any] = dict(config or {})
        self.events: List[TraceEvent] = []
        self.tool_calls: List[ToolCall] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.confidence: Optional[float] = None
        self.confidence_breakdown: Dict[str, Any] = {}
        self.selected_task: Optional[str] = None
        self.selected_tools: List[str] = []
        self.result_digest: Dict[str, Any] = {}
        self.status: str = "running"
        self.app: Dict[str, str] = {
            "name": APP_NAME,
            "version": APP_VERSION,
            "problem_statement": PROBLEM_STATEMENT_ID,
        }
        self._t0: float = time.perf_counter()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- recording
    def step(self, step: str, component: str, message: str = "", status: str = "ok",
             duration_ms: Optional[float] = None, **details: Any) -> TraceEvent:
        event = TraceEvent(
            step=step,
            component=component,
            status=status,
            message=message,
            duration_ms=duration_ms,
            details=details,
        )
        with self._lock:
            self.events.append(event)
            if status == "warning":
                self.warnings.append(message or f"{step}:{component}")
            elif status == "error":
                self.errors.append(message or f"{step}:{component}")
        return event

    @contextmanager
    def timed_step(self, step: str, component: str, message: str = "") -> Iterator[TraceEvent]:
        """Context manager that records wall-clock duration around a step."""
        event = TraceEvent(step=step, component=component, message=message)
        start = time.perf_counter()
        try:
            yield event
        except Exception as exc:  # noqa: BLE001 - re-raised below
            event.status = "error"
            event.message = f"{type(exc).__name__}: {exc}"
            event.duration_ms = (time.perf_counter() - start) * 1000.0
            with self._lock:
                self.events.append(event)
                self.errors.append(event.message)
            raise
        else:
            event.duration_ms = (time.perf_counter() - start) * 1000.0
            with self._lock:
                self.events.append(event)

    def record_tool_call(self, tool_id: str, tool_name: str, model_ref: str, task: str,
                         parameters: Optional[Dict[str, Any]] = None,
                         duration_ms: Optional[float] = None, status: str = "ok",
                         summary: str = "") -> ToolCall:
        call = ToolCall(
            tool_id=tool_id,
            tool_name=tool_name,
            model_ref=model_ref,
            task=task,
            parameters=dict(parameters or {}),
            duration_ms=duration_ms,
            status=status,
            summary=summary,
        )
        with self._lock:
            self.tool_calls.append(call)
            if tool_id not in self.selected_tools:
                self.selected_tools.append(tool_id)
        return call

    def set_outcome(self, confidence: float, breakdown: Dict[str, Any],
                    result_digest: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self.confidence = round(float(confidence), 4)
            self.confidence_breakdown = dict(breakdown)
            self.result_digest = dict(result_digest or {})

    def finalize(self, status: str = "completed") -> "ExecutionSummary":
        with self._lock:
            self.status = status if status != "running" else ("error" if self.errors else "completed")
        return self

    # ------------------------------------------------------------------ output
    @property
    def duration_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            payload: Dict[str, Any] = {
                "app": dict(self.app),
                "session_id": self.session_id,
                "started_at": self.started_at,
                "duration_ms": round(self.duration_ms, 2),
                "status": self.status,
                "query": self.query,
                "selected_task": self.selected_task,
                "selected_tools": list(self.selected_tools),
                "inputs": [dict(i) for i in self.inputs],
                "config": dict(self.config),
                "execution_trace": [event.as_dict() for event in self.events],
                "tool_calls": [call.as_dict() for call in self.tool_calls],
                "warnings": list(self.warnings),
                "errors": list(self.errors),
                "confidence": self.confidence,
                "confidence_breakdown": dict(self.confidence_breakdown),
                "result_digest": dict(self.result_digest),
            }
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        get_logger("summary").info("Execution summary saved to %s", path)
        return path
