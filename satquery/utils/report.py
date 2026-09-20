"""Report generation for SatQuery AI: JSON payloads and lightweight PDF export.

The PDF writer is dependency-free (a minimal but valid PDF 1.4 generator) so
that downloadable reports work on any machine without extra packages.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from satquery.config import APP_NAME, APP_VERSION, PROBLEM_STATEMENT_ID

# --------------------------------------------------------------------- JSON --


def _flatten(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Flatten nested dicts/lists into dotted-key scalar rows for tables."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _flatten(value, f"{prefix}{key}." if prefix else f"{key}.")
    elif isinstance(obj, (list, tuple)) and obj and all(isinstance(v, (int, float, str, bool)) for v in obj):
        yield prefix.rstrip("."), ", ".join(str(v) for v in obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        yield prefix.rstrip("."), obj


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a flat ``key -> scalar`` view of a metrics dictionary."""
    flat: Dict[str, Any] = {}
    for key, value in flatten_dict(metrics):
        flat[key] = value
    return flat


def flatten_dict(metrics: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return list(_flatten(metrics))


def build_report_payload(summary, result) -> Dict[str, Any]:
    """Assemble the complete downloadable report payload.

    ``summary`` is a :class:`satquery.utils.logger.ExecutionSummary` and
    ``result`` an :class:`satquery.agent.controller.AgentResult`.
    """
    images = [img.to_dict() for img in getattr(result, "images", [])]
    payload: Dict[str, Any] = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "problem_statement": PROBLEM_STATEMENT_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": summary.session_id,
        "query": summary.query,
        "selected_task": summary.selected_task,
        "selected_tools": list(summary.selected_tools),
        "confidence": summary.confidence,
        "confidence_breakdown": dict(summary.confidence_breakdown),
        "answer": getattr(result, "answer", ""),
        "metrics": getattr(result, "metrics", {}),
        "inputs": images,
        "warnings": list(summary.warnings),
        "errors": list(summary.errors),
        "execution_summary": summary.to_dict(),
    }
    return payload


def save_json_report(payload: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- PDF --
_PAGE_W, _PAGE_H = 595.0, 842.0  # A4 portrait, points
_MARGIN = 54.0
_LINE_GAP = 6.0


def _latin(text: str) -> str:
    replacements = {
        "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
        "•": "-", "×": "x", "≥": ">=", "≤": "<=", "→": "->", "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class _PdfPage:
    def __init__(self) -> None:
        self.ops: List[str] = []
        self.y = _PAGE_H - _MARGIN

    def text(self, content: str, font: str = "F1", size: float = 10.0,
             x: float = _MARGIN, color: Tuple[float, float, float] = (0, 0, 0)) -> None:
        r, g, b = color
        self.ops.append(
            f"BT /{font} {size:.1f} Tf {r:.2f} {g:.2f} {b:.2f} rg "
            f"1 0 0 1 {x:.1f} {self.y:.1f} Tm ({_escape(_latin(content))}) Tj ET"
        )

    def line(self, x1: float, y1: float, x2: float, y2: float,
             width: float = 0.7, color: Tuple[float, float, float] = (0.55, 0.55, 0.55)) -> None:
        r, g, b = color
        self.ops.append(
            f"{width:.1f} w {r:.2f} {g:.2f} {b:.2f} RG "
            f"{x1:.1f} {(_PAGE_H - y1):.1f} m {x2:.1f} {(_PAGE_H - y2):.1f} l S"
        )

    def advance(self, amount: float) -> None:
        self.y -= amount

    def at_bottom(self, needed: float = 24.0) -> bool:
        return self.y - needed < _MARGIN

    def content_stream(self) -> bytes:
        return "\n".join(self.ops).encode("latin-1", errors="replace")


class PdfBuilder:
    """Minimal multi-page PDF writer (Helvetica only, text + rules)."""

    CHAR_WIDTH_FACTOR = 0.52  # rough average Helvetica glyph width / size

    def __init__(self) -> None:
        self.pages: List[_PdfPage] = [_PdfPage()]

    # ----------------------------------------------------------------- layout
    @property
    def page(self) -> _PdfPage:
        return self.pages[-1]

    def _usable_width(self) -> float:
        return _PAGE_W - 2 * _MARGIN

    def wrap(self, text: str, size: float = 10.0) -> List[str]:
        max_chars = max(20, int(self._usable_width() / (size * self.CHAR_WIDTH_FACTOR)))
        words = _latin(str(text)).split()
        if not words:
            return [""]
        lines, current = [], words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= max_chars:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def text(self, content: str, size: float = 10.0, bold: bool = False,
             color: Tuple[float, float, float] = (0, 0, 0), indent: float = 0.0,
             gap: float = _LINE_GAP) -> None:
        font = "F2" if bold else "F1"
        lines = self.wrap(content, size)
        for index, line in enumerate(lines):
            if self.page.at_bottom(size + gap):
                self.new_page()
            self.page.text(line, font=font, size=size, x=_MARGIN + indent, color=color)
            self.page.advance(size + gap)
        self.page.advance(gap * 0.5)

    def heading(self, content: str) -> None:
        if self.page.at_bottom(40):
            self.new_page()
        self.page.advance(4)
        self.text(content, size=12.5, bold=True, color=(0.10, 0.22, 0.45), gap=2)
        self.page.line(_MARGIN, self.page.y + 6, _PAGE_W - _MARGIN, self.page.y + 6, width=0.9)
        self.page.advance(10)

    def key_value(self, key: str, value: Any, indent: float = 0.0) -> None:
        self.text(f"{key}: {value}", size=9.5, indent=indent, gap=1.5)

    def bullets(self, items: Iterable[str], indent: float = 10.0) -> None:
        for item in items:
            self.text(f"- {item}", size=9.5, indent=indent, gap=1.5)
        self.page.advance(4)

    def spacer(self, amount: float = 8.0) -> None:
        self.page.advance(amount)

    def new_page(self) -> None:
        self.pages.append(_PdfPage())

    # ----------------------------------------------------------------- output
    def build(self) -> bytes:
        objects: List[bytes] = []

        def add(body: bytes) -> int:
            objects.append(body)
            return len(objects)

        kids_placeholder = add(b"")  # object 1 reserved: catalog
        pages_obj = add(b"")         # object 2 reserved: pages tree
        font_regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

        page_ids: List[int] = []
        for pdf_page in self.pages:
            content_id = add(pdf_page.content_stream())
            page_id = add(
                f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 {_PAGE_W:.0f} {_PAGE_H:.0f}] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_regular} 0 R "
                f"/F2 {font_bold} 0 R >> >> >>".encode()
            )
            page_ids.append(page_id)

        objects[0] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        objects[1] = (
            f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
        )

        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        body = bytearray()
        offsets: List[int] = []
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(header) + len(body))
            body += f"{index} 0 obj\n".encode("latin-1") + obj + b"\nendobj\n"
        xref_pos = len(header) + len(body)
        xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
        for offset in offsets:
            xref += f"{offset:010d} 00000 n \n"
        trailer = (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        )
        return header + bytes(body) + xref.encode("latin-1") + trailer.encode("latin-1")


# ------------------------------------------------------------- report build --
def _fmt_confidence(value: Optional[float]) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def render_pdf_report(payload: Dict[str, Any]) -> bytes:
    """Render the report payload into PDF bytes."""
    pdf = PdfBuilder()
    app = payload.get("app", {})

    pdf.text(f"{app.get('name', APP_NAME)} - Execution Report", size=17, bold=True, color=(0.05, 0.18, 0.38))
    pdf.text(
        f"Problem statement {payload.get('problem_statement', PROBLEM_STATEMENT_ID)}   |   "
        f"Session {payload.get('session_id', 'n/a')}   |   "
        f"Generated {payload.get('generated_at', 'n/a')}   |   v{app.get('version', APP_VERSION)}",
        size=8.5, color=(0.35, 0.35, 0.35),
    )
    pdf.spacer(6)

    pdf.heading("1. Query and Routing")
    pdf.key_value("Natural-language query", payload.get("query", ""))
    pdf.key_value("Selected task", payload.get("selected_task", "n/a"))
    pdf.key_value("Tools executed", ", ".join(payload.get("selected_tools", [])) or "none")
    pdf.key_value("Overall confidence", _fmt_confidence(payload.get("confidence")))
    breakdown = payload.get("confidence_breakdown", {})
    for key in ("data_quality", "model", "agreement"):
        if key in breakdown:
            pdf.key_value(f"  confidence.{key}", _fmt_confidence(breakdown.get(key)), indent=10)
    pdf.spacer(4)

    pdf.heading("2. Answer")
    pdf.text(str(payload.get("answer", "")), size=10.5)
    pdf.spacer(2)

    pdf.heading("3. Inputs and Validation")
    images = payload.get("inputs", [])
    if not images:
        pdf.text("No image inputs recorded.", size=9.5)
    for img in images:
        pdf.text(f"- {img.get('name', 'image')}", size=10, bold=True, gap=1)
        pdf.key_value("  format", f"{img.get('format')} | {img.get('width')}x{img.get('height')} | {img.get('bands')} band(s)")
        pdf.key_value("  modality", f"{img.get('modality')} (confidence {img.get('modality_confidence')})")
        pdf.key_value("  CRS", f"{img.get('crs')} (valid: {img.get('crs_valid')})")
        if img.get("issues"):
            for issue in img["issues"]:
                pdf.key_value("  issue", issue, indent=14)
        pdf.spacer(2)
    warnings = payload.get("warnings") or []
    errors = payload.get("errors") or []
    if warnings:
        pdf.text("Warnings", size=10, bold=True, gap=1)
        pdf.bullets(warnings)
    if errors:
        pdf.text("Errors", size=10, bold=True, gap=1)
        pdf.bullets(errors)

    pdf.heading("4. Key Metrics")
    rows = flatten_dict(payload.get("metrics", {}))[:60]
    if not rows:
        pdf.text("No metrics recorded.", size=9.5)
    for key, value in rows:
        pdf.key_value(key, value, indent=6)

    pdf.heading("5. Auditable Execution Trace")
    summary = payload.get("execution_summary", {})
    tool_calls = summary.get("tool_calls", [])
    if tool_calls:
        for call in tool_calls:
            pdf.text(
                f"tool: {call.get('tool_name')} ({call.get('tool_id')})  |  model: {call.get('model_ref')}  |  "
                f"status: {call.get('status')}  |  {call.get('duration_ms', 0):.1f} ms",
                size=9, gap=1,
            )
            params = call.get("parameters", {})
            if params:
                pdf.text(f"  parameters: {json.dumps(params, default=str)}", size=8, indent=12, gap=1)
            if call.get("summary"):
                pdf.text(f"  summary: {call.get('summary')}", size=8, indent=12, gap=3)
    else:
        pdf.text("No tool calls recorded.", size=9.5)
    pdf.spacer(2)
    events = summary.get("execution_trace", [])
    for event in events:
        duration = event.get("duration_ms")
        duration_text = f"{duration:.0f} ms" if isinstance(duration, (int, float)) else "-"
        pdf.text(
            f"[{event.get('status', 'ok').upper():7s}] {event.get('step')} ({event.get('component')}) "
            f"- {event.get('message', '')} [{duration_text}]",
            size=8, indent=6, gap=1,
        )

    pdf.spacer(8)
    pdf.page.line(_MARGIN, pdf.page.y, _PAGE_W - _MARGIN, pdf.page.y, width=0.5)
    pdf.page.advance(10)
    pdf.text(
        f"Generated by {APP_NAME} v{APP_VERSION} - agentic multi-modal remote-sensing framework (SIH-167).",
        size=8, color=(0.4, 0.4, 0.4),
    )
    return pdf.build()


def save_pdf_report(payload: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_pdf_report(payload))
    return path
