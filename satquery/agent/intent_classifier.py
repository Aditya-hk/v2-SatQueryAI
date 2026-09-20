"""Natural-language intent classification and task routing for SatQuery AI.

Rules + fuzzy keyword scoring (offline, deterministic). The classifier resolves:

* the target task - single-image analysis, bi-temporal change analysis, or
  cross-modal (optical-SAR) fusion;
* the sub-capability - VQA, captioning, or region grounding;
* explicit parameters - change threshold overrides, region-of-interest terms,
  requested top-k regions.

If the selected task is not supported by the uploaded input bundle, the
classifier proposes a graceful fallback task that *is* supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from satquery.models.registry import (
    TASK_CHANGE_ANALYSIS,
    TASK_CROSS_MODAL,
    TASK_SINGLE_IMAGE,
)
from satquery.utils.logger import get_logger

logger = get_logger("intent")

# ------------------------------------------------------------------ lexicon ---
TASK_KEYWORDS: Dict[str, List[str]] = {
    TASK_CHANGE_ANALYSIS: [
        "change", "changed", "difference", "differ", "between the two", "between these",
        "two dates", "both dates", "compare", "compared", "over time", "time", "dates",
        "before and after", "after", "before", "growth", "shrunk", "expanded", "increase",
        "decrease", "increased", "decreased", "loss", "gained", "deforestation",
        "urban expansion", "urban growth", "new construction", "flood", "flooded",
        "receded", "dried", "t1", "t2",
    ],
    TASK_CROSS_MODAL: [
        "optical and sar", "sar and optical", "optical+sar", "sar+optical",
        "using optical", "using sar", "use the optical", "use both", "jointly",
        "together", "combine", "combined", "both sensors", "both images", "fusion",
        "cross-modal", "cross modal", "multisensor", "multi-sensor", "radar and optical",
        "cartosat", "risat", "sentinel-1 and sentinel-2", "s1 and s2",
    ],
    TASK_SINGLE_IMAGE: [
        "describe", "caption", "what is shown", "what is visible", "what do you see",
        "summar", "overview", "identify", "detect", "highlight", "locate", "where is",
        "where are", "find", "show me", "mark", "how many", "count", "is there",
        "are there", "any ", "what type", "which class", "land cover", "land-cover",
        "what covers", "what kind", "classify", "segment", "map of", "analyse",
        "analyze", "interpret", "what changed in this", "explanation", "explain",
    ],
}

GROUNDING_TERMS: Tuple[str, ...] = (
    "water", "lake", "river", "reservoir", "pond", "flood", "vegetation", "forest",
    "tree", "crop", "farm", "grass", "park", "field", "built", "urban", "building",
    "city", "settlement", "house", "industr", "road", "bare", "soil", "sand", "rock",
    "desert", "dune", "ship", "vessel", "cloud", "clouds",
)

_CAPTION_PATTERNS = (
    r"\bcaption\b", r"\bdescrib", r"\bwhat (is|do) (shown|see|visible)", r"\boverview\b",
    r"\bsummar", r"\bcontent", r"\bland.?cover", r"\bscene\b",
)
_GROUNDING_PATTERNS = (
    r"\b(highlight|locate|where|find|show me|mark|point out|outline|draw)\b",
)
_VQA_PATTERNS = (
    r"\b(is|are|does|do|can|has|have|what|which|how|how many|count)\b.*\?",
    r"^\s*(is|are|does|can|has|what|which|how many|count)\b",
)

_CHANGE_THRESHOLD_PATTERN = re.compile(
    r"(?:threshold|sensitivit\w+)\s*(?:of|at|to|=|:)?\s*(0?\.\d+|1(?:\.0+)?)", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\b(\d{1,3})\s*(?:regions?|areas?|zones?|patches?|bodies)\b", re.IGNORECASE)


@dataclass
class IntentResult:
    """Structured outcome of natural-language intent classification."""

    task: str
    sub_capability: str  # "vqa" | "captioning" | "grounding" | "change_description" | "change_vqa"
    confidence: float
    matched_keywords: List[str] = field(default_factory=list)
    grounding_term: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "sub_capability": self.sub_capability,
            "confidence": round(self.confidence, 4),
            "matched_keywords": self.matched_keywords,
            "grounding_term": self.grounding_term,
            "parameters": dict(self.parameters),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "explanation": self.explanation,
        }


# --------------------------------------------------------------- utilities ----
def _count_hits(text: str, keywords: List[str]) -> List[str]:
    q = text.lower()
    return [kw for kw in keywords if kw in q]


def _regex_hits(text: str, patterns: Tuple[str, ...]) -> List[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def _top_term(text: str) -> Optional[str]:
    q = text.lower()
    for term in GROUNDING_TERMS:
        if term in q:
            return term
    return None


def classify_intent(query: str) -> IntentResult:
    """Classify the natural-language query into task + sub-capability.

    Scoring: change-related and cross-modal cues outrank generic VQA phrasing
    (pair context), while single-image phrases (``describe``, ``where is``,
    ``how many``) route to single-image analysis with captioning/grounding
    sub-capabilities when applicable.
    """
    query = (query or "").strip()
    hits_change = _count_hits(query, TASK_KEYWORDS[TASK_CHANGE_ANALYSIS])
    hits_cross = _count_hits(query, TASK_KEYWORDS[TASK_CROSS_MODAL])
    hits_single = _count_hits(query, TASK_KEYWORDS[TASK_SINGLE_IMAGE])

    score_change = 1.3 * len(hits_change)
    score_cross = 1.25 * len(hits_cross)
    score_single = 1.0 * len(hits_single)

    if _regex_hits(query, _GROUNDING_PATTERNS):
        score_single += 0.6
    if _regex_hits(query, _CAPTION_PATTERNS):
        score_single += 0.4

    total = score_change + score_cross + score_single
    if total <= 0.0:
        return IntentResult(
            task=TASK_SINGLE_IMAGE,
            sub_capability="vqa",
            confidence=0.35,
            explanation="No strong task cues found; defaulting to single-image VQA.",
        )

    if score_change >= score_cross and score_change >= score_single:
        task = TASK_CHANGE_ANALYSIS
        score = score_change
    elif score_cross >= score_single:
        task = TASK_CROSS_MODAL
        score = score_cross
    else:
        task = TASK_SINGLE_IMAGE
        score = score_single

    raw = score / total
    margin = (score - max(v for v in (score_change, score_cross, score_single) if v != score) or 1.0)
    confidence = min(0.97, 0.55 + 0.42 * raw + 0.05 * min(margin, 2.0))

    # ------------------------------------------------------- sub-capability
    sub_capability = "vqa"
    grounding_term = _top_term(query)
    if task == TASK_SINGLE_IMAGE:
        if _regex_hits(query, _GROUNDING_PATTERNS) and grounding_term:
            sub_capability = "grounding"
        elif _regex_hits(query, _CAPTION_PATTERNS):
            sub_capability = "captioning"
        elif _regex_hits(query, _VQA_PATTERNS):
            sub_capability = "vqa"
        else:
            sub_capability = "vqa"
    elif task == TASK_CHANGE_ANALYSIS:
        # "What changed between these dates?" / "Describe the change" are
        # description requests; yes/no or counting forms are change-VQA.
        yes_no = bool(re.match(r"^\s*(is|are|was|were|has|have|had|does|do|did|can)\b", query.lower())) \
            or "how many" in query.lower() or "how much" in query.lower() \
            or query.lower().startswith("which")
        sub_capability = "change_vqa" if yes_no else "change_description"
    elif task == TASK_CROSS_MODAL:
        if _regex_hits(query, _GROUNDING_PATTERNS) and grounding_term:
            sub_capability = "grounding"
        else:
            sub_capability = "vqa"

    # ---------------------------------------------------------- parameters
    parameters: Dict[str, Any] = {}
    match = _CHANGE_THRESHOLD_PATTERN.search(query)
    if match:
        try:
            parameters["change_threshold"] = float(match.group(1))
        except ValueError:
            pass
    match = _NUMBER_PATTERN.search(query)
    if match:
        parameters["max_regions"] = int(match.group(1))
    if grounding_term:
        parameters["grounding_term"] = grounding_term

    matched = sorted(set(hits_change + hits_cross + hits_single))[:8]
    return IntentResult(
        task=task,
        sub_capability=sub_capability,
        confidence=confidence,
        matched_keywords=matched,
        grounding_term=grounding_term,
        parameters=parameters,
        explanation=(
            f"Routed to '{task}' (sub-capability '{sub_capability}') from keyword cues "
            f"{matched}."
        ),
    )


def fallback_task_for(task: str, images: Optional[List[Any]] = None) -> Optional[Tuple[str, str]]:
    """If the requested task cannot run on the current inputs, propose a fallback.

    Returns ``(task, reason)`` or ``None`` when the task is executable as-is.
    """
    modalities = [getattr(img, "modality", "optical") for img in (images or [])]
    n = len(images or [])
    if task == TASK_CHANGE_ANALYSIS and n != 2:
        # Hard validation stop (per the problem statement's edge-case table):
        # change analysis is only defined for exactly two aligned images.
        reason = f"Change analysis requires exactly 2 images (T1, T2); {n} provided. "
        if n == 0:
            return None, reason + "No images to analyse."
        if n == 1:
            return None, reason + "Upload the second acquisition (T2) or rephrase as a single-image query."
        return None, reason + "Provide exactly two spatially aligned images."
    if task == TASK_CROSS_MODAL and not (n == 2 and "sar" in modalities and "optical" in modalities):
        reason = (f"Cross-modal analysis needs a co-registered optical + SAR pair; provided "
                  f"{n} image(s) with modalities {sorted(set(modalities))}. ")
        if n == 0:
            return None, reason + "No images to analyse."
        if n == 1:
            return TASK_SINGLE_IMAGE, reason + "Falling back to single-image analysis."
        if n == 2:
            return TASK_SINGLE_IMAGE, reason + "Falling back to single-image analysis on the first image."
        return TASK_SINGLE_IMAGE, reason
    if task == TASK_SINGLE_IMAGE and n == 0:
        return None, "No images uploaded; nothing to analyse."
    return None
