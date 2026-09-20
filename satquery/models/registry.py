"""Specialist Model Registry for SatQuery AI.

The registry is the single source of truth the agent consults when routing a
query: it declares every specialist tool (id, name, task, input requirement,
model reference, default parameters) and resolves them lazily via a factory
callable, so heavy model weights are only instantiated when a run needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from satquery.utils.logger import get_logger

logger = get_logger("registry")

TASK_SINGLE_IMAGE = "single_image_analysis"
TASK_CHANGE_ANALYSIS = "bi_temporal_change"
TASK_CROSS_MODAL = "cross_modal_fusion"

INPUT_REQUIREMENTS: Dict[str, str] = {
    "single_image": "1 image (optical/multispectral or SAR)",
    "bi_temporal_pair": "2 spatially aligned images of the same area (T1, T2)",
    "cross_modal_pair": "co-registered optical/multispectral + SAR pair",
}


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of one specialist tool."""

    tool_id: str
    name: str
    task: str
    description: str
    input_requirement: str
    model_ref: str
    default_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "task": self.task,
            "description": self.description,
            "input_requirement": self.input_requirement,
            "model_ref": self.model_ref,
            "default_params": dict(self.default_params),
        }


class SpecialistModelRegistry:
    """Registry of specialist tools with lazy, cached instantiation.

    Example
    -------
    >>> registry = SpecialistModelRegistry.default()
    >>> registry.select_tools("single_image_vqa", images=[img])
    ['single_image_vqa', 'scene_captioner']
    >>> tool = registry.get_tool("single_image_vqa", images=[img])
    >>> result = tool.run(img, question="Describe the land cover.")
    """

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}
        self._factories: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._instances: Dict[str, Any] = {}

    # ------------------------------------------------------------ registration
    def register(self, spec: ToolSpec, factory: Callable[[Dict[str, Any]], Any]) -> None:
        if spec.tool_id in self._specs:
            raise ValueError(f"Tool '{spec.tool_id}' is already registered.")
        if not callable(factory):
            raise TypeError(f"Factory for '{spec.tool_id}' must be callable.")
        self._specs[spec.tool_id] = spec
        self._factories[spec.tool_id] = factory
        logger.info("Registered tool '%s' (%s)", spec.tool_id, spec.model_ref)

    # ---------------------------------------------------------------- queries
    def has_tool(self, tool_id: str) -> bool:
        return tool_id in self._specs

    def list_tools(self) -> List[ToolSpec]:
        return [self._specs[key] for key in sorted(self._specs)]

    def tools_for_task(self, task: str) -> List[ToolSpec]:
        return [spec for spec in self.list_tools() if spec.task == task]

    def describe(self) -> Dict[str, Any]:
        return {
            "tasks": {
                TASK_SINGLE_IMAGE: INPUT_REQUIREMENTS["single_image"],
                TASK_CHANGE_ANALYSIS: INPUT_REQUIREMENTS["bi_temporal_pair"],
                TASK_CROSS_MODAL: INPUT_REQUIREMENTS["cross_modal_pair"],
            },
            "tools": [spec.to_dict() for spec in self.list_tools()],
        }

    # --------------------------------------------------------------- routing
    def select_tools(self, task: str, images: Optional[List[Any]] = None) -> List[str]:
        """Choose tool ids for a task given the validated input bundle.

        Rules
        -----
        * single image -> VQA tool, plus captioner (optical) or SAR analyser.
        * bi-temporal  -> change analyser; change VQA answers follow-up
          questions about the change scene.
        * cross-modal  -> fusion tool; grounding also available on the optical
          twin for region queries.
        """
        modalities = [getattr(img, "modality", "optical") for img in (images or [])]
        sar_present = any(m == "sar" for m in modalities)
        optical_present = any(m != "sar" for m in modalities)

        if task == TASK_CROSS_MODAL:
            return ["optical_sar_fusion"]
        if task == TASK_CHANGE_ANALYSIS:
            return ["change_analyser"]
        if task == TASK_SINGLE_IMAGE:
            tools = ["single_image_vqa"]
            if optical_present:
                tools.append("scene_captioner")
            if sar_present:
                tools.append("sar_structure_analyser")
            return tools
        raise ValueError(f"Unknown task '{task}'.")

    def supports(self, task: str, images: Optional[List[Any]] = None) -> bool:
        try:
            return bool(self.select_tools(task, images))
        except ValueError:
            return False

    # ---------------------------------------------------------- instantiation
    def get_tool(self, tool_id: str, images: Optional[List[Any]] = None) -> Any:
        """Return a live tool instance, instantiating it on first use."""
        if tool_id not in self._specs:
            raise KeyError(f"Tool '{tool_id}' is not registered.")
        if tool_id in self._instances:
            return self._instances[tool_id]
        instance = self._factories[tool_id]({"images": images or []})
        self._instances[tool_id] = instance
        logger.info("Instantiated tool '%s'", tool_id)
        return instance

    def reset(self) -> None:
        """Drop cached instances (used by tests and the GUI reload button)."""
        self._instances.clear()


# ------------------------------------------------------------------ default --
def build_default_registry() -> SpecialistModelRegistry:
    """Registry pre-loaded with all SatQuery specialist tools."""
    from satquery.models.change_detection import BiTemporalChangeAnalyzer
    from satquery.models.cross_modal_fusion import OpticalSARFusionTool
    from satquery.models.single_image_vqa import (
        OpticalSingleImageTool,
        SARStructureTool,
        SceneCaptionerTool,
    )

    registry = SpecialistModelRegistry()

    registry.register(
        ToolSpec(
            tool_id="single_image_vqa",
            name="Single-Image VQA",
            task=TASK_SINGLE_IMAGE,
            description="Visual question answering over one optical/multispectral or SAR image.",
            input_requirement=INPUT_REQUIREMENTS["single_image"],
            model_ref="satquery-vqa (RS-adapted ViT-B/16 + BigEarthNet LoRA, offline head)",
        ),
        lambda _: OpticalSingleImageTool(),
    )
    registry.register(
        ToolSpec(
            tool_id="scene_captioner",
            name="Scene Captioner",
            task=TASK_SINGLE_IMAGE,
            description="Land-cover captioning and text-guided region grounding for optical scenes.",
            input_requirement=INPUT_REQUIREMENTS["single_image"],
            model_ref="satquery-captioner (RS-adapted, BigEarthNet label grounding)",
        ),
        lambda _: SceneCaptionerTool(),
    )
    registry.register(
        ToolSpec(
            tool_id="sar_structure_analyser",
            name="SAR Structure Analyser",
            task=TASK_SINGLE_IMAGE,
            description="Structural interpretation of SAR imagery (backscatter, texture, built-up cues).",
            input_requirement=INPUT_REQUIREMENTS["single_image"],
            model_ref="satquery-sar-struct (speckle-robust texture head)",
        ),
        lambda _: SARStructureTool(),
    )
    registry.register(
        ToolSpec(
            tool_id="change_analyser",
            name="Bi-Temporal Change Analyser",
            task=TASK_CHANGE_ANALYSIS,
            description="Change detection, change description, change-VQA and spatial change maps.",
            input_requirement=INPUT_REQUIREMENTS["bi_temporal_pair"],
            model_ref="satquery-change (index differencing + CDVQA-style QA head)",
        ),
        lambda _: BiTemporalChangeAnalyzer(),
    )
    registry.register(
        ToolSpec(
            tool_id="optical_sar_fusion",
            name="Optical-SAR Fusion",
            task=TASK_CROSS_MODAL,
            description="Joint spectral + structural feature extraction from co-registered optical+SAR pairs.",
            input_requirement=INPUT_REQUIREMENTS["cross_modal_pair"],
            model_ref="satquery-fusion (early band fusion + late embedding agreement)",
        ),
        lambda _: OpticalSARFusionTool(),
    )
    return registry


_DEFAULT_REGISTRY: Optional[SpecialistModelRegistry] = None


def default_registry() -> SpecialistModelRegistry:
    """Process-wide default registry (created on first call)."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY
