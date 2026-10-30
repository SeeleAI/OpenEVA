from typing import TYPE_CHECKING

from .configuration_eva01 import EVA01Config
from .modeling_eva01 import EVA01ForConditionalGeneration
from .mesh_runtime import decode_mesh_with_material, export_mesh_glb, load_decoder
from .processing_eva01 import EVA01Processor

if TYPE_CHECKING:
    from .unified_generation import EVA01UnifiedGeneration, GenerationResult

__all__ = [
    "EVA01Config",
    "EVA01ForConditionalGeneration",
    "EVA01Processor",
    "decode_mesh_with_material",
    "export_mesh_glb",
    "load_decoder",
    "EVA01UnifiedGeneration",
    "GenerationResult",
]


def __getattr__(name: str):
    if name in {"EVA01UnifiedGeneration", "GenerationResult"}:
        from .unified_generation import EVA01UnifiedGeneration, GenerationResult

        return {
            "EVA01UnifiedGeneration": EVA01UnifiedGeneration,
            "GenerationResult": GenerationResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
