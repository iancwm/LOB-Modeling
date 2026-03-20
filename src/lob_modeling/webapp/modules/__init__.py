"""Model module registry and wrappers."""

from .registry import get_module, list_modules, register_module, registry
from .base import ModelModule, ParameterSpec, VisualizationSpec, SimulationResult, EducationalContent

# Register available model modules
from .wrappers.kyle_wrapper import KyleModelModule

register_module("kyle", KyleModelModule)

__all__ = [
    "get_module",
    "list_modules", 
    "register_module",
    "registry",
    "ModelModule",
    "ParameterSpec",
    "VisualizationSpec",
    "SimulationResult",
    "EducationalContent",
]
