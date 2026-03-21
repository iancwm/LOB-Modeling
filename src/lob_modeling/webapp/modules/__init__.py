"""Model module registry and wrappers."""

from .base import (
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)
from .registry import get_module, list_modules, register_module, registry

# Register available model modules
from .wrappers.kyle_wrapper import KyleModelModule
from .wrappers.almgren_chriss_wrapper import AlmgrenChrissModule

register_module("kyle", KyleModelModule)
register_module("almgren_chriss", AlmgrenChrissModule)

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
