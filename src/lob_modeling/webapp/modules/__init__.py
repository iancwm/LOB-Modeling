"""Model module registry and wrappers."""

from .base import (
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)
from .registry import get_module, list_modules, register_module, registry
from .wrappers.almgren_chriss_wrapper import AlmgrenChrissModule
from .wrappers.asset_option_wrapper import AssetOptionModule
from .wrappers.criscuolo_waehlbroeck_wrapper import CriscuoloWaehlbroeckModule
from .wrappers.de_prado_wrapper import DePradoModule
from .wrappers.glosten_milgrom_wrapper import GlostenMilgromModule

# Register available model modules
from .wrappers.kyle_wrapper import KyleModelModule

register_module("kyle", KyleModelModule)
register_module("almgren_chriss", AlmgrenChrissModule)
register_module("glosten_milgrom", GlostenMilgromModule)
register_module("de_prado", DePradoModule)
register_module("criscuolo_waehlbroeck", CriscuoloWaehlbroeckModule)
register_module("asset_option", AssetOptionModule)

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
