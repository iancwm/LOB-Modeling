"""Module registry for model discovery and registration."""

from typing import List, Type

from .base import ModelModule


class ModuleRegistry:
    """Registry for model modules."""

    def __init__(self):
        """Initialize the module registry."""
        self._modules: dict[str, Type[ModelModule]] = {}

    def register(self, model_id: str, module_class: Type[ModelModule]) -> None:
        """Register a model module.

        Args:
            model_id: Unique identifier for the model.
            module_class: Model module class to register.
        """
        self._modules[model_id] = module_class

    def get(self, model_id: str) -> ModelModule:
        """Get a module instance by ID.

        Args:
            model_id: Unique identifier for the model.

        Returns:
            Module instance.

        Raises:
            KeyError: If model_id is not found.
        """
        if model_id not in self._modules:
            raise KeyError(f"Model '{model_id}' not found")
        return self._modules[model_id]()

    def list_modules(self) -> List[str]:
        """List all registered module IDs.

        Returns:
            List of module IDs.
        """
        return list(self._modules.keys())

    def __contains__(self, model_id: str) -> bool:
        """Check if a model is registered.

        Args:
            model_id: Unique identifier for the model.

        Returns:
            True if model is registered, False otherwise.
        """
        return model_id in self._modules


# Global registry instance
registry = ModuleRegistry()


def get_module(model_id: str) -> ModelModule:
    """Get a module instance by ID.

    Args:
        model_id: Unique identifier for the model.

    Returns:
        Module instance.
    """
    return registry.get(model_id)


def list_modules() -> List[str]:
    """List all registered module IDs.

    Returns:
        List of module IDs.
    """
    return registry.list_modules()


def register_module(model_id: str, module_class: Type[ModelModule]) -> None:
    """Register a model module.

    Args:
        model_id: Unique identifier for the model.
        module_class: Model module class to register.
    """
    registry.register(model_id, module_class)
