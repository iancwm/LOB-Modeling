"""Tests for the module registry."""

from typing import Any, Dict, List

import pytest

from src.lob_modeling.webapp.modules.base import (
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)
from src.lob_modeling.webapp.modules.registry import (
    ModuleRegistry,
    get_module,
    list_modules,
    register_module,
    registry,
)


class TestModuleRegistry:
    """Tests for ModuleRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModuleRegistry()

        class TestModule(ModelModule):
            @property
            def model_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test module"

            @property
            def parameters(self) -> Dict[str, ParameterSpec]:
                return {}

            @property
            def visualizations(self) -> List[VisualizationSpec]:
                return []

            def simulate(self, params: Dict[str, Any]) -> SimulationResult:
                return SimulationResult(
                    time_series={"x": [1]},
                    metrics={"y": 1.0},
                )

            def get_educational_content(self) -> EducationalContent:
                return EducationalContent(
                    learning_objectives=[],
                    background_theory="",
                    equations=[],
                    interpretation_guide="",
                )

        self.TestModule = TestModule

    def test_register(self):
        """Test registering a module."""
        self.registry.register("test", self.TestModule)
        assert "test" in self.registry._modules

    def test_get_registered_module(self):
        """Test getting a registered module."""
        self.registry.register("test", self.TestModule)
        module = self.registry.get("test")
        assert isinstance(module, self.TestModule)
        assert module.model_id == "test"

    def test_get_unregistered_module(self):
        """Test getting an unregistered module raises KeyError."""
        with pytest.raises(KeyError, match="Model 'nonexistent' not found"):
            self.registry.get("nonexistent")

    def test_list_modules(self):
        """Test listing registered modules."""
        self.registry.register("test1", self.TestModule)
        self.registry.register("test2", self.TestModule)
        modules = self.registry.list_modules()
        assert modules == ["test1", "test2"]

    def test_list_empty_modules(self):
        """Test listing when no modules registered."""
        modules = self.registry.list_modules()
        assert modules == []

    def test_contains_registered_module(self):
        """Test checking if registry contains a module."""
        self.registry.register("test", self.TestModule)
        assert "test" in self.registry

    def test_contains_unregistered_module(self):
        """Test checking if registry contains an unregistered module."""
        assert "nonexistent" not in self.registry


class TestRegistryFunctions:
    """Tests for module-level registry functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry before each test
        registry._modules = {}

        class TestModule(ModelModule):
            @property
            def model_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test"

            @property
            def description(self) -> str:
                return "Test module"

            @property
            def parameters(self) -> Dict[str, ParameterSpec]:
                return {}

            @property
            def visualizations(self) -> List[VisualizationSpec]:
                return []

            def simulate(self, params: Dict[str, Any]) -> SimulationResult:
                return SimulationResult(
                    time_series={"x": [1]},
                    metrics={"y": 1.0},
                )

            def get_educational_content(self) -> EducationalContent:
                return EducationalContent(
                    learning_objectives=[],
                    background_theory="",
                    equations=[],
                    interpretation_guide="",
                )

        self.TestModule = TestModule

    def test_register_module(self):
        """Test register_module function."""
        register_module("test", self.TestModule)
        assert "test" in registry

    def test_get_module(self):
        """Test get_module function."""
        register_module("test", self.TestModule)
        module = get_module("test")
        assert isinstance(module, self.TestModule)

    def test_get_module_not_found(self):
        """Test get_module with nonexistent module."""
        with pytest.raises(KeyError):
            get_module("nonexistent")

    def test_list_modules(self):
        """Test list_modules function."""
        register_module("test1", self.TestModule)
        register_module("test2", self.TestModule)
        modules = list_modules()
        assert len(modules) == 2
        assert "test1" in modules
        assert "test2" in modules
