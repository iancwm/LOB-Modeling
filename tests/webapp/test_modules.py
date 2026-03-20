"""Tests for the module base classes."""

import pytest
from typing import Any, Dict, List

from src.lob_modeling.webapp.modules.base import (
    ModelModule,
    ParameterSpec,
    VisualizationSpec,
    SimulationResult,
    EducationalContent,
)


class TestParameterSpec:
    """Tests for ParameterSpec class."""

    def test_init_with_required_fields(self):
        """Test ParameterSpec initialization with required fields."""
        spec = ParameterSpec(
            name="test_param",
            type_=float,
            default=1.0,
        )
        assert spec.name == "test_param"
        assert spec.type_ == float
        assert spec.default == 1.0
        assert spec.min_value is None
        assert spec.max_value is None
        assert spec.description == ""

    def test_init_with_all_fields(self):
        """Test ParameterSpec initialization with all fields."""
        spec = ParameterSpec(
            name="test_param",
            type_=int,
            default=5,
            min_value=0,
            max_value=10,
            description="A test parameter",
        )
        assert spec.name == "test_param"
        assert spec.type_ == int
        assert spec.default == 5
        assert spec.min_value == 0
        assert spec.max_value == 10
        assert spec.description == "A test parameter"

    def test_to_dict(self):
        """Test ParameterSpec to_dict conversion."""
        spec = ParameterSpec(
            name="price",
            type_=float,
            default=10.5,
            min_value=0.0,
            max_value=100.0,
            description="Security price",
        )
        result = spec.to_dict()
        assert result == {
            "name": "price",
            "type": "float",
            "default": 10.5,
            "min": 0.0,
            "max": 100.0,
            "description": "Security price",
        }


class TestVisualizationSpec:
    """Tests for VisualizationSpec class."""

    def test_init_with_required_fields(self):
        """Test VisualizationSpec initialization with required fields."""
        spec = VisualizationSpec(
            id="test_viz",
            title="Test Visualization",
            type="line",
        )
        assert spec.id == "test_viz"
        assert spec.title == "Test Visualization"
        assert spec.type == "line"
        assert spec.description == ""
        assert spec.data_mapping == {}
        assert spec.axes == {}
        assert spec.annotations == []
        assert spec.styling == {}

    def test_init_with_all_fields(self):
        """Test VisualizationSpec initialization with all fields."""
        spec = VisualizationSpec(
            id="price_chart",
            title="Price Over Time",
            type="multi_line",
            description="Shows price discovery",
            data_mapping={"x": "time", "y": ["price"]},
            axes={"x": {"label": "Time"}, "y": {"label": "Price"}},
            annotations=[{"type": "line", "value": 0}],
            styling={"color": "blue"},
        )
        assert spec.id == "price_chart"
        assert spec.title == "Price Over Time"
        assert spec.type == "multi_line"
        assert spec.description == "Shows price discovery"
        assert spec.data_mapping == {"x": "time", "y": ["price"]}
        assert spec.axes == {"x": {"label": "Time"}, "y": {"label": "Price"}}
        assert spec.annotations == [{"type": "line", "value": 0}]
        assert spec.styling == {"color": "blue"}

    def test_to_dict(self):
        """Test VisualizationSpec to_dict conversion."""
        spec = VisualizationSpec(
            id="order_flow",
            title="Order Flow",
            type="bar",
            description="Order flow visualization",
        )
        result = spec.to_dict()
        assert result == {
            "id": "order_flow",
            "title": "Order Flow",
            "type": "bar",
            "description": "Order flow visualization",
            "data_mapping": {},
            "axes": {},
            "annotations": [],
            "styling": {},
        }


class TestSimulationResult:
    """Tests for SimulationResult class."""

    def test_init_with_required_fields(self):
        """Test SimulationResult initialization with required fields."""
        result = SimulationResult(
            time_series={"time": [0, 1, 2], "price": [1.0, 1.1, 1.2]},
            metrics={"final_price": 1.2},
        )
        assert result.time_series == {"time": [0, 1, 2], "price": [1.0, 1.1, 1.2]}
        assert result.metrics == {"final_price": 1.2}
        assert result.metadata == {}

    def test_init_with_metadata(self):
        """Test SimulationResult initialization with metadata."""
        result = SimulationResult(
            time_series={"time": [0, 1, 2]},
            metrics={"metric": 1.0},
            metadata={"model_id": "test"},
        )
        assert result.metadata == {"model_id": "test"}

    def test_to_dict(self):
        """Test SimulationResult to_dict conversion."""
        result = SimulationResult(
            time_series={"time": [0, 1, 2], "value": [1.0, 2.0, 3.0]},
            metrics={"mean": 2.0},
            metadata={"source": "test"},
        )
        assert result.to_dict() == {
            "time_series": {"time": [0, 1, 2], "value": [1.0, 2.0, 3.0]},
            "metrics": {"mean": 2.0},
            "metadata": {"source": "test"},
        }


class TestEducationalContent:
    """Tests for EducationalContent class."""

    def test_init(self):
        """Test EducationalContent initialization."""
        content = EducationalContent(
            learning_objectives=["Objective 1", "Objective 2"],
            background_theory="Theory text",
            equations=[{"label": "E=mc^2", "equation": "E=mc^2"}],
            interpretation_guide="Guide text",
        )
        assert content.learning_objectives == ["Objective 1", "Objective 2"]
        assert content.background_theory == "Theory text"
        assert content.equations == [{"label": "E=mc^2", "equation": "E=mc^2"}]
        assert content.interpretation_guide == "Guide text"

    def test_to_dict(self):
        """Test EducationalContent to_dict conversion."""
        content = EducationalContent(
            learning_objectives=["Learn something"],
            background_theory="Background",
            equations=[],
            interpretation_guide="Interpret",
        )
        result = content.to_dict()
        assert result == {
            "learning_objectives": ["Learn something"],
            "background_theory": "Background",
            "equations": [],
            "interpretation_guide": "Interpret",
        }


class TestModelModule:
    """Tests for ModelModule abstract base class."""

    def test_abstract_methods(self):
        """Test that ModelModule cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelModule()

    def test_concrete_implementation(self):
        """Test that a concrete implementation can be created."""
        class TestModule(ModelModule):
            @property
            def model_id(self) -> str:
                return "test"

            @property
            def display_name(self) -> str:
                return "Test Module"

            @property
            def description(self) -> str:
                return "A test module"

            @property
            def parameters(self) -> Dict[str, ParameterSpec]:
                return {}

            @property
            def visualizations(self) -> List[VisualizationSpec]:
                return []

            def simulate(self, params: Dict[str, Any]) -> SimulationResult:
                return SimulationResult(
                    time_series={"x": [1, 2, 3]},
                    metrics={"y": 1.0},
                )

            def get_educational_content(self) -> EducationalContent:
                return EducationalContent(
                    learning_objectives=[],
                    background_theory="",
                    equations=[],
                    interpretation_guide="",
                )

        module = TestModule()
        assert module.model_id == "test"
        assert module.display_name == "Test Module"
        assert module.description == "A test module"
