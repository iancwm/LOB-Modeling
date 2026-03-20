"""Base interface for all model modules."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ParameterSpec:
    """Specification for a model parameter."""

    def __init__(
        self,
        name: str,
        type_: type,
        default: Any,
        min_value: float | None = None,
        max_value: float | None = None,
        description: str = "",
    ):
        """Initialize parameter specification.

        Args:
            name: Parameter name.
            type_: Parameter type.
            default: Default value.
            min_value: Minimum allowed value (for numeric types).
            max_value: Maximum allowed value (for numeric types).
            description: Parameter description.
        """
        self.name = name
        self.type_ = type_
        self.default = default
        self.min_value = min_value
        self.max_value = max_value
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "type": self.type_.__name__,
            "default": self.default,
            "min": self.min_value,
            "max": self.max_value,
            "description": self.description,
        }


class VisualizationSpec:
    """Specification for a model visualization."""

    def __init__(
        self,
        id: str,
        title: str,
        type: str,
        description: str = "",
        data_mapping: Dict[str, Any] | None = None,
        axes: Dict[str, Any] | None = None,
        annotations: List[Dict[str, Any]] | None = None,
        styling: Dict[str, Any] | None = None,
    ):
        """Initialize visualization specification.

        Args:
            id: Unique identifier within model.
            title: Display title.
            type: Chart type (e.g., 'multi_line', 'stacked_bar').
            description: Optional description.
            data_mapping: How to map simulation data to chart.
            axes: Axis configuration.
            annotations: Optional annotations.
            styling: Optional styling overrides.
        """
        self.id = id
        self.title = title
        self.type = type
        self.description = description
        self.data_mapping = data_mapping or {}
        self.axes = axes or {}
        self.annotations = annotations or []
        self.styling = styling or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "description": self.description,
            "data_mapping": self.data_mapping,
            "axes": self.axes,
            "annotations": self.annotations,
            "styling": self.styling,
        }


class SimulationResult:
    """Result of a model simulation."""

    def __init__(
        self,
        time_series: Dict[str, List[float]],
        metrics: Dict[str, float],
        metadata: Dict[str, Any] | None = None,
    ):
        """Initialize simulation result.

        Args:
            time_series: Time series data for visualization.
            metrics: Computed metrics from simulation.
            metadata: Additional metadata.
        """
        self.time_series = time_series
        self.metrics = metrics
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "time_series": self.time_series,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


class EducationalContent:
    """Educational content for a model."""

    def __init__(
        self,
        learning_objectives: List[str],
        background_theory: str,
        equations: List[Dict[str, str]],
        interpretation_guide: str,
    ):
        """Initialize educational content.

        Args:
            learning_objectives: List of learning objectives.
            background_theory: Background theory text.
            equations: List of equations with descriptions.
            interpretation_guide: Guide for interpreting results.
        """
        self.learning_objectives = learning_objectives
        self.background_theory = background_theory
        self.equations = equations
        self.interpretation_guide = interpretation_guide

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "learning_objectives": self.learning_objectives,
            "background_theory": self.background_theory,
            "equations": self.equations,
            "interpretation_guide": self.interpretation_guide,
        }


class ModelModule(ABC):
    """Base interface for all model modules."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier (e.g., 'kyle', 'almgren_chriss')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description for educational context."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, ParameterSpec]:
        """Schema defining parameters, types, ranges, defaults."""
        pass

    @property
    @abstractmethod
    def visualizations(self) -> List[VisualizationSpec]:
        """List of model-specific chart definitions."""
        pass

    @abstractmethod
    def simulate(self, params: Dict[str, Any]) -> SimulationResult:
        """Run simulation with given parameters.

        Args:
            params: Dictionary of parameter name to value.

        Returns:
            Simulation result with time series and metrics.
        """
        pass

    @abstractmethod
    def get_educational_content(self) -> EducationalContent:
        """Return theory, equations, interpretation guide.

        Returns:
            Educational content for the model.
        """
        pass
