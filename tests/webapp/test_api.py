"""Tests for REST API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.lob_modeling.webapp.main import create_app
from src.lob_modeling.webapp.modules.base import (
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)
from src.lob_modeling.webapp.modules.registry import registry


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_module():
    """Create a mock model module."""

    class MockModule(ModelModule):
        @property
        def model_id(self) -> str:
            return "mock"

        @property
        def display_name(self) -> str:
            return "Mock Model"

        @property
        def description(self) -> str:
            return "A mock model for testing"

        @property
        def parameters(self):
            return {
                "param1": ParameterSpec(
                    name="param1",
                    type_=float,
                    default=1.0,
                    min_value=0.0,
                    max_value=10.0,
                    description="Test parameter",
                )
            }

        @property
        def visualizations(self):
            return [
                VisualizationSpec(
                    id="test_viz",
                    title="Test",
                    type="line",
                )
            ]

        def simulate(self, params):
            return SimulationResult(
                time_series={"time": [0, 1, 2], "value": [1.0, 2.0, 3.0]},
                metrics={"final": 3.0},
            )

        def get_educational_content(self):
            return EducationalContent(
                learning_objectives=["Learn"],
                background_theory="Theory",
                equations=[],
                interpretation_guide="Guide",
            )

    return MockModule


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestListModels:
    """Tests for list models endpoint."""

    def test_list_models_empty(self, client):
        """Test listing models when registry is empty."""
        # Note: The registry always has at least the Kyle module registered
        # This test verifies the endpoint works even with registered modules
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_list_models_with_registered_module(self, client, mock_module):
        """Test listing models with additional registered module."""
        # Get initial count
        initial_response = client.get("/models")
        initial_count = len(initial_response.json()["models"])

        # Register mock module
        registry.register("mock", mock_module)
        try:
            response = client.get("/models")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert len(data["models"]) == initial_count + 1
            # Find our mock model
            mock_model = next((m for m in data["models"] if m["id"] == "mock"), None)
            assert mock_model is not None
            assert mock_model["displayName"] == "Mock Model"
            assert mock_model["description"] == "A mock model for testing"
        finally:
            # Clean up
            registry._modules.pop("mock", None)


class TestGetModel:
    """Tests for get model endpoint."""

    def test_get_model_not_found(self, client):
        """Test getting a nonexistent model."""
        response = client.get("/models/nonexistent")
        assert response.status_code == 404

    def test_get_model_success(self, client, mock_module):
        """Test getting a model successfully."""
        registry.register("mock", mock_module)
        try:
            response = client.get("/models/mock")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "mock"
            assert data["displayName"] == "Mock Model"
            assert "parameters" in data
            assert "param1" in data["parameters"]
            assert "visualizations" in data
            assert "educationalContent" in data
        finally:
            registry._modules = {}


class TestSimulate:
    """Tests for simulate endpoint."""

    def test_simulate_model_not_found(self, client):
        """Test simulating a nonexistent model."""
        response = client.post("/models/nonexistent/simulate", json={})
        assert response.status_code == 404

    def test_simulate_success(self, client, mock_module):
        """Test successful simulation."""
        registry.register("mock", mock_module)
        try:
            params = {"param1": 5.0}
            response = client.post("/models/mock/simulate", json=params)
            assert response.status_code == 200
            data = response.json()
            assert "simulationId" in data
            assert "results" in data
            assert "time_series" in data["results"]
            assert "metrics" in data["results"]
        finally:
            registry._modules = {}

    def test_simulate_with_default_params(self, client, mock_module):
        """Test simulation with default parameters."""
        registry.register("mock", mock_module)
        try:
            response = client.post("/models/mock/simulate", json={})
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
        finally:
            registry._modules = {}


class TestCreateStream:
    """Tests for create stream endpoint."""

    def test_create_stream_model_not_found(self, client):
        """Test creating stream for nonexistent model."""
        response = client.post("/models/nonexistent/stream", json={})
        assert response.status_code == 404

    def test_create_stream_success(self, client, mock_module):
        """Test successful stream creation."""
        registry.register("mock", mock_module)
        try:
            params = {"param1": 5.0}
            response = client.post("/models/mock/stream", json=params)
            assert response.status_code == 200
            data = response.json()
            assert "sessionId" in data
            assert "websocketUrl" in data
            assert data["websocketUrl"].startswith("/ws/")
        finally:
            registry._modules = {}


class TestDependencies:
    """Tests for dependency injection."""

    def test_get_dependencies(self, client):
        """Test getting dependency status."""
        response = client.get("/api/dependencies")
        assert response.status_code == 200
        data = response.json()
        assert "session_store" in data
        assert "ws_manager" in data
        # Dependencies are initialized by lifespan
        assert data["session_store"] is True
        assert data["ws_manager"] is True
