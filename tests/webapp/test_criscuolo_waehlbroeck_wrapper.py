"""Tests for the Criscuolo-Waehlbroeck wrapper module."""

import unittest
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from lob_modeling.webapp.modules.base import (
    EducationalContent,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)
from lob_modeling.webapp.modules.wrappers.criscuolo_waehlbroeck_wrapper import (
    CriscuoloWaehlbroeckModule,
)


class TestCriscuoloWaehlbroeckModule(unittest.TestCase):
    """Test cases for the Criscuolo-Waehlbroeck module wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = CriscuoloWaehlbroeckModule()

    def test_module_init(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.module)
        self.assertIsInstance(self.module, CriscuoloWaehlbroeckModule)

    def test_model_id(self):
        """Test that model_id returns correct identifier."""
        self.assertEqual(self.module.model_id, "criscuolo_waehlbroeck")

    def test_display_name(self):
        """Test that display_name returns human-readable name."""
        self.assertEqual(self.module.display_name, "Criscuolo & Waehlbroeck (2014)")

    def test_description(self):
        """Test that description is non-empty and descriptive."""
        description = self.module.description
        self.assertIsNotNone(description)
        self.assertTrue(len(description) > 0)
        self.assertIn("volatility", description.lower())

    def test_parameters_schema(self):
        """Test that parameters schema is correctly defined."""
        params = self.module.parameters
        self.assertIsInstance(params, dict)
        
        # Check required parameters exist
        expected_params = ["KAPPA", "THETA", "GAMMA", "V_0", "T", "N", "S_0"]
        for param_name in expected_params:
            self.assertIn(param_name, params)
            self.assertIsInstance(params[param_name], ParameterSpec)

    def test_parameter_defaults(self):
        """Test that parameter defaults are within expected ranges."""
        params = self.module.parameters
        
        # Test KAPPA
        self.assertEqual(params["KAPPA"].default, 3.0)
        self.assertEqual(params["KAPPA"].type_, float)
        
        # Test THETA
        self.assertEqual(params["THETA"].default, 0.01)
        
        # Test N (integer)
        self.assertEqual(params["N"].default, 10)
        self.assertEqual(params["N"].type_, int)
        
        # Test V_0
        self.assertEqual(params["V_0"].default, 0.5)

    def test_visualizations_schema(self):
        """Test that visualizations are correctly defined."""
        viz = self.module.visualizations
        self.assertIsInstance(viz, list)
        self.assertGreater(len(viz), 0)
        
        for vis_spec in viz:
            self.assertIsInstance(vis_spec, VisualizationSpec)
            self.assertIsNotNone(vis_spec.id)
            self.assertIsNotNone(vis_spec.title)
            self.assertIsNotNone(vis_spec.type)

    def test_visualization_ids(self):
        """Test that expected visualization IDs are present."""
        viz = self.module.visualizations
        viz_ids = [v.id for v in viz]
        
        self.assertIn("volatility_path", viz_ids)
        self.assertIn("execution_schedule", viz_ids)

    def test_simulate_with_default_params(self):
        """Test simulation runs with default parameters."""
        params = {
            "KAPPA": 3.0,
            "THETA": 0.01,
            "GAMMA": 0.1,
            "V_0": 0.5,
            "T": 0.5,
            "N": 10,
            "S_0": 100.0,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertIsNotNone(result.time_series)
        self.assertIsNotNone(result.metrics)

    def test_simulate_returns_correct_structure(self):
        """Test that simulation returns expected data structure."""
        params = {
            "KAPPA": 3.0,
            "THETA": 0.01,
            "GAMMA": 0.1,
            "V_0": 0.5,
            "T": 0.5,
            "N": 10,
            "S_0": 100.0,
        }
        
        result = self.module.simulate(params)
        
        # Check time_series structure
        self.assertIn("time", result.time_series)
        self.assertIn("volatility", result.time_series)
        self.assertIn("trade_rate", result.time_series)
        self.assertIn("share_turnover", result.time_series)
        
        # Check metrics structure
        self.assertIn("avg_volatility", result.metrics)
        self.assertIn("total_cost", result.metrics)
        self.assertIn("optimal_participation", result.metrics)
        self.assertIn("execution_time_years", result.metrics)

    def test_volatility_non_negative(self):
        """Test that volatility values are non-negative."""
        params = {
            "KAPPA": 3.0,
            "THETA": 0.01,
            "GAMMA": 0.1,
            "V_0": 0.5,
            "T": 0.5,
            "N": 20,
            "S_0": 100.0,
        }
        
        result = self.module.simulate(params)
        
        volatility = result.time_series["volatility"]
        
        # All volatility values should be non-negative
        for vol in volatility:
            self.assertGreaterEqual(vol, 0)

    def test_trade_rate_range(self):
        """Test that trade rates are in valid range [0, 1]."""
        params = {
            "KAPPA": 3.0,
            "THETA": 0.01,
            "GAMMA": 0.1,
            "V_0": 0.5,
            "T": 0.5,
            "N": 15,
            "S_0": 100.0,
        }
        
        result = self.module.simulate(params)
        
        trade_rates = result.time_series["trade_rate"]
        
        # All trade rates should be between 0 and 1
        for rate in trade_rates:
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)

    def test_simulate_with_custom_params(self):
        """Test simulation with custom parameters."""
        params = {
            "KAPPA": 5.0,
            "THETA": 0.02,
            "GAMMA": 0.15,
            "V_0": 0.8,
            "T": 1.0,
            "N": 20,
            "S_0": 150.0,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(len(result.time_series["time"]), 20)

    def test_simulate_metadata(self):
        """Test that simulation result includes correct metadata."""
        params = {"N": 15}
        
        result = self.module.simulate(params)
        
        self.assertIn("model_id", result.metadata)
        self.assertEqual(result.metadata["model_id"], "criscuolo_waehlbroeck")
        self.assertIn("parameters", result.metadata)
        self.assertEqual(
            result.metadata["method"], "stochastic_volatility_optimization"
        )

    def test_get_educational_content(self):
        """Test that educational content is correctly provided."""
        content = self.module.get_educational_content()
        
        self.assertIsInstance(content, EducationalContent)
        self.assertIsNotNone(content.learning_objectives)
        self.assertIsNotNone(content.background_theory)
        self.assertIsNotNone(content.equations)
        self.assertIsNotNone(content.interpretation_guide)

    def test_educational_content_learning_objectives(self):
        """Test that learning objectives are meaningful."""
        content = self.module.get_educational_content()
        
        objectives = content.learning_objectives
        self.assertGreater(len(objectives), 0)
        
        # Check objectives are non-empty strings
        for obj in objectives:
            self.assertIsInstance(obj, str)
            self.assertTrue(len(obj) > 0)
        
        # Check objectives mention key concepts
        all_objectives = " ".join(objectives).lower()
        self.assertTrue("volatility" in all_objectives or "execution" in all_objectives)

    def test_educational_content_equations(self):
        """Test that equations are properly structured."""
        content = self.module.get_educational_content()
        
        equations = content.equations
        self.assertGreater(len(equations), 0)
        
        # Check equation structure
        for eq in equations:
            self.assertIsInstance(eq, dict)
            self.assertIn("label", eq)
            self.assertIn("equation", eq)
            self.assertIn("description", eq)

    def test_time_series_length(self):
        """Test that time series length matches N parameter."""
        params = {
            "KAPPA": 3.0,
            "THETA": 0.01,
            "GAMMA": 0.1,
            "V_0": 0.5,
            "T": 0.5,
            "N": 12,
            "S_0": 100.0,
        }
        
        result = self.module.simulate(params)
        
        # Time series length should match N
        self.assertEqual(len(result.time_series["time"]), 12)
        self.assertEqual(len(result.time_series["volatility"]), 12)
        self.assertEqual(len(result.time_series["trade_rate"]), 12)

    def test_execution_time_metric(self):
        """Test that execution_time_years matches T parameter."""
        params = {
            "KAPPA": 3.0,
            "THETA": 0.01,
            "GAMMA": 0.1,
            "V_0": 0.5,
            "T": 0.75,
            "N": 10,
            "S_0": 100.0,
        }
        
        result = self.module.simulate(params)
        
        self.assertAlmostEqual(result.metrics["execution_time_years"], 0.75, places=5)


if __name__ == "__main__":
    unittest.main()
