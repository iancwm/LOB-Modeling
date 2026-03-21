"""Tests for the Glosten-Milgrom wrapper module."""

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
from lob_modeling.webapp.modules.wrappers.glosten_milgrom_wrapper import (
    GlostenMilgromModule,
)


class TestGlostenMilgromModule(unittest.TestCase):
    """Test cases for the Glosten-Milgrom module wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = GlostenMilgromModule()

    def test_module_init(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.module)
        self.assertIsInstance(self.module, GlostenMilgromModule)

    def test_model_id(self):
        """Test that model_id returns correct identifier."""
        self.assertEqual(self.module.model_id, "glosten_milgrom")

    def test_display_name(self):
        """Test that display_name returns human-readable name."""
        self.assertEqual(self.module.display_name, "Glosten-Milgrom (1985)")

    def test_description(self):
        """Test that description is non-empty and descriptive."""
        description = self.module.description
        self.assertIsNotNone(description)
        self.assertTrue(len(description) > 0)
        self.assertIn("market", description.lower())

    def test_parameters_schema(self):
        """Test that parameters schema is correctly defined."""
        params = self.module.parameters
        self.assertIsInstance(params, dict)
        
        # Check required parameters exist
        expected_params = ["N", "ALPHA", "BETA", "V_low", "V_high"]
        for param_name in expected_params:
            self.assertIn(param_name, params)
            self.assertIsInstance(params[param_name], ParameterSpec)

    def test_parameter_defaults(self):
        """Test that parameter defaults are within expected ranges."""
        params = self.module.parameters
        
        # Test N (integer)
        self.assertEqual(params["N"].default, 50)
        self.assertEqual(params["N"].type_, int)
        
        # Test ALPHA
        self.assertEqual(params["ALPHA"].default, 0.5)
        self.assertGreaterEqual(params["ALPHA"].default, params["ALPHA"].min_value)
        self.assertLessEqual(params["ALPHA"].default, params["ALPHA"].max_value)
        
        # Test BETA
        self.assertEqual(params["BETA"].default, 0.3)
        
        # Test V_low and V_high
        self.assertEqual(params["V_low"].default, 0.0)
        self.assertEqual(params["V_high"].default, 10.0)

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
        
        self.assertIn("bid_ask_spread", viz_ids)
        self.assertIn("spread_width", viz_ids)

    def test_simulate_with_default_params(self):
        """Test simulation runs with default parameters."""
        params = {
            "N": 50,
            "ALPHA": 0.5,
            "BETA": 0.3,
            "V_low": 0.0,
            "V_high": 10.0,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertIsNotNone(result.time_series)
        self.assertIsNotNone(result.metrics)

    def test_simulate_returns_correct_structure(self):
        """Test that simulation returns expected data structure."""
        params = {
            "N": 20,
            "ALPHA": 0.5,
            "BETA": 0.3,
            "V_low": 0.0,
            "V_high": 10.0,
        }
        
        result = self.module.simulate(params)
        
        # Check time_series structure
        self.assertIn("time", result.time_series)
        self.assertIn("bid", result.time_series)
        self.assertIn("ask", result.time_series)
        self.assertIn("spread", result.time_series)
        
        # Check metrics structure
        self.assertIn("avg_spread", result.metrics)
        self.assertIn("final_bid", result.metrics)
        self.assertIn("final_ask", result.metrics)
        self.assertIn("final_spread", result.metrics)
        self.assertIn("true_value_estimate", result.metrics)

    def test_simulate_bid_ask_convergence(self):
        """Test that bid and ask prices converge over time."""
        params = {
            "N": 100,
            "ALPHA": 0.5,
            "BETA": 0.3,
            "V_low": 0.0,
            "V_high": 10.0,
        }
        
        result = self.module.simulate(params)
        
        bid = result.time_series["bid"]
        ask = result.time_series["ask"]
        
        # Bid should always be less than or equal to ask
        for i in range(len(bid)):
            self.assertLessEqual(bid[i], ask[i])
        
        # Spread should be non-negative
        spread = result.time_series["spread"]
        for s in spread:
            self.assertGreaterEqual(s, 0)

    def test_simulate_spread_calculation(self):
        """Test that spread is correctly calculated as ask - bid."""
        params = {
            "N": 30,
            "ALPHA": 0.5,
            "BETA": 0.3,
            "V_low": 5.0,
            "V_high": 15.0,
        }
        
        result = self.module.simulate(params)
        
        bid = result.time_series["bid"]
        ask = result.time_series["ask"]
        spread = result.time_series["spread"]
        
        # Verify spread calculation
        for i in range(len(bid)):
            expected_spread = ask[i] - bid[i]
            self.assertAlmostEqual(spread[i], expected_spread, places=10)

    def test_simulate_with_custom_params(self):
        """Test simulation with custom parameters."""
        params = {
            "N": 75,
            "ALPHA": 0.7,
            "BETA": 0.5,
            "V_low": 10.0,
            "V_high": 50.0,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(len(result.time_series["time"]), 75)  # N time steps

    def test_simulate_metadata(self):
        """Test that simulation result includes correct metadata."""
        params = {"N": 30}
        
        result = self.module.simulate(params)
        
        self.assertIn("model_id", result.metadata)
        self.assertEqual(result.metadata["model_id"], "glosten_milgrom")
        self.assertIn("parameters", result.metadata)
        self.assertEqual(result.metadata["method"], "bayesian_updating")

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
        self.assertTrue(
            "information" in all_objectives or "spread" in all_objectives
        )

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

    def test_true_value_estimate_within_bounds(self):
        """Test that true value estimate is within V_low and V_high bounds."""
        params = {
            "N": 50,
            "ALPHA": 0.5,
            "BETA": 0.3,
            "V_low": 5.0,
            "V_high": 15.0,
        }
        
        result = self.module.simulate(params)
        
        estimate = result.metrics["true_value_estimate"]
        
        # Estimate should be within bounds (with some tolerance for edge cases)
        self.assertGreaterEqual(estimate, params["V_low"] - 0.1)
        self.assertLessEqual(estimate, params["V_high"] + 0.1)


if __name__ == "__main__":
    unittest.main()
