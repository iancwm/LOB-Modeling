"""Tests for the Asset Option wrapper module."""

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
from lob_modeling.webapp.modules.wrappers.asset_option_wrapper import (
    AssetOptionModule,
)


class TestAssetOptionModule(unittest.TestCase):
    """Test cases for the Asset Option module wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = AssetOptionModule()

    def test_module_init(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.module)
        self.assertIsInstance(self.module, AssetOptionModule)

    def test_model_id(self):
        """Test that model_id returns correct identifier."""
        self.assertEqual(self.module.model_id, "asset_option")

    def test_display_name(self):
        """Test that display_name returns human-readable name."""
        self.assertEqual(self.module.display_name, "Asset or Nothing Option")

    def test_description(self):
        """Test that description is non-empty and descriptive."""
        description = self.module.description
        self.assertIsNotNone(description)
        self.assertTrue(len(description) > 0)
        self.assertIn("option", description.lower())

    def test_parameters_schema(self):
        """Test that parameters schema is correctly defined."""
        params = self.module.parameters
        self.assertIsInstance(params, dict)
        
        # Check required parameters exist
        expected_params = ["s", "n", "k", "T", "K", "F", "SIGMA", "r"]
        for param_name in expected_params:
            self.assertIn(param_name, params)
            self.assertIsInstance(params[param_name], ParameterSpec)

    def test_parameter_defaults(self):
        """Test that parameter defaults are within expected ranges."""
        params = self.module.parameters
        
        # Test s (spot price)
        self.assertEqual(params["s"].default, 100.0)
        self.assertEqual(params["s"].type_, float)
        
        # Test n (tree steps)
        self.assertEqual(params["n"].default, 50)
        self.assertEqual(params["n"].type_, int)
        
        # Test K (strike)
        self.assertEqual(params["K"].default, 100.0)
        
        # Test SIGMA (volatility)
        self.assertEqual(params["SIGMA"].default, 0.3)
        
        # Test r (risk-free rate)
        self.assertEqual(params["r"].default, 0.05)

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
        
        self.assertIn("option_tree", viz_ids)
        self.assertIn("asset_paths", viz_ids)

    def test_simulate_with_default_params(self):
        """Test simulation runs with default parameters."""
        params = {
            "s": 100.0,
            "n": 50,
            "k": 252,
            "T": 1.0,
            "K": 100.0,
            "F": 1.0,
            "SIGMA": 0.3,
            "r": 0.05,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertIsNotNone(result.time_series)
        self.assertIsNotNone(result.metrics)

    def test_simulate_returns_correct_structure(self):
        """Test that simulation returns expected data structure."""
        params = {
            "s": 100.0,
            "n": 30,
            "k": 252,
            "T": 1.0,
            "K": 100.0,
            "F": 1.0,
            "SIGMA": 0.3,
            "r": 0.05,
        }
        
        result = self.module.simulate(params)
        
        # Check time_series structure
        self.assertIn("time", result.time_series)
        self.assertIn("option_value", result.time_series)
        self.assertIn("asset_price", result.time_series)
        
        # Check metrics structure
        self.assertIn("option_price", result.metrics)
        self.assertIn("intrinsic_value", result.metrics)
        self.assertIn("time_value", result.metrics)
        self.assertIn("moneyness", result.metrics)
        self.assertIn("strike", result.metrics)
        self.assertIn("spot", result.metrics)

    def test_option_price_non_negative(self):
        """Test that option price is non-negative."""
        params = {
            "s": 100.0,
            "n": 50,
            "k": 252,
            "T": 1.0,
            "K": 100.0,
            "F": 1.0,
            "SIGMA": 0.3,
            "r": 0.05,
        }
        
        result = self.module.simulate(params)
        
        self.assertGreaterEqual(result.metrics["option_price"], 0)

    def test_moneyness_calculation(self):
        """Test that moneyness is correctly calculated as s/K."""
        params = {
            "s": 120.0,
            "n": 30,
            "k": 252,
            "T": 1.0,
            "K": 100.0,
            "F": 1.0,
            "SIGMA": 0.3,
            "r": 0.05,
        }
        
        result = self.module.simulate(params)
        
        # Moneyness = spot / strike
        expected_moneyness = params["s"] / params["K"]
        self.assertAlmostEqual(result.metrics["moneyness"], expected_moneyness, places=10)

    def test_intrinsic_value_calculation(self):
        """Test that intrinsic value is correctly calculated."""
        params = {
            "s": 110.0,
            "n": 30,
            "k": 252,
            "T": 1.0,
            "K": 100.0,
            "F": 1.0,
            "SIGMA": 0.3,
            "r": 0.05,
        }
        
        result = self.module.simulate(params)
        
        # Intrinsic value = max(s - K, 0) for call-like payoff
        expected_intrinsic = max(params["s"] - params["K"], 0)
        self.assertAlmostEqual(
            result.metrics["intrinsic_value"], expected_intrinsic, places=10
        )

    def test_out_of_the_money_intrinsic_value(self):
        """Test intrinsic value when option is out of the money."""
        params = {
            "s": 90.0,
            "n": 30,
            "k": 252,
            "T": 1.0,
            "K": 100.0,
            "F": 1.0,
            "SIGMA": 0.3,
            "r": 0.05,
        }
        
        result = self.module.simulate(params)
        
        # Intrinsic value should be 0 when s < K
        self.assertEqual(result.metrics["intrinsic_value"], 0)

    def test_simulate_with_custom_params(self):
        """Test simulation with custom parameters."""
        params = {
            "s": 150.0,
            "n": 40,
            "k": 252,
            "T": 2.0,
            "K": 140.0,
            "F": 1.5,
            "SIGMA": 0.4,
            "r": 0.03,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(len(result.time_series["time"]), 41)  # n + 1

    def test_simulate_metadata(self):
        """Test that simulation result includes correct metadata."""
        params = {"s": 100.0, "n": 25}
        
        result = self.module.simulate(params)
        
        self.assertIn("model_id", result.metadata)
        self.assertEqual(result.metadata["model_id"], "asset_option")
        self.assertIn("parameters", result.metadata)
        self.assertEqual(result.metadata["method"], "binomial_tree")
        self.assertEqual(result.metadata["option_type"], "asset_or_nothing_call")

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
        self.assertTrue("option" in all_objectives or "binomial" in all_objectives)

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

    def test_spot_strike_metadata(self):
        """Test that spot and strike are correctly stored in metrics."""
        params = {
            "s": 125.0,
            "n": 30,
            "k": 252,
            "T": 1.0,
            "K": 120.0,
            "F": 1.0,
            "SIGMA": 0.3,
            "r": 0.05,
        }
        
        result = self.module.simulate(params)
        
        self.assertAlmostEqual(result.metrics["spot"], 125.0, places=10)
        self.assertAlmostEqual(result.metrics["strike"], 120.0, places=10)

    def test_option_value_monotonicity(self):
        """Test that option values are non-decreasing toward expiry (simplified)."""
        params = {
            "s": 100.0,
            "n": 20,
            "k": 252,
            "T": 1.0,
            "K": 100.0,
            "F": 1.0,
            "SIGMA": 0.3,
            "r": 0.05,
        }
        
        result = self.module.simulate(params)
        
        option_values = result.time_series["option_value"]
        
        # Option values should generally increase toward expiry
        # (due to time value decay, this is a simplified check)
        self.assertGreater(option_values[-1], option_values[0])


if __name__ == "__main__":
    unittest.main()
