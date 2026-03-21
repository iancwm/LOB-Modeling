"""Tests for the De Prado wrapper module."""

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
from lob_modeling.webapp.modules.wrappers.de_prado_wrapper import (
    DePradoModule,
)


class TestDePradoModule(unittest.TestCase):
    """Test cases for the De Prado VPIN module wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = DePradoModule()

    def test_module_init(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.module)
        self.assertIsInstance(self.module, DePradoModule)

    def test_model_id(self):
        """Test that model_id returns correct identifier."""
        self.assertEqual(self.module.model_id, "de_prado")

    def test_display_name(self):
        """Test that display_name returns human-readable name."""
        self.assertEqual(self.module.display_name, "De Prado et al. (2012)")

    def test_description(self):
        """Test that description is non-empty and descriptive."""
        description = self.module.description
        self.assertIsNotNone(description)
        self.assertTrue(len(description) > 0)
        self.assertIn("vpin", description.lower())

    def test_parameters_schema(self):
        """Test that parameters schema is correctly defined."""
        params = self.module.parameters
        self.assertIsInstance(params, dict)
        
        # Check required parameters exist
        expected_params = ["n_buckets", "mu", "epsilon", "alpha", "delta", "n_trades"]
        for param_name in expected_params:
            self.assertIn(param_name, params)
            self.assertIsInstance(params[param_name], ParameterSpec)

    def test_parameter_defaults(self):
        """Test that parameter defaults are within expected ranges."""
        params = self.module.parameters
        
        # Test n_buckets (integer)
        self.assertEqual(params["n_buckets"].default, 50)
        self.assertEqual(params["n_buckets"].type_, int)
        
        # Test mu
        self.assertEqual(params["mu"].default, 0.7)
        self.assertGreaterEqual(params["mu"].default, params["mu"].min_value)
        self.assertLessEqual(params["mu"].default, params["mu"].max_value)
        
        # Test epsilon
        self.assertEqual(params["epsilon"].default, 0.3)
        
        # Test alpha
        self.assertEqual(params["alpha"].default, 0.5)

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
        
        self.assertIn("vpin_evolution", viz_ids)
        self.assertIn("order_imbalance", viz_ids)

    def test_simulate_with_default_params(self):
        """Test simulation runs with default parameters."""
        params = {
            "n_buckets": 50,
            "mu": 0.7,
            "epsilon": 0.3,
            "alpha": 0.5,
            "delta": 0.3,
            "n_trades": 1000,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertIsNotNone(result.time_series)
        self.assertIsNotNone(result.metrics)

    def test_simulate_returns_correct_structure(self):
        """Test that simulation returns expected data structure."""
        params = {
            "n_buckets": 20,
            "mu": 0.7,
            "epsilon": 0.3,
            "alpha": 0.5,
            "delta": 0.3,
            "n_trades": 500,
        }
        
        result = self.module.simulate(params)
        
        # Check time_series structure
        self.assertIn("bucket", result.time_series)
        self.assertIn("vpin", result.time_series)
        self.assertIn("buy_volume", result.time_series)
        self.assertIn("sell_volume", result.time_series)
        
        # Check metrics structure
        self.assertIn("avg_vpin", result.metrics)
        self.assertIn("max_vpin", result.metrics)
        self.assertIn("total_buy_volume", result.metrics)
        self.assertIn("total_sell_volume", result.metrics)
        self.assertIn("order_imbalance", result.metrics)
        self.assertIn("informed_trading_estimate", result.metrics)

    def test_vpin_range(self):
        """Test that VPIN values are in valid range [0, 1]."""
        params = {
            "n_buckets": 30,
            "mu": 0.7,
            "epsilon": 0.3,
            "alpha": 0.5,
            "delta": 0.3,
            "n_trades": 600,
        }
        
        result = self.module.simulate(params)
        
        vpin_values = result.time_series["vpin"]
        
        # All VPIN values should be between 0 and 1
        for vpin in vpin_values:
            self.assertGreaterEqual(vpin, 0.0)
            self.assertLessEqual(vpin, 1.0)

    def test_vpin_calculation(self):
        """Test that VPIN is correctly calculated."""
        params = {
            "n_buckets": 10,
            "mu": 0.7,
            "epsilon": 0.3,
            "alpha": 0.5,
            "delta": 0.3,
            "n_trades": 200,
        }
        
        result = self.module.simulate(params)
        
        buy_vol = result.time_series["buy_volume"]
        sell_vol = result.time_series["sell_volume"]
        vpin = result.time_series["vpin"]
        
        # Verify VPIN calculation: |buy - sell| / (buy + sell)
        for i in range(len(buy_vol)):
            total = buy_vol[i] + sell_vol[i]
            if total > 0:
                expected_vpin = abs(buy_vol[i] - sell_vol[i]) / total
                self.assertAlmostEqual(vpin[i], expected_vpin, places=10)
            else:
                self.assertEqual(vpin[i], 0)

    def test_simulate_with_custom_params(self):
        """Test simulation with custom parameters."""
        params = {
            "n_buckets": 75,
            "mu": 1.0,
            "epsilon": 0.5,
            "alpha": 0.7,
            "delta": 0.4,
            "n_trades": 2000,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(len(result.time_series["bucket"]), 75)

    def test_simulate_metadata(self):
        """Test that simulation result includes correct metadata."""
        params = {"n_buckets": 25}
        
        result = self.module.simulate(params)
        
        self.assertIn("model_id", result.metadata)
        self.assertEqual(result.metadata["model_id"], "de_prado")
        self.assertIn("parameters", result.metadata)
        self.assertEqual(result.metadata["method"], "vpin_simulation")

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
        self.assertTrue("vpin" in all_objectives or "order" in all_objectives)

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

    def test_metrics_consistency(self):
        """Test that metrics are consistent with time series data."""
        params = {
            "n_buckets": 40,
            "mu": 0.7,
            "epsilon": 0.3,
            "alpha": 0.5,
            "delta": 0.3,
            "n_trades": 800,
        }
        
        result = self.module.simulate(params)
        
        # avg_vpin should match mean of vpin values
        vpin_values = result.time_series["vpin"]
        expected_avg = sum(vpin_values) / len(vpin_values)
        self.assertAlmostEqual(result.metrics["avg_vpin"], expected_avg, places=10)
        
        # max_vpin should match max of vpin values
        self.assertAlmostEqual(
            result.metrics["max_vpin"], max(vpin_values), places=10
        )


if __name__ == "__main__":
    unittest.main()
