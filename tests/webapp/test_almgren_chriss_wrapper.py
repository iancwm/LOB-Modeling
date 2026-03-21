"""Tests for the Almgren-Chriss wrapper module."""

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
from lob_modeling.webapp.modules.wrappers.almgren_chriss_wrapper import (
    AlmgrenChrissModule,
)


class TestAlmgrenChrissModule(unittest.TestCase):
    """Test cases for the Almgren-Chriss module wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = AlmgrenChrissModule()

    def test_module_init(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.module)
        self.assertIsInstance(self.module, AlmgrenChrissModule)

    def test_model_id(self):
        """Test that model_id returns correct identifier."""
        self.assertEqual(self.module.model_id, "almgren_chriss")

    def test_display_name(self):
        """Test that display_name returns human-readable name."""
        self.assertEqual(self.module.display_name, "Almgren-Chriss (2000)")

    def test_description(self):
        """Test that description is non-empty and descriptive."""
        description = self.module.description
        self.assertIsNotNone(description)
        self.assertTrue(len(description) > 0)
        self.assertIn("execution", description.lower())

    def test_parameters_schema(self):
        """Test that parameters schema is correctly defined."""
        params = self.module.parameters
        self.assertIsInstance(params, dict)
        
        # Check required parameters exist
        expected_params = ["ALPHA", "ETA", "GAMMA", "LAMBDA", "SIGMA", "N", "T", "X"]
        for param_name in expected_params:
            self.assertIn(param_name, params)
            self.assertIsInstance(params[param_name], ParameterSpec)

    def test_parameter_defaults(self):
        """Test that parameter defaults are within expected ranges."""
        params = self.module.parameters
        
        # Test ALPHA
        self.assertEqual(params["ALPHA"].default, 1.0)
        self.assertEqual(params["ALPHA"].type_, float)
        
        # Test ETA
        self.assertEqual(params["ETA"].default, 5e-6)
        
        # Test N (integer)
        self.assertEqual(params["N"].default, 50)
        self.assertEqual(params["N"].type_, int)

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
        
        self.assertIn("inventory_decay", viz_ids)
        self.assertIn("trade_schedule", viz_ids)

    def test_simulate_with_default_params(self):
        """Test simulation runs with default parameters."""
        params = {
            "ALPHA": 1.0,
            "ETA": 5e-6,
            "GAMMA": 5e-5,
            "LAMBDA": 0.00009,
            "SIGMA": 0.495,
            "N": 50,
            "T": 0.025,
            "X": 500,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertIsNotNone(result.time_series)
        self.assertIsNotNone(result.metrics)

    def test_simulate_returns_correct_structure(self):
        """Test that simulation returns expected data structure."""
        params = {
            "ALPHA": 1.0,
            "ETA": 5e-6,
            "GAMMA": 5e-5,
            "LAMBDA": 0.00009,
            "SIGMA": 0.495,
            "N": 10,
            "T": 0.025,
            "X": 100,
        }
        
        result = self.module.simulate(params)
        
        # Check time_series structure
        self.assertIn("time", result.time_series)
        self.assertIn("inventory", result.time_series)
        self.assertIn("trades", result.time_series)
        
        # Check metrics structure
        self.assertIn("expected_shortfall", result.metrics)
        self.assertIn("variance_shortfall", result.metrics)
        self.assertIn("total_traded", result.metrics)
        self.assertIn("avg_trade_size", result.metrics)

    def test_simulate_inventory_decay(self):
        """Test that inventory decays from initial value to zero."""
        initial_shares = 500
        params = {
            "ALPHA": 1.0,
            "ETA": 5e-6,
            "GAMMA": 5e-5,
            "LAMBDA": 0.00009,
            "SIGMA": 0.495,
            "N": 50,
            "T": 0.025,
            "X": initial_shares,
        }
        
        result = self.module.simulate(params)
        
        inventory = result.time_series["inventory"]
        
        # Initial inventory should match X
        self.assertEqual(inventory[0], initial_shares)
        
        # Final inventory should be 0 or close to 0
        self.assertLessEqual(inventory[-1], 1)  # Allow for rounding

    def test_simulate_total_traded(self):
        """Test that total traded equals initial shares."""
        initial_shares = 500
        params = {
            "ALPHA": 1.0,
            "ETA": 5e-6,
            "GAMMA": 5e-5,
            "LAMBDA": 0.00009,
            "SIGMA": 0.495,
            "N": 50,
            "T": 0.025,
            "X": initial_shares,
        }
        
        result = self.module.simulate(params)
        
        # Total traded should equal initial shares
        self.assertAlmostEqual(
            result.metrics["total_traded"], initial_shares, places=0
        )

    def test_simulate_with_custom_params(self):
        """Test simulation with custom parameters."""
        params = {
            "ALPHA": 1.5,
            "ETA": 1e-5,
            "GAMMA": 1e-4,
            "LAMBDA": 0.0001,
            "SIGMA": 0.6,
            "N": 25,
            "T": 0.05,
            "X": 1000,
        }
        
        result = self.module.simulate(params)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(len(result.time_series["time"]), 26)  # N + 1

    def test_simulate_metadata(self):
        """Test that simulation result includes correct metadata."""
        params = {"N": 10}
        
        result = self.module.simulate(params)
        
        self.assertIn("model_id", result.metadata)
        self.assertEqual(result.metadata["model_id"], "almgren_chriss")
        self.assertIn("parameters", result.metadata)

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


if __name__ == "__main__":
    unittest.main()
