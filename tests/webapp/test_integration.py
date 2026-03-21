"""Integration tests for all model wrappers."""

import unittest
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from lob_modeling.webapp.modules import (
    AlmgrenChrissModule,
    AssetOptionModule,
    CriscuoloWaehlbroeckModule,
    DePradoModule,
    GlostenMilgromModule,
    KyleModelModule,
    get_module,
    list_modules,
)
from lob_modeling.webapp.modules.base import SimulationResult


class TestModelComparison(unittest.TestCase):
    """Integration tests for model comparison functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # List of all expected model modules
        self.expected_models = [
            "kyle",
            "almgren_chriss",
            "glosten_milgrom",
            "de_prado",
            "criscuolo_waehlbroeck",
            "asset_option",
        ]

    def test_all_models_registered(self):
        """Test that all expected models are registered."""
        registered = list_modules()
        
        for model_id in self.expected_models:
            self.assertIn(model_id, registered)

    def test_all_models_simulate_success(self):
        """Test that all models can run simulations successfully."""
        # Default parameters for each model
        model_params = {
            "kyle": {"V_0": 5.0, "SIGMA_G": 0.4, "SIGMA_T": 0.2, "N": 20},
            "almgren_chriss": {
                "ALPHA": 1.0,
                "ETA": 5e-6,
                "GAMMA": 5e-5,
                "LAMBDA": 0.00009,
                "SIGMA": 0.495,
                "N": 20,
                "T": 0.025,
                "X": 500,
            },
            "glosten_milgrom": {
                "N": 20,
                "ALPHA": 0.5,
                "BETA": 0.3,
                "V_low": 0.0,
                "V_high": 10.0,
            },
            "de_prado": {
                "n_buckets": 20,
                "mu": 0.7,
                "epsilon": 0.3,
                "alpha": 0.5,
                "delta": 0.3,
                "n_trades": 400,
            },
            "criscuolo_waehlbroeck": {
                "KAPPA": 3.0,
                "THETA": 0.01,
                "GAMMA": 0.1,
                "V_0": 0.5,
                "T": 0.5,
                "N": 10,
                "S_0": 100.0,
            },
            "asset_option": {
                "s": 100.0,
                "n": 20,
                "k": 252,
                "T": 1.0,
                "K": 100.0,
                "F": 1.0,
                "SIGMA": 0.3,
                "r": 0.05,
            },
        }
        
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                params = model_params[model_id]
                
                result = module.simulate(params)
                
                self.assertIsInstance(result, SimulationResult)
                self.assertIsNotNone(result.time_series)
                self.assertIsNotNone(result.metrics)

    def test_all_models_return_consistent_structure(self):
        """Test that all models return consistent result structure."""
        model_params = {
            "kyle": {"N": 10},
            "almgren_chriss": {"N": 10, "X": 100},
            "glosten_milgrom": {"N": 10},
            "de_prado": {"n_buckets": 10, "n_trades": 200},
            "criscuolo_waehlbroeck": {"N": 10},
            "asset_option": {"n": 10},
        }
        
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                params = model_params[model_id]
                
                result = module.simulate(params)
                
                # All results should have time_series, metrics, metadata
                self.assertIn("time_series", result.to_dict())
                self.assertIn("metrics", result.to_dict())
                self.assertIn("metadata", result.to_dict())
                
                # Metadata should contain model_id
                self.assertIn("model_id", result.metadata)
                self.assertEqual(result.metadata["model_id"], model_id)

    def test_all_models_have_educational_content(self):
        """Test that all models provide educational content."""
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                
                content = module.get_educational_content()
                
                self.assertIsNotNone(content)
                self.assertIsNotNone(content.learning_objectives)
                self.assertIsNotNone(content.background_theory)
                self.assertIsNotNone(content.equations)
                self.assertIsNotNone(content.interpretation_guide)

    def test_all_models_have_visualizations(self):
        """Test that all models provide visualizations."""
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                
                viz = module.visualizations
                
                self.assertIsInstance(viz, list)
                self.assertGreater(len(viz), 0)
                
                for vis_spec in viz:
                    self.assertIsNotNone(vis_spec.id)
                    self.assertIsNotNone(vis_spec.title)
                    self.assertIsNotNone(vis_spec.type)

    def test_all_models_have_parameters(self):
        """Test that all models define parameters."""
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                
                params = module.parameters
                
                self.assertIsInstance(params, dict)
                self.assertGreater(len(params), 0)
                
                # Each parameter should have required fields
                for param_name, param_spec in params.items():
                    self.assertIsNotNone(param_spec.name)
                    self.assertIsNotNone(param_spec.type_)
                    self.assertIsNotNone(param_spec.default)

    def test_model_metadata_consistency(self):
        """Test that model metadata is consistent across models."""
        model_params = {
            "kyle": {"N": 10},
            "almgren_chriss": {"N": 10},
            "glosten_milgrom": {"N": 10},
            "de_prado": {"n_buckets": 10},
            "criscuolo_waehlbroeck": {"N": 10},
            "asset_option": {"n": 10},
        }
        
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                params = model_params[model_id]
                
                result = module.simulate(params)
                
                # All models should include model_id in metadata
                self.assertIn("model_id", result.metadata)
                self.assertEqual(result.metadata["model_id"], model_id)


class TestParameterValidation(unittest.TestCase):
    """Integration tests for parameter validation across all models."""

    def setUp(self):
        """Set up test fixtures."""
        self.expected_models = [
            "kyle",
            "almgren_chriss",
            "glosten_milgrom",
            "de_prado",
            "criscuolo_waehlbroeck",
            "asset_option",
        ]

    def test_models_accept_default_parameters(self):
        """Test that all models accept their default parameters."""
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                
                # Get default parameters
                params = {
                    name: spec.default
                    for name, spec in module.parameters.items()
                }
                
                # Should run without error
                result = module.simulate(params)
                self.assertIsInstance(result, SimulationResult)

    def test_models_accept_min_parameters(self):
        """Test that models accept minimum parameter values."""
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                
                # Get min parameters (use min_value if available, else default)
                params = {}
                for name, spec in module.parameters.items():
                    if spec.min_value is not None:
                        params[name] = spec.min_value
                    else:
                        params[name] = spec.default
                
                # Should run without error
                result = module.simulate(params)
                self.assertIsInstance(result, SimulationResult)

    def test_models_accept_max_parameters(self):
        """Test that models accept maximum parameter values."""
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                
                # Get max parameters (use max_value if available, else default)
                params = {}
                for name, spec in module.parameters.items():
                    if spec.max_value is not None:
                        params[name] = spec.max_value
                    else:
                        params[name] = spec.default
                
                # Should run without error
                result = module.simulate(params)
                self.assertIsInstance(result, SimulationResult)

    def test_parameter_bounds_are_valid(self):
        """Test that parameter bounds are logically valid."""
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                
                for name, spec in module.parameters.items():
                    # If both min and max are defined, min should be <= max
                    if spec.min_value is not None and spec.max_value is not None:
                        self.assertLessEqual(
                            spec.min_value,
                            spec.max_value,
                            f"Parameter {name} in {model_id}: min > max",
                        )
                    
                    # Default should be within bounds if defined
                    if spec.min_value is not None:
                        self.assertGreaterEqual(
                            spec.default,
                            spec.min_value,
                            f"Parameter {name} in {model_id}: default < min",
                        )
                    if spec.max_value is not None:
                        self.assertLessEqual(
                            spec.default,
                            spec.max_value,
                            f"Parameter {name} in {model_id}: default > max",
                        )


class TestErrorHandling(unittest.TestCase):
    """Integration tests for error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.expected_models = [
            "kyle",
            "almgren_chriss",
            "glosten_milgrom",
            "de_prado",
            "criscuolo_waehlbroeck",
            "asset_option",
        ]

    def test_models_handle_empty_params_gracefully(self):
        """Test that models handle empty parameter dicts (use defaults)."""
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                
                # Empty params should use defaults
                result = module.simulate({})
                self.assertIsInstance(result, SimulationResult)

    def test_models_handle_partial_params(self):
        """Test that models handle partial parameter dicts."""
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                
                # Provide only one parameter
                first_param = next(iter(module.parameters.keys()))
                default_value = module.parameters[first_param].default
                
                params = {first_param: default_value}
                
                # Should use defaults for missing params
                result = module.simulate(params)
                self.assertIsInstance(result, SimulationResult)

    def test_model_ids_are_unique(self):
        """Test that all model IDs are unique."""
        model_ids = self.expected_models
        unique_ids = set(model_ids)
        
        self.assertEqual(len(model_ids), len(unique_ids))

    def test_display_names_are_unique(self):
        """Test that all display names are unique."""
        display_names = []
        for model_id in self.expected_models:
            module = get_module(model_id)
            display_names.append(module.display_name)
        
        unique_names = set(display_names)
        self.assertEqual(
            len(display_names), len(unique_names), "Display names should be unique"
        )

    def test_simulation_results_are_serializable(self):
        """Test that simulation results can be serialized to dict."""
        model_params = {
            "kyle": {"N": 10},
            "almgren_chriss": {"N": 10},
            "glosten_milgrom": {"N": 10},
            "de_prado": {"n_buckets": 10},
            "criscuolo_waehlbroeck": {"N": 10},
            "asset_option": {"n": 10},
        }
        
        for model_id in self.expected_models:
            with self.subTest(model_id=model_id):
                module = get_module(model_id)
                params = model_params[model_id]
                
                result = module.simulate(params)
                
                # Should be serializable to dict
                result_dict = result.to_dict()
                
                self.assertIsInstance(result_dict, dict)
                self.assertIn("time_series", result_dict)
                self.assertIn("metrics", result_dict)


if __name__ == "__main__":
    unittest.main()
