"""Asset or Nothing Option wrapper for the webapp module system."""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path to import existing models
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from lob_modeling.models.asset_option import asset_or_nothing_call

from ..base import (
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)


class AssetOptionModule(ModelModule):
    """Asset or Nothing Option model module for the webapp."""

    @property
    def model_id(self) -> str:
        """Unique identifier for the Asset Option model."""
        return "asset_option"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return "Asset or Nothing Option"

    @property
    def description(self) -> str:
        """Brief description for educational context."""
        return (
            "Binomial tree pricing model for asset or nothing call options. "
            "Pays the asset value if the asset price exceeds the strike at expiry."
        )

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        """Schema defining parameters, types, ranges, defaults."""
        return {
            "s": ParameterSpec(
                name="s",
                type_=float,
                default=100.0,
                min_value=10.0,
                max_value=500.0,
                description="Initial asset price",
            ),
            "n": ParameterSpec(
                name="n",
                type_=int,
                default=50,
                min_value=10,
                max_value=200,
                description="Number of binomial tree steps",
            ),
            "k": ParameterSpec(
                name="k",
                type_=int,
                default=252,
                min_value=50,
                max_value=500,
                description="Size of partitions (trading days)",
            ),
            "T": ParameterSpec(
                name="T",
                type_=float,
                default=1.0,
                min_value=0.1,
                max_value=5.0,
                description="Time to expiry in years",
            ),
            "K": ParameterSpec(
                name="K",
                type_=float,
                default=100.0,
                min_value=50.0,
                max_value=500.0,
                description="Strike price",
            ),
            "F": ParameterSpec(
                name="F",
                type_=float,
                default=1.0,
                min_value=0.5,
                max_value=2.0,
                description="Payoff multiplier",
            ),
            "SIGMA": ParameterSpec(
                name="SIGMA",
                type_=float,
                default=0.3,
                min_value=0.05,
                max_value=1.0,
                description="Annual volatility",
            ),
            "r": ParameterSpec(
                name="r",
                type_=float,
                default=0.05,
                min_value=0.0,
                max_value=0.2,
                description="Risk-free interest rate",
            ),
        }

    @property
    def visualizations(self) -> List[VisualizationSpec]:
        """List of model-specific chart definitions."""
        return [
            VisualizationSpec(
                id="option_tree",
                title="Binomial Tree - Option Values",
                type="heatmap",
                description="Shows option values at each node of the binomial tree",
                data_mapping={"x": "time", "y": "step", "z": "option_value"},
                axes={
                    "x": {"label": "Time Step", "format": "integer"},
                    "y": {"label": "Node", "format": "integer"},
                    "z": {"label": "Option Value", "format": "currency"},
                },
            ),
            VisualizationSpec(
                id="asset_paths",
                title="Asset Price Paths",
                type="multi_line",
                description="Shows simulated asset price paths through the tree",
                data_mapping={"x": "time", "y": ["asset_price"]},
                axes={
                    "x": {"label": "Time Step", "format": "integer"},
                    "y": {"label": "Asset Price ($)", "format": "currency"},
                },
            ),
        ]

    def simulate(self, params: Dict[str, Any]) -> SimulationResult:
        """Run simulation with given parameters.

        Args:
            params: Dictionary of parameter name to value.

        Returns:
            Simulation result with time series and metrics.
        """
        # Extract parameters with defaults
        s = params.get("s", 100.0)
        n = params.get("n", 50)
        k = params.get("k", 252)
        T = params.get("T", 1.0)
        K = params.get("K", 100.0)
        F = params.get("F", 1.0)
        SIGMA = params.get("SIGMA", 0.3)
        r = params.get("r", 0.05)

        # Run binomial tree pricing
        option_tree = asset_or_nothing_call(s, n, k, T, K, F, SIGMA, r)

        # Extract option values at each time step (initial node)
        option_values = [option_tree[0, i] for i in range(n + 1)]
        
        # Generate asset price paths (simplified - show expected path)
        import numpy as np
        dt = T / n
        time_steps = list(range(n + 1))
        
        # Calculate expected asset price at each step (risk-neutral)
        asset_prices = [s * np.exp((r - 0.5 * SIGMA**2) * dt * i) for i in time_steps]

        # Convert to SimulationResult format
        time_series = {
            "time": time_steps,
            "option_value": [float(x) for x in option_values],
            "asset_price": [float(x) for x in asset_prices],
        }

        # Calculate metrics
        initial_option_value = option_tree[0, 0]
        intrinsic_value = max(s - K, 0)
        time_value = initial_option_value - intrinsic_value if initial_option_value > intrinsic_value else 0
        
        metrics = {
            "option_price": float(initial_option_value),
            "intrinsic_value": float(intrinsic_value),
            "time_value": float(time_value),
            "moneyness": float(s / K),
            "strike": float(K),
            "spot": float(s),
        }

        return SimulationResult(
            time_series=time_series,
            metrics=metrics,
            metadata={
                "model_id": self.model_id,
                "parameters": params,
                "method": "binomial_tree",
                "option_type": "asset_or_nothing_call",
            },
        )

    def get_educational_content(self) -> EducationalContent:
        """Return theory, equations, interpretation guide.

        Returns:
            Educational content for the Asset Option model.
        """
        return EducationalContent(
            learning_objectives=[
                "Understand the binomial tree approach to option pricing",
                "Learn how asset or nothing options differ from vanilla options",
                "Observe how option values evolve through the tree",
            ],
            background_theory=(
                "The Asset or Nothing Option is an exotic option that pays the asset "
                "value if the asset price exceeds the strike at expiry, otherwise it "
                "pays nothing. This differs from a vanilla call option which pays "
                "(S - K)⁺. The binomial tree model discretizes time and models the "
                "asset price as moving up or down at each step. Risk-neutral pricing "
                "is used to compute the option value by working backwards from expiry."
            ),
            equations=[
                {
                    "label": "Payoff",
                    "equation": "Payoff = S_T if S_T > K, else 0",
                    "description": "Asset or nothing call payoff at expiry",
                },
                {
                    "label": "Risk-Neutral Probability",
                    "equation": "p = (e^(rΔt) - d) / (u - d)",
                    "description": "Risk-neutral probability of up move",
                },
            ],
            interpretation_guide=(
                "The option value chart shows how the option value evolves through the "
                "binomial tree. The option price at time 0 is the fair value of the "
                "option. The moneyness ratio (S/K) indicates whether the option is "
                "in-the-money (>1), at-the-money (=1), or out-of-the-money (<1). "
                "Higher volatility increases the option value due to greater upside potential."
            ),
        )
