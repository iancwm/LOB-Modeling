"""Kyle Model (1985) wrapper for the webapp module system."""

import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path to import existing models
# From: src/lob_modeling/webapp/modules/wrappers/kyle_wrapper.py
# Need to add: src/ to sys.path
src_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from lob_modeling.models.kyle import KyleModel as OriginalKyleModel  # noqa: E402

from ..base import (  # noqa: E402
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)


class KyleModelModule(ModelModule):
    """Kyle Model module for the webapp."""

    @property
    def model_id(self) -> str:
        """Unique identifier for the Kyle Model."""
        return "kyle"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return "Kyle Model (1985)"

    @property
    def description(self) -> str:
        """Brief description for educational context."""
        return (
            "Single dealer model with asymmetric information. "
            "Demonstrates how informed traders balance profit against "
            "information revelation and how market makers set prices "
            "based on order flow."
        )

    @property
    def parameters(self) -> dict[str, ParameterSpec]:
        """Schema defining parameters, types, ranges, defaults."""
        return {
            "V_0": ParameterSpec(
                name="V_0",
                type_=float,
                default=5.0,
                min_value=0.0,
                max_value=100.0,
                description="Initial security value",
            ),
            "SIGMA_G": ParameterSpec(
                name="SIGMA_G",
                type_=float,
                default=0.4,
                min_value=0.01,
                max_value=10.0,
                description="Volatility of security guess at time N",
            ),
            "SIGMA_T": ParameterSpec(
                name="SIGMA_T",
                type_=float,
                default=0.2,
                min_value=0.01,
                max_value=10.0,
                description="True variance of security at time 0",
            ),
            "N": ParameterSpec(
                name="N",
                type_=int,
                default=50,
                min_value=10,
                max_value=500,
                description="Number of discretized time periods",
            ),
        }

    @property
    def visualizations(self) -> list[VisualizationSpec]:
        """List of model-specific chart definitions."""
        return [
            VisualizationSpec(
                id="price_discovery",
                title="Price Discovery Over Time",
                type="multi_line",
                description="Shows convergence of market price to true value",
                data_mapping={
                    "x": "time",
                    "y": ["true_value", "market_price"],
                },
                axes={
                    "x": {"label": "Time Period", "format": "integer"},
                    "y": {"label": "Price ($)", "format": "currency"},
                },
            ),
            VisualizationSpec(
                id="order_flow",
                title="Order Flow Dynamics",
                type="stacked_bar",
                description="Informed and noise trader order flow",
                data_mapping={
                    "x": "time",
                    "y": ["informed_order", "noise_order"],
                },
                axes={
                    "x": {"label": "Time Period", "format": "integer"},
                    "y": {"label": "Order Size", "format": "float"},
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
        V_0 = params.get("V_0", 5.0)
        SIGMA_G = params.get("SIGMA_G", 0.4)
        SIGMA_T = params.get("SIGMA_T", 0.2)
        N = params.get("N", 50)

        # Create model instance
        model = OriginalKyleModel(
            V_0=V_0,
            SIGMA_G=SIGMA_G,
            SIGMA_T=SIGMA_T,
            N=N,
        )

        # Run multiperiod simulation
        result = model.multiperiod_price(plot=False)

        # Convert to SimulationResult format
        time_series = {
            "time": list(range(N + 1)),
            "true_value": [V_0] * (N + 1),  # Simplified for now
            "market_price": result.get("price_changes", [0] * (N + 1)),
            "informed_order": result.get("informed_orders", [0] * (N + 1)),
            "noise_order": result.get("noise_orders", [0] * (N + 1)),
        }

        metrics = {
            "final_price": (
                time_series["market_price"][-1] if time_series["market_price"] else 0
            ),
            "price_variance": (
                sum(result.get("SIGMA", [0])) / len(result.get("SIGMA", [1]))
                if result.get("SIGMA")
                else 0
            ),
        }

        return SimulationResult(
            time_series=time_series,
            metrics=metrics,
            metadata={
                "model_id": self.model_id,
                "parameters": params,
            },
        )

    def get_educational_content(self) -> EducationalContent:
        """Return theory, equations, interpretation guide.

        Returns:
            Educational content for the Kyle Model.
        """
        return EducationalContent(
            learning_objectives=[
                "Understand how asymmetric information affects price formation",
                "Observe how informed traders balance profit vs. information revelation",
                "See how market makers set prices based on order flow",
            ],
            background_theory=(
                "The Kyle model (1985) describes a market with asymmetric information "
                "where three types of agents interact: an informed trader who knows the "
                "true value of the asset, noise traders who trade randomly, and a "
                "market maker who sets prices based on observed order flow. The model "
                "demonstrates how private information is gradually incorporated into "
                "prices through trading."
            ),
            equations=[
                {
                    "label": "Market Price",
                    "equation": "p_n = p_{n-1} + λ(α_n + u_n)",
                    "description": "Market maker updates price based on order flow",
                },
                {
                    "label": "Informed Trader's Order",
                    "equation": "α_n = β(V - p_{n-1})",
                    "description": "Informed trader's optimal order size",
                },
            ],
            interpretation_guide=(
                "The price discovery chart shows how the market price converges to the "
                "true value over time. The order flow chart shows the informed trader's "
                "strategy of balancing profit against information revelation. Notice how "
                "the informed trader trades more aggressively early on and reduces trading "
                "as the market price approaches the true value."
            ),
        )
