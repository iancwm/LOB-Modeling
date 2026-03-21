"""Almgren-Chriss (2000) wrapper for the webapp module system."""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path to import existing models
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from lob_modeling.models.almgren_chriss import AlmgrenChriss2000

from ..base import (
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)


class AlmgrenChrissModule(ModelModule):
    """Almgren-Chriss model module for the webapp."""

    @property
    def model_id(self) -> str:
        """Unique identifier for the Almgren-Chriss model."""
        return "almgren_chriss"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return "Almgren-Chriss (2000)"

    @property
    def description(self) -> str:
        """Brief description for educational context."""
        return (
            "Optimal execution model with linear impact costs. "
            "Demonstrates how to minimize trading costs while managing "
            "market impact and risk."
        )

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        """Schema defining parameters, types, ranges, defaults."""
        return {
            "ALPHA": ParameterSpec(
                name="ALPHA",
                type_=float,
                default=1.0,
                min_value=0.1,
                max_value=2.0,
                description="Power of temporary impact function",
            ),
            "ETA": ParameterSpec(
                name="ETA",
                type_=float,
                default=5e-6,
                min_value=1e-8,
                max_value=1e-4,
                description="Linear coefficient of temporary impact",
            ),
            "GAMMA": ParameterSpec(
                name="GAMMA",
                type_=float,
                default=5e-5,
                min_value=1e-7,
                max_value=1e-3,
                description="Linear coefficient of permanent impact",
            ),
            "LAMBDA": ParameterSpec(
                name="LAMBDA",
                type_=float,
                default=0.00009,
                min_value=0.00001,
                max_value=0.001,
                description="Risk aversion measure",
            ),
            "SIGMA": ParameterSpec(
                name="SIGMA",
                type_=float,
                default=0.495,
                min_value=0.1,
                max_value=2.0,
                description="Annual volatility",
            ),
            "N": ParameterSpec(
                name="N",
                type_=int,
                default=50,
                min_value=10,
                max_value=200,
                description="Number of time steps",
            ),
            "T": ParameterSpec(
                name="T",
                type_=float,
                default=0.025,
                min_value=0.001,
                max_value=0.1,
                description="Expiry time in days",
            ),
            "X": ParameterSpec(
                name="X",
                type_=int,
                default=500,
                min_value=100,
                max_value=10000,
                description="Initial number of shares",
            ),
        }

    @property
    def visualizations(self) -> List[VisualizationSpec]:
        """List of model-specific chart definitions."""
        return [
            VisualizationSpec(
                id="inventory_decay",
                title="Inventory Decay Over Time",
                type="line",
                description="Shows how inventory decreases over the execution period",
                data_mapping={"x": "time", "y": ["inventory"]},
                axes={
                    "x": {"label": "Trading Step", "format": "integer"},
                    "y": {"label": "Shares", "format": "integer"},
                },
            ),
            VisualizationSpec(
                id="trade_schedule",
                title="Optimal Trade Schedule",
                type="bar",
                description="Shows optimal number of shares to sell at each step",
                data_mapping={"x": "time", "y": ["trades"]},
                axes={
                    "x": {"label": "Trading Step", "format": "integer"},
                    "y": {"label": "Shares Traded", "format": "integer"},
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
        ALPHA = params.get("ALPHA", 1.0)
        ETA = params.get("ETA", 5e-6)
        GAMMA = params.get("GAMMA", 5e-5)
        LAMBDA = params.get("LAMBDA", 0.00009)
        SIGMA = params.get("SIGMA", 0.495)
        N = params.get("N", 50)
        T = params.get("T", 0.025)
        X = params.get("X", 500)

        # Create model instance
        model = AlmgrenChriss2000(
            ALPHA=ALPHA,
            ETA=ETA,
            GAMMA=GAMMA,
            LAMBDA=LAMBDA,
            SIGMA=SIGMA,
            N=N,
            T=T,
            X=X,
        )

        # Run optimization (quadratic programming approach)
        opt_sale, inventory, expected_shortfall, variance_shortfall = model.basic_almgren(
            plot=False
        )

        # Convert to SimulationResult format
        time_steps = list(range(len(inventory)))
        time_series = {
            "time": time_steps,
            "inventory": [int(x) for x in inventory],
            "trades": [int(x) for x in opt_sale],
        }

        metrics = {
            "expected_shortfall": float(expected_shortfall),
            "variance_shortfall": float(variance_shortfall),
            "total_traded": float(sum(opt_sale)),
            "avg_trade_size": float(sum(opt_sale) / len(opt_sale)) if len(opt_sale) > 0 else 0,
        }

        return SimulationResult(
            time_series=time_series,
            metrics=metrics,
            metadata={
                "model_id": self.model_id,
                "parameters": params,
                "method": "quadratic_programming",
            },
        )

    def get_educational_content(self) -> EducationalContent:
        """Return theory, equations, interpretation guide.

        Returns:
            Educational content for the Almgren-Chriss model.
        """
        return EducationalContent(
            learning_objectives=[
                "Understand the trade-off between market impact and timing risk",
                "Learn how to optimize execution schedules for large trades",
                "Observe the effect of risk aversion on trading strategy",
            ],
            background_theory=(
                "The Almgren-Chriss model (2000) provides a framework for optimal "
                "execution of large portfolio transactions. The model balances the "
                "cost of market impact (trading too fast) against the risk of price "
                "fluctuations (trading too slow). The optimal strategy minimizes a "
                "combination of expected cost and variance, weighted by the trader's "
                "risk aversion parameter."
            ),
            equations=[
                {
                    "label": "Implementation Shortfall",
                    "equation": "IS = 0.5γX² + εΣvₙ + ((η-0.5γ)/τ)Σvₙ²",
                    "description": "Total cost of execution including permanent impact, "
                    "fixed costs, and temporary impact",
                },
                {
                    "label": "Variance of IS",
                    "equation": "Var(IS) = τσ² Σ(Xₙ)²",
                    "description": "Viance of implementation shortfall depends on "
                    "volatility and remaining inventory",
                },
            ],
            interpretation_guide=(
                "The inventory decay chart shows how the optimal strategy gradually "
                "reduces holdings over time. A more risk-averse trader (higher λ) will "
                "trade more aggressively early on. The trade schedule shows the optimal "
                "number of shares to sell at each step. Notice how the strategy balances "
                "market impact against timing risk."
            ),
        )
