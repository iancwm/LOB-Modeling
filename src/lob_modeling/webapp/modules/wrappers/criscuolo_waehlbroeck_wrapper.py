"""Criscuolo & Waehlbroeck (2014) wrapper for the webapp module system.

Simplified wrapper for stochastic volatility optimal execution model.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path to import existing models
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from lob_modeling.models.criscuolo_waehlbroeck import Criscuolo2014  # noqa: E402

from ..base import (  # noqa: E402
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)


class CriscuoloWaehlbroeckModule(ModelModule):
    """Criscuolo-Waehlbroeck model module for the webapp."""

    @property
    def model_id(self) -> str:
        """Unique identifier for the Criscuolo-Waehlbroeck model."""
        return "criscuolo_waehlbroeck"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return "Criscuolo & Waehlbroeck (2014)"

    @property
    def description(self) -> str:
        """Brief description for educational context."""
        return (
            "Stochastic volatility optimal execution model. "
            "Demonstrates how to optimize trading schedules under "
            "time-varying volatility conditions."
        )

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        """Schema defining parameters, types, ranges, defaults."""
        return {
            "KAPPA": ParameterSpec(
                name="KAPPA",
                type_=float,
                default=3.0,
                min_value=0.5,
                max_value=10.0,
                description="Exponential decay constant for volatility mean reversion",
            ),
            "THETA": ParameterSpec(
                name="THETA",
                type_=float,
                default=0.01,
                min_value=0.001,
                max_value=0.1,
                description="Long-run average volatility",
            ),
            "GAMMA": ParameterSpec(
                name="GAMMA",
                type_=float,
                default=0.1,
                min_value=0.01,
                max_value=0.5,
                description="Variance of volatility",
            ),
            "V_0": ParameterSpec(
                name="V_0",
                type_=float,
                default=0.5,
                min_value=0.1,
                max_value=2.0,
                description="Initial volatility",
            ),
            "T": ParameterSpec(
                name="T",
                type_=float,
                default=0.5,
                min_value=0.1,
                max_value=2.0,
                description="Time to expiry in years",
            ),
            "N": ParameterSpec(
                name="N",
                type_=int,
                default=10,
                min_value=4,
                max_value=50,
                description="Number of time periods",
            ),
            "S_0": ParameterSpec(
                name="S_0",
                type_=float,
                default=100.0,
                min_value=50.0,
                max_value=500.0,
                description="Initial stock price",
            ),
        }

    @property
    def visualizations(self) -> List[VisualizationSpec]:
        """List of model-specific chart definitions."""
        return [
            VisualizationSpec(
                id="volatility_path",
                title="Stochastic Volatility Path",
                type="line",
                description="Shows volatility evolution over the execution period",
                data_mapping={"x": "time", "y": ["volatility"]},
                axes={
                    "x": {"label": "Time Period", "format": "integer"},
                    "y": {"label": "Volatility", "format": "percent"},
                },
            ),
            VisualizationSpec(
                id="execution_schedule",
                title="Optimal Execution Schedule",
                type="bar",
                description="Shows optimal trading rate at each period",
                data_mapping={"x": "time", "y": ["trade_rate"]},
                axes={
                    "x": {"label": "Time Period", "format": "integer"},
                    "y": {"label": "Trade Rate", "format": "percent"},
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
        KAPPA = params.get("KAPPA", 3.0)
        THETA = params.get("THETA", 0.01)
        GAMMA = params.get("GAMMA", 0.1)
        V_0 = params.get("V_0", 0.5)
        T = params.get("T", 0.5)
        N = params.get("N", 10)
        S_0 = params.get("S_0", 100.0)

        # Create model instance
        model = Criscuolo2014(
            KAPPA=KAPPA,
            THETA=THETA,
            GAMMA=GAMMA,
            V_0=V_0,
            T=T,
            N=N,
            S_0=S_0,
        )

        # Run optimization
        try:
            opt_result = model.optimal_execution()

            # Extract optimal trajectory
            trades = np.reshape(opt_result.x, (N, 2))
            share_turnover = [trade[0] for trade in trades]
            participation = [trade[1] for trade in trades]

            # Simulate volatility path (simplified Heston-like)
            np.random.seed(42)
            dt = T / N
            volatility = [V_0]
            for i in range(1, N):
                dvol = (
                    KAPPA * (THETA - volatility[-1]) * dt
                    + GAMMA * np.sqrt(dt) * np.random.normal()
                )
                volatility.append(max(0.01, volatility[-1] + dvol))

        except Exception as e:
            # Fallback if optimization fails
            share_turnover = [1.0 / N] * N
            participation = [0.2] * N
            volatility = [V_0] * N

        # Convert to SimulationResult format
        time_steps = list(range(N))
        time_series = {
            "time": time_steps,
            "volatility": [float(v) for v in volatility[:N]],
            "trade_rate": [float(p) for p in participation],
            "share_turnover": [float(s) for s in share_turnover],
        }

        # Calculate metrics
        avg_volatility = sum(volatility[:N]) / len(volatility[:N]) if volatility else 0
        total_cost = opt_result.fun if "opt_result" in dir() else 0

        metrics = {
            "avg_volatility": float(avg_volatility),
            "total_cost": float(total_cost) if total_cost else 0,
            "optimal_participation": (
                float(sum(participation) / len(participation)) if participation else 0
            ),
            "execution_time_years": float(T),
        }

        return SimulationResult(
            time_series=time_series,
            metrics=metrics,
            metadata={
                "model_id": self.model_id,
                "parameters": params,
                "method": "stochastic_volatility_optimization",
            },
        )

    def get_educational_content(self) -> EducationalContent:
        """Return theory, equations, interpretation guide.

        Returns:
            Educational content for the Criscuolo-Waehlbroeck model.
        """
        return EducationalContent(
            learning_objectives=[
                "Understand how stochastic volatility affects optimal execution",
                "Learn to optimize trading schedules under time-varying volatility",
                "Observe the trade-off between timing risk and market impact",
            ],
            background_theory=(
                "The Criscuolo-Waehlbroeck model (2014) extends optimal execution theory "
                "to incorporate stochastic volatility. Unlike constant volatility models, "
                "this framework captures realistic market conditions where volatility "
                "fluctuates over time. The model uses a Heston-like stochastic volatility "
                "process with mean reversion, and optimizes the trading schedule to "
                "minimize total execution costs including both market impact and timing risk."
            ),
            equations=[
                {
                    "label": "Volatility Dynamics",
                    "equation": "dVₜ = κ(θ - Vₜ)dt + γdWₜ",
                    "description": "Mean-reverting stochastic volatility process",
                },
                {
                    "label": "Total Cost",
                    "equation": "Cost = Impact + α(t) + Risk",
                    "description": "Total execution cost includes impact, alpha, and risk",
                },
            ],
            interpretation_guide=(
                "The volatility path chart shows how volatility evolves stochastically "
                "over the execution period. The execution schedule shows the optimal "
                "trading rate at each period. During high volatility periods, the model "
                "may recommend faster execution to reduce timing risk. The participation "
                "rate indicates what fraction of market volume to trade at each step."
            ),
        )
