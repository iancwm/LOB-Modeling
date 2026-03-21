"""De Prado et al. (2012) wrapper for the webapp module system.

Simplified wrapper for VPIN (Volume-synchronized Probability of Informed Trading)
simulation without requiring external data files.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path to import existing models
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from ..base import (  # noqa: E402
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)


class DePradoModule(ModelModule):
    """De Prado VPIN model module for the webapp."""

    @property
    def model_id(self) -> str:
        """Unique identifier for the De Prado model."""
        return "de_prado"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return "De Prado et al. (2012)"

    @property
    def description(self) -> str:
        """Brief description for educational context."""
        return (
            "VPIN (Volume-synchronized Probability of Informed Trading) model. "
            "Demonstrates how to detect informed trading and optimize execution "
            "horizon based on order flow toxicity."
        )

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        """Schema defining parameters, types, ranges, defaults."""
        return {
            "n_buckets": ParameterSpec(
                name="n_buckets",
                type_=int,
                default=50,
                min_value=10,
                max_value=200,
                description="Number of volume buckets for VPIN calculation",
            ),
            "mu": ParameterSpec(
                name="mu",
                type_=float,
                default=0.7,
                min_value=0.1,
                max_value=2.0,
                description="Rate of informed trades",
            ),
            "epsilon": ParameterSpec(
                name="epsilon",
                type_=float,
                default=0.3,
                min_value=0.1,
                max_value=1.0,
                description="Rate of uninformed trades",
            ),
            "alpha": ParameterSpec(
                name="alpha",
                type_=float,
                default=0.5,
                min_value=0.1,
                max_value=0.9,
                description="Probability that news/event arrives",
            ),
            "delta": ParameterSpec(
                name="delta",
                type_=float,
                default=0.3,
                min_value=0.1,
                max_value=0.9,
                description="Probability of bad news",
            ),
            "n_trades": ParameterSpec(
                name="n_trades",
                type_=int,
                default=1000,
                min_value=100,
                max_value=5000,
                description="Number of trades to simulate",
            ),
        }

    @property
    def visualizations(self) -> List[VisualizationSpec]:
        """List of model-specific chart definitions."""
        return [
            VisualizationSpec(
                id="vpin_evolution",
                title="VPIN Evolution Over Time",
                type="line",
                description="Shows VPIN values across volume buckets",
                data_mapping={"x": "bucket", "y": ["vpin"]},
                axes={
                    "x": {"label": "Volume Bucket", "format": "integer"},
                    "y": {"label": "VPIN", "format": "percent"},
                },
            ),
            VisualizationSpec(
                id="order_imbalance",
                title="Order Imbalance",
                type="bar",
                description="Shows buy vs sell volume imbalance",
                data_mapping={"x": "bucket", "y": ["buy_volume", "sell_volume"]},
                axes={
                    "x": {"label": "Volume Bucket", "format": "integer"},
                    "y": {"label": "Volume", "format": "integer"},
                },
            ),
        ]

    def simulate(self, params: Dict[str, Any]) -> SimulationResult:
        """Run VPIN simulation with given parameters.

        Args:
            params: Dictionary of parameter name to value.

        Returns:
            Simulation result with time series and metrics.
        """
        # Extract parameters with defaults
        n_buckets = params.get("n_buckets", 50)
        mu = params.get("mu", 0.7)
        epsilon = params.get("epsilon", 0.3)
        alpha = params.get("alpha", 0.5)
        delta = params.get("delta", 0.3)
        n_trades = params.get("n_trades", 1000)

        # Simulate order flow
        np.random.seed(42)  # For reproducibility

        # Generate synthetic trade data
        trades = np.random.choice([-1, 1], size=n_trades, p=[0.5, 0.5])
        volumes = np.random.exponential(scale=100, size=n_trades)

        # Adjust for informed trading
        informed_mask = np.random.random(n_trades) < (alpha * mu)
        trades[informed_mask] = np.random.choice([-1, 1], size=sum(informed_mask))

        # Calculate VPIN-like metric for each bucket
        bucket_size = n_trades // n_buckets
        vpin_values = []
        buy_volumes = []
        sell_volumes = []

        for i in range(n_buckets):
            start_idx = i * bucket_size
            end_idx = start_idx + bucket_size

            bucket_trades = trades[start_idx:end_idx]
            bucket_volumes = volumes[start_idx:end_idx]

            buy_vol = sum(bucket_volumes[bucket_trades == 1])
            sell_vol = sum(bucket_volumes[bucket_trades == -1])

            # VPIN = |Buy - Sell| / (Buy + Sell)
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                vpin = abs(buy_vol - sell_vol) / total_vol
            else:
                vpin = 0

            vpin_values.append(vpin)
            buy_volumes.append(buy_vol)
            sell_volumes.append(sell_vol)

        # Convert to SimulationResult format
        bucket_ids = list(range(1, n_buckets + 1))
        time_series = {
            "bucket": bucket_ids,
            "vpin": [float(x) for x in vpin_values],
            "buy_volume": [float(x) for x in buy_volumes],
            "sell_volume": [float(x) for x in sell_volumes],
        }

        # Calculate metrics
        avg_vpin = sum(vpin_values) / len(vpin_values) if vpin_values else 0
        max_vpin = max(vpin_values) if vpin_values else 0
        total_buy = sum(buy_volumes)
        total_sell = sum(sell_volumes)

        metrics = {
            "avg_vpin": float(avg_vpin),
            "max_vpin": float(max_vpin),
            "total_buy_volume": float(total_buy),
            "total_sell_volume": float(total_sell),
            "order_imbalance": (
                float(abs(total_buy - total_sell) / (total_buy + total_sell))
                if (total_buy + total_sell) > 0
                else 0
            ),
            "informed_trading_estimate": float(alpha * mu),
        }

        return SimulationResult(
            time_series=time_series,
            metrics=metrics,
            metadata={
                "model_id": self.model_id,
                "parameters": params,
                "method": "vpin_simulation",
            },
        )

    def get_educational_content(self) -> EducationalContent:
        """Return theory, equations, interpretation guide.

        Returns:
            Educational content for the De Prado model.
        """
        return EducationalContent(
            learning_objectives=[
                "Understand how VPIN measures order flow toxicity",
                "Learn to detect informed trading from order flow",
                "Observe how VPIN relates to market quality and execution risk",
            ],
            background_theory=(
                "The De Prado model (2012) introduces VPIN (Volume-synchronized Probability "
                "of Informed Trading) as a measure of order flow toxicity. VPIN quantifies "
                "the imbalance between buy and sell volume, which can indicate the presence "
                "of informed traders. High VPIN values suggest elevated toxicity and higher "
                "execution risk. The model uses volume buckets rather than time intervals, "
                "making it more robust to variations in trading activity."
            ),
            equations=[
                {
                    "label": "VPIN",
                    "equation": "VPIN = |V_buy - V_sell| / (V_buy + V_sell)",
                    "description": "VPIN measures the relative imbalance between buy and sell volume",
                },
                {
                    "label": "Order Imbalance",
                    "equation": "OI = Σ(V_buy - V_sell)",
                    "description": "Cumulative order imbalance across buckets",
                },
            ],
            interpretation_guide=(
                "The VPIN evolution chart shows how order flow toxicity changes across volume "
                "buckets. Higher VPIN values (closer to 1) indicate greater imbalance and "
                "potential informed trading. The order imbalance chart shows the absolute buy "
                "and sell volumes. Persistent imbalances may indicate information asymmetry "
                "in the market. Traders can use VPIN to adjust execution strategies during "
                "periods of high toxicity."
            ),
        )
