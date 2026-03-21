"""Glosten-Milgrom (1985) wrapper for the webapp module system."""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path to import existing models
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from lob_modeling.models.glosten_milgrom import GlostenAndMilgromSimplest

from ..base import (
    EducationalContent,
    ModelModule,
    ParameterSpec,
    SimulationResult,
    VisualizationSpec,
)


class GlostenMilgromModule(ModelModule):
    """Glosten-Milgrom model module for the webapp."""

    @property
    def model_id(self) -> str:
        """Unique identifier for the Glosten-Milgrom model."""
        return "glosten_milgrom"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return "Glosten-Milgrom (1985)"

    @property
    def description(self) -> str:
        """Brief description for educational context."""
        return (
            "Specialist market model with asymmetric information. "
            "Demonstrates how market makers set bid-ask spreads based on "
            "order flow and Bayesian updating."
        )

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        """Schema defining parameters, types, ranges, defaults."""
        return {
            "N": ParameterSpec(
                name="N",
                type_=int,
                default=50,
                min_value=10,
                max_value=200,
                description="Number of trades to simulate",
            ),
            "ALPHA": ParameterSpec(
                name="ALPHA",
                type_=float,
                default=0.5,
                min_value=0.1,
                max_value=0.9,
                description="Probability of low price (1-ALPHA = high price)",
            ),
            "BETA": ParameterSpec(
                name="BETA",
                type_=float,
                default=0.3,
                min_value=0.01,
                max_value=0.99,
                description="Proportion of informed traders",
            ),
            "V_low": ParameterSpec(
                name="V_low",
                type_=float,
                default=0.0,
                min_value=0.0,
                max_value=50.0,
                description="Low possible asset value",
            ),
            "V_high": ParameterSpec(
                name="V_high",
                type_=float,
                default=10.0,
                min_value=0.0,
                max_value=100.0,
                description="High possible asset value",
            ),
        }

    @property
    def visualizations(self) -> List[VisualizationSpec]:
        """List of model-specific chart definitions."""
        return [
            VisualizationSpec(
                id="bid_ask_spread",
                title="Bid-Ask Spread Evolution",
                type="multi_line",
                description="Shows how bid and ask prices evolve over trades",
                data_mapping={"x": "time", "y": ["bid", "ask"]},
                axes={
                    "x": {"label": "Trades", "format": "integer"},
                    "y": {"label": "Price ($)", "format": "currency"},
                },
            ),
            VisualizationSpec(
                id="spread_width",
                title="Spread Width Over Time",
                type="line",
                description="Shows the bid-ask spread (ask - bid) over time",
                data_mapping={"x": "time", "y": ["spread"]},
                axes={
                    "x": {"label": "Trades", "format": "integer"},
                    "y": {"label": "Spread ($)", "format": "currency"},
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
        N = params.get("N", 50)
        ALPHA = params.get("ALPHA", 0.5)
        BETA = params.get("BETA", 0.3)
        V_low = params.get("V_low", 0.0)
        V_high = params.get("V_high", 10.0)

        # Create model instance (runs bidask_price in __init__)
        model = GlostenAndMilgromSimplest(
            N=N,
            ALPHA=ALPHA,
            BETA=BETA,
            V_low=V_low,
            V_high=V_high,
        )

        # Convert to SimulationResult format
        time_steps = list(range(len(model.bid)))
        spread = [model.ask[i] - model.bid[i] for i in range(len(model.bid))]
        
        time_series = {
            "time": time_steps,
            "bid": model.bid,
            "ask": model.ask,
            "spread": spread,
        }

        # Calculate metrics
        avg_spread = sum(spread) / len(spread) if spread else 0
        final_bid = model.bid[-1] if model.bid else 0
        final_ask = model.ask[-1] if model.ask else 0
        
        metrics = {
            "avg_spread": float(avg_spread),
            "final_bid": float(final_bid),
            "final_ask": float(final_ask),
            "final_spread": float(final_ask - final_bid),
            "true_value_estimate": float((final_bid + final_ask) / 2),
        }

        return SimulationResult(
            time_series=time_series,
            metrics=metrics,
            metadata={
                "model_id": self.model_id,
                "parameters": params,
                "method": "bayesian_updating",
            },
        )

    def get_educational_content(self) -> EducationalContent:
        """Return theory, equations, interpretation guide.

        Returns:
            Educational content for the Glosten-Milgrom model.
        """
        return EducationalContent(
            learning_objectives=[
                "Understand how asymmetric information creates bid-ask spreads",
                "Learn how market makers use Bayesian updating to estimate value",
                "Observe how spreads narrow as more information is revealed",
            ],
            background_theory=(
                "The Glosten-Milgrom model (1985) explains bid-ask spreads in dealer "
                "markets with asymmetric information. The market maker observes buy "
                "and sell orders but doesn't know which traders are informed. Using "
                "Bayesian updating, the market maker adjusts bid and ask prices based "
                "on the order flow. Informed traders know the true value, while "
                "uninformed traders trade randomly. The spread compensates the market "
                "maker for losses to informed traders."
            ),
            equations=[
                {
                    "label": "Bid Price",
                    "equation": "bidₜ = E[V | sell order at t, history]",
                    "description": "Bid price is expected value conditional on sell order",
                },
                {
                    "label": "Ask Price",
                    "equation": "askₜ = E[V | buy order at t, history]",
                    "description": "Ask price is expected value conditional on buy order",
                },
            ],
            interpretation_guide=(
                "The bid-ask spread chart shows how prices evolve as the market maker "
                "learns from order flow. Initially, the spread is wide due to uncertainty. "
                "As more trades are observed, the market maker's estimate converges toward "
                "the true value, and the spread typically narrows. The proportion of "
                "informed traders (β) affects how quickly prices adjust."
            ),
        )
