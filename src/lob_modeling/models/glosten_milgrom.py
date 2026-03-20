"""Glosten-Milgrom (1985) - Specialist market with asymmetric information.

This module implements the Glosten-Milgrom model of bid-ask spreads in a
specialist market where informed and uninformed traders interact.
"""

import random
from typing import List

import numpy as np
import plotly.graph_objects as go


class GlostenAndMilgromSimplest:
    """Simplified Glosten-Milgrom model for bid-ask spread dynamics.

    This class implements a simplified version of the Glosten-Milgrom (1985)
    model where the market maker uses Bayesian updating to estimate the true
    value of an asset based on observed buy and sell orders.

    Attributes:
        N: Number of trades observed in time t.
        ALPHA: Probability of low vs high price (1-ALPHA = high price).
        BETA: Proportion of informed traders.
        V_low: The low possible asset value.
        V_high: The high possible asset value.
        ask: List of ask prices over time.
        bid: List of bid prices over time.
        lob: List of limit order book states (buy=1, sell=-1).
    """

    def __init__(
        self,
        N: int = 50,
        ALPHA: float = 0.5,
        BETA: float = 0.3,
        V_low: float = 0,
        V_high: float = 10,
    ) -> None:
        """Initialize the Glosten-Milgrom model with specified parameters.

        Args:
            N: Number of trades to simulate. Defaults to 50.
            ALPHA: Probability of low price. Defaults to 0.5.
            BETA: Proportion of informed traders. Defaults to 0.3.
            V_low: Low possible asset value. Defaults to 0.
            V_high: High possible asset value. Defaults to 10.
        """
        self.N = N
        self.ALPHA = float(ALPHA)
        self.BETA = float(BETA)
        self.V_low = float(V_low)
        self.V_high = float(V_high)

        self.ask: List[float] = []
        self.bid: List[float] = []
        self.lob = [1 if random.random() < 0.5 else -1 for _ in range(0, self.N + 1)]
        self.bidask_price()

    def bidask_price(self) -> None:
        """Compute bid and ask prices using Bayesian updating.

        Uses Bayes' rule to update the market maker's estimate of the asset
        value after each trade. The numerator is the prior probability of the
        asset value (high or low) multiplied by the conditional probability of
        observing the sequence of buy and sell orders given that value. The
        denominator is the total probability of the observed order sequence.

        Updates:
            self.bid: Appends bid prices for each period.
            self.ask: Appends ask prices for each period.
        """
        for n in range(0, self.N):
            buy_orders = sum([x if x > 0 else 0 for x in self.lob[0:n]])
            sell_orders = n - buy_orders

            bid_numerator = (
                self.V_low
                * (((1 - self.BETA) ** buy_orders) * ((1 + self.BETA) ** (sell_orders + 1)))
                + self.V_high
                * (
                    (1 - self.ALPHA)
                    * ((1 + self.BETA) ** buy_orders)
                    * ((1 - self.BETA) ** (sell_orders + 1))
                )
            )
            bid_denominator = (
                (1 - self.ALPHA)
                * ((1 + self.BETA) ** buy_orders)
                * ((1 - self.BETA) ** (sell_orders + 1))
                + self.ALPHA
                * ((1 - self.BETA) ** buy_orders)
                * ((1 + self.BETA) ** (sell_orders + 1))
            )

            ask_numerator = (
                self.V_low
                * (((1 - self.BETA) ** (buy_orders + 1)) * ((1 + self.BETA) ** sell_orders))
                + self.V_high
                * (
                    (1 - self.ALPHA)
                    * ((1 + self.BETA) ** (buy_orders + 1))
                    * ((1 - self.BETA) ** sell_orders)
                )
            )
            ask_denominator = (
                (1 - self.ALPHA)
                * ((1 + self.BETA) ** (buy_orders + 1))
                * ((1 - self.BETA) ** sell_orders)
                + self.ALPHA
                * ((1 - self.BETA) ** (buy_orders + 1))
                * ((1 + self.BETA) ** sell_orders)
            )

            self.bid.append(bid_numerator / bid_denominator)
            self.ask.append(ask_numerator / ask_denominator)

    def plot(self) -> None:
        """Create an interactive plot of bid and ask prices over time.

        Displays a Plotly figure showing the evolution of bid and ask prices
        as a function of the number of trades/time.
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.arange(self.N),
                y=self.bid,
                mode="lines",
                name="Bid Price",
                line=dict(color="blue", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(self.N),
                y=self.ask,
                mode="lines",
                name="Ask Price",
                line=dict(color="red", width=2),
            )
        )

        fig.update_layout(
            title="Simplified Glosten-Milgrom",
            xaxis_title="Trades/Time",
            yaxis_title="Price",
            hovermode="x unified",
            template="plotly_white",
            height=600,
        )

        fig.show()
