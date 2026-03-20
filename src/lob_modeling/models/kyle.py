"""Kyle Model (1985) - Single dealer model with asymmetric information.

This module implements the Kyle model of insider trading, where an informed
trader interacts with a market maker and noise traders in a limit order book.
"""

import math
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go


class KyleModel:
    """Kyle Model for market making with asymmetric information.

    The model features a single market maker, one informed trader, and many
    noise traders. The market maker sets prices based on observed order flow.

    Attributes:
        V_0: Initial security value.
        V_N: True ending value of security (randomly generated).
        SIGMA_G: Volatility of security guess at time N.
        SIGMA_T: True variance of security at time 0.
        SIGMA: Variance of net order flow of noise traders.
        ERR: Error tolerance for numerical solution of SIGMA[0].
        N: Number of discretized time periods.
        MAX_ITER: Maximum iterations for SIGMA[0] convergence.
    """

    def __init__(
        self,
        V_0: float = 5,
        V_N: float = 5,
        SIGMA_G: float = 0.4,
        SIGMA_T: float = 0.2,
        SIGMA: float = 2,
        ERR: float = 0.05,
        N: int = 50,
        MAX_ITER: int = 100,
    ) -> None:
        """Initialize the Kyle Model with specified parameters.

        Args:
            V_0: Initial security value. Defaults to 5.
            V_N: True ending value of security. Defaults to 5.
            SIGMA_G: Volatility of security guess at time N. Defaults to 0.4.
            SIGMA_T: True variance of security at time 0. Defaults to 0.2.
            SIGMA: Variance of net order flow of noise traders. Defaults to 2.
            ERR: Error tolerance for numerical solution. Defaults to 0.05.
            N: Number of discretized time periods. Defaults to 50.
            MAX_ITER: Maximum iterations for convergence. Defaults to 100.
        """
        self.V_0 = float(V_0)
        self.SIGMA_G = float(SIGMA_G)
        self.SIGMA_T = float(SIGMA_T)
        self.SIGMA = float(SIGMA)
        self.ERR = float(ERR)
        self.N = N
        self.MAX_ITER = MAX_ITER
        self.V_N = np.random.normal(self.V_0, self.SIGMA_G, 1)

    def one_period_price(self) -> Dict[str, float]:
        """Calculate the one-period Kyle model equilibrium.

        Computes the market maker's expected price after observing the limit
        order book (informed order + noise orders), and the informed trader's
        expected profit.

        Returns:
            A dictionary containing:
                - 'MM Price': Market maker's price after observing order flow.
                - 'Informed Profit': Expected profit of the informed trader.
        """
        ALPHA = self.V_0 * (self.SIGMA / math.sqrt(self.SIGMA_G))
        BETA = self.SIGMA / math.sqrt(self.SIGMA_G)
        informed_order = (BETA * self.V_N) + ALPHA
        net_order = informed_order + np.random.normal(0, self.SIGMA, 1)
        mm_price = (
            ((math.sqrt(self.SIGMA_G) / (2 * self.SIGMA)) * net_order) + self.V_0
        )[0]
        informed_profit = (
            (((self.V_N - self.V_0) ** 2) * self.SIGMA) / (2 * math.sqrt(self.SIGMA_G))
        )[0]
        print(f"Market Maker Price: {mm_price}")
        print(f"Informed Trader Expected Profit: {informed_profit}")
        return {"MM Price": mm_price, "Informed Profit": informed_profit}

    def multiperiod_price(self, plot: bool = True) -> Dict[str, List[float]]:
        """Solve the multi-period Kyle model using difference equations.

        Computes optimal parameters and trader's expected profit as a function
        of time by solving the system of difference equations backwards from
        the final period.

        Args:
            plot: If True, creates an interactive Plotly visualization of
                order sizes. Defaults to True.

        Returns:
            A dictionary containing arrays of:
                - 'ALPHA': Trader position size parameters.
                - 'BETA': Uninformed trader size parameters.
                - 'DELTA': Price change parameters.
                - 'LAMBDA': Volatility movement parameters.
                - 'SIGMA': Volatility parameters over time.
                - 'price_changes': Price changes at each period.
                - 'informed_orders': Informed trader orders.
                - 'noise_orders': Noise trader orders.
        """
        dT = 1 / self.N
        ALPHA = np.zeros(self.N + 1)
        BETA = np.zeros(self.N + 1)
        DELTA = np.zeros(self.N + 1)
        LAMBDA = np.zeros(self.N + 1)
        SIGMA = np.zeros(self.N + 1)
        price_changes = np.zeros(self.N + 1)
        informed_orders = np.zeros(self.N + 1)
        noise_orders = np.random.normal(0, (self.SIGMA**2) * (1 / self.N), self.N + 1)
        price_changes[0] = self.V_0
        informed_orders[0] = 0
        BETA[self.N] = 0
        DELTA[self.N] = 0
        SIGMA[self.N] = self.SIGMA_G
        LAMBDA[self.N] = math.sqrt(SIGMA[self.N]) / (self.SIGMA * math.sqrt(2 * dT))
        price_changes[self.N] = LAMBDA[self.N] * noise_orders[self.N]

        iter_count = 0

        while (abs(SIGMA[0] - self.SIGMA_T) > self.ERR) and (
            iter_count < self.MAX_ITER
        ):
            for n in range(self.N, 0, -1):
                ALPHA[n] = (LAMBDA[n] * (self.SIGMA**2)) / SIGMA[n]
                denominator = 1 - (ALPHA[n] * LAMBDA[n] * dT)

                if abs(denominator) < 1e-10:
                    SIGMA[n - 1] = SIGMA[n]
                else:
                    SIGMA[n - 1] = SIGMA[n] / denominator

                BETA[n - 1] = 1 / (4 * LAMBDA[n] * (1 - (BETA[n] * LAMBDA[n])))

                if SIGMA[n - 1] < 1e-8:
                    SIGMA[n - 1] = 1e-8

                lambda_coeffs = [
                    ((self.SIGMA**2) * BETA[n] * dT) / SIGMA[n],
                    -((self.SIGMA**2) * dT) / SIGMA[n],
                    -BETA[n],
                    0.5,
                ]

                if np.any(~np.isfinite(lambda_coeffs)):
                    lambda_roots = np.array([0.1])
                else:
                    lambda_roots = np.roots(lambda_coeffs)

                if len(lambda_roots) < 3:
                    LAMBDA[n - 1] = max(lambda_roots)
                else:
                    LAMBDA[n - 1] = np.median(lambda_roots)

                DELTA[n - 1] = 1 / (4 * LAMBDA[n] * (1 - (BETA[n] * LAMBDA[n])))

            SIGMA[self.N] -= 0.007
            if SIGMA[self.N] < 1e-8:
                SIGMA[self.N] = 1e-8
            iter_count += 1

        ALPHA[0] = (1 - (2 * BETA[0] * LAMBDA[0])) / (
            dT * ((2 * LAMBDA[0]) * (1 - (BETA[0] * LAMBDA[0])))
        )

        for i in range(1, self.N):
            informed_orders[i] = (
                BETA[i] * (self.V_N[0] - np.cumsum(price_changes[:i])[i - 1])
            ) / self.N
            price_changes[i + 1] = LAMBDA[i + 1] * (
                informed_orders[i] + noise_orders[i]
            )

        if plot:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.N + 1),
                    y=np.cumsum(informed_orders),
                    mode="lines",
                    name="Informed Orders",
                    line=dict(color="blue", width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.N + 1),
                    y=np.cumsum(noise_orders),
                    mode="lines",
                    name="Noise Orders",
                    line=dict(color="orange", width=2),
                )
            )

            fig.update_layout(
                title="Order Sizes of Market Participants",
                xaxis_title="Time",
                yaxis_title="Order Size",
                hovermode="x unified",
                template="plotly_white",
                height=600,
            )

            fig.show()

        return {
            "ALPHA": ALPHA,
            "BETA": BETA,
            "DELTA": DELTA,
            "LAMBDA": LAMBDA,
            "SIGMA": SIGMA,
            "price_changes": price_changes,
            "informed_orders": informed_orders,
            "noise_orders": noise_orders,
        }
