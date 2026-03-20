"""Criscuolo & Waehlbroeck (2014) - Stochastic volatility optimal execution.

This module implements the Criscuolo & Waehlbroeck (2014) model for optimal
execution with stochastic volatility, a time-dependent variant of the Heston
model with mean reversion.
"""

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize


class Criscuolo2014:
    """Criscuolo-Waehlbroeck stochastic volatility optimal execution model.

    This class implements the stochastic volatility optimal execution model
    from Criscuolo & Waehlbroeck (2014). The model captures realistic market
    conditions by incorporating stochastic volatility, market impact costs,
    and constrained optimization.

    Attributes:
        KAPPA: Exponential decay constant for volatility mean reversion.
        THETA: Long-run average volatility.
        GAMMA: Variance of volatility.
        RHO: Correlation between stock price and volatility.
        V_0: Initial volatility.
        r: Risk-free rate.
        T: Time to expiry in years.
        N: Number of discretized time periods.
        S_0: Initial stock price.
        ALPHA_INFINITY: Magnitude of alpha term (in basis points).
        MU_1: Speed at which alpha reaches maximum value.
        MU_2: Length until alpha approaches zero.
        ALPHA: Parameter for impact from share turnover.
        BETA: Parameter for impact from trader schedule/execution.
        XI: Additional parameter for market impact.
        VOL_RATIO: Ratio of initial volatility to sqrt(theta).
    """

    def __init__(
        self,
        KAPPA: float = 3,
        THETA: float = 0.01,
        GAMMA: float = 0.1,
        RHO: float = 0.0,
        V_0: float = 0.5,
        r: float = 0.05,
        T: float = 0.5,
        N: int = 4,
        S_0: float = 100,
        ALPHA_INFINITY: float = 0.4,
        MU_1: float = 0.4,
        MU_2: float = 0.8,
        ALPHA: float = 1.5,
        BETA: float = 0.3,
        XI: float = 365,
    ) -> None:
        """Initialize the Criscuolo-Waehlbroeck model with specified parameters.

        Args:
            KAPPA: Exponential decay constant. Defaults to 3.
            THETA: Long-run average volatility. Defaults to 0.01.
            GAMMA: Variance of volatility. Defaults to 0.1.
            RHO: Correlation between stock and volatility. Defaults to 0.0.
            V_0: Initial volatility. Defaults to 0.5.
            r: Risk-free rate. Defaults to 0.05.
            T: Time to expiry in years. Defaults to 0.5.
            N: Number of time periods. Defaults to 4.
            S_0: Initial stock price. Defaults to 100.
            ALPHA_INFINITY: Alpha magnitude in bps. Defaults to 0.4.
            MU_1: Speed to max alpha. Defaults to 0.4.
            MU_2: Length to alpha decay. Defaults to 0.8.
            ALPHA: Share turnover impact parameter. Defaults to 1.5.
            BETA: Execution schedule impact parameter. Defaults to 0.3.
            XI: Additional impact parameter. Defaults to 365.
        """
        self.KAPPA = float(KAPPA)
        self.THETA = float(THETA)
        self.GAMMA = float(GAMMA)
        self.RHO = float(RHO)
        self.V_0 = float(V_0)
        self.r = float(r)
        self.T = float(T)
        self.N = int(N)
        self.S_0 = float(S_0)
        self.ALPHA_INFINITY = float(ALPHA_INFINITY)
        self.MU_1 = float(MU_1)
        self.MU_2 = float(MU_2)
        self.ALPHA = float(ALPHA)
        self.BETA = float(BETA)
        self.XI = float(XI)
        self.VOL_RATIO = self.V_0 / math.sqrt(self.THETA)

    def optimal_execution(self) -> Any:
        """Compute optimal trading trajectory minimizing total cost.

        Uses scipy's SLSQP optimizer to find the trading trajectory that
        minimizes the total cost (alpha cost + impact cost) under stochastic
        volatility.

        Returns:
            Optimization result object containing optimal trajectory and
            share turnover.
        """

        def total_cost(trades: np.ndarray) -> float:
            """Calculate total execution cost for given trading trajectory.

            Args:
                trades: Flattened array of trade parameters with shape (2N,)
                    containing [share_turnover, participation] pairs.

            Returns:
                Total cost value. Returns 1e10 for invalid inputs.
            """
            try:
                trades = np.reshape(trades, (self.N, 2))
                share_turnover = [trade[0] for trade in trades]
                participation = [trade[1] for trade in trades]

                if any(x <= 0 for x in share_turnover) or any(
                    x <= 0 for x in participation
                ):
                    return 1e10
                if any(x > 1 for x in share_turnover) or any(x > 1 for x in participation):
                    return 1e10

                turnover_dif = np.diff(share_turnover, prepend=share_turnover[0])
                inst_time = np.cumsum(turnover_dif / participation)
                inst_time_diff = np.diff(inst_time, prepend=inst_time[0])

                if any(x <= 0 for x in inst_time_diff):
                    return 1e10

                alpha_cost = (
                    self.ALPHA_INFINITY
                    * (1 / trades[self.N - 1][0])
                    * np.sum(
                        [
                            (trades[k][1] * self.MU_2)
                            * (
                                math.exp(-1 * (inst_time[k - 1] / self.MU_2))
                                - math.exp(-1 * (inst_time[k] / self.MU_2))
                            )
                            + (
                                (self.MU_1 / (self.MU_1 + self.MU_2))
                                * (
                                    math.exp(
                                        -1 * inst_time[k] * ((1 / self.MU_1) + (1 / self.MU_2))
                                    )
                                    - math.exp(
                                        -1 * inst_time[k - 1]
                                        * ((1 / self.MU_1) + (1 / self.MU_2))
                                    )
                                )
                            )
                            for k in range(0, len(share_turnover) - 1)
                        ]
                    )
                )

                F_func = [
                    math.exp(-0.5 * (self.KAPPA * inst_time[k]))
                    * math.sqrt((self.VOL_RATIO**2) - 1 + math.exp(self.KAPPA * inst_time[k]))
                    for k in range(0, len(share_turnover) - 1)
                ]

                F_func_diff = np.diff(F_func, prepend=F_func[0])
                stochastic_vol = [
                    math.sqrt(self.THETA)
                    + ((3 * self.GAMMA) / (16 * self.KAPPA * math.sqrt(self.THETA)))
                    + (
                        (2 * math.sqrt(self.THETA)) / (self.KAPPA * inst_time_diff[k])
                    )
                    * (
                        math.log((1 + F_func[k]) / (1 + F_func[k - 1]))
                        - F_func_diff[k]
                    )
                    + (
                        (2 * math.sqrt(self.THETA) * (self.GAMMA**2))
                        / (16 * (self.KAPPA**2) * F_func_diff[k] * self.THETA)
                    )
                    * (
                        (
                            3 * math.log((1 + F_func[k]) / (1 + F_func[k - 1]))
                            + (
                                (F_func_diff[k] * 2 * (self.VOL_RATIO**2) - 3)
                                / (((self.VOL_RATIO**2) - 1) ** 2)
                            )
                            - (
                                ((self.VOL_RATIO**4) * F_func_diff[k])
                                / (F_func[k] * F_func[k - 1] * (((self.VOL_RATIO**2) - 1) ** 2))
                            )
                        )
                    )
                    for k in range(0, len(share_turnover) - 1)
                ]

                impact_cost = (self.XI / (self.ALPHA - 1)) * np.sum(
                    [
                        stochastic_vol[k]
                        * (trades[k][1] ** self.BETA)
                        * (
                            (trades[k][0] ** (self.ALPHA - 1))
                            - (trades[k - 1][0] ** (self.ALPHA - 1))
                        )
                        for k in range(0, len(share_turnover) - 1)
                    ]
                ) + (
                    (self.GAMMA * self.RHO) / (2 * trades[self.N - 1][0])
                ) * (
                    self.KAPPA * trades[self.N - 1][0]
                    - np.sum(share_turnover[0 : self.N - 1])
                )

                total_cost = impact_cost + alpha_cost

                if np.isnan(total_cost) or np.isinf(total_cost):
                    return 1e10

                return total_cost
            except Exception:
                return 1e10

        initial_share_turnover = np.ones(self.N) / self.N
        initial_participation = 0.2 * np.ones(self.N)
        initial_trades = np.column_stack([initial_share_turnover, initial_participation])

        opt_sale = minimize(
            total_cost,
            initial_trades.flatten(),
            method="SLSQP",
            bounds=[(0.01, 1.0) for _ in range(self.N * 2)],
            options={"maxiter": 1000, "ftol": 1e-6},
        )
        return opt_sale
