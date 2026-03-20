"""Almgren-Chriss (2000) - Optimal execution of portfolio transactions.

This module implements deviations from the seminal Almgren & Chriss (2000) model
for optimal execution, including both dynamic programming and quadratic
programming solutions.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize


class AlmgrenChriss2000:
    """Almgren-Chriss model for optimal trade execution.

    This class implements the Almgren & Chriss (2000) framework for optimal
    execution of portfolio transactions, with both dynamic programming and
    quadratic programming solvers.

    Attributes:
        ALPHA: Power of temporary impact function.
        ETA: Linear coefficient of temporary impact function.
        GAMMA: Linear coefficient of permanent impact function.
        BETA: Power of permanent impact function.
        LAMBDA: Risk aversion measure.
        SIGMA: Annual volatility.
        EPSILON: Bid-ask spread and fees (fixed cost per share).
        MU: Expected drift term (set to 0 if no drift).
        N: Number of time steps/buckets.
        T: Expiry time in days (when holdings must be depleted).
        TAU: Time increment (sqrt(T/N)).
        KAPPA: Intermediate parameter for optimization.
        X: Initial number of shares held.
    """

    def __init__(
        self,
        ALPHA: float = 1,
        ETA: float = 5e-6,
        GAMMA: float = 5e-5,
        BETA: float = 1,
        LAMBDA: float = 0.00009,
        SIGMA: float = 0.495,
        EPSILON: float = 0.0625,
        MU: float = 0,
        N: int = 50,
        T: float = 0.025,
        X: int = 500,
    ) -> None:
        """Initialize the Almgren-Chriss model with specified parameters.

        Args:
            ALPHA: Power of temporary impact function. Defaults to 1.
            ETA: Linear coefficient of temporary impact. Defaults to 5e-6.
            GAMMA: Linear coefficient of permanent impact. Defaults to 5e-5.
            BETA: Power of permanent impact function. Defaults to 1.
            LAMBDA: Risk aversion measure. Defaults to 0.00009.
            SIGMA: Annual volatility. Defaults to 0.495.
            EPSILON: Bid-ask spread + fees ($/share). Defaults to 0.0625.
            MU: Expected drift term. Defaults to 0.
            N: Number of time steps. Defaults to 50.
            T: Expiry time in days. Defaults to 0.025.
            X: Initial number of shares. Defaults to 500.
        """
        self.ALPHA = float(ALPHA)
        self.ETA = float(ETA)
        self.GAMMA = float(GAMMA)
        self.BETA = float(BETA)
        self.LAMBDA = float(LAMBDA)
        self.SIGMA = float(SIGMA)
        self.EPSILON = float(EPSILON)
        self.MU = float(MU)
        self.N = int(N)
        self.T = float(T)
        self.TAU = math.sqrt(self.T / self.N)
        self.KAPPA = math.sqrt(
            (self.LAMBDA * (self.SIGMA**2))
            / (self.ETA * (1 + ((0.5 * self.GAMMA * self.TAU) / self.ETA)))
        )
        self.X = int(X)

    def temp_impact(self, x: float) -> float:
        """Calculate temporary market impact.

        Args:
            x: Trading rate.

        Returns:
            Temporary impact cost.
        """
        return self.ETA * (x**self.ALPHA)

    def perm_impact(self, x: float) -> float:
        """Calculate permanent market impact.

        Args:
            x: Trading rate.

        Returns:
            Permanent impact cost.
        """
        return self.GAMMA * (x**self.BETA)

    def hamiltonian(self, x: int, n: int) -> float:
        """Compute the Hamiltonian for the optimization problem.

        Args:
            x: Current inventory level.
            n: Number of shares to trade.

        Returns:
            Hamiltonian value for the given state.
        """
        eq = (
            self.LAMBDA * n * self.perm_impact(n / self.TAU)
            + self.LAMBDA * (x - n) * self.TAU * self.temp_impact(n / self.TAU)
            + (0.5 * (self.LAMBDA**2) * (self.SIGMA**2) * self.TAU * ((x - n) ** 2))
        )
        return eq

    def variance_IS(self, opt_sale: np.ndarray) -> float:
        """Calculate variance of implementation shortfall.

        Args:
            opt_sale: Array of optimal sales at each time step.

        Returns:
            Variance of the implementation shortfall.
        """
        variance_shortfall = 0
        step = -1
        while step < len(opt_sale) - 1:
            step += 1
            temp = (self.X - np.sum(opt_sale[0:step])) ** 2
            variance_shortfall += temp

        variance_shortfall = self.TAU * (self.SIGMA**2) * variance_shortfall
        return variance_shortfall

    def bellman_solve(
        self, plot: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Solve for optimal execution strategy using dynamic programming.

        Uses stochastic control (Bellman equation) to find the optimal trading
        strategy that minimizes expected cost.

        Args:
            plot: If True, creates an interactive Plotly visualization of
                the execution strategy. Defaults to False.

        Returns:
            A tuple containing:
                - value_func: Value function array (N x (X+1)).
                - opt_moves: Optimal trading moves array (N x (X+1)).
                - inventory: Inventory trajectory over time.
                - opt_sale: Optimal sales at each time step.
                - expected_shortfall: Expected implementation shortfall.
                - variance_shortfall: Variance of implementation shortfall.
        """
        value_func = np.zeros(shape=(self.N, self.X + 1), dtype="float64")
        opt_moves = np.zeros(shape=(self.N, self.X + 1), dtype="int")
        inventory = np.zeros(shape=(self.N, 1), dtype="int")
        inventory[0] = self.X
        opt_sale = []

        for x in range(self.X + 1):
            value_func[self.N - 1, x] = np.exp(x * self.temp_impact((x / self.TAU)))
            opt_moves[self.N - 1, x] = x

        for step in range(self.N - 2, -1, -1):
            for x in range(self.X + 1):
                best_value = value_func[step + 1, 0] * np.exp(
                    self.hamiltonian(x, x)
                )
                best_n = x

                for n in range(x):
                    current_value = value_func[step + 1, x - n] * np.exp(
                        self.hamiltonian(x, n)
                    )
                    if current_value < best_value:
                        best_value = current_value
                        best_n = n

                value_func[step, x] = best_value
                opt_moves[step, x] = best_n

        for step in range(1, self.N):
            inventory[step] = inventory[step - 1] - opt_moves[step, inventory[step - 1]]
            opt_sale.append(opt_moves[step - 1, inventory[step - 1]])

        expected_shortfall = (
            0.5 * self.GAMMA * (self.X**2)
            + self.EPSILON * np.sum(opt_sale)
            + ((self.ETA - 0.5 * self.GAMMA) / self.TAU)
            * np.sum([sale**2 for sale in opt_sale])
        )

        variance_shortfall = self.variance_IS(opt_sale)

        opt_sale = np.asarray(opt_sale)
        if plot:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(inventory))),
                    y=inventory,
                    mode="lines",
                    name="Inventory",
                    line=dict(color="blue", width=2),
                )
            )
            fig.update_layout(
                title="Optimal Execution - Dynamic Programming",
                xaxis_title="Trading Step",
                yaxis_title="Number of Shares",
                hovermode="x unified",
                template="plotly_white",
                height=600,
            )
            fig.show()

        return (
            value_func,
            opt_moves,
            inventory,
            opt_sale,
            expected_shortfall,
            variance_shortfall,
        )

    def basic_almgren(
        self, plot: bool = True
    ) -> Tuple[np.ndarray, List[int], float, float]:
        """Solve for optimal execution using quadratic programming.

        Implements the baseline quadratic cost model from Almgren & Chriss (2000)
        with linear impact costs.

        Args:
            plot: If True, creates an interactive Plotly visualization of
                the execution strategy. Defaults to True.

        Returns:
            A tuple containing:
                - opt_sale: Optimal sales at each time step.
                - inventory: Inventory trajectory over time.
                - expected_shortfall: Expected implementation shortfall.
                - variance_shortfall: Variance of implementation shortfall.
        """

        def optimal_objective(sale: np.ndarray) -> float:
            """Compute the quadratic objective function for optimization.

            Args:
                sale: Array of sales at each time step.

            Returns:
                Total objective cost (expected cost + risk penalty).
            """
            expected_cost = (
                0.5 * self.GAMMA * (self.X**2)
                - (
                    self.MU
                    * np.sum(
                        [
                            self.TAU * (self.X - inventory)
                            for inventory in np.cumsum(sale)
                        ]
                    )
                )
                + self.EPSILON * np.sum([abs(order) for order in sale])
                + ((self.ETA - 0.5 * self.GAMMA) / self.TAU)
                * np.sum([order**2 for order in sale])
            )
            variance_cost = self.variance_IS(sale)
            objective_cost = expected_cost + self.LAMBDA * variance_cost
            return objective_cost

        trades = np.zeros((self.N, 1))
        BOUNDS = tuple((0.0, self.X) for _ in range(len(trades)))
        CONSTRAINTS = {"type": "eq", "fun": lambda x: np.sum(x) - self.X}
        opt_sale = minimize(
            optimal_objective,
            trades.flatten(),
            method="SLSQP",
            bounds=BOUNDS,
            constraints=CONSTRAINTS,
        )
        opt_sale = np.array(opt_sale.x)
        inventory = [self.X]
        for t in range(len(opt_sale)):
            inventory.append(inventory[t] - opt_sale[t])

        expected_shortfall = (
            0.5 * self.GAMMA * (self.X**2)
            + self.EPSILON * np.sum(opt_sale)
            + ((self.ETA - 0.5 * self.GAMMA) / self.TAU)
            * np.sum([sale**2 for sale in opt_sale])
        )
        variance_shortfall = self.variance_IS(opt_sale)

        if plot:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(inventory))),
                    y=inventory,
                    mode="lines",
                    name="Inventory",
                    line=dict(color="blue", width=2),
                )
            )
            fig.update_layout(
                title="Optimal Execution - Quadratic Programming",
                xaxis_title="Trading Step",
                yaxis_title="Number of Shares",
                hovermode="x unified",
                template="plotly_white",
                height=600,
            )
            fig.show()

        return opt_sale, inventory, expected_shortfall, variance_shortfall

    def reinforcement_learn(self) -> None:
        """Placeholder for future reinforcement learning implementation.

        TODO: Implement multi-agent reinforcement learning approach based on
        Bao & Liu's recent paper.
        """
        pass
