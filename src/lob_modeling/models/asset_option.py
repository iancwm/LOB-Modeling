"""Asset or Nothing Option pricing model.

This module implements the asset or nothing option pricing model using a
binomial tree approach.
"""

from math import e

import numpy as np


def asset_or_nothing_call(
    s: float,
    n: int,
    k: int,
    T: float,
    K: float,
    F: float,
    SIGMA: float,
    r: float,
) -> np.ndarray:
    """Price an asset or nothing call option using binomial tree.

    An asset or nothing call option pays the asset value if the asset price
    exceeds the strike at expiry, otherwise it pays nothing.

    Args:
        s: Initial asset price.
        n: Number of partitions for expiry (n >> 0 for good estimate).
        k: Size of partitions.
        T: Time to expiry in years.
        K: Strike price.
        F: Payoff if S(t) > K for t <= T.
        SIGMA: Annual volatility.
        r: Risk-free interest rate.

    Returns:
        2D numpy array containing option prices at each node of the binomial
        tree.
    """
    u = e ** (SIGMA * (T / k))
    d = 1 / u
    BETA = e ** (-(r * T) / k)
    p = (1 + ((r * T) / k) - d) / (u - d)
    asset_option = np.zeros([n + 1, n + 1])

    for moves in range(n + 1):
        for downfactors in range(moves + 1):
            asset_option[downfactors, moves] = (
                s * (u ** (moves - downfactors)) * (d**downfactors)
            )

    for moves in range(n - 1, -1, -1):
        for downfactors in range(0, moves + 1):
            if asset_option[downfactors, moves] - K <= 0:
                asset_option[downfactors, moves] = BETA * (
                    (p * asset_option[downfactors, moves + 1])
                    + ((1 - p) * asset_option[downfactors + 1, moves + 1])
                )
            else:
                asset_option[downfactors, moves] = F

    return asset_option
