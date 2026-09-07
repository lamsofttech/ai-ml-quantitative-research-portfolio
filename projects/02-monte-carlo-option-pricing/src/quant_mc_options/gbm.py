"""Vectorized, reproducible geometric Brownian motion simulation.

Under the risk-neutral measure, the asset price solves
``dS = (r - q) S dt + sigma S dW``. Because GBM has a closed-form solution,
terminal values and full paths are drawn exactly (no Euler discretization
error) from

``S_T = S_0 * exp((mu - 0.5 * sigma**2) * T + sigma * sqrt(T) * Z)``,  Z ~ N(0, 1)

which is exact for any step size, so path granularity only affects
visualization, not simulation bias.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_market_params(s0: float, sigma: float, maturity: float) -> None:
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if maturity <= 0:
        raise ValueError("maturity (T) must be positive")


def simulate_terminal(
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> NDArray:
    """Draw terminal GBM values ``S_T`` using the exact lognormal solution.

    With ``antithetic=True`` each standard normal draw ``z`` is paired with
    ``-z``, halving Monte Carlo variance for symmetric payoffs at no extra
    simulation cost. ``n_paths`` is rounded up to an even number in that case
    so every draw has its antithetic partner.
    """
    _validate_market_params(s0, sigma, maturity)
    if n_paths < 1:
        raise ValueError("n_paths must be positive")

    if antithetic:
        half = -(-n_paths // 2)  # ceil division
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)

    drift = (mu - 0.5 * sigma**2) * maturity
    diffusion = sigma * np.sqrt(maturity) * z
    return s0 * np.exp(drift + diffusion)


def simulate_paths(
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> NDArray:
    """Simulate ``n_paths`` GBM sample paths on a grid of ``n_steps`` intervals.

    Returns an array of shape ``(n_paths, n_steps + 1)`` including the
    starting value at column 0. Uses the exact per-step lognormal transition,
    so this is unbiased regardless of how coarse the grid is.
    """
    _validate_market_params(s0, sigma, maturity)
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if n_paths < 1:
        raise ValueError("n_paths must be positive")

    dt = maturity / n_steps
    if antithetic:
        half = -(-n_paths // 2)
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))

    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return s0 * np.exp(log_paths)
