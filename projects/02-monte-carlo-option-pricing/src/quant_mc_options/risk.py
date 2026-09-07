"""Value at Risk, Expected Shortfall, and Monte Carlo sensitivity analysis.

VaR and CVaR here are computed on a *simulated* profit-and-loss (P&L) sample,
so results only quantify risk relative to the assumed GBM model (constant
drift and volatility, no jumps or regime changes) -- see the project README
for how that assumption can fail in real markets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from .monte_carlo import price_european


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1), e.g. 0.95 for a 95% confidence level")


def historical_var(pnl: NDArray, alpha: float = 0.95) -> float:
    """Empirical Value at Risk from a simulated or historical P&L sample.

    Returns a positive number: the loss threshold exceeded with probability
    ``1 - alpha``. Uses the empirical quantile directly, so it makes no
    distributional assumption beyond what generated ``pnl``.
    """
    _validate_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size < 2:
        raise ValueError("pnl must contain at least 2 samples")
    losses = -pnl
    return float(np.quantile(losses, alpha))


def historical_cvar(pnl: NDArray, alpha: float = 0.95) -> float:
    """Expected Shortfall (CVaR): mean loss in the tail beyond the VaR threshold.

    CVaR is coherent (subadditive) where VaR is not, and it summarizes how
    severe losses are *beyond* the VaR cutoff, which VaR alone does not.
    """
    _validate_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float)
    losses = -pnl
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else float(var)


def gbm_parametric_var(
    s0: float, mu: float, sigma: float, horizon: float, alpha: float = 0.95
) -> float:
    """Closed-form VaR of a long position in the underlying itself under GBM.

    Because ``ln(S_T / S0)`` is exactly Normal under GBM, this position's VaR
    has a closed form, giving a benchmark for the Monte Carlo empirical VaR
    of the same position (see ``historical_var`` applied to underlying P&L).
    This closed form does **not** extend to nonlinear payoffs like options.
    """
    _validate_alpha(alpha)
    if s0 <= 0 or sigma < 0 or horizon <= 0:
        raise ValueError("s0 must be positive, sigma non-negative, horizon positive")
    z = norm.ppf(1 - alpha)
    s_quantile = s0 * np.exp((mu - 0.5 * sigma**2) * horizon + sigma * np.sqrt(horizon) * z)
    return float(s0 - s_quantile)


@dataclass(frozen=True)
class MCGreeks:
    """Finite-difference Greeks estimated by repricing under common random numbers.

    Reusing the identical random draws for the base case and every bumped
    case cancels most Monte Carlo sampling noise from the *difference*, which
    a naive finite difference on independent simulations would not do.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def mc_greeks(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    maturity: float,
    option_type: str,
    n_paths: int,
    seed: int,
    q: float = 0.0,
    ds: float | None = None,
    dsigma: float = 0.01,
    dt: float | None = None,
    dr: float = 1e-4,
) -> MCGreeks:
    """Estimate option Greeks by central-difference bump-and-reprice with common random numbers."""
    ds = ds if ds is not None else 0.01 * s0
    dt = dt if dt is not None else min(1e-3, 0.01 * maturity)

    def reprice(s0_: float, sigma_: float, maturity_: float, r_: float) -> float:
        rng = np.random.default_rng(seed)
        return price_european(s0_, k, r_, sigma_, maturity_, option_type, n_paths, rng, q=q).price

    base = reprice(s0, sigma, maturity, r)
    up_s, down_s = reprice(s0 + ds, sigma, maturity, r), reprice(s0 - ds, sigma, maturity, r)
    up_sigma = reprice(s0, sigma + dsigma, maturity, r)
    down_sigma = reprice(s0, sigma - dsigma, maturity, r)
    up_t = reprice(s0, sigma, maturity + dt, r)
    down_t = reprice(s0, sigma, maturity - dt, r)
    up_r = reprice(s0, sigma, maturity, r + dr)
    down_r = reprice(s0, sigma, maturity, r - dr)

    return MCGreeks(
        delta=(up_s - down_s) / (2 * ds),
        gamma=(up_s - 2 * base + down_s) / (ds**2),
        vega=(up_sigma - down_sigma) / (2 * dsigma),
        theta=-(up_t - down_t) / (2 * dt),
        rho=(up_r - down_r) / (2 * dr),
    )
