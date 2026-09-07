"""Closed-form Black-Scholes-Merton pricing and Greeks.

Serves as the analytical benchmark that the Monte Carlo estimator in
:mod:`quant_mc_options.monte_carlo` is checked against. Supports a
continuous dividend yield ``q`` (set to 0 for a non-dividend-paying asset).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

VALID_OPTION_TYPES = ("call", "put")


def _validate_inputs(s0, k, sigma, maturity, option_type: str) -> None:
    if np.any(np.asarray(s0) <= 0):
        raise ValueError("s0 must be positive")
    if np.any(np.asarray(k) <= 0):
        raise ValueError("k (strike) must be positive")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be non-negative")
    if np.any(np.asarray(maturity) <= 0):
        raise ValueError("maturity (T) must be positive")
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {VALID_OPTION_TYPES}")


def _d1_d2(s0: ArrayLike, k: ArrayLike, r: float, sigma: ArrayLike, maturity: ArrayLike, q: float):
    s0 = np.asarray(s0, dtype=float)
    k = np.asarray(k, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    vol_sqrt_t = sigma * np.sqrt(maturity)
    d1 = (np.log(s0 / k) + (r - q + 0.5 * sigma**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(
    s0: ArrayLike,
    k: ArrayLike,
    r: float,
    sigma: ArrayLike,
    maturity: ArrayLike,
    option_type: str = "call",
    q: float = 0.0,
) -> NDArray:
    """European option price under Black-Scholes-Merton with dividend yield ``q``."""
    _validate_inputs(s0, k, sigma, maturity, option_type)
    s0 = np.asarray(s0, dtype=float)
    k = np.asarray(k, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    d1, d2 = _d1_d2(s0, k, r, sigma, maturity, q)
    disc_s = s0 * np.exp(-q * maturity)
    disc_k = k * np.exp(-r * maturity)
    if option_type == "call":
        price = disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    else:
        price = disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
    return float(price) if price.ndim == 0 else price


@dataclass(frozen=True)
class Greeks:
    """Per-unit sensitivities of the option price to each market input."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def bs_greeks(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    maturity: float,
    option_type: str = "call",
    q: float = 0.0,
) -> Greeks:
    """Closed-form Greeks. ``vega``, ``theta`` are per unit of volatility/year;
    ``theta`` is the rate of price change as calendar time passes (dV/dt = -dV/dT)."""
    _validate_inputs(s0, k, sigma, maturity, option_type)
    d1, d2 = _d1_d2(s0, k, r, sigma, maturity, q)
    d1, d2 = float(d1), float(d2)
    disc_q = np.exp(-q * maturity)
    disc_r = np.exp(-r * maturity)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (s0 * sigma * np.sqrt(maturity))
    vega = s0 * disc_q * pdf_d1 * np.sqrt(maturity)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -s0 * disc_q * pdf_d1 * sigma / (2 * np.sqrt(maturity))
            - r * k * disc_r * norm.cdf(d2)
            + q * s0 * disc_q * norm.cdf(d1)
        )
        rho = k * maturity * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (
            -s0 * disc_q * pdf_d1 * sigma / (2 * np.sqrt(maturity))
            + r * k * disc_r * norm.cdf(-d2)
            - q * s0 * disc_q * norm.cdf(-d1)
        )
        rho = -k * maturity * disc_r * norm.cdf(-d2)

    return Greeks(delta=float(delta), gamma=float(gamma), vega=float(vega),
                   theta=float(theta), rho=float(rho))
