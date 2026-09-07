"""Monte Carlo European option pricing with variance reduction and confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .gbm import simulate_terminal

VALID_OPTION_TYPES = ("call", "put")


def _payoff(s_t: np.ndarray, k: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(s_t - k, 0.0)
    if option_type == "put":
        return np.maximum(k - s_t, 0.0)
    raise ValueError(f"option_type must be one of {VALID_OPTION_TYPES}")


@dataclass(frozen=True)
class MCPriceResult:
    """Monte Carlo price estimate with its sampling uncertainty.

    ``ci95`` is a normal-approximation 95% confidence interval for the
    *estimator*, not a prediction interval for any single payoff outcome.
    It narrows at the standard Monte Carlo rate of ``1/sqrt(n_paths)``.
    """

    price: float
    std_error: float
    ci95: tuple[float, float]
    n_paths: int
    n_effective: int

    @property
    def half_width(self) -> float:
        return self.ci95[1] - self.price


def price_european(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    maturity: float,
    option_type: str,
    n_paths: int,
    rng: np.random.Generator,
    q: float = 0.0,
    antithetic: bool = True,
) -> MCPriceResult:
    """Price a European option by Monte Carlo simulation under the risk-neutral measure.

    Uses the risk-neutral drift ``mu = r - q`` so the discounted payoff is a
    martingale and the sample mean is an unbiased price estimator. With
    ``antithetic=True``, payoffs are paired (z, -z) and averaged per pair
    before computing the sample mean and standard error, which is the
    statistically correct way to estimate variance under antithetic sampling
    (the paired averages, not the raw payoffs, are the i.i.d. sample).
    """
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")

    mu = r - q
    s_t = simulate_terminal(s0, mu, sigma, maturity, n_paths, rng, antithetic=antithetic)
    discounted = np.exp(-r * maturity) * _payoff(s_t, k, option_type)

    if antithetic:
        half = len(discounted) // 2
        sample = (discounted[:half] + discounted[half : 2 * half]) / 2.0
    else:
        sample = discounted

    price = float(sample.mean())
    std_error = float(sample.std(ddof=1) / np.sqrt(len(sample)))
    z_975 = float(norm.ppf(0.975))
    ci95 = (price - z_975 * std_error, price + z_975 * std_error)

    return MCPriceResult(
        price=price,
        std_error=std_error,
        ci95=ci95,
        n_paths=n_paths,
        n_effective=len(sample),
    )
