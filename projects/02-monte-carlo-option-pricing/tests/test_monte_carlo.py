import numpy as np
import pytest

from quant_mc_options.black_scholes import bs_price
from quant_mc_options.monte_carlo import price_european


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize(
    ("s0", "k", "r", "sigma", "maturity"),
    [(100.0, 100.0, 0.03, 0.2, 1.0), (100.0, 110.0, 0.03, 0.35, 0.5)],
)
def test_price_converges_to_black_scholes(option_type, s0, k, r, sigma, maturity):
    rng = np.random.default_rng(42)
    result = price_european(s0, k, r, sigma, maturity, option_type, 300_000, rng)
    benchmark = bs_price(s0, k, r, sigma, maturity, option_type)
    # 6 standard errors is an extremely wide, non-flaky band under a fixed
    # seed; it is here to catch a broken pricing formula, not to certify
    # convergence rate (see the walk-forward-style diagnostics in examples.py
    # for how the error shrinks with n_paths).
    assert abs(result.price - benchmark) < 6 * result.std_error


def test_ci95_contains_black_scholes_price():
    rng = np.random.default_rng(7)
    s0, k, r, sigma, maturity = 100.0, 100.0, 0.02, 0.25, 1.0
    result = price_european(s0, k, r, sigma, maturity, "call", 500_000, rng)
    benchmark = bs_price(s0, k, r, sigma, maturity, "call")
    lo, hi = result.ci95
    assert lo < benchmark < hi


def test_antithetic_variance_reduction_for_monotone_payoff():
    # max(S_T - K, 0) is non-decreasing in the driving normal z, so pairing
    # (z, -z) provably cannot increase estimator variance for a call payoff.
    s0, k, r, sigma, maturity = 100.0, 100.0, 0.03, 0.3, 1.0
    plain = price_european(
        s0, k, r, sigma, maturity, "call", 200_000, np.random.default_rng(1), antithetic=False
    )
    antithetic = price_european(
        s0, k, r, sigma, maturity, "call", 200_000, np.random.default_rng(1), antithetic=True
    )
    assert antithetic.std_error < plain.std_error


def test_standard_error_shrinks_with_more_paths():
    s0, k, r, sigma, maturity = 100.0, 100.0, 0.03, 0.2, 1.0
    small = price_european(s0, k, r, sigma, maturity, "call", 10_000, np.random.default_rng(5))
    large = price_european(s0, k, r, sigma, maturity, "call", 400_000, np.random.default_rng(5))
    assert large.std_error < small.std_error
    assert large.n_effective > small.n_effective


def test_rejects_invalid_option_type():
    with pytest.raises(ValueError):
        price_european(100.0, 100.0, 0.03, 0.2, 1.0, "exotic", 1000, np.random.default_rng(0))


def test_rejects_too_few_paths():
    with pytest.raises(ValueError):
        price_european(100.0, 100.0, 0.03, 0.2, 1.0, "call", 1, np.random.default_rng(0))
