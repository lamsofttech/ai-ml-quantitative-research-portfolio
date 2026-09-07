import numpy as np
import pytest
from scipy.stats import norm

from quant_mc_options.black_scholes import bs_greeks
from quant_mc_options.gbm import simulate_terminal
from quant_mc_options.risk import (
    gbm_parametric_var,
    historical_cvar,
    historical_var,
    mc_greeks,
)


def test_cvar_is_at_least_var():
    rng = np.random.default_rng(1)
    pnl = rng.standard_normal(50_000) * 10
    var = historical_var(pnl, alpha=0.95)
    cvar = historical_cvar(pnl, alpha=0.95)
    assert cvar >= var


def test_historical_var_matches_known_normal_quantile():
    rng = np.random.default_rng(2)
    sigma = 20.0
    pnl = rng.normal(loc=0.0, scale=sigma, size=1_000_000)
    var_95 = historical_var(pnl, alpha=0.95)
    theoretical = -norm.ppf(0.05) * sigma
    assert var_95 == pytest.approx(theoretical, rel=0.02)


def test_gbm_parametric_var_matches_empirical_var():
    s0, mu, sigma, horizon, alpha = 100.0, 0.05, 0.25, 1.0, 0.95
    rng = np.random.default_rng(3)
    s_t = simulate_terminal(s0, mu, sigma, horizon, 500_000, rng)
    empirical = historical_var(s_t - s0, alpha=alpha)
    parametric = gbm_parametric_var(s0, mu, sigma, horizon, alpha=alpha)
    # Quantile estimation is noisier than mean estimation, hence the wider
    # relative tolerance compared to the price-convergence tests.
    assert empirical == pytest.approx(parametric, rel=0.05)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_var_functions_reject_invalid_alpha(alpha):
    pnl = np.array([1.0, -2.0, 3.0, -4.0])
    with pytest.raises(ValueError):
        historical_var(pnl, alpha=alpha)
    with pytest.raises(ValueError):
        gbm_parametric_var(100.0, 0.05, 0.2, 1.0, alpha=alpha)


def test_historical_var_requires_enough_samples():
    with pytest.raises(ValueError):
        historical_var(np.array([1.0]))


def test_mc_greeks_match_black_scholes_closed_form():
    s0, k, r, sigma, maturity = 100.0, 100.0, 0.03, 0.2, 1.0
    mc = mc_greeks(s0, k, r, sigma, maturity, "call", n_paths=300_000, seed=11)
    bs = bs_greeks(s0, k, r, sigma, maturity, "call")

    assert mc.delta == pytest.approx(bs.delta, abs=0.02)
    assert mc.vega == pytest.approx(bs.vega, abs=1.0)
    assert mc.gamma == pytest.approx(bs.gamma, abs=0.01)
    assert mc.rho == pytest.approx(bs.rho, abs=1.0)
    assert mc.theta == pytest.approx(bs.theta, abs=1.0)
