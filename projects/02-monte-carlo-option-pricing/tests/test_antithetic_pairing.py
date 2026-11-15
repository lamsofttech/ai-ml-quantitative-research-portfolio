"""Deterministic checks for pair ordering, actual budgets and uncertainty."""

import numpy as np
import pytest
from scipy.stats import norm

from quant_mc_options.gbm import simulate_paths, simulate_terminal
from quant_mc_options.monte_carlo import price_european


@pytest.mark.parametrize("requested", [3, 4, 5, 7, 8])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_pricer_matches_manual_complete_pairs(requested, option_type):
    seed, s0, k, r, sigma, maturity = 19, 100.0, 102.0, 0.03, 0.25, 0.75
    pairs = (requested + 1) // 2
    z = np.random.default_rng(seed).standard_normal(pairs)
    drift = (r - 0.5 * sigma**2) * maturity
    up = s0 * np.exp(drift + sigma * np.sqrt(maturity) * z)
    down = s0 * np.exp(drift - sigma * np.sqrt(maturity) * z)
    sign = 1 if option_type == "call" else -1
    pair_payoffs = np.exp(-r * maturity) * (
        np.maximum(sign * (up - k), 0) + np.maximum(sign * (down - k), 0)
    ) / 2
    expected_price = pair_payoffs.mean()
    expected_se = pair_payoffs.std(ddof=1) / np.sqrt(pairs)
    result = price_european(
        s0, k, r, sigma, maturity, option_type, requested, np.random.default_rng(seed)
    )
    assert result.n_paths == 2 * pairs
    assert result.n_effective == pairs
    assert result.price == pytest.approx(expected_price)
    assert result.std_error == pytest.approx(expected_se)
    np.testing.assert_allclose(result.ci95, [
        expected_price - norm.ppf(0.975) * expected_se,
        expected_price + norm.ppf(0.975) * expected_se,
    ])


def test_one_antithetic_pair_cannot_estimate_standard_error():
    with pytest.raises(ValueError, match="at least two antithetic pairs"):
        price_european(100, 100, 0.03, 0.2, 1, "call", 2, np.random.default_rng(1))


@pytest.mark.parametrize("n_paths", [2, 3, 7])
def test_plain_sampling_preserves_exact_requested_budget(n_paths):
    result = price_european(
        100, 100, 0.03, 0.2, 1, "call", n_paths, np.random.default_rng(1), antithetic=False
    )
    assert result.n_paths == result.n_effective == n_paths
    assert np.isfinite(result.std_error)
    assert np.all(np.isfinite(result.ci95))


def test_low_level_odd_terminal_budget_keeps_shape_and_partial_pair_layout():
    terminal = simulate_terminal(100, 0.03, 0.2, 1, 7, np.random.default_rng(2))
    assert terminal.shape == (7,)
    # Four positive draws, then three negative partners; the fourth has no partner.
    np.testing.assert_allclose(terminal[:3] * terminal[4:], 100**2 * np.exp(2 * 0.01))


def test_low_level_odd_paths_keep_shape_and_partial_pair_layout():
    paths = simulate_paths(100, 0.03, 0.2, 1, 5, 7, np.random.default_rng(2))
    assert paths.shape == (7, 6)
    expected = 100**2 * np.exp(2 * 0.01 * np.linspace(0, 1, 6))
    np.testing.assert_allclose(paths[:3] * paths[4:], np.broadcast_to(expected, (3, 6)))
