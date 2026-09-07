import numpy as np
import pytest

from quant_mc_options.gbm import simulate_paths, simulate_terminal


def test_terminal_mean_matches_lognormal_theory():
    s0, mu, sigma, maturity = 100.0, 0.07, 0.2, 1.0
    rng = np.random.default_rng(1)
    s_t = simulate_terminal(s0, mu, sigma, maturity, 500_000, rng)

    theoretical_mean = s0 * np.exp(mu * maturity)
    assert s_t.mean() == pytest.approx(theoretical_mean, rel=0.01)

    theoretical_log_var = sigma**2 * maturity
    assert np.log(s_t).var() == pytest.approx(theoretical_log_var, rel=0.02)


def test_antithetic_pairs_have_invariant_product():
    # For paired draws z and -z, S_up * S_down = s0**2 * exp(2 * drift) exactly,
    # independent of the random draw -- a deterministic structural check.
    s0, mu, sigma, maturity = 50.0, 0.05, 0.3, 2.0
    rng = np.random.default_rng(2)
    s_t = simulate_terminal(s0, mu, sigma, maturity, 1000, rng, antithetic=True)

    half = len(s_t) // 2
    products = s_t[:half] * s_t[half : 2 * half]
    expected = s0**2 * np.exp(2 * (mu - 0.5 * sigma**2) * maturity)
    np.testing.assert_allclose(products, expected, rtol=1e-10)


def test_terminal_odd_path_count_is_handled():
    rng = np.random.default_rng(3)
    s_t = simulate_terminal(100.0, 0.05, 0.2, 1.0, 7, rng, antithetic=True)
    assert len(s_t) == 7


@pytest.mark.parametrize(
    ("s0", "sigma", "maturity", "n_paths"),
    [(0.0, 0.2, 1.0, 10), (100.0, -0.1, 1.0, 10), (100.0, 0.2, 0.0, 10), (100.0, 0.2, 1.0, 0)],
)
def test_terminal_rejects_invalid_inputs(s0, sigma, maturity, n_paths):
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        simulate_terminal(s0, 0.05, sigma, maturity, n_paths, rng)


def test_paths_shape_and_start_at_s0():
    rng = np.random.default_rng(5)
    paths = simulate_paths(100.0, 0.05, 0.2, 1.0, n_steps=50, n_paths=200, rng=rng)
    assert paths.shape == (200, 51)
    np.testing.assert_allclose(paths[:, 0], 100.0)


def test_paths_terminal_column_matches_terminal_distribution():
    # simulate_paths and simulate_terminal draw different random-number call
    # patterns (a grid of steps vs. one draw per path), so even with the same
    # seed they land on different realizations. Both use the exact GBM
    # transition, though, so their terminal distributions must agree.
    s0, mu, sigma, maturity = 100.0, 0.05, 0.25, 1.0
    path_terminal = simulate_paths(
        s0, mu, sigma, maturity, n_steps=60, n_paths=30_000, rng=np.random.default_rng(6)
    )[:, -1]
    direct_terminal = simulate_terminal(
        s0, mu, sigma, maturity, 30_000, rng=np.random.default_rng(7)
    )

    theoretical_mean = s0 * np.exp(mu * maturity)
    assert path_terminal.mean() == pytest.approx(theoretical_mean, rel=0.01)
    assert direct_terminal.mean() == pytest.approx(theoretical_mean, rel=0.01)
    assert path_terminal.mean() == pytest.approx(direct_terminal.mean(), rel=0.02)


@pytest.mark.parametrize("n_steps,n_paths", [(0, 10), (10, 0)])
def test_paths_rejects_invalid_step_or_path_count(n_steps, n_paths):
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        simulate_paths(100.0, 0.05, 0.2, 1.0, n_steps, n_paths, rng)
