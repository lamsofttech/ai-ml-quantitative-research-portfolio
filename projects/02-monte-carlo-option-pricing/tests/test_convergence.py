import numpy as np
import pytest

from quant_mc_options.convergence import (
    PARAMETERS,
    binomial_proportion_interval,
    run_coverage_grid,
    run_study,
)
from quant_mc_options.monte_carlo import price_european


def test_reproducible_and_independent_of_budget_order():
    first = run_study((200, 800), replications=4, seed=17)
    assert first == run_study((800, 200), replications=4, seed=17)
    assert first != run_study((200, 800), replications=4, seed=18)
    assert len(first["rows"]) == 8


def test_rmse_and_coverage_match_direct_pricer_replications():
    study = run_study((200,), replications=4, seed=17)
    for row in study["rows"]:
        option = row["option_type"]
        results = [price_european(
            **PARAMETERS, option_type=option, n_paths=200,
            rng=np.random.default_rng(np.random.SeedSequence(
                [17, int(option == "put"), int(row["antithetic"]), 200, replication]
            )), antithetic=row["antithetic"],
        ) for replication in range(4)]
        benchmark = row["black_scholes_price"]
        expected = np.sqrt(np.mean([(r.price - benchmark)**2 for r in results]))
        assert row["rmse"] == pytest.approx(expected)
        assert row["covered_replications"] == sum(
            r.ci95[0] <= benchmark <= r.ci95[1] for r in results
        )
        assert row["n_effective"] == (100 if row["antithetic"] else 200)


@pytest.mark.parametrize("kwargs", [
    {"path_counts": ()}, {"path_counts": (3,)}, {"path_counts": (5,)},
    {"path_counts": (4, 4)}, {"replications": 1}, {"replications": True},
    {"seed": -1}, {"seed": True},
])
def test_invalid_study_configuration(kwargs):
    with pytest.raises(ValueError):
        run_study(**kwargs)


@pytest.mark.parametrize(("successes", "trials", "expected"), [
    (0, 10, (0.0, 0.3084971078)),
    (5, 10, (0.1870860284, 0.8129139716)),
    (10, 10, (0.6915028922, 1.0)),
])
def test_exact_binomial_interval_handles_boundaries(successes, trials, expected):
    assert binomial_proportion_interval(successes, trials) == pytest.approx(expected)


@pytest.mark.parametrize(("successes", "trials", "confidence_level"), [
    (-1, 10, 0.95), (11, 10, 0.95), (1, 0, 0.95),
    (True, 10, 0.95), (1, True, 0.95), (1, 10, 0.0), (1, 10, 1.0),
])
def test_invalid_binomial_interval(successes, trials, confidence_level):
    with pytest.raises(ValueError):
        binomial_proportion_interval(successes, trials, confidence_level)


def test_coverage_grid_is_reproducible_and_order_independent():
    first = run_coverage_grid(
        strikes=(80.0, 100.0), maturities=(0.5, 1.0),
        n_paths=200, replications=4, seed=19,
    )
    reordered = run_coverage_grid(
        strikes=(100.0, 80.0), maturities=(1.0, 0.5),
        n_paths=200, replications=4, seed=19,
    )
    assert first == reordered
    assert first != run_coverage_grid(
        strikes=(80.0, 100.0), maturities=(0.5, 1.0),
        n_paths=200, replications=4, seed=20,
    )
    assert len(first["rows"]) == 16
    assert first["strikes"] == [80.0, 100.0]
    assert first["maturities"] == [0.5, 1.0]


def test_coverage_grid_reports_binomial_uncertainty_for_each_configuration():
    study = run_coverage_grid(
        strikes=(100.0,), maturities=(1.0,), n_paths=200, replications=4, seed=23
    )
    assert len(study["rows"]) == 4
    for row in study["rows"]:
        count = row["covered_replications"]
        assert row["ci95_coverage"] == count / 4
        assert row["coverage_binomial_ci95"] == pytest.approx(
            binomial_proportion_interval(count, 4)
        )
        assert row["n_effective"] == (100 if row["antithetic"] else 200)


@pytest.mark.parametrize("kwargs", [
    {"strikes": ()}, {"strikes": (0.0,)}, {"strikes": (100.0, 100.0)},
    {"maturities": ()}, {"maturities": (-1.0,)}, {"maturities": (1.0, 1.0)},
    {"n_paths": 3}, {"n_paths": 5}, {"n_paths": True},
    {"replications": 1}, {"replications": True}, {"seed": -1}, {"seed": True},
])
def test_invalid_coverage_grid_configuration(kwargs):
    with pytest.raises(ValueError):
        run_coverage_grid(**kwargs)
