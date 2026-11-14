import numpy as np
import pytest

from quant_mc_options.convergence import PARAMETERS, run_study
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
