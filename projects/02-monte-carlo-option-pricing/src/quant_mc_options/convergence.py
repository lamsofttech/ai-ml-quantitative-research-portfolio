"""Replicated, seeded convergence study against a known Black-Scholes price."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta

from .black_scholes import bs_price
from .monte_carlo import price_european

PARAMETERS = {"s0": 100.0, "k": 100.0, "r": 0.03, "sigma": 0.2,
              "maturity": 1.0, "q": 0.0}


def binomial_proportion_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return an equal-tailed Clopper-Pearson interval for a binomial rate."""
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials < 1
        or not 0 <= successes <= trials
    ):
        raise ValueError("successes and trials must be integers with 0 <= successes <= trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    alpha = 1.0 - confidence_level
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return lower, upper


def run_study(
    path_counts: tuple[int, ...] = (1000, 4000, 16000, 64000),
    replications: int = 50,
    seed: int = 20261114,
) -> dict:
    """Compare RMSE, standard error and empirical CI coverage across fresh streams.

    All path budgets are even so antithetic samples have complete pairs.
    A single error need not decrease with n; convergence concerns replication RMSE.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(replications, bool) or not isinstance(replications, int) or replications < 2:
        raise ValueError("replications must be an integer >= 2")
    if not path_counts or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 4 or n % 2 for n in path_counts
    ):
        raise ValueError("path_counts must contain even integers >= 4")
    if len(set(path_counts)) != len(path_counts):
        raise ValueError("path_counts must be unique")
    rows = []
    for option_index, option in enumerate(("call", "put")):
        benchmark = bs_price(**PARAMETERS, option_type=option)
        for antithetic in (False, True):
            for n in sorted(path_counts):
                estimates, errors, half_widths, covered = [], [], [], []
                for replication in range(replications):
                    stream = np.random.SeedSequence(
                        [seed, option_index, int(antithetic), n, replication]
                    )
                    result = price_european(
                        **PARAMETERS, option_type=option, n_paths=n,
                        rng=np.random.default_rng(stream), antithetic=antithetic,
                    )
                    estimates.append(result.price)
                    errors.append(result.std_error)
                    half_widths.append(result.half_width)
                    covered.append(result.ci95[0] <= benchmark <= result.ci95[1])
                difference = np.asarray(estimates) - benchmark
                rows.append({
                    "option_type": option, "antithetic": antithetic,
                    "n_paths": n, "n_effective": n // 2 if antithetic else n,
                    "normal_draws_per_replication": n // 2 if antithetic else n,
                    "black_scholes_price": benchmark,
                    "mean_price": float(np.mean(estimates)),
                    "bias": float(np.mean(difference)),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "empirical_price_sd": float(np.std(estimates, ddof=1)),
                    "mean_reported_se": float(np.mean(errors)),
                    "mean_ci95_half_width": float(np.mean(half_widths)),
                    "covered_replications": int(sum(covered)),
                    "ci95_coverage": float(np.mean(covered)),
                })
    return {
        "schema_version": 1, "data": "simulated risk-neutral GBM; no market observations",
        "parameters": PARAMETERS.copy(), "path_counts": sorted(path_counts),
        "replications": replications, "seed": seed,
        "rng": "NumPy default_rng(SeedSequence([seed, option_index, antithetic, n, replication]))",
        "option_index": {"call": 0, "put": 1},
        "evaluation_period": "not applicable; simulated one-year maturity",
        "rows": rows,
    }


def run_coverage_grid(
    strikes: tuple[float, ...] = (80.0, 100.0, 120.0),
    maturities: tuple[float, ...] = (0.25, 1.0, 2.0),
    n_paths: int = 4000,
    replications: int = 200,
    seed: int = 20261116,
) -> dict:
    """Measure interval coverage over an explicit strike/maturity grid.

    Coverage is a binomial proportion across independent replications. Each
    reported rate therefore includes a two-sided 95% Clopper-Pearson interval;
    it is not treated as an exactly known calibration probability.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(replications, bool) or not isinstance(replications, int) or replications < 2:
        raise ValueError("replications must be an integer >= 2")
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 4 or n_paths % 2:
        raise ValueError("n_paths must be an even integer >= 4")

    def normalize_grid(values: tuple[float, ...], name: str) -> tuple[float, ...]:
        if not values:
            raise ValueError(f"{name} must not be empty")
        normalized = tuple(sorted(float(value) for value in values))
        if any(not np.isfinite(value) or value <= 0.0 for value in normalized):
            raise ValueError(f"{name} must contain positive finite values")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must contain unique values")
        return normalized

    strikes = normalize_grid(strikes, "strikes")
    maturities = normalize_grid(maturities, "maturities")
    rows = []
    common = {key: value for key, value in PARAMETERS.items() if key not in {"k", "maturity"}}
    for option_index, option in enumerate(("call", "put")):
        for antithetic in (False, True):
            for strike_index, strike in enumerate(strikes):
                for maturity_index, maturity in enumerate(maturities):
                    benchmark = bs_price(
                        **common, k=strike, maturity=maturity, option_type=option
                    )
                    estimates, errors, covered = [], [], []
                    for replication in range(replications):
                        stream = np.random.SeedSequence([
                            seed,
                            option_index,
                            int(antithetic),
                            strike_index,
                            maturity_index,
                            n_paths,
                            replication,
                        ])
                        result = price_european(
                            **common,
                            k=strike,
                            maturity=maturity,
                            option_type=option,
                            n_paths=n_paths,
                            rng=np.random.default_rng(stream),
                            antithetic=antithetic,
                        )
                        estimates.append(result.price)
                        errors.append(result.std_error)
                        covered.append(result.ci95[0] <= benchmark <= result.ci95[1])
                    covered_count = int(sum(covered))
                    difference = np.asarray(estimates) - benchmark
                    rows.append({
                        "option_type": option,
                        "antithetic": antithetic,
                        "strike": strike,
                        "maturity": maturity,
                        "n_paths": n_paths,
                        "n_effective": n_paths // 2 if antithetic else n_paths,
                        "black_scholes_price": benchmark,
                        "mean_price": float(np.mean(estimates)),
                        "bias": float(np.mean(difference)),
                        "rmse": float(np.sqrt(np.mean(difference**2))),
                        "mean_reported_se": float(np.mean(errors)),
                        "covered_replications": covered_count,
                        "ci95_coverage": covered_count / replications,
                        "coverage_binomial_ci95": list(
                            binomial_proportion_interval(covered_count, replications)
                        ),
                    })
    return {
        "schema_version": 1,
        "study_type": "strike-maturity-coverage-grid",
        "data": "simulated risk-neutral GBM; no market observations",
        "parameters": common,
        "strikes": list(strikes),
        "maturities": list(maturities),
        "n_paths": n_paths,
        "replications": replications,
        "seed": seed,
        "coverage_interval": "two-sided 95% Clopper-Pearson exact binomial interval",
        "rng": (
            "NumPy default_rng(SeedSequence([seed, option_index, antithetic, "
            "strike_index, maturity_index, n_paths, replication]))"
        ),
        "option_index": {"call": 0, "put": 1},
        "evaluation_period": "not applicable; simulated maturity grid",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-version", required=True, help="Git commit containing this runner")
    parser.add_argument(
        "--study",
        choices=("convergence", "coverage-grid"),
        default="convergence",
        help="Reproduce the original convergence study or the strike/maturity coverage grid",
    )
    args = parser.parse_args()
    report = run_study() if args.study == "convergence" else run_coverage_grid()
    source = Path(__file__).parent
    report["provenance"] = {
        "code_version": args.code_version,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("convergence.py", "monte_carlo.py", "gbm.py", "black_scholes.py")
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "data_version": (
            "analytic-gbm-atm-european-v1"
            if args.study == "convergence"
            else "analytic-gbm-strike-maturity-grid-v1"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['rows'])} summary rows to {args.output}")


if __name__ == "__main__":
    main()
