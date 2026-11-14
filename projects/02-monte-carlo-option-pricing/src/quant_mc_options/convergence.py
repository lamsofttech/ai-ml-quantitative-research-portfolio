"""Replicated, seeded convergence study against a known Black-Scholes price."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from .black_scholes import bs_price
from .monte_carlo import price_european

PARAMETERS = {"s0": 100.0, "k": 100.0, "r": 0.03, "sigma": 0.2,
              "maturity": 1.0, "q": 0.0}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-version", required=True, help="Git commit containing this runner")
    args = parser.parse_args()
    report = run_study()
    source = Path(__file__).parent
    report["provenance"] = {
        "code_version": args.code_version,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("convergence.py", "monte_carlo.py", "gbm.py", "black_scholes.py")
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "data_version": "analytic-gbm-atm-european-v1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['rows'])} summary rows to {args.output}")


if __name__ == "__main__":
    main()
