"""Demonstration: Monte Carlo option pricing, Black-Scholes comparison, and risk analysis.

All results here use simulated market inputs (an assumed S0, r, sigma), not
fitted-to-market or live data. See README.md for what that does and does not
demonstrate.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from quant_mc_options import (  # noqa: E402
    bs_greeks,
    bs_price,
    gbm_parametric_var,
    historical_cvar,
    historical_var,
    mc_greeks,
    price_european,
    simulate_paths,
    simulate_terminal,
)

# A single fixed, published seed everywhere below: every number in this
# script -- and every figure -- is exactly reproducible from this file alone.
SEED = 2026

S0, K, R, SIGMA, MATURITY = 100.0, 100.0, 0.03, 0.20, 1.0


def price_and_compare() -> None:
    rng = np.random.default_rng(SEED)
    mc = price_european(S0, K, R, SIGMA, MATURITY, "call", 200_000, rng)
    bs = bs_price(S0, K, R, SIGMA, MATURITY, "call")
    print("== European call price ==")
    print(f"Black-Scholes (closed form): {bs:.4f}")
    print(f"Monte Carlo ({mc.n_paths:,} paths): {mc.price:.4f}")
    print(f"  95% CI: ({mc.ci95[0]:.4f}, {mc.ci95[1]:.4f})  std error: {mc.std_error:.4f}")
    diff = abs(mc.price - bs)
    print(f"  |MC - BS| = {diff:.4f} ({diff / mc.std_error:.2f} std errors)")


def convergence_plot(output: Path) -> None:
    bs = bs_price(S0, K, R, SIGMA, MATURITY, "call")
    path_counts = np.unique(np.logspace(2, 5.5, 15).astype(int))
    errors, half_widths = [], []
    for n in path_counts:
        rng = np.random.default_rng(SEED)
        result = price_european(S0, K, R, SIGMA, MATURITY, "call", int(n), rng)
        errors.append(abs(result.price - bs))
        half_widths.append(result.half_width)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(path_counts, errors, "o-", label="|MC price - Black-Scholes price|")
    ax.loglog(path_counts, half_widths, "--", label="95% CI half-width")
    ax.loglog(
        path_counts,
        half_widths[0] * np.sqrt(path_counts[0] / path_counts),
        ":",
        color="gray",
        label=r"theoretical $O(1/\sqrt{n})$",
    )
    ax.set(xlabel="number of simulated paths", ylabel="absolute error")
    ax.set_title("Monte Carlo convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=160)


def path_plot(output: Path) -> None:
    rng = np.random.default_rng(SEED)
    paths = simulate_paths(S0, R, SIGMA, MATURITY, n_steps=252, n_paths=40, rng=rng)
    grid = np.linspace(0, MATURITY, paths.shape[1])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, paths.T, linewidth=0.8, alpha=0.7)
    ax.axhline(K, color="black", linestyle="--", label="strike K")
    ax.set(xlabel="time (years)", ylabel="simulated asset price")
    ax.set_title("Simulated GBM sample paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "gbm_paths.png", dpi=160)


def risk_analysis() -> None:
    print("\n== Value at Risk (1-year horizon, long underlying position) ==")
    rng = np.random.default_rng(SEED)
    s_t = simulate_terminal(S0, R, SIGMA, MATURITY, 500_000, rng)
    pnl = s_t - S0
    for alpha in (0.95, 0.99):
        empirical = historical_var(pnl, alpha)
        parametric = gbm_parametric_var(S0, R, SIGMA, MATURITY, alpha)
        cvar = historical_cvar(pnl, alpha)
        print(
            f"alpha={alpha:.2f}: empirical VaR={empirical:.2f}  "
            f"closed-form VaR={parametric:.2f}  CVaR={cvar:.2f}"
        )


def greeks_comparison() -> None:
    print("\n== Greeks: Monte Carlo (common random numbers) vs. Black-Scholes ==")
    mc = mc_greeks(S0, K, R, SIGMA, MATURITY, "call", n_paths=300_000, seed=SEED)
    bs = bs_greeks(S0, K, R, SIGMA, MATURITY, "call")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        print(f"  {name:>6}: MC={getattr(mc, name): .4f}   BS={getattr(bs, name): .4f}")


def main() -> None:
    output = Path(__file__).parent / "artifacts"
    output.mkdir(exist_ok=True)

    price_and_compare()
    risk_analysis()
    greeks_comparison()
    convergence_plot(output)
    path_plot(output)
    print(f"\nFigures written to {output}")


if __name__ == "__main__":
    main()
