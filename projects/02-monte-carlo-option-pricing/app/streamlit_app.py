"""Interactive Monte Carlo option pricing and risk analysis app.

Run locally with:
    streamlit run app/streamlit_app.py
or via the project's Dockerfile (see README.md).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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

st.set_page_config(page_title="Monte Carlo Option Pricing", layout="wide")


@st.cache_data(show_spinner=False)
def compute_price(s0, k, r, sigma, maturity, option_type, n_paths, seed, antithetic):
    rng = np.random.default_rng(seed)
    mc = price_european(s0, k, r, sigma, maturity, option_type, n_paths, rng, antithetic=antithetic)
    bs = bs_price(s0, k, r, sigma, maturity, option_type)
    return mc, bs


@st.cache_data(show_spinner=False)
def compute_greeks(s0, k, r, sigma, maturity, option_type, n_paths, seed):
    mc = mc_greeks(s0, k, r, sigma, maturity, option_type, n_paths=n_paths, seed=seed)
    bs = bs_greeks(s0, k, r, sigma, maturity, option_type)
    return mc, bs


@st.cache_data(show_spinner=False)
def compute_risk(s0, r, sigma, horizon, alpha, n_paths, seed):
    rng = np.random.default_rng(seed)
    s_t = simulate_terminal(s0, r, sigma, horizon, n_paths, rng)
    pnl = s_t - s0
    return {
        "empirical_var": historical_var(pnl, alpha),
        "parametric_var": gbm_parametric_var(s0, r, sigma, horizon, alpha),
        "cvar": historical_cvar(pnl, alpha),
        "terminal": s_t,
    }


@st.cache_data(show_spinner=False)
def compute_paths(s0, r, sigma, maturity, n_steps, n_display, seed):
    rng = np.random.default_rng(seed)
    return simulate_paths(s0, r, sigma, maturity, n_steps, n_display, rng)


def main() -> None:
    st.title("Monte Carlo Option Pricing and Risk Analysis")
    st.caption(
        "Simulated market inputs, not live or historical market data. "
        "Educational demonstration of pricing and risk methodology, not investment advice."
    )

    with st.sidebar:
        st.header("Market and contract inputs")
        s0 = st.number_input("Spot price S0", min_value=0.01, value=100.0, step=1.0)
        k = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
        r = st.slider("Risk-free rate r", min_value=-0.02, max_value=0.15, value=0.03, step=0.005)
        sigma = st.slider(
            "Volatility (sigma)", min_value=0.01, max_value=1.0, value=0.20, step=0.01
        )
        maturity = st.slider(
            "Maturity T (years)", min_value=0.05, max_value=3.0, value=1.0, step=0.05
        )
        option_type = st.selectbox("Option type", ["call", "put"])

        st.header("Simulation settings")
        n_paths = st.select_slider(
            "Number of simulated paths",
            options=[1_000, 5_000, 20_000, 50_000, 100_000, 300_000],
            value=50_000,
        )
        antithetic = st.checkbox("Antithetic variance reduction", value=True)
        seed = st.number_input("Random seed", min_value=0, value=2026, step=1)

        st.header("Risk settings")
        alpha = st.select_slider(
            "VaR / CVaR confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.95
        )

    mc_result, bs_val = compute_price(
        s0, k, r, sigma, maturity, option_type, n_paths, seed, antithetic
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Black-Scholes price", f"{bs_val:.4f}")
    col2.metric(
        "Monte Carlo price",
        f"{mc_result.price:.4f}",
        delta=f"{mc_result.price - bs_val:+.4f} vs. Black-Scholes",
    )
    col3.metric("95% CI half-width", f"{mc_result.half_width:.4f}")
    st.caption(
        f"95% confidence interval: ({mc_result.ci95[0]:.4f}, {mc_result.ci95[1]:.4f}) "
        f"from {mc_result.n_effective:,} effective samples. "
        "This interval quantifies simulation (sampling) error only, not model risk."
    )

    st.subheader("Greeks: Monte Carlo (common random numbers) vs. Black-Scholes")
    mc_g, bs_g = compute_greeks(s0, k, r, sigma, maturity, option_type, min(n_paths, 100_000), seed)
    greeks_df = pd.DataFrame(
        {
            "Monte Carlo": [mc_g.delta, mc_g.gamma, mc_g.vega, mc_g.theta, mc_g.rho],
            "Black-Scholes": [bs_g.delta, bs_g.gamma, bs_g.vega, bs_g.theta, bs_g.rho],
        },
        index=["delta", "gamma", "vega", "theta", "rho"],
    )
    st.dataframe(greeks_df.style.format("{:.4f}"), width="stretch")

    st.subheader(f"Value at Risk — long underlying position, {maturity:g}-year horizon")
    risk = compute_risk(s0, r, sigma, maturity, alpha, n_paths, seed)
    rcol1, rcol2, rcol3 = st.columns(3)
    rcol1.metric(f"Empirical VaR ({alpha:.0%})", f"{risk['empirical_var']:.2f}")
    rcol2.metric("Closed-form GBM VaR", f"{risk['parametric_var']:.2f}")
    rcol3.metric(f"CVaR / Expected Shortfall ({alpha:.0%})", f"{risk['cvar']:.2f}")
    st.caption(
        "VaR and CVaR are computed under the assumed GBM model (constant drift/volatility). "
        "They do not capture jumps, volatility clustering, or regime changes seen in real markets."
    )

    hist_df = pd.DataFrame({"terminal price": risk["terminal"]})
    st.bar_chart(np.histogram(hist_df["terminal price"], bins=60)[0])

    st.subheader("Simulated sample paths")
    n_display = st.slider("Paths to display", min_value=5, max_value=200, value=40)
    paths = compute_paths(s0, r, sigma, maturity, 252, n_display, seed)
    grid = np.linspace(0, maturity, paths.shape[1])
    chart_df = pd.DataFrame(paths.T, index=grid)
    st.line_chart(chart_df, height=320)

    st.divider()
    st.caption(
        "Evidence classification: all figures use simulated GBM paths with user-chosen "
        "parameters, not fitted or historical market data. Monte Carlo estimates are "
        "reported with confidence intervals; results are numerical/statistical, not a "
        "claim about the profitability of any trading strategy."
    )


if __name__ == "__main__":
    main()
