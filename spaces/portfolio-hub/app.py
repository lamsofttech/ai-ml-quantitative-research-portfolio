"""AI and Machine Learning for Quantitative Research — portfolio hub.

A Gradio front door to the portfolio: an overview of the identity, career
objective, and project roadmap, plus a live demo of the one project that
currently has an interactive interface (Monte Carlo option pricing). The
pricing/risk math itself is not reimplemented here — it is installed
straight from the project's own package (see requirements.txt), so this
Space can never drift from the tested, reviewed source of truth on GitHub.
"""

import matplotlib

matplotlib.use("Agg")

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from quant_mc_options import (
    bs_greeks,
    bs_price,
    gbm_parametric_var,
    historical_cvar,
    historical_var,
    mc_greeks,
    price_european,
    simulate_paths,
)

REPO_URL = "https://github.com/lamsofttech/ai-ml-quantitative-research-portfolio"

OVERVIEW_MD = f"""
# AI and Machine Learning for Quantitative Research

**Building intelligent, reproducible systems for quantitative modeling, financial analysis, forecasting, and risk management.**

I am pursuing an M.S. in Quantitative Methods with a concentration in Mathematical Finance, and my career
goal is to become an AI Engineer in Quantitative Research. This portfolio presents reproducible projects
combining machine learning, statistics, mathematical modeling, financial analysis, and production software
engineering. Each project progresses from a clearly defined research question to tested code, documented
results, containerized execution, and automated CI/CD — full source, tests, and READMEs are on
[GitHub]({REPO_URL}).

## Project roadmap

| Project | Focus | Status |
|---|---|---|
| [Numerical Methods for Quantitative Research]({REPO_URL}/tree/main/projects/01-numerical-methods) | Lagrange interpolation, error, stability, Python and R | Implemented foundation |
| [Monte Carlo Option Pricing and Risk Analysis]({REPO_URL}/tree/main/projects/02-monte-carlo-option-pricing) | GBM, Black–Scholes, confidence intervals, VaR — **live demo in the next tab** | Implemented |
| [Financial Time-Series Forecasting]({REPO_URL}/tree/main/projects/03-financial-time-series) | Baselines, walk-forward validation, leakage control | Planned |
| [Machine Learning for Credit-Risk Prediction]({REPO_URL}/tree/main/projects/04-credit-risk-ml) | Calibration, imbalance, fairness, explainability | Planned |
| [Financial News and SEC Filing Analysis]({REPO_URL}/tree/main/projects/05-financial-nlp) | Sentiment, topics, transformers | Planned |

## Important disclaimer

This portfolio is for education and research. It is not investment advice. Simulated or historical
performance does not guarantee future results. A research model is never presented here as a profitable
trading strategy without appropriate out-of-sample evidence, costs, and risk analysis.

**Author:** Lameck Nyakweba — M.S. Quantitative Methods, Mathematical Finance concentration;
aspiring AI Engineer in Quantitative Research.
"""

PRICING_NOTE = f"""
Every number below comes from the exact same `quant_mc_options` package used in the project's own test
suite — this Space installs it directly from the GitHub repository rather than re-implementing the math.
Inputs are simulated market parameters, not live or historical market data; see the
[project README]({REPO_URL}/tree/main/projects/02-monte-carlo-option-pricing) for the full methodology,
evidence classification, and limitations.
"""


def run_pricing(s0, k, r, sigma, maturity, option_type, n_paths, seed, alpha, antithetic):
    seed = int(seed)
    n_paths = int(n_paths)

    rng = np.random.default_rng(seed)
    mc = price_european(s0, k, r, sigma, maturity, option_type, n_paths, rng, antithetic=antithetic)
    bs = bs_price(s0, k, r, sigma, maturity, option_type)
    diff_se = abs(mc.price - bs) / mc.std_error

    price_md = (
        f"### Price\n\n"
        f"| | Value |\n|---|---|\n"
        f"| Black-Scholes (closed form) | {bs:.4f} |\n"
        f"| Monte Carlo ({mc.n_paths:,} paths) | {mc.price:.4f} |\n"
        f"| 95% confidence interval | ({mc.ci95[0]:.4f}, {mc.ci95[1]:.4f}) |\n"
        f"| Standard error | {mc.std_error:.4f} |\n"
        f"| abs(MC - BS) in std errors | {diff_se:.2f} |\n"
    )

    g_mc = mc_greeks(s0, k, r, sigma, maturity, option_type, n_paths=min(n_paths, 100_000), seed=seed)
    g_bs = bs_greeks(s0, k, r, sigma, maturity, option_type)
    greeks_md = "### Greeks: Monte Carlo (common random numbers) vs. Black-Scholes\n\n"
    greeks_md += "| Greek | Monte Carlo | Black-Scholes |\n|---|---|---|\n"
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        greeks_md += f"| {name} | {getattr(g_mc, name):.4f} | {getattr(g_bs, name):.4f} |\n"

    rng_risk = np.random.default_rng(seed)
    paths = simulate_paths(s0, r, sigma, maturity, n_steps=252, n_paths=min(n_paths, 50_000), rng=rng_risk)
    s_t = paths[:, -1]
    pnl = s_t - s0
    empirical = historical_var(pnl, alpha)
    parametric = gbm_parametric_var(s0, r, sigma, maturity, alpha)
    cvar = historical_cvar(pnl, alpha)
    risk_md = (
        f"### Value at Risk — long underlying position, {maturity:g}-year horizon\n\n"
        f"| | Value |\n|---|---|\n"
        f"| Empirical VaR ({alpha:.0%}) | {empirical:.2f} |\n"
        f"| Closed-form GBM VaR ({alpha:.0%}) | {parametric:.2f} |\n"
        f"| CVaR / Expected Shortfall ({alpha:.0%}) | {cvar:.2f} |\n\n"
        f"_Computed under the assumed GBM model (constant drift/volatility); does not capture jumps, "
        f"volatility clustering, or regime changes seen in real markets._"
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    display_paths = paths[: min(40, paths.shape[0])]
    grid = np.linspace(0, maturity, display_paths.shape[1])
    ax.plot(grid, display_paths.T, linewidth=0.8, alpha=0.7)
    ax.axhline(k, color="black", linestyle="--", label="strike K")
    ax.set(xlabel="time (years)", ylabel="simulated asset price")
    ax.set_title("Simulated GBM sample paths")
    ax.legend()
    fig.tight_layout()

    return price_md, greeks_md, risk_md, fig


with gr.Blocks(title="AI and ML for Quantitative Research") as demo:  # noqa: SIM117
    # Nested `with` is the idiomatic Gradio Blocks layout API, not a
    # candidate for collapsing into a single `with` statement.
    with gr.Tabs():
        with gr.Tab("Overview"):
            gr.Markdown(OVERVIEW_MD)

        with gr.Tab("Monte Carlo Option Pricing"):
            gr.Markdown("## Monte Carlo Option Pricing and Risk Analysis — live demo")
            gr.Markdown(PRICING_NOTE)
            with gr.Row():
                with gr.Column(scale=1):
                    s0 = gr.Number(label="Spot price S0", value=100.0, minimum=0.01)
                    k = gr.Number(label="Strike K", value=100.0, minimum=0.01)
                    r = gr.Slider(label="Risk-free rate r", minimum=-0.02, maximum=0.15, value=0.03, step=0.005)
                    sigma = gr.Slider(label="Volatility (sigma)", minimum=0.01, maximum=1.0, value=0.20, step=0.01)
                    maturity = gr.Slider(label="Maturity T (years)", minimum=0.05, maximum=3.0, value=1.0, step=0.05)
                    option_type = gr.Radio(["call", "put"], label="Option type", value="call")
                    n_paths = gr.Dropdown(
                        [1_000, 5_000, 20_000, 50_000, 100_000, 300_000],
                        label="Number of simulated paths",
                        value=50_000,
                    )
                    antithetic = gr.Checkbox(label="Antithetic variance reduction", value=True)
                    seed = gr.Number(label="Random seed", value=2026, precision=0)
                    alpha = gr.Dropdown([0.90, 0.95, 0.975, 0.99], label="VaR / CVaR confidence level", value=0.95)
                    run_btn = gr.Button("Run simulation", variant="primary")
                with gr.Column(scale=1):
                    price_out = gr.Markdown()
                    greeks_out = gr.Markdown()
                    risk_out = gr.Markdown()
                    plot_out = gr.Plot()

            inputs = [s0, k, r, sigma, maturity, option_type, n_paths, seed, alpha, antithetic]
            outputs = [price_out, greeks_out, risk_out, plot_out]
            run_btn.click(fn=run_pricing, inputs=inputs, outputs=outputs)
            demo.load(fn=run_pricing, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
