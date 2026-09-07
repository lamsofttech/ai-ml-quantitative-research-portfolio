# Monte Carlo Option Pricing and Risk Analysis

## Research question

How accurately, and with what quantified uncertainty, can vectorized Monte
Carlo simulation price a European option and estimate the risk of a position
under a geometric Brownian motion (GBM) model — and how does it compare to
the Black–Scholes closed-form solution? Monte Carlo is the tool of choice
once a payoff or risk measure has no closed form (path-dependent options,
multi-asset baskets, non-Gaussian risk models); this project validates the
methodology against the one case where an exact answer is known.

## Mathematical foundation

Under the risk-neutral measure with continuously compounded rate $r$ and
dividend yield $q$, the asset price follows

$$dS_t = (r-q)S_t\,dt + \sigma S_t\,dW_t,$$

which has the exact solution

$$S_T = S_0\exp\!\left[(r-q-\tfrac12\sigma^2)T + \sigma\sqrt{T}\,Z\right],\qquad Z\sim N(0,1).$$

A European call/put with strike $K$ and maturity $T$ has price
$V_0=e^{-rT}\,\mathbb E[\text{payoff}(S_T)]$. The Monte Carlo estimator draws
$n$ i.i.d. copies of $Z$, averages the discounted payoff, and reports a
95% confidence interval from the sample standard error, which shrinks at
the canonical Monte Carlo rate $O(1/\sqrt n)$. The closed-form Black–Scholes
price,

$$C_0 = S_0e^{-qT}N(d_1) - Ke^{-rT}N(d_2),\qquad
d_{1,2}=\frac{\ln(S_0/K)+(r-q\pm\tfrac12\sigma^2)T}{\sigma\sqrt T},$$

is the benchmark the simulator is checked against.

## Implementation

- `src/quant_mc_options/gbm.py` — exact (not Euler-discretized) GBM terminal
  and path simulation, vectorized over paths, with optional antithetic
  variates for variance reduction.
- `src/quant_mc_options/black_scholes.py` — closed-form price and Greeks
  (delta, gamma, vega, theta, rho).
- `src/quant_mc_options/monte_carlo.py` — Monte Carlo European option price
  with a statistically correct antithetic standard error (computed from
  paired-sample averages, not raw payoffs) and a 95% confidence interval.
- `src/quant_mc_options/risk.py` — empirical Value at Risk and Expected
  Shortfall (CVaR) from a simulated P&L sample, a closed-form GBM VaR for
  the underlying (for cross-checking the simulator), and Greeks estimated by
  finite-difference repricing under **common random numbers**, which cancels
  most Monte Carlo noise from the difference instead of adding independent
  sampling noise on top of it.
- `examples.py` — a fully reproducible (fixed-seed) demonstration: price
  comparison, a convergence plot showing error and CI half-width shrinking
  with path count, VaR/CVaR, Greeks, and a sample-path visualization.
- `app/streamlit_app.py` — interactive version of the same demonstration
  with adjustable market inputs, simulation size, and risk confidence level.

## Reproduce

```bash
python -m pip install -e ".[dev]"
pytest
python examples.py
```

Run the interactive app locally:

```bash
python -m pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Or via the project's own container (see [Dockerfile](Dockerfile) — this
project owns its dependencies and image rather than sharing the
repository-root environment):

```bash
docker build -t mc-option-pricing .
docker run --rm -p 8501:8501 mc-option-pricing
```

## Interpretation and limitations

- **What the results show:** the Monte Carlo estimator is unbiased and its
  reported uncertainty is well calibrated — the closed-form Black–Scholes
  price consistently falls inside the simulation's 95% CI, and the CI
  half-width shrinks at the theoretical $1/\sqrt n$ rate (see
  `artifacts/convergence.png` after running `examples.py`). Empirical VaR
  computed from simulated paths agrees with the closed-form GBM VaR, and
  Monte Carlo Greeks under common random numbers agree with the closed-form
  Greeks within simulation noise.
- **What the results do not show:** none of this is evidence about real
  option markets. GBM assumes constant volatility and no jumps, which is
  well known to conflict with the volatility smile/skew, fat tails, and
  volatility clustering observed empirically. This project validates a
  *simulation method*, not a *market model*; using it to price or hedge a
  real position would additionally require calibrating (or replacing) the
  volatility assumption against actual market data.
- **Model risk:** VaR and CVaR here are conditional on the GBM assumption.
  A model that fits calmer historical periods can understate risk in a
  regime shift — VaR is not a worst-case bound.
- **No trading claim:** nothing in this project estimates or claims trading
  profitability. It contains no transaction costs, slippage, or strategy
  logic — it prices and risk-analyzes a single static position.

## Evidence classification

- Data: entirely simulated GBM paths from user-chosen parameters; no
  historical or live market data is used.
- Result type: numerical/statistical validation (estimator accuracy,
  calibrated uncertainty), not a financial forecast or profitability claim.
- Leakage risk: not applicable — there is no historical dataset or model
  fitting step to leak into. Reproducibility is instead guaranteed by fixed
  random seeds throughout `examples.py` and the app.
- Serving and monitoring: the Streamlit app is containerized and includes a
  Docker healthcheck; there is no trained model to drift or retrain, so
  production model-monitoring (as used for the forecasting and credit-risk
  projects) does not apply here in the same sense.
