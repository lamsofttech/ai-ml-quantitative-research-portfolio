# Case study: validating Monte Carlo option-pricing uncertainty

## Employer summary

I built and validated a vectorized Monte Carlo system for European option
pricing and risk analysis. The research question was not simply whether a
simulation could produce a price, but whether its reported uncertainty behaved
as expected across repeated experiments, path budgets, strikes, maturities,
payoffs and sampling methods.

The simulator uses exact geometric Brownian motion (GBM) terminal sampling and
is benchmarked against the Black-Scholes closed-form value under matching
assumptions. In the convergence study, increasing the path budget from 1,000 to
64,000 reduced the mean reported standard error by approximately the expected
factor of eight for all four payoff/method combinations. In the broader
36-configuration coverage study, observed 95% confidence-interval coverage
ranged from 94% to 98%, and every exact binomial interval for the unknown
coverage probability included the nominal 95% target.

This is evidence that the numerical estimator and its uncertainty calculation
behave consistently in the tested GBM setting. It is not evidence that GBM fits
real option markets, that the system predicts prices, or that it produces a
profitable trading strategy.

## Research question

How accurately, and with what quantified uncertainty, can Monte Carlo
simulation price European calls and puts under GBM, and does its confidence
interval behave consistently across sampling budgets and contract settings?

This matters in quantitative research because Monte Carlo methods are often
used when no closed form is available. Before trusting a more complex payoff,
the implementation should first be checked in a setting where the correct price
is known and repeated simulations can expose bias, sampling variability and
misreported standard errors.

## Design decisions

| Decision | Reason |
|---|---|
| Use exact GBM terminal sampling | Removes Euler time-discretization error, isolating Monte Carlo sampling error. |
| Benchmark against Black-Scholes | Provides a known value under the same assumptions for numerical validation. |
| Run independent replications | A single seeded result cannot estimate RMSE or empirical confidence-interval coverage. |
| Increase paths by factors of four | Standard error should scale as `1/sqrt(n)`, so each increase should approximately halve it. |
| Compare plain and antithetic sampling at equal path budgets | Tests variance reduction without silently giving one method more simulated terminal values. |
| Compute antithetic SE from pair averages | The positive and negative draws are correlated; treating all payoffs as independent would misstate uncertainty. |
| Add a strike/maturity grid | Tests whether interval behavior extends beyond the at-the-money, one-year baseline. |
| Report exact binomial intervals for coverage | A finite coverage count is uncertain and should not be presented as an exactly known probability. |

The experiment runner is in
[convergence.py](src/quant_mc_options/convergence.py), and the pricing engine is
in [monte_carlo.py](src/quant_mc_options/monte_carlo.py). Regression coverage
includes the [study tests](tests/test_convergence.py) and the
[antithetic-pairing tests](tests/test_antithetic_pairing.py).

## Evidence 1: convergence with path count

The [replicated convergence report](results/README.md) and
[machine-readable artifact](results/convergence.json) compare calls and puts,
plain and antithetic sampling, and path budgets of 1,000, 4,000, 16,000 and
64,000. Each configuration uses 50 independent streams.

| Payoff | Sampling | Mean SE at 1,000 paths | Mean SE at 64,000 paths | RMSE at 1,000 paths | RMSE at 64,000 paths |
|---|---|---:|---:|---:|---:|
| Call | Plain | 0.448147 | 0.055733 | 0.413412 | 0.053631 |
| Call | Antithetic | 0.334079 | 0.041582 | 0.376187 | 0.037176 |
| Put | Plain | 0.294824 | 0.036960 | 0.274552 | 0.030564 |
| Put | Antithetic | 0.214175 | 0.026704 | 0.191026 | 0.028768 |

A 64-fold path increase should reduce standard error by `sqrt(64) = 8`.
The recorded endpoints closely follow that relationship. Observed coverage in
this 50-replication study ranged from 46/50 to 49/50. Because one success
changes the rate by two percentage points, those coverage values are useful
diagnostics but are too coarse to establish general calibration.

Antithetic sampling had lower observed RMSE at every tested budget for both
payoffs in this fixed experiment. That does not prove it improves every payoff,
and no wall-clock timing comparison was performed.

## Evidence 2: coverage across contracts

The [strike/maturity coverage report](results/coverage-grid.md) and
[artifact](results/coverage-grid.json) evaluate 36 configurations:

- strikes 80, 100 and 120 with spot fixed at 100;
- maturities 0.25, 1 and 2 years;
- calls and puts;
- plain and antithetic sampling;
- 4,000 terminal paths and 200 independent replications per configuration.

Observed coverage ranged from 188/200 (`94%`) to 196/200 (`98%`). Every
two-sided 95% Clopper-Pearson interval contained the nominal `95%` target. The
widest recorded interval was approximately `0.8975–0.9686`, which shows why a
point estimate such as `0.94` must not be treated as the exact underlying
coverage probability.

Within this fixed grid, antithetic sampling had a lower maximum RMSE and lower
maximum absolute bias for both payoff groups. Absolute RMSE is not directly
comparable across every contract because their prices and payoff scales differ.

## Reproduce the evidence

For the convergence baseline, check out
[`ba4e31c5212fba8a6a38d79ee785fc5bc022a537`](https://github.com/lamsofttech/ai-ml-quantitative-research-portfolio/commit/ba4e31c5212fba8a6a38d79ee785fc5bc022a537):

```bash
python -m pip install -e "projects/02-monte-carlo-option-pricing[dev]"
python -m pytest projects/02-monte-carlo-option-pricing/tests
python -m quant_mc_options.convergence \
  --code-version ba4e31c5212fba8a6a38d79ee785fc5bc022a537 \
  --output projects/02-monte-carlo-option-pricing/results/convergence.json
```

For the coverage grid, check out
[`3bfa3c1bd45bea79b8dd3ef72c55f13c699b8c3f`](https://github.com/lamsofttech/ai-ml-quantitative-research-portfolio/commit/3bfa3c1bd45bea79b8dd3ef72c55f13c699b8c3f):

```bash
python -m quant_mc_options.convergence --study coverage-grid \
  --code-version 3bfa3c1bd45bea79b8dd3ef72c55f13c699b8c3f \
  --output projects/02-monte-carlo-option-pricing/results/coverage-grid.json
python scripts/validate_case_study_links.py
```

Each recorded artifact was reproduced twice byte-for-byte in its documented
Python environment. Cross-version floating-point and random-number-generator
bitwise equality is not promised. See the [project README](README.md) for the
full mathematical foundation and the [Streamlit interface](app/streamlit_app.py)
and [container definition](Dockerfile) for serving details.

## Limitations

- Black-Scholes and the simulator share constant-volatility, lognormal GBM
  assumptions. Agreement validates the implementation under those assumptions;
  it is not independent evidence that the market follows GBM.
- The experiments exclude volatility smiles, jumps, stochastic volatility,
  transaction costs, liquidity, hedging error and regime shifts.
- Coverage is estimated from finite replications. Exact binomial intervals
  quantify that uncertainty but do not prove calibration for untested contracts.
- Antithetic variance reduction is payoff-dependent, and this study is not a
  speed benchmark.
- The project contains no strategy, backtest or profitability claim.

## Interview talking points

1. **Why should quadrupling paths halve standard error?** Independent Monte
   Carlo error scales as `1/sqrt(n)`; multiplying `n` by four divides SE by two.
2. **Why can one realized error increase at a larger path budget?** The rate is
   a distributional statement, not a promise that every random draw improves
   monotonically.
3. **Why use pair averages for antithetic SE?** The two payoffs in a pair share
   one underlying random draw and are correlated; the independent units are the
   pair averages.
4. **Why attach an interval to empirical coverage?** `188/200` is a finite
   binomial observation, not exact knowledge that future coverage equals 94%.
5. **What would you test next?** Add timing-normalized comparisons and a payoff
   where Black-Scholes is unavailable, while retaining an independently
   testable benchmark or convergence reference.

## Skills demonstrated

Vectorized simulation; mathematical finance; uncertainty quantification;
variance reduction; independent experimental replication; exact binomial
inference; numerical benchmarking; reproducible random streams; testing;
containerized serving; and evidence-aware model-risk communication.
