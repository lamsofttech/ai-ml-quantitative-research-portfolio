# Strike/maturity coverage study — November 16, 2026

This is simulated numerical validation, not market evidence. The study compares
Monte Carlo confidence intervals with the matching Black-Scholes value under the
same GBM assumptions.

## Design

- Strikes: 80, 100 and 120 with spot fixed at 100.
- Maturities: 0.25, 1 and 2 years.
- Payoffs/methods: call and put; plain and antithetic sampling.
- Budget: 4,000 terminal paths per replication; 200 independent replications
  for each of the 36 configurations.
- Root seed: 20261116. Exact stream construction, environment versions, source
  hashes and all row-level results are in [coverage-grid.json](coverage-grid.json).
- Coverage uncertainty: two-sided 95% Clopper-Pearson exact binomial interval.

## Results

| Payoff | Sampling | Observed coverage across 9 contracts | Largest absolute bias | RMSE range |
|---|---|---:|---:|---:|
| Call | Plain | 94.0%-98.0% | 0.035305 | 0.020399-0.422041 |
| Call | Antithetic | 94.0%-96.0% | 0.025204 | 0.018948-0.255912 |
| Put | Plain | 95.0%-98.0% | 0.013112 | 0.005532-0.289259 |
| Put | Antithetic | 94.0%-98.0% | 0.008949 | 0.006312-0.132306 |

The 36 observed coverage rates ranged from 188/200 (94.0%) to 196/200
(98.0%). Every configuration's exact binomial interval contains the nominal
95% target. The widest reported interval is approximately 0.8975 to 0.9686,
showing why a point estimate such as 94% must not be read as an exact calibration
probability. Inclusion of 95% is consistency evidence, not proof that every
future contract or model is calibrated.

Absolute RMSE is not directly comparable across all rows because option values
and payoff scales change with strike and maturity. In this fixed experiment,
antithetic sampling has a lower maximum RMSE and maximum absolute bias within
both payoff groups, but the study is not a timing benchmark and does not prove
that antithetic variates help every payoff family.

## Reproduce

Install project 02, check out code commit
`3bfa3c1bd45bea79b8dd3ef72c55f13c699b8c3f`, then run:

```bash
python -m quant_mc_options.convergence --study coverage-grid \
  --code-version 3bfa3c1bd45bea79b8dd3ef72c55f13c699b8c3f \
  --output projects/02-monte-carlo-option-pricing/results/coverage-grid.json
```

Two executions in Python 3.12.14 with NumPy 2.5.5 and SciPy 1.18.1 produced
byte-identical JSON. Cross-version bitwise equality is not promised. The
artifact SHA-256 is
`61d8ce315b1791b3664b98597c434a34c1ef78f945cc6a50c8d9adc4b0d2cf93`.

## Limitations

The simulator and benchmark share constant-volatility, lognormal GBM
assumptions. This validates numerical interval behavior under those assumptions;
it does not show that GBM fits market prices, does not include volatility smiles,
jumps, transaction costs or hedging error, and makes no profitability claim.
Two hundred replications still leave material binomial uncertainty. The normal
pricing interval can also be less accurate for sparse, highly skewed payoffs
outside this grid.

## Read for 10-15 minutes

Read this report, then trace `run_coverage_grid` and
`binomial_proportion_interval` in `src/quant_mc_options/convergence.py`.

Question: if 188 of 200 intervals contain the benchmark, why should we report
the interval 0.8975-0.9686 as well as the point estimate 0.94?
