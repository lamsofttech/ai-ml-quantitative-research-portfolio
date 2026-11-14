# Replicated convergence baseline — November 14, 2026

Simulated risk-neutral GBM, European call and put; no real market data.

Inputs: S0=100, K=100, r=0.03, q=0, sigma=0.20, maturity=1 year.
Budgets: 1,000/4,000/16,000/64,000 terminal paths; 50 independent runs per configuration.
Root seed: 20261114. Exact stream construction, environment and source hashes are in [convergence.json](convergence.json).

| Option | Method | Paths | RMSE | Mean reported SE | 95% CI coverage |
|---|---|---:|---:|---:|---:|
| call | plain | 1,000 | 0.413412 | 0.448147 | 47/50 |
| call | plain | 4,000 | 0.231151 | 0.223169 | 46/50 |
| call | plain | 16,000 | 0.117112 | 0.111721 | 47/50 |
| call | plain | 64,000 | 0.053631 | 0.055733 | 49/50 |
| call | antithetic | 1,000 | 0.376187 | 0.334079 | 47/50 |
| call | antithetic | 4,000 | 0.169696 | 0.167092 | 47/50 |
| call | antithetic | 16,000 | 0.087987 | 0.083284 | 46/50 |
| call | antithetic | 64,000 | 0.037176 | 0.041582 | 49/50 |
| put | plain | 1,000 | 0.274552 | 0.294824 | 49/50 |
| put | plain | 4,000 | 0.164939 | 0.147456 | 46/50 |
| put | plain | 16,000 | 0.070518 | 0.073980 | 46/50 |
| put | plain | 64,000 | 0.030564 | 0.036960 | 49/50 |
| put | antithetic | 1,000 | 0.191026 | 0.214175 | 48/50 |
| put | antithetic | 4,000 | 0.080403 | 0.106543 | 49/50 |
| put | antithetic | 16,000 | 0.048534 | 0.053494 | 47/50 |
| put | antithetic | 64,000 | 0.028768 | 0.026704 | 47/50 |

## Interpretation

In this fixed experiment, each fourfold path increase approximately halves the mean reported standard error. The 64,000-path call RMSE was 0.053631 for plain sampling and 0.037176 for antithetic sampling. Antithetic sampling reduced observed RMSE at every tested budget for both payoffs; this is not a speed benchmark or a claim about other payoff families.

Observed CI coverage ranges from 46/50 to 49/50 (92% to 98%). With only 50 replications, each success changes coverage by two percentage points; these observations do not establish general 95% calibration. RMSE and bias are also estimates across a finite number of runs. A single realization can move away from the benchmark as n grows.

All path counts are even, giving complete antithetic pairs and n/2 independent pair averages. Odd path counts are not validated by this experiment. Exact GBM terminal sampling removes time-discretization error, but not Monte Carlo sampling error or market-model risk. The Black-Scholes benchmark and simulator share assumptions, so agreement is numerical validation, not independent evidence that GBM fits markets.

## Reproduce

Install project 02, check out code commit `ba4e31c5212fba8a6a38d79ee785fc5bc022a537`, then run:

```bash
python -m quant_mc_options.convergence --code-version ba4e31c5212fba8a6a38d79ee785fc5bc022a537 --output convergence.json
```

Two local executions produced byte-identical JSON with the recorded environment. Cross-version bitwise equality is not promised.

## Read for 10-15 minutes

Read this table and the run_study function in ../src/quant_mc_options/convergence.py. RMSE measures observed error across independent runs, whereas reported SE estimates uncertainty from the simulated sample in each run.

Question: Why should quadrupling the path budget roughly halve SE, while a single absolute pricing error can still increase?
