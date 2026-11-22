# Degree-16 Lebesgue local peak refinement — November 22, 2026

This deterministic follow-up estimates how much the 1601-point uniform grid
still understates each degree-16 Lebesgue peak. Starting from the largest grid
value for each node design, it searches the two neighboring grid cells with an
80-decimal-digit bounded golden-section calculation. No historical, private or
random data is used.

## Decision rule

The residual grid bias is called operationally small when the locally refined
estimate is no more than 0.05% above the 1601-point estimate. The node-design
conclusion remains stable when the refined equidistant estimate is more than
100 times the refined Chebyshev-Lobatto estimate. These are transparent
sensitivity criteria, not mathematical proof thresholds.

Each local search stops when its bracket is at most $10^{-30}$ wide or after
300 iterations. Both searches reached the width tolerance after 132 iterations.

## Results

| Node design | 1601-point estimate | Locally refined estimate | Residual | Relative residual | Refined location |
|---|---:|---:|---:|---:|---:|
| Equidistant | 934.421374 | 934.534111 | 0.112738 | 0.01206% | 0.96672795 |
| Chebyshev-Lobatto | 2.724667 | 2.724709 | 0.00004156 | 0.001525% | -0.09794287 |

Both relative residuals are below the predefined 0.05% threshold. The refined
equidistant-to-Chebyshev ratio is 342.98, so the conclusion remains unchanged:
at degree 16, equidistant interpolation is far more sensitive to errors in the
supplied values than Chebyshev-Lobatto interpolation.

The local refinement is useful because it separates two ideas. A uniform grid
can miss the top of a peak between points, while a bounded local search can
estimate that top more closely. The refined result is still a numerical
estimate, not a symbolic proof that no larger peak exists elsewhere.

## Reproduce

Check out implementation commit
`68a8c0618e99edf058ad34fd5f37edf45bf7c6e2`, install project 01 and run:

```bash
python -m quant_numerical.stability \
  --code-version 68a8c0618e99edf058ad34fd5f37edf45bf7c6e2 \
  --output projects/01-numerical-methods/results/stability.json \
  --refinement-output projects/01-numerical-methods/results/grid-refinement.json \
  --peak-output projects/01-numerical-methods/results/peak-refinement.json
```

Two executions on Python 3.12.14 with NumPy 2.3.5 produced byte-identical JSON.
The peak-refinement artifact SHA-256 is
`ec926709dcfe1aa532c18335c1c5485697fc5b4271fe300f06fd3ad587d6264c`.
Exact row values, criteria, source hashes and environment are recorded in
[peak-refinement.json](peak-refinement.json).

## Limitations

Golden-section search assumes the selected neighboring-cell bracket contains a
single relevant local peak. The 1601-point grid supplies that bracket, so the
method does not independently prove that another unselected interval lacks a
higher maximum. Only degree 16, two symmetric node designs and one interval are
checked. Interior Chebyshev-Lobatto nodes still use the platform's float64
cosine values before exact Decimal conversion.

## Read for 10-15 minutes

Read this report, then read `_decimal_golden_section_maximum` and
`run_peak_refinement_study` in `src/quant_numerical/stability.py`. Finally read
`test_degree_16_local_peak_refinement_has_small_residual_grid_bias` in
`tests/test_stability.py`.

Question: what uncertainty does local peak refinement reduce, and why does it
still not prove the global continuous Lebesgue constant?
