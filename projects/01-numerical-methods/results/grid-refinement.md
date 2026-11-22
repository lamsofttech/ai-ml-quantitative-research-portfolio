# Degree-16 Lebesgue grid-refinement sensitivity — November 21, 2026

This deterministic sensitivity check asks whether the degree-16 conditioning
conclusion changes when the evaluation grid is refined. It compares the same
equidistant and Chebyshev-Lobatto node sets on 401, 801 and 1601 equally spaced
points over $[-1,1]$. No historical, private or random data is used.

## Decision rule

The estimate is called operationally stable when the 401-point and 1601-point
Lebesgue estimates differ by no more than 0.5%. The node-design conclusion is
called stable when the equidistant estimate remains more than 100 times the
Chebyshev-Lobatto estimate on every grid. These are explicit sensitivity
criteria, not confidence intervals or proofs of the continuous maxima.

Symmetric node sets can have equal maxima at positive and negative locations.
The code therefore compares absolute maximizing locations; a sign change does
not represent movement to a different part of the interval.

## Results

| Node design | Grid points | Grid spacing | Lebesgue estimate | Maximizing location |
|---|---:|---:|---:|---:|
| Equidistant | 401 | 0.00500 | 933.089921 | -0.96500 |
| Equidistant | 801 | 0.00250 | 934.233969 | -0.96750 |
| Equidistant | 1601 | 0.00125 | 934.421374 | 0.96625 |
| Chebyshev-Lobatto | 401 | 0.00500 | 2.723812 | -0.10000 |
| Chebyshev-Lobatto | 801 | 0.00250 | 2.724667 | -0.09750 |
| Chebyshev-Lobatto | 1601 | 0.00125 | 2.724667 | -0.09750 |

From 401 to 1601 points, the equidistant estimate changes by 0.1425% and
the Chebyshev-Lobatto estimate changes by 0.0314%. Their absolute maximizing
locations move by 0.00125 and 0.00250 respectively, no more than one coarse
grid interval.

The equidistant-to-Chebyshev estimate ratio is 342.57, 342.88 and 342.95 on
the three grids. Therefore both the 0.5% estimate criterion and the
greater-than-100-times node-design criterion pass. The original conclusion is
not an artifact of using only 401 evaluation points: at degree 16, the
equidistant nodes remain far more sensitive to supplied-value perturbations.

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
The grid-refinement artifact SHA-256 is
`0b60f9659db19f505485b5fff75e24093a5260e823a5d32b84cd05fff2e079ea`.
Exact row values, criteria, source hashes and environment are recorded in
[grid-refinement.json](grid-refinement.json).
The follow-up [local peak-refinement report](peak-refinement.md) estimates the
remaining bias beyond the 1601-point grid.

## Limitations

Grid refinement is evidence about numerical sensitivity, not a proof of the
continuous Lebesgue constants. Only degree 16, two symmetric node designs, one
interval and three nested uniform grids are checked. The 0.5% tolerance and
100-times comparison are transparent operational criteria, not universal
mathematical thresholds. Interior Chebyshev-Lobatto nodes still use the
platform's float64 cosine values before exact Decimal conversion.

## Read for 10-15 minutes

Read this report, then read `run_grid_refinement_study` in
`src/quant_numerical/stability.py` and
`test_degree_16_lebesgue_conclusion_is_stable_under_grid_refinement` in
`tests/test_stability.py`.

Question: why does the 1601-point estimate being close to the 401-point
estimate strengthen the conditioning conclusion without proving the exact
continuous Lebesgue constant?
