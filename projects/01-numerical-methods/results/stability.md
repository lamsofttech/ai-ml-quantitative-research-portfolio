# Interpolation error and stability baseline — November 17, 2026

This deterministic experiment separates polynomial approximation error from
float64 representation/evaluation error. It uses the known Runge function

$$f(x)=\frac{1}{1+25x^2},\qquad -1\le x\le 1,$$

so every interpolated value can be compared with a known truth. No historical
or private data is used.

## Design

- Equally spaced nodes including -1 and 1.
- Polynomial degrees 4, 8, 12 and 16 (5, 9, 13 and 17 nodes).
- 401 equally spaced evaluation points.
- An 80-decimal-digit barycentric calculation defines the reference
  interpolation polynomial.
- The production `lagrange_interpolate` function evaluates the same nodes and
  values in NumPy float64.

The two error components are:

1. **Approximation error:** true Runge value minus the high-precision polynomial.
2. **Floating-point error:** float64 production result minus the same
   high-precision polynomial.

Exact configuration, source hashes, environment and row-level values are in
[stability.json](stability.json).

## Results

| Degree | Nodes | Maximum approximation error | Maximum floating-point error | Maximum total float64 error | Total-error location |
|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 0.438357 | 3.33e-16 | 0.438357 | 0.795 |
| 8 | 9 | 1.045174 | 1.44e-15 | 1.045174 | -0.920 |
| 12 | 13 | 3.657717 | 6.61e-14 | 3.657717 | -0.955 |
| 16 | 17 | 14.363078 | 7.82e-13 | 14.363078 | -0.965 |

Adding equally spaced nodes makes the maximum error worse for this function,
and the largest errors move toward the interval endpoints. At degree 16, the
maximum approximation error is about 18 trillion times the maximum discrepancy
between float64 and the high-precision polynomial. The total float64 error is
therefore dominated by the interpolation polynomial and node design—not by
ordinary double-precision roundoff.

This is the Runge phenomenon. More nodes do not automatically mean a better
global polynomial approximation when they are equally spaced. A better next
comparison is a node design concentrated near the endpoints, such as
Chebyshev-Lobatto nodes.

## Reproduce

Install project 01, check out code commit
`bc50e5f827b0a8d44491f8e0cdbc7faeaf925b2d`, then run:

```bash
python -m quant_numerical.stability \
  --code-version bc50e5f827b0a8d44491f8e0cdbc7faeaf925b2d \
  --output projects/01-numerical-methods/results/stability.json
```

Two executions on Python 3.12.14 with NumPy 2.5.5 produced byte-identical JSON.
Cross-version bitwise equality is not promised. Artifact SHA-256:
`7982e07eaa7315af7a2dd50e57951e911e7bcd7a0591f9d83be7c12619ad097c`.

## Limitations

This baseline studies one difficult function, one interval and one equally
spaced node design. The 401-point grid estimates the continuous maximum error;
it does not mathematically prove the exact supremum between grid points. The
80-digit reference is a strong numerical benchmark, not a symbolic proof.
Absolute errors are reported; relative error near small function values would
answer a different question. No extrapolation is evaluated.

## Read for 10-15 minutes

Read this report and then trace `run_stability_study` in
`src/quant_numerical/stability.py`, especially the three error arrays.

Question: why does degree 16 perform much worse than degree 4 even though its
float64 discrepancy remains below $8\times10^{-13}$?
