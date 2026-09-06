# Numerical Methods for Quantitative Research

## Research question

How accurately and reliably can polynomial interpolation approximate a smooth function between observed values, and what numerical risks appear as the number or spacing of nodes changes?

## Mathematical foundation

Given distinct nodes \(x_0,\ldots,x_n\) and values \(y_j=f(x_j)\), the interpolating polynomial is

$$P_n(x)=\sum_{j=0}^{n}y_jL_j(x),\qquad L_j(x)=\prod_{k\ne j}\frac{x-x_k}{x_j-x_k}.$$

Each basis polynomial is one at its own node and zero at every other node. The error for sufficiently differentiable \(f\) is

$$f(x)-P_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}\prod_{j=0}^{n}(x-x_j)$$

for some \(\xi\) in the interval containing the nodes and \(x\).

## Implementation

- Python uses the barycentric form, which is more stable and efficient for repeated evaluation than directly expanding the polynomial.
- R provides a transparent basis-polynomial implementation for learning and comparison.
- Tests verify exactness at nodes, recovery of a known quadratic, scalar behavior, weights, and invalid-input handling.
- The demonstration approximates \(e^x\) on \([0,1]\) using simulated function values.

## Reproduce

```bash
python -m pip install -e ".[dev]"
pytest
python examples.py
Rscript r/lagrange_interpolation.R
```

## Interpretation and limitations

Interpolation describes a numerical approximation, not a predictive financial model. Error depends on smoothness, node placement, floating-point precision, and polynomial degree. High-degree interpolation at equally spaced nodes can oscillate near interval boundaries (Runge phenomenon). Extrapolation outside the observed interval is especially unreliable. Later work will compare node designs, piecewise splines, and error bounds.

## Evidence classification

- Data: simulated evaluations of known mathematical functions.
- Result: numerical accuracy and stability, not financial profitability.
- Leakage risk: not applicable to this deterministic demonstration; any later forecasting use must use time-aware splits.

