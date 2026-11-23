# Case study: choosing stable interpolation nodes

## Employer summary

I built and validated a reproducible numerical experiment that asks a practical
question: when a global polynomial is fitted between observed values, does
adding more equally spaced observations necessarily improve the result?

For the Runge function on `[-1, 1]`, the answer was no. At degree 16, equally
spaced nodes produced a maximum approximation error of `14.363078` and a
locally refined Lebesgue estimate of `934.534111`. Chebyshev-Lobatto nodes
reduced the measured maximum approximation error to `0.036713` and the refined
conditioning estimate to `2.724709`. The refined conditioning ratio was
`342.98x`.

The important engineering result is not that one set of numbers looks better.
It is that the conclusion survived high-precision reference calculations, an
adversarial perturbation experiment, three evaluation-grid resolutions and a
bounded local peak search. The work demonstrates numerical analysis,
experimental design, reproducibility and honest limitation reporting. It does
not claim a financial forecast or trading advantage.

## Research question

How accurately and reliably can polynomial interpolation approximate a smooth
function between observed values, and how much does node placement affect
error amplification as polynomial degree increases?

This matters in quantitative work because an interpolated curve can look
plausible while being highly sensitive to small errors in its inputs. A method
that is accurate on one dataset but poorly conditioned can fail when inputs are
rounded, measured with noise or slightly revised.

## Design decisions

| Decision | Reason |
|---|---|
| Use the known Runge function | Every approximation can be compared with a known truth; no private or historical data is required. |
| Compare equal polynomial degrees | Equidistant and Chebyshev-Lobatto designs receive the same number of nodes. |
| Separate approximation and float64 error | This distinguishes a poor polynomial design from implementation roundoff. |
| Use an 80-decimal-digit reference | The production NumPy result is checked against a much higher-precision numerical benchmark. |
| Measure the Lebesgue function | It quantifies worst-case amplification of errors in supplied values independently of the selected function values. |
| Construct a bounded adversarial perturbation | It demonstrates that the grid-estimated amplification bound is attainable at the selected point. |
| Refine the grid and then the local peak | It tests whether the conclusion is an artifact of missing a peak between grid points. |

The [implementation](src/quant_numerical/stability.py) records configuration,
environment versions and source hashes in machine-readable artifacts. The
[regression tests](tests/test_stability.py) check reproducibility, numerical
criteria, invalid inputs and the command-line interface.

## Evidence and interpretation

The [stability report](results/stability.md) and its
[machine-readable artifact](results/stability.json) contain the base study.
At degree 16:

| Node design | Maximum approximation error | Maximum float64 discrepancy | 401-point Lebesgue estimate | Adversarial output change for `1e-6` input bound |
|---|---:|---:|---:|---:|
| Equidistant | 14.363078 | 7.82e-13 | 933.090 | 9.330899e-4 |
| Chebyshev-Lobatto | 0.036713 | 4.79e-16 | 2.724 | 2.723812e-6 |

Approximation error dominates float64 discrepancy, so the Runge failure is not
fixed by simply evaluating the same equidistant polynomial at higher precision.
The deliberately aligned perturbations attain the recorded grid amplification
to the shown precision, connecting the conditioning diagnostic to an observable
output change.

The [grid-refinement report](results/grid-refinement.md) and
[artifact](results/grid-refinement.json) compare 401, 801 and 1601 evaluation
points. From 401 to 1601 points, the estimates changed by only `0.1425%` for
equidistant nodes and `0.0314%` for Chebyshev-Lobatto nodes. The conditioning
ratio remained between `342.57x` and `342.95x`.

The [local peak report](results/peak-refinement.md) and
[artifact](results/peak-refinement.json) then search the cells around the
1601-point maxima. Residual grid bias was `0.01206%` and `0.001525%`; the
refined ratio was `342.98x`. These checks make the practical comparison robust
to the tested grid choices, but they are numerical evidence rather than a
symbolic proof of the global continuous maxima.

## Reproduce the study

Check out implementation commit
[`68a8c0618e99edf058ad34fd5f37edf45bf7c6e2`](https://github.com/lamsofttech/ai-ml-quantitative-research-portfolio/commit/68a8c0618e99edf058ad34fd5f37edf45bf7c6e2),
then run from the repository root:

```bash
python -m pip install -e "projects/01-numerical-methods[dev]"
python -m pytest projects/01-numerical-methods/tests
python -m quant_numerical.stability \
  --code-version 68a8c0618e99edf058ad34fd5f37edf45bf7c6e2 \
  --output projects/01-numerical-methods/results/stability.json \
  --refinement-output projects/01-numerical-methods/results/grid-refinement.json \
  --peak-output projects/01-numerical-methods/results/peak-refinement.json
python scripts/validate_case_study_links.py
```

The existing artifacts were generated twice on Python 3.12.14 and were
byte-identical. Cross-version and cross-platform bitwise equality are not
promised. See the [project README](README.md) for the package, CLI, R and app
interfaces.

## Limitations

- The study covers one difficult function, one interval, two node designs and
  degrees 4, 8, 12 and 16; it does not establish a universal ranking.
- Grid and local optimization reduce discretization uncertainty but do not
  prove the exact continuous Lebesgue constants or rule out every other peak.
- The 80-digit calculation is a high-precision numerical benchmark, not a
  symbolic proof.
- The perturbation is intentionally worst-case. Typical random measurement
  noise will not usually align with every cardinal-basis sign.
- No historical prices, model fitting, extrapolation, transaction costs or
  trading strategy are evaluated.

## Interview talking points

1. **Why did more nodes make the equidistant result worse?** High-degree global
   interpolation on equally spaced nodes develops large endpoint oscillations;
   Chebyshev-Lobatto nodes cluster where that instability is strongest.
2. **Why report a Lebesgue estimate as well as function error?** Function error
   describes one selected function. The Lebesgue function describes the node
   design's worst-case sensitivity to errors in any supplied values.
3. **How did you avoid mistaking a coarse-grid artifact for a result?** I used
   three nested grids, predefined decision criteria and an 80-digit bounded
   local search around the finest-grid peaks.
4. **What would you build next?** Compare piecewise splines and noisy-data
   experiments under the same evidence contract, without claiming the current
   global-polynomial result generalizes automatically.

## Skills demonstrated

Numerical analysis; polynomial interpolation; high-precision benchmarking;
conditioning and error decomposition; adversarial test construction;
sensitivity analysis; bounded optimization; automated testing; reproducible
artifacts; and evidence-aware technical communication.
