# Numerical Methods for Quantitative Research

## Research question

How accurately and reliably can polynomial interpolation approximate a smooth function between observed values, and what numerical risks appear as the number or spacing of nodes changes?

## Mathematical foundation

Given distinct nodes \(x_0,\ldots,x_n\) and values \(y_j=f(x_j)\), the interpolating polynomial is

$$P_n(x)=\sum_{j=0}^{n}y_jL_j(x),\qquad L_j(x)=\prod_{k\ne j}\frac{x-x_k}{x_j-x_k}.$$

Each basis polynomial is one at its own node and zero at every other node. The error for sufficiently differentiable \(f\) is

$$f(x)-P_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}\prod_{j=0}^{n}(x-x_j)$$

for some \(\xi\) in the interval containing the nodes and \(x\).

## Neville's method

Lagrange's barycentric form above is fast and numerically stable, but it does not expose its
own intermediate work — the final weighted sum gives no way to check the calculation by hand.
Neville's method computes the *same* interpolated value through an explicit recursive table
whose every entry is a small, independently verifiable calculation, which is why it exists here
as a second, complementary implementation rather than a replacement.

**Recurrence:**

$$Q_{i,0}=f(x_i),\qquad Q_{i,j}=\frac{(x-x_{i-j})\,Q_{i,j-1}-(x-x_i)\,Q_{i-1,j-1}}{x_i-x_{i-j}}.$$

`Q[n-1, n-1]` (the bottom-right corner of the lower-triangular table) is the interpolated value.
Distinct x-nodes are required: `x_i - x_{i-j}` is a divisor in every recursive step.

**Assignment data used as the default in every interface below:**

| \(x\) | 1.0 | 1.3 | 1.6 | 1.9 | 2.2 | 2.5 |
|---|---|---|---|---|---|---|
| \(f(x)\) | 0.7651977 | 0.6200860 | 0.4554022 | 0.2818186 | 0.1103623 | -0.0483838 |

Evaluated at \(x = 1.5\): \(f(1.5) = 0.5118276664\) (\(\approx 0.5118277\) to seven decimal places).
This value is never hard-coded anywhere in the implementation — `tests/test_neville.py` checks
it against the recurrence's own output, and a second, independent test confirms a different
target on the same data produces a different result.

**Why calculation evidence matters:** a final answer alone does not let anyone verify *how* it
was reached — a transcription error, a sign flip, or the wrong recurrence would produce a
plausible-looking but wrong number with no way to catch it. Every interface below (CLI, tests,
and the Gradio app) therefore prints or displays the complete table **and** every substituted
calculation, not just the final estimate.

## Implementation

- Python's Lagrange interpolation uses the barycentric form, which is more stable and efficient for repeated evaluation than directly expanding the polynomial.
- Python's Neville's method (`src/quant_numerical/neville.py`) builds the full recursive table and records a human-readable substituted formula for every entry — see `NevilleResult`/`NevilleStep`.
- R provides a transparent basis-polynomial implementation of Lagrange interpolation for learning and comparison.
- Tests verify exactness at nodes, recovery of a known quadratic, scalar behavior, weights, and invalid-input handling for both methods, plus the exact assignment values for Neville's method.
- The Lagrange demonstration (`examples.py`) approximates \(e^x\) on \([0,1]\) using simulated function values.
- `nevilles_method.py` is a standalone command-line runner for Neville's method (no Gradio required); `app.py` is an interactive Gradio interface over the same engine.

## Reproduce

Create and activate a virtual environment (Windows PowerShell shown; see the repository root
README for macOS/Linux):

```powershell
cd "C:\Users\Local User\Desktop\MATHEMATICAL\projects\01-numerical-methods"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the tests, the Lagrange demonstration, and the R comparison:

```powershell
pytest
python examples.py
Rscript r/lagrange_interpolation.R
```

Run Neville's method from the command line, with the assignment's own data (reproduces
`f(1.5) = 0.5118276664`):

```powershell
python nevilles_method.py
```

Or with custom data and a CSV export of the complete table:

```powershell
python nevilles_method.py `
  --x "1.0,1.3,1.6,1.9,2.2,2.5" `
  --y "0.7651977,0.6200860,0.4554022,0.2818186,0.1103623,-0.0483838" `
  --target 1.5 `
  --csv artifacts/neville_complete_table.csv
```

Run the interactive Gradio interface (editable inputs, the full table, every calculation step,
and CSV / source-code downloads):

```powershell
python -m pip install -e ".[app]"
python app.py
```

This opens at `http://127.0.0.1:7860/` by default.

### Deploying the Gradio app to Hugging Face Spaces

1. Create a new Space with SDK **Gradio**.
2. Upload `app.py`, the `src/quant_numerical/` package (at minimum `__init__.py` and
   `neville.py`), and `pyproject.toml`, preserving the `src/quant_numerical/...` path so
   `app.py`'s `sys.path.insert(..., "src")` import keeps working — or add a Space
   `requirements.txt` that installs this package directly from GitHub, the same way
   `spaces/portfolio-hub/requirements.txt` installs `quant-mc-option-pricing` (project 02) in
   this repository, and delete the `sys.path` line from `app.py` in favor of a normal
   `from quant_numerical.neville import ...`.
3. Add a Space `requirements.txt` with `gradio`, `pandas`, and `numpy` (matching the versions
   pinned in this project's `pyproject.toml` under `[project.optional-dependencies].app`).
4. Set `app_file: app.py` in the Space's README front matter (see
   `spaces/portfolio-hub/README.md` in this repository for the exact format).

## Interpretation and limitations

Interpolation describes a numerical approximation, not a predictive financial model. Error depends on smoothness, node placement, floating-point precision, and polynomial degree. High-degree interpolation at equally spaced nodes can oscillate near interval boundaries (Runge phenomenon). Extrapolation outside the observed interval is especially unreliable. Later work will compare node designs, piecewise splines, and error bounds.

## Evidence classification

- Data: simulated evaluations of known mathematical functions.
- Result: numerical accuracy and stability, not financial profitability.
- Leakage risk: not applicable to this deterministic demonstration; any later forecasting use must use time-aware splits.

