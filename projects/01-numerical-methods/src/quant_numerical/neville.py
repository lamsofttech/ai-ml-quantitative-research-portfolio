"""Neville's method: recursive polynomial interpolation with a full calculation trace.

Unlike the barycentric Lagrange form in :mod:`quant_numerical.interpolation` (fast, stable,
but opaque about intermediate work), Neville's method builds the interpolating polynomial's
value at a single target point through an explicit triangular table of recursive estimates.
Every entry is cheap to verify by hand, which is the point of this module: it exists to make
the *process*, not just the final answer, checkable.

Recurrence (Burden & Faires' notation, matching this project's assignment brief)::

    Q[i, 0] = f(x_i)
    Q[i, j] = ((x_target - x[i-j]) * Q[i, j-1] - (x_target - x[i]) * Q[i-1, j-1])
              / (x[i] - x[i-j])

``Q[n-1, n-1]`` (the bottom-right corner of the table) is the interpolating polynomial's
value at ``x_target``. Distinct x-nodes are required: ``x[i] - x[i-j]`` is a divisor in every
recursive step, so a repeated x-value would divide by zero.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

# The assignment's own data, kept in one place so the CLI runner, the tests, and the Gradio
# interface all default to it instead of retyping the same numbers three times.
ASSIGNMENT_X: tuple[float, ...] = (1.0, 1.3, 1.6, 1.9, 2.2, 2.5)
ASSIGNMENT_Y: tuple[float, ...] = (
    0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623, -0.0483838
)
ASSIGNMENT_TARGET: float = 1.5
ASSIGNMENT_EXPECTED: float = 0.5118276664


@dataclass(frozen=True)
class NevilleStep:
    """One computed entry ``Q[i, j]`` of the Neville table, with its full substitution.

    The five ``*_input`` fields below are ``None`` for a leaf entry (``j == 0``, which is
    simply ``f(x_i)``, nothing to substitute) and hold the recurrence's five numeric inputs
    for every other entry -- broken out as their own values, not just baked into ``formula``'s
    text, specifically so a CSV export can be recomputed and cross-checked in a spreadsheet
    rather than only read.
    """

    i: int
    j: int
    value: float
    formula: str
    x_target_input: float | None = None
    x_i_minus_j_input: float | None = None
    q_i_jminus1_input: float | None = None
    x_i_input: float | None = None
    q_iminus1_jminus1_input: float | None = None


@dataclass(frozen=True, eq=False)
class NevilleResult:
    """Complete output of :func:`neville_interpolate`: the table, every step, and the estimate."""

    x_nodes: NDArray
    y_nodes: NDArray
    x_target: float
    table: NDArray  # (n, n) lower-triangular; entries above the diagonal are NaN (never computed)
    steps: list[NevilleStep] = field(default_factory=list)
    estimate: float = 0.0


def _validate_nodes(x_nodes: ArrayLike, y_nodes: ArrayLike) -> tuple[NDArray, NDArray]:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_nodes and y_nodes must be one-dimensional")
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("x_nodes and y_nodes must have the same non-zero length")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nodes must contain only finite values")
    if len(np.unique(x)) != len(x):
        raise ValueError(
            "x_nodes must be distinct: a repeated value divides by zero in Neville's recurrence"
        )
    return x, y


def neville_interpolate(x_nodes: ArrayLike, y_nodes: ArrayLike, x_target: float) -> NevilleResult:
    """Run Neville's method and return the full table plus every recursive substitution.

    Parameters
    ----------
    x_nodes, y_nodes:
        Distinct interpolation nodes ``x_i`` and their observed values ``f(x_i)``. Must be the
        same non-zero length.
    x_target:
        The point at which to approximate ``f``.

    Returns
    -------
    NevilleResult
        ``result.estimate`` is the interpolated value; ``result.table`` is the full triangular
        table (NaN above the diagonal); ``result.steps`` records every ``Q[i, j]`` in
        computation order with a human-readable substituted formula, suitable for display or a
        CSV export.

    Raises
    ------
    ValueError
        If the inputs are empty, mismatched in length, non-finite, or contain a duplicate
        x-value (which would divide by zero).
    """
    x, y = _validate_nodes(x_nodes, y_nodes)
    target = float(x_target)
    if not np.isfinite(target):
        raise ValueError("x_target must be finite")

    n = len(x)
    table = np.full((n, n), np.nan)
    steps: list[NevilleStep] = []

    for i in range(n):
        table[i, 0] = y[i]
        leaf_formula = f"Q[{i},0] = f(x_{i}) = {y[i]:.10f}"
        steps.append(NevilleStep(i=i, j=0, value=float(y[i]), formula=leaf_formula))

    for j in range(1, n):
        for i in range(j, n):
            left = table[i, j - 1]  # Q[i, j-1]
            right = table[i - 1, j - 1]  # Q[i-1, j-1]
            x_left = x[i - j]
            x_right = x[i]
            value = ((target - x_left) * left - (target - x_right) * right) / (x_right - x_left)
            table[i, j] = value
            formula = (
                f"Q[{i},{j}] = (({target:g} - {x_left:g})({left:.10f}) "
                f"- ({target:g} - {x_right:g})({right:.10f})) "
                f"/ ({x_right:g} - {x_left:g}) = {value:.10f}"
            )
            steps.append(
                NevilleStep(
                    i=i,
                    j=j,
                    value=float(value),
                    formula=formula,
                    x_target_input=target,
                    x_i_minus_j_input=float(x_left),
                    q_i_jminus1_input=float(left),
                    x_i_input=float(x_right),
                    q_iminus1_jminus1_input=float(right),
                )
            )

    estimate = float(table[n - 1, n - 1])
    return NevilleResult(
        x_nodes=x, y_nodes=y, x_target=target, table=table, steps=steps, estimate=estimate
    )


def table_grid(result: NevilleResult) -> list[list[str]]:
    """Render the triangular table as a header row plus one row per node, for display or export.

    Cells above the diagonal (not computed) are empty strings rather than ``NaN`` so the grid
    reads cleanly in a Gradio Dataframe or a plain terminal.
    """
    n = len(result.x_nodes)
    header = ["x_i", *[f"Q[i,{j}]" for j in range(n)]]
    rows = [header]
    for i in range(n):
        row = [f"{result.x_nodes[i]:g}"]
        for j in range(n):
            row.append(f"{result.table[i, j]:.10f}" if j <= i else "")
        rows.append(row)
    return rows


def format_table(result: NevilleResult) -> str:
    """Format the triangular table as fixed-width text, suitable for terminal output."""
    grid = table_grid(result)
    widths = [max(len(row[col]) for row in grid) for col in range(len(grid[0]))]
    lines = [
        "  ".join(cell.rjust(width) for cell, width in zip(row, widths, strict=True))
        for row in grid
    ]
    return "\n".join(lines)


CSV_HEADER = [
    "i",
    "j",
    "value",
    "x_target",
    "x_i_minus_j",
    "Q_i_jminus1",
    "x_i",
    "Q_iminus1_jminus1",
    "substitution",
]


def write_csv(result: NevilleResult, path: str | Path) -> Path:
    """Write every computed step to a CSV file and return its path.

    Deliberately more than a display log: alongside the human-readable ``substitution``
    string, each recursive row's five numeric inputs are their own columns
    (``x_target``, ``x_i_minus_j``, ``Q_i_jminus1``, ``x_i``, ``Q_iminus1_jminus1``) --
    blank for a leaf row (``j == 0``), since ``Q[i,0]`` has nothing to substitute. Opened in
    Excel, this is ready to test: build the recurrence yourself in a spare column, e.g.
    ``=((D2-E2)*F2-(D2-G2)*H2)/(G2-E2)`` against this file's own column order, and confirm it
    matches the stored ``value`` -- not just a transcript to read, something to check.
    """
    out_path = Path(path)
    if out_path.parent != Path():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value: float | None) -> str:
        return "" if value is None else f"{value:.10f}"

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for step in result.steps:
            writer.writerow(
                [
                    step.i,
                    step.j,
                    f"{step.value:.10f}",
                    fmt(step.x_target_input),
                    fmt(step.x_i_minus_j_input),
                    fmt(step.q_i_jminus1_input),
                    fmt(step.x_i_input),
                    fmt(step.q_iminus1_jminus1_input),
                    step.formula,
                ]
            )
    return out_path


# ---------------------------------------------------------------------------------------
# Command-line entry point. This file needs only numpy beyond the standard library, so it
# can be downloaded on its own -- no other file from this project -- and validated
# independently: `pip install numpy` then `python neville.py`.
# ---------------------------------------------------------------------------------------


def _parse_floats(raw: str) -> list[float]:
    values = [chunk.strip() for chunk in raw.split(",")]
    return [float(chunk) for chunk in values if chunk != ""]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Neville's method with no arguments to reproduce the assignment's own "
            "data and its expected result, f(1.5) = 0.5118276664."
        ),
    )
    parser.add_argument(
        "--x", type=str, default=None, help="Comma-separated x-values (default: assignment data)"
    )
    parser.add_argument(
        "--y", type=str, default=None, help="Comma-separated f(x)-values (default: assignment data)"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Evaluation point (default: assignment target, 1.5)",
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Optional path to write the complete table as CSV"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    x_values = _parse_floats(args.x) if args.x is not None else list(ASSIGNMENT_X)
    y_values = _parse_floats(args.y) if args.y is not None else list(ASSIGNMENT_Y)
    target = args.target if args.target is not None else ASSIGNMENT_TARGET

    try:
        result = neville_interpolate(x_values, y_values, target)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Neville's method interpolation at x = {target:g}\n")
    print("Complete recursive table (blank cells are not part of the recurrence):\n")
    print(format_table(result))

    print("\nEvery calculation step:\n")
    for step in result.steps:
        print(f"  {step.formula}")

    print(f"\nf({target:g}) ~= {result.estimate:.10f}")

    if args.csv:
        csv_path = write_csv(result, args.csv)
        print(f"\nWrote complete table ({len(result.steps)} rows) to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
