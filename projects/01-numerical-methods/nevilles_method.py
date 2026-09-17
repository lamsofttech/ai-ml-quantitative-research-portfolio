"""Standalone command-line runner for Neville's method.

Run with no arguments to reproduce the assignment's own data and its expected result,
``f(1.5) = 0.5118276664``:

    python nevilles_method.py

Or supply custom data (PowerShell line-continuation shown; drop the backticks on bash/zsh):

    python nevilles_method.py `
      --x "1.0,1.3,1.6,1.9,2.2,2.5" `
      --y "0.7651977,0.6200860,0.4554022,0.2818186,0.1103623,-0.0483838" `
      --target 1.5 `
      --csv neville_complete_table.csv

The recurrence itself lives in ``quant_numerical.neville`` (this project's tested package,
also used by the Gradio interface in ``app.py`` and by ``tests/test_neville.py``) -- this
script is a thin runner around it, kept in the project root next to ``examples.py`` and
imported the same way that file already imports the package, so it works whether or not the
package has been ``pip install``-ed. Run it from inside this project directory (with ``src/``
present alongside it), exactly like ``examples.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from quant_numerical.neville import (  # noqa: E402
    ASSIGNMENT_TARGET,
    ASSIGNMENT_X,
    ASSIGNMENT_Y,
    format_table,
    neville_interpolate,
    write_csv,
)


def _parse_floats(raw: str) -> list[float]:
    values = [chunk.strip() for chunk in raw.split(",")]
    return [float(chunk) for chunk in values if chunk != ""]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
