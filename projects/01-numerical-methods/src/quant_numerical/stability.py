"""High-precision error decomposition for polynomial interpolation node designs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from .interpolation import lagrange_interpolate


def _runge(x: Decimal) -> Decimal:
    """Runge's function, evaluated without binary floating-point arithmetic."""
    return Decimal(1) / (Decimal(1) + Decimal(25) * x * x)


def _decimal_barycentric_weights(nodes: list[Decimal]) -> list[Decimal]:
    weights = []
    for index, node in enumerate(nodes):
        product = Decimal(1)
        for other_index, other in enumerate(nodes):
            if index != other_index:
                product *= node - other
        weights.append(Decimal(1) / product)
    return weights


def _decimal_barycentric_value(
    nodes: list[Decimal],
    values: list[Decimal],
    weights: list[Decimal],
    point: Decimal,
) -> Decimal:
    for node, value in zip(nodes, values, strict=True):
        if point == node:
            return value
    terms = [weight / (point - node) for node, weight in zip(nodes, weights, strict=True)]
    numerator = sum((term * value for term, value in zip(terms, values, strict=True)), Decimal(0))
    return numerator / sum(terms, Decimal(0))


def _max_error(errors: list[Decimal], grid: list[Decimal]) -> tuple[Decimal, Decimal, int]:
    index = max(range(len(errors)), key=errors.__getitem__)
    return errors[index], grid[index], index


def _interpolation_nodes(design: str, degree: int) -> list[Decimal]:
    if design == "equidistant":
        return [
            Decimal(-1) + Decimal(2 * index) / Decimal(degree)
            for index in range(degree + 1)
        ]
    if design == "chebyshev-lobatto":
        # The sign/order convention makes the nodes increase from -1 to 1.
        # Interior cosine nodes are first generated in float64 and then converted
        # exactly to Decimal, so the reference and production evaluators use the
        # identical represented node set. Endpoints are set exactly.
        nodes = [
            Decimal.from_float(-math.cos(math.pi * index / degree))
            for index in range(degree + 1)
        ]
        nodes[0] = Decimal(-1)
        nodes[-1] = Decimal(1)
        return nodes
    raise ValueError(f"unsupported node design: {design}")


def run_stability_study(
    degrees: tuple[int, ...] = (4, 8, 12, 16),
    node_designs: tuple[str, ...] = ("equidistant", "chebyshev-lobatto"),
    grid_size: int = 401,
    decimal_precision: int = 80,
) -> dict:
    """Separate interpolation approximation error from float64 numerical error.

    The high-precision polynomial and production implementation use the same
    represented nodes and Runge-function values. Their difference captures
    representation and evaluation error; the high-precision polynomial's
    difference from the true function is the polynomial approximation error.
    """
    if not degrees or any(
        isinstance(degree, bool) or not isinstance(degree, int) or degree < 1
        for degree in degrees
    ):
        raise ValueError("degrees must contain positive integers")
    if len(set(degrees)) != len(degrees):
        raise ValueError("degrees must be unique")
    supported_designs = {"equidistant", "chebyshev-lobatto"}
    if (
        not node_designs
        or any(
            not isinstance(design, str) or design not in supported_designs
            for design in node_designs
        )
        or len(set(node_designs)) != len(node_designs)
    ):
        raise ValueError(
            "node_designs must contain unique values from: "
            "equidistant, chebyshev-lobatto"
        )
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 3:
        raise ValueError("grid_size must be an integer >= 3")
    if (
        isinstance(decimal_precision, bool)
        or not isinstance(decimal_precision, int)
        or decimal_precision < 30
    ):
        raise ValueError("decimal_precision must be an integer >= 30")

    ordered_degrees = tuple(sorted(degrees))
    ordered_designs = tuple(
        design for design in ("equidistant", "chebyshev-lobatto") if design in node_designs
    )
    with localcontext() as context:
        context.prec = decimal_precision
        denominator = Decimal(grid_size - 1)
        grid = [Decimal(-1) + Decimal(2 * index) / denominator for index in range(grid_size)]
        true_values = [_runge(point) for point in grid]
        rows = []

        for design in ordered_designs:
            for degree in ordered_degrees:
                nodes = _interpolation_nodes(design, degree)
                values = [_runge(node) for node in nodes]
                weights = _decimal_barycentric_weights(nodes)
                decimal_estimates = [
                    _decimal_barycentric_value(nodes, values, weights, point) for point in grid
                ]
                float_estimates = np.asarray(
                    lagrange_interpolate(
                        [float(node) for node in nodes],
                        [float(value) for value in values],
                        [float(point) for point in grid],
                    )
                )
                float_as_decimal = [Decimal.from_float(float(value)) for value in float_estimates]

                approximation = [
                    abs(true - estimate)
                    for true, estimate in zip(true_values, decimal_estimates, strict=True)
                ]
                floating_point = [
                    abs(float_estimate - decimal_estimate)
                    for float_estimate, decimal_estimate in zip(
                        float_as_decimal, decimal_estimates, strict=True
                    )
                ]
                total_float64 = [
                    abs(float_estimate - true)
                    for float_estimate, true in zip(float_as_decimal, true_values, strict=True)
                ]

                max_approximation, approximation_argmax, _ = _max_error(approximation, grid)
                max_floating, floating_argmax, _ = _max_error(floating_point, grid)
                max_total, total_argmax, total_index = _max_error(total_float64, grid)
                rows.append({
                    "node_design": design,
                    "degree": degree,
                    "n_nodes": degree + 1,
                    "max_approximation_error": float(max_approximation),
                    "approximation_error_argmax": float(approximation_argmax),
                    "max_floating_point_error": float(max_floating),
                    "floating_point_error_argmax": float(floating_argmax),
                    "max_total_float64_error": float(max_total),
                    "total_error_argmax": float(total_argmax),
                    "approximation_error_at_total_argmax": float(approximation[total_index]),
                    "floating_point_error_at_total_argmax": float(floating_point[total_index]),
                })

    return {
        "schema_version": 2,
        "study_type": "interpolation-node-design-comparison",
        "data": "deterministic simulated evaluations of the known Runge function",
        "function": "f(x) = 1 / (1 + 25 x^2)",
        "interval": [-1.0, 1.0],
        "node_designs": list(ordered_designs),
        "node_definition": {
            "equidistant": "equally spaced, including both endpoints",
            "chebyshev-lobatto": (
                "x_j = -cos(pi*j/degree), including both endpoints; interior nodes "
                "represented in float64 and converted exactly for the Decimal reference"
            ),
        },
        "degrees": list(ordered_degrees),
        "grid_size": grid_size,
        "decimal_precision": decimal_precision,
        "float_evaluator": "production barycentric lagrange_interpolate using NumPy float64",
        "error_decomposition": {
            "approximation": "abs(true Runge value - high-precision interpolation polynomial)",
            "floating_point": (
                "abs(float64 production result - high-precision interpolation polynomial)"
            ),
            "total_float64": "abs(float64 production result - true Runge value)",
        },
        "random_seed": "not applicable; deterministic nodes and evaluation grid",
        "evaluation_period": "not applicable; deterministic numerical experiment",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-version", required=True, help="Git commit containing this runner")
    args = parser.parse_args(argv)

    report = run_stability_study()
    source = Path(__file__).parent
    report["provenance"] = {
        "code_version": args.code_version,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("stability.py", "interpolation.py")
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "data_version": "analytic-runge-node-comparison-v1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['rows'])} stability rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
