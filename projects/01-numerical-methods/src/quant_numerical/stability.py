"""High-precision error decomposition for polynomial interpolation node designs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections.abc import Callable
from decimal import Decimal, localcontext
from functools import partial
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


def _decimal_cardinal_values(
    nodes: list[Decimal],
    weights: list[Decimal],
    point: Decimal,
) -> list[Decimal]:
    """Return all Lagrange cardinal-basis values at one point."""
    if point in nodes:
        exact_index = nodes.index(point)
        return [Decimal(index == exact_index) for index in range(len(nodes))]
    terms = [weight / (point - node) for node, weight in zip(nodes, weights, strict=True)]
    denominator = sum(terms, Decimal(0))
    return [term / denominator for term in terms]


def _decimal_lebesgue_value(
    nodes: list[Decimal],
    weights: list[Decimal],
    point: Decimal,
) -> Decimal:
    """Return the Lebesgue function for a represented node set at one point."""
    cardinal_values = _decimal_cardinal_values(nodes, weights, point)
    return sum((abs(value) for value in cardinal_values), Decimal(0))


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
    """Compare approximation, float64 evaluation and conditioning by node design.

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
        perturbation_magnitude = Decimal("1e-6")
        rows = []

        for design in ordered_designs:
            for degree in ordered_degrees:
                nodes = _interpolation_nodes(design, degree)
                values = [_runge(node) for node in nodes]
                weights = _decimal_barycentric_weights(nodes)
                decimal_estimates = [
                    _decimal_barycentric_value(nodes, values, weights, point) for point in grid
                ]
                lebesgue_values = [
                    _decimal_lebesgue_value(nodes, weights, point) for point in grid
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
                max_lebesgue, lebesgue_argmax, _ = _max_error(lebesgue_values, grid)
                cardinal_values = _decimal_cardinal_values(
                    nodes, weights, lebesgue_argmax
                )
                perturbations = [
                    perturbation_magnitude if value >= 0 else -perturbation_magnitude
                    for value in cardinal_values
                ]
                perturbed_values = [
                    value + perturbation
                    for value, perturbation in zip(values, perturbations, strict=True)
                ]
                baseline_at_target = _decimal_barycentric_value(
                    nodes, values, weights, lebesgue_argmax
                )
                perturbed_at_target = _decimal_barycentric_value(
                    nodes, perturbed_values, weights, lebesgue_argmax
                )
                observed_change = abs(perturbed_at_target - baseline_at_target)
                observed_amplification = observed_change / perturbation_magnitude
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
                    "estimated_lebesgue_constant": float(max_lebesgue),
                    "lebesgue_argmax": float(lebesgue_argmax),
                    "perturbation_magnitude": float(perturbation_magnitude),
                    "perturbation_target": float(lebesgue_argmax),
                    "observed_output_change": float(observed_change),
                    "observed_amplification": float(observed_amplification),
                    "lebesgue_bound_gap": float(abs(max_lebesgue - observed_amplification)),
                })

    return {
        "schema_version": 4,
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
        "conditioning_diagnostic": {
            "lebesgue_function": "Lambda(x) = sum_j abs(L_j(x))",
            "estimated_lebesgue_constant": (
                "maximum Lebesgue-function value on the recorded evaluation grid"
            ),
            "interpretation": (
                "worst-case amplification bound for perturbations in interpolation values; "
                "larger values indicate greater sensitivity"
            ),
        },
        "perturbation_experiment": {
            "magnitude": float(perturbation_magnitude),
            "pattern": (
                "at each design/degree grid maximum, perturb each node value by "
                "+/- magnitude using the sign of its cardinal basis value"
            ),
            "purpose": (
                "construct a deterministic worst-case sign pattern that attains the "
                "recorded grid-based Lebesgue amplification bound"
            ),
            "scope": (
                "bounded synthetic value perturbations; not random measurement noise "
                "or a probability statement"
            ),
        },
        "random_seed": "not applicable; deterministic nodes and evaluation grid",
        "evaluation_period": "not applicable; deterministic numerical experiment",
        "rows": rows,
    }


def run_grid_refinement_study(
    degree: int = 16,
    grid_sizes: tuple[int, ...] = (401, 801, 1601),
    node_designs: tuple[str, ...] = ("equidistant", "chebyshev-lobatto"),
    decimal_precision: int = 80,
    relative_stability_tolerance: float = 0.005,
) -> dict:
    """Measure how evaluation-grid resolution affects Lebesgue estimates.

    The continuous Lebesgue constant is approximated by the maximum on each
    equally spaced grid. Symmetric node sets can produce equal maxima at
    positive and negative locations, so location comparisons use absolute
    coordinates and do not treat a sign flip as movement.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    if (
        not grid_sizes
        or any(
            isinstance(grid_size, bool)
            or not isinstance(grid_size, int)
            or grid_size < 3
            for grid_size in grid_sizes
        )
        or len(set(grid_sizes)) != len(grid_sizes)
    ):
        raise ValueError("grid_sizes must contain unique integers >= 3")
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
    if (
        isinstance(decimal_precision, bool)
        or not isinstance(decimal_precision, int)
        or decimal_precision < 30
    ):
        raise ValueError("decimal_precision must be an integer >= 30")
    if (
        isinstance(relative_stability_tolerance, bool)
        or not isinstance(relative_stability_tolerance, (int, float))
        or not math.isfinite(relative_stability_tolerance)
        or relative_stability_tolerance <= 0
    ):
        raise ValueError("relative_stability_tolerance must be finite and positive")

    ordered_grid_sizes = tuple(sorted(grid_sizes))
    ordered_designs = tuple(
        design for design in ("equidistant", "chebyshev-lobatto") if design in node_designs
    )
    rows = []
    with localcontext() as context:
        context.prec = decimal_precision
        for design in ordered_designs:
            nodes = _interpolation_nodes(design, degree)
            weights = _decimal_barycentric_weights(nodes)
            for grid_size in ordered_grid_sizes:
                denominator = Decimal(grid_size - 1)
                grid = [
                    Decimal(-1) + Decimal(2 * index) / denominator
                    for index in range(grid_size)
                ]
                lebesgue_values = [
                    _decimal_lebesgue_value(nodes, weights, point) for point in grid
                ]
                maximum, argmax, _ = _max_error(lebesgue_values, grid)
                rows.append({
                    "node_design": design,
                    "degree": degree,
                    "grid_size": grid_size,
                    "grid_spacing": float(Decimal(2) / denominator),
                    "estimated_lebesgue_constant": float(maximum),
                    "lebesgue_argmax": float(argmax),
                    "absolute_lebesgue_argmax": float(abs(argmax)),
                })

    summaries = []
    for design in ordered_designs:
        design_rows = [row for row in rows if row["node_design"] == design]
        coarsest = design_rows[0]
        finest = design_rows[-1]
        relative_change = abs(
            finest["estimated_lebesgue_constant"]
            - coarsest["estimated_lebesgue_constant"]
        ) / finest["estimated_lebesgue_constant"]
        summaries.append({
            "node_design": design,
            "coarsest_grid_size": coarsest["grid_size"],
            "finest_grid_size": finest["grid_size"],
            "coarsest_estimate": coarsest["estimated_lebesgue_constant"],
            "finest_estimate": finest["estimated_lebesgue_constant"],
            "coarse_to_fine_relative_change": relative_change,
            "absolute_argmax_movement": abs(
                finest["absolute_lebesgue_argmax"]
                - coarsest["absolute_lebesgue_argmax"]
            ),
            "stable_within_tolerance": relative_change
            <= relative_stability_tolerance,
        })

    design_ratios = []
    if set(ordered_designs) == {"equidistant", "chebyshev-lobatto"}:
        for grid_size in ordered_grid_sizes:
            by_design = {
                row["node_design"]: row["estimated_lebesgue_constant"]
                for row in rows
                if row["grid_size"] == grid_size
            }
            design_ratios.append({
                "grid_size": grid_size,
                "equidistant_to_chebyshev_ratio": (
                    by_design["equidistant"] / by_design["chebyshev-lobatto"]
                ),
            })

    all_estimates_stable = all(
        summary["stable_within_tolerance"] for summary in summaries
    )
    lower_conditioning_conclusion_stable = (
        bool(design_ratios)
        and all(
            row["equidistant_to_chebyshev_ratio"] > 100
            for row in design_ratios
        )
    )
    return {
        "schema_version": 1,
        "study_type": "lebesgue-grid-refinement-sensitivity",
        "data": "deterministic simulated evaluations of interpolation cardinal bases",
        "interval": [-1.0, 1.0],
        "degree": degree,
        "node_designs": list(ordered_designs),
        "grid_sizes": list(ordered_grid_sizes),
        "decimal_precision": decimal_precision,
        "relative_stability_tolerance": relative_stability_tolerance,
        "location_comparison": (
            "absolute coordinates because symmetric node sets can have equal +/- maxima"
        ),
        "rows": rows,
        "summaries": summaries,
        "design_ratios": design_ratios,
        "conclusion": {
            "all_estimates_stable_within_tolerance": all_estimates_stable,
            "equidistant_over_100x_chebyshev_on_every_grid": (
                lower_conditioning_conclusion_stable
            ),
            "stable": all_estimates_stable and lower_conditioning_conclusion_stable,
            "scope": (
                "an operational grid-sensitivity check, not a proof of the continuous "
                "Lebesgue constants"
            ),
        },
        "random_seed": "not applicable; deterministic nodes and evaluation grids",
        "evaluation_period": "not applicable; deterministic numerical experiment",
    }


def _decimal_golden_section_maximum(
    function: Callable[[Decimal], Decimal],
    left: Decimal,
    right: Decimal,
    tolerance: Decimal,
    max_iterations: int,
) -> tuple[Decimal, Decimal, int, Decimal]:
    """Return a deterministic local maximum from a bounded Decimal interval."""
    ratio = (Decimal(5).sqrt() - Decimal(1)) / Decimal(2)
    inner_left = right - ratio * (right - left)
    inner_right = left + ratio * (right - left)
    value_left = function(inner_left)
    value_right = function(inner_right)
    iterations = 0

    while right - left > tolerance and iterations < max_iterations:
        if value_left < value_right:
            left = inner_left
            inner_left = inner_right
            value_left = value_right
            inner_right = left + ratio * (right - left)
            value_right = function(inner_right)
        else:
            right = inner_right
            inner_right = inner_left
            value_right = value_left
            inner_left = right - ratio * (right - left)
            value_left = function(inner_left)
        iterations += 1

    midpoint = (left + right) / Decimal(2)
    candidates = [
        (function(left), left),
        (function(right), right),
        (value_left, inner_left),
        (value_right, inner_right),
        (function(midpoint), midpoint),
    ]
    maximum, argmax = max(candidates, key=lambda candidate: candidate[0])
    return maximum, argmax, iterations, right - left


def run_peak_refinement_study(
    degree: int = 16,
    base_grid_size: int = 1601,
    node_designs: tuple[str, ...] = ("equidistant", "chebyshev-lobatto"),
    decimal_precision: int = 80,
    local_tolerance: str = "1e-30",
    max_iterations: int = 300,
    relative_residual_tolerance: float = 0.0005,
) -> dict:
    """Refine each uniform-grid Lebesgue peak inside its neighboring cells.

    A bounded golden-section search starts one base-grid step on each side of
    the recorded maximum. The result estimates the residual grid bias without
    adding a SciPy dependency to this independently installable project.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    if (
        isinstance(base_grid_size, bool)
        or not isinstance(base_grid_size, int)
        or base_grid_size < 3
    ):
        raise ValueError("base_grid_size must be an integer >= 3")
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
    if (
        isinstance(decimal_precision, bool)
        or not isinstance(decimal_precision, int)
        or decimal_precision < 30
    ):
        raise ValueError("decimal_precision must be an integer >= 30")
    if not isinstance(local_tolerance, str):
        raise ValueError("local_tolerance must be a positive finite Decimal string")
    try:
        decimal_tolerance = Decimal(local_tolerance)
    except ArithmeticError as error:
        raise ValueError(
            "local_tolerance must be a positive finite Decimal string"
        ) from error
    if not decimal_tolerance.is_finite() or decimal_tolerance <= 0:
        raise ValueError("local_tolerance must be a positive finite Decimal string")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if (
        isinstance(relative_residual_tolerance, bool)
        or not isinstance(relative_residual_tolerance, (int, float))
        or not math.isfinite(relative_residual_tolerance)
        or relative_residual_tolerance <= 0
    ):
        raise ValueError("relative_residual_tolerance must be finite and positive")

    ordered_designs = tuple(
        design for design in ("equidistant", "chebyshev-lobatto") if design in node_designs
    )
    rows = []
    with localcontext() as context:
        context.prec = decimal_precision
        denominator = Decimal(base_grid_size - 1)
        grid = [
            Decimal(-1) + Decimal(2 * index) / denominator
            for index in range(base_grid_size)
        ]
        spacing = Decimal(2) / denominator
        for design in ordered_designs:
            nodes = _interpolation_nodes(design, degree)
            weights = _decimal_barycentric_weights(nodes)
            function = partial(_decimal_lebesgue_value, nodes, weights)
            grid_values = [function(point) for point in grid]
            grid_maximum, grid_argmax, grid_index = _max_error(grid_values, grid)
            bracket_left = grid[max(0, grid_index - 1)]
            bracket_right = grid[min(base_grid_size - 1, grid_index + 1)]
            refined_maximum, refined_argmax, iterations, final_width = (
                _decimal_golden_section_maximum(
                    function,
                    bracket_left,
                    bracket_right,
                    decimal_tolerance,
                    max_iterations,
                )
            )
            if refined_maximum < grid_maximum:
                refined_maximum = grid_maximum
                refined_argmax = grid_argmax
            residual = refined_maximum - grid_maximum
            relative_residual = residual / refined_maximum
            rows.append({
                "node_design": design,
                "degree": degree,
                "base_grid_size": base_grid_size,
                "base_grid_spacing": float(spacing),
                "grid_estimate": float(grid_maximum),
                "grid_argmax": float(grid_argmax),
                "search_bracket": [float(bracket_left), float(bracket_right)],
                "refined_estimate": float(refined_maximum),
                "refined_argmax": float(refined_argmax),
                "absolute_residual": float(residual),
                "relative_residual": float(relative_residual),
                "absolute_location_shift": float(abs(refined_argmax - grid_argmax)),
                "iterations": iterations,
                "final_bracket_width": float(final_width),
                "converged": final_width <= decimal_tolerance,
                "residual_within_tolerance": (
                    relative_residual <= Decimal(str(relative_residual_tolerance))
                ),
            })

    refined_design_ratio = None
    if set(ordered_designs) == {"equidistant", "chebyshev-lobatto"}:
        by_design = {row["node_design"]: row["refined_estimate"] for row in rows}
        refined_design_ratio = (
            by_design["equidistant"] / by_design["chebyshev-lobatto"]
        )
    all_converged = all(row["converged"] for row in rows)
    all_residuals_within_tolerance = all(
        row["residual_within_tolerance"] for row in rows
    )
    lower_conditioning_conclusion_stable = (
        refined_design_ratio is not None and refined_design_ratio > 100
    )
    return {
        "schema_version": 1,
        "study_type": "lebesgue-local-peak-refinement",
        "data": "deterministic simulated evaluations of interpolation cardinal bases",
        "interval": [-1.0, 1.0],
        "degree": degree,
        "node_designs": list(ordered_designs),
        "base_grid_size": base_grid_size,
        "decimal_precision": decimal_precision,
        "method": "bounded golden-section maximization in the neighboring grid cells",
        "local_tolerance": local_tolerance,
        "max_iterations": max_iterations,
        "relative_residual_tolerance": relative_residual_tolerance,
        "rows": rows,
        "refined_equidistant_to_chebyshev_ratio": refined_design_ratio,
        "conclusion": {
            "all_local_searches_converged": all_converged,
            "all_residuals_within_tolerance": all_residuals_within_tolerance,
            "equidistant_over_100x_chebyshev_after_refinement": (
                lower_conditioning_conclusion_stable
            ),
            "stable": (
                all_converged
                and all_residuals_within_tolerance
                and lower_conditioning_conclusion_stable
            ),
            "scope": (
                "a deterministic local sensitivity check around grid-selected peaks, "
                "not a proof of the global continuous supremum"
            ),
        },
        "random_seed": "not applicable; deterministic nodes and evaluations",
        "evaluation_period": "not applicable; deterministic numerical experiment",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--refinement-output",
        type=Path,
        help="optional path for the degree-16 grid-refinement sensitivity artifact",
    )
    parser.add_argument(
        "--peak-output",
        type=Path,
        help="optional path for the degree-16 local peak-refinement artifact",
    )
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
        "data_version": "analytic-runge-node-perturbation-v3",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['rows'])} stability rows to {args.output}")
    if args.refinement_output is not None:
        refinement = run_grid_refinement_study()
        refinement["provenance"] = {
            "code_version": args.code_version,
            "source_sha256": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("stability.py", "interpolation.py")
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "data_version": "analytic-lebesgue-grid-refinement-v1",
        }
        args.refinement_output.parent.mkdir(parents=True, exist_ok=True)
        args.refinement_output.write_text(
            json.dumps(refinement, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Wrote {len(refinement['rows'])} grid-refinement rows to "
            f"{args.refinement_output}"
        )
    if args.peak_output is not None:
        peak_refinement = run_peak_refinement_study()
        peak_refinement["provenance"] = {
            "code_version": args.code_version,
            "source_sha256": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("stability.py", "interpolation.py")
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "data_version": "analytic-lebesgue-local-peak-refinement-v1",
        }
        args.peak_output.parent.mkdir(parents=True, exist_ok=True)
        args.peak_output.write_text(
            json.dumps(peak_refinement, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Wrote {len(peak_refinement['rows'])} local peak-refinement rows to "
            f"{args.peak_output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
