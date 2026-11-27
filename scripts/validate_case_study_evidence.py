"""Verify employer-facing case-study claims against committed JSON evidence."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT01_CASE_STUDY = ROOT / "projects/01-numerical-methods/CASE_STUDY.md"
DEFAULT_STABILITY = ROOT / "projects/01-numerical-methods/results/stability.json"
DEFAULT_GRID_REFINEMENT = (
    ROOT / "projects/01-numerical-methods/results/grid-refinement.json"
)
DEFAULT_PEAK_REFINEMENT = (
    ROOT / "projects/01-numerical-methods/results/peak-refinement.json"
)
DEFAULT_CASE_STUDY = ROOT / "projects/02-monte-carlo-option-pricing/CASE_STUDY.md"
DEFAULT_CONVERGENCE = (
    ROOT / "projects/02-monte-carlo-option-pricing/results/convergence.json"
)
DEFAULT_COVERAGE_GRID = (
    ROOT / "projects/02-monte-carlo-option-pricing/results/coverage-grid.json"
)


class EvidenceConsistencyError(ValueError):
    """The narrative and its machine-readable evidence disagree."""


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise EvidenceConsistencyError(f"Artifact must contain a rows list: {path}")
    return data


def _parse_convergence_table(markdown: str) -> dict[tuple[str, bool], tuple[str, ...]]:
    rows: dict[tuple[str, bool], tuple[str, ...]] = {}
    for line in markdown.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[0] not in {"Call", "Put"}:
            continue
        if cells[1] not in {"Plain", "Antithetic"}:
            continue
        key = (cells[0].lower(), cells[1] == "Antithetic")
        if key in rows:
            raise EvidenceConsistencyError(f"Duplicate convergence table row: {key}")
        rows[key] = tuple(cells[2:])
    if len(rows) != 4:
        raise EvidenceConsistencyError(
            f"Expected four convergence table rows in case study, found {len(rows)}"
        )
    return rows


def _require_match(pattern: str, markdown: str, description: str) -> tuple[str, ...]:
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise EvidenceConsistencyError(f"Missing {description} in case study")
    return match.groups()


def _parse_stability_table(markdown: str) -> dict[str, tuple[float, ...]]:
    rows: dict[str, tuple[float, ...]] = {}
    for line in markdown.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] not in {"Equidistant", "Chebyshev-Lobatto"}:
            continue
        key = cells[0].lower()
        if key in rows:
            raise EvidenceConsistencyError(f"Duplicate stability table row: {key}")
        try:
            rows[key] = tuple(float(value) for value in cells[1:])
        except ValueError as error:
            raise EvidenceConsistencyError(f"Non-numeric stability table row: {cells}") from error
    if len(rows) != 2:
        raise EvidenceConsistencyError(
            f"Expected two stability table rows in case study, found {len(rows)}"
        )
    return rows


def _require_exact_numbers(
    actual: tuple[float, ...], expected: tuple[float, ...], description: str
) -> None:
    if actual != expected:
        raise EvidenceConsistencyError(
            f"{description} mismatch: expected {expected}, found {actual}"
        )


def validate_project01_case_study(
    case_study: Path = DEFAULT_PROJECT01_CASE_STUDY,
    stability_artifact: Path = DEFAULT_STABILITY,
    grid_artifact: Path = DEFAULT_GRID_REFINEMENT,
    peak_artifact: Path = DEFAULT_PEAK_REFINEMENT,
) -> dict[str, int]:
    """Compare Project 01 narrative values with all three numerical artifacts."""
    markdown = case_study.read_text(encoding="utf-8")
    stability = _load_json(stability_artifact)
    grid = _load_json(grid_artifact)
    peak = _load_json(peak_artifact)

    stability_rows = stability["rows"]
    if len(stability_rows) != 8:
        raise EvidenceConsistencyError(
            f"Expected 8 stability rows, found {len(stability_rows)}"
        )
    degree_16: dict[str, dict] = {}
    for row in stability_rows:
        if int(row["degree"]) == 16:
            design = str(row["node_design"])
            if design in degree_16:
                raise EvidenceConsistencyError(f"Duplicate degree-16 row: {design}")
            degree_16[design] = row
    if set(degree_16) != {"equidistant", "chebyshev-lobatto"}:
        raise EvidenceConsistencyError(
            f"Missing degree-16 stability designs: {sorted(degree_16)}"
        )

    peak_rows = peak["rows"]
    if len(peak_rows) != 2:
        raise EvidenceConsistencyError(
            f"Expected 2 peak-refinement rows, found {len(peak_rows)}"
        )
    peak_by_design = {str(row["node_design"]): row for row in peak_rows}
    if set(peak_by_design) != {"equidistant", "chebyshev-lobatto"}:
        raise EvidenceConsistencyError(
            f"Missing peak-refinement designs: {sorted(peak_by_design)}"
        )

    employer_summary = tuple(
        float(value)
        for value in _require_match(
            r"equally\s+spaced nodes produced a maximum approximation error of `([0-9.]+)`"
            r".*?locally refined Lebesgue estimate of `([0-9.]+)`\. "
            r"Chebyshev-Lobatto nodes.*?maximum approximation error to `([0-9.]+)`"
            r".*?refined\s+conditioning estimate to `([0-9.]+)`\. The refined "
            r"conditioning ratio was\s+`([0-9.]+)x`",
            markdown,
            "Project 01 employer summary values",
        )
    )
    expected_employer_summary = (
        round(float(degree_16["equidistant"]["max_approximation_error"]), 6),
        round(float(peak_by_design["equidistant"]["refined_estimate"]), 6),
        round(float(degree_16["chebyshev-lobatto"]["max_approximation_error"]), 6),
        round(float(peak_by_design["chebyshev-lobatto"]["refined_estimate"]), 6),
        round(float(peak["refined_equidistant_to_chebyshev_ratio"]), 2),
    )
    _require_exact_numbers(
        employer_summary, expected_employer_summary, "Project 01 employer summary"
    )

    stability_table = _parse_stability_table(markdown)
    for design in ("equidistant", "chebyshev-lobatto"):
        row = degree_16[design]
        expected = (
            round(float(row["max_approximation_error"]), 6),
            float(f"{float(row['max_floating_point_error']):.2e}"),
            round(float(row["estimated_lebesgue_constant"]), 3),
            float(f"{float(row['observed_output_change']):.6e}"),
        )
        _require_exact_numbers(
            stability_table[design], expected, f"Degree-16 stability row {design}"
        )

    grid_rows = grid["rows"]
    summaries = grid.get("summaries")
    ratios = grid.get("design_ratios")
    if len(grid_rows) != 6 or not isinstance(summaries, list) or len(summaries) != 2:
        raise EvidenceConsistencyError("Grid artifact must contain 6 rows and 2 summaries")
    if not isinstance(ratios, list) or len(ratios) != 3:
        raise EvidenceConsistencyError("Grid artifact must contain 3 design ratios")
    summary_by_design = {str(row["node_design"]): row for row in summaries}
    narrative_grid_changes = tuple(
        float(value)
        for value in _require_match(
            r"estimates changed by only `([0-9.]+)%` for\s+equidistant nodes and "
            r"`([0-9.]+)%` for Chebyshev-Lobatto nodes",
            markdown,
            "grid-refinement changes",
        )
    )
    expected_grid_changes = (
        round(float(summary_by_design["equidistant"]["coarse_to_fine_relative_change"]) * 100, 4),
        round(
            float(
                summary_by_design["chebyshev-lobatto"]["coarse_to_fine_relative_change"]
            )
            * 100,
            4,
        ),
    )
    _require_exact_numbers(
        narrative_grid_changes, expected_grid_changes, "Grid-refinement changes"
    )

    narrative_ratio_range = tuple(
        float(value)
        for value in _require_match(
            r"conditioning\s+ratio remained between `([0-9.]+)x` and `([0-9.]+)x`",
            markdown,
            "grid design-ratio range",
        )
    )
    ratio_values = [float(row["equidistant_to_chebyshev_ratio"]) for row in ratios]
    expected_ratio_range = (round(min(ratio_values), 2), round(max(ratio_values), 2))
    _require_exact_numbers(
        narrative_ratio_range, expected_ratio_range, "Grid design-ratio range"
    )

    narrative_residuals = tuple(
        float(value)
        for value in _require_match(
            r"Residual grid bias was `([0-9.]+)%` and `([0-9.]+)%`; the\s+"
            r"refined ratio was `([0-9.]+)x`",
            markdown,
            "peak-refinement residuals",
        )
    )
    expected_residuals = (
        round(float(peak_by_design["equidistant"]["relative_residual"]) * 100, 5),
        round(float(peak_by_design["chebyshev-lobatto"]["relative_residual"]) * 100, 6),
        round(float(peak["refined_equidistant_to_chebyshev_ratio"]), 2),
    )
    _require_exact_numbers(
        narrative_residuals, expected_residuals, "Peak-refinement residuals"
    )

    if not bool(grid["conclusion"]["stable"]) or not bool(peak["conclusion"]["stable"]):
        raise EvidenceConsistencyError(
            "Case study describes stable conclusions, but an artifact conclusion is false"
        )

    return {
        "stability_rows": len(stability_rows),
        "grid_rows": len(grid_rows),
        "peak_rows": len(peak_rows),
    }


def validate_project02_case_study(
    case_study: Path = DEFAULT_CASE_STUDY,
    convergence_artifact: Path = DEFAULT_CONVERGENCE,
    coverage_artifact: Path = DEFAULT_COVERAGE_GRID,
) -> dict[str, int]:
    """Compare headline narrative values with the committed experiment rows."""
    markdown = case_study.read_text(encoding="utf-8")
    convergence = _load_json(convergence_artifact)
    coverage = _load_json(coverage_artifact)

    convergence_rows = convergence["rows"]
    if len(convergence_rows) != 16:
        raise EvidenceConsistencyError(
            f"Expected 16 convergence rows, found {len(convergence_rows)}"
        )

    by_key: dict[tuple[str, bool, int], dict] = {}
    for row in convergence_rows:
        key = (row["option_type"], bool(row["antithetic"]), int(row["n_paths"]))
        if key in by_key:
            raise EvidenceConsistencyError(f"Duplicate convergence artifact row: {key}")
        by_key[key] = row

    table = _parse_convergence_table(markdown)
    for option_type in ("call", "put"):
        for antithetic in (False, True):
            low_key = (option_type, antithetic, 1000)
            high_key = (option_type, antithetic, 64000)
            if low_key not in by_key or high_key not in by_key:
                raise EvidenceConsistencyError(
                    f"Missing convergence endpoint rows for {(option_type, antithetic)}"
                )
            low = by_key[low_key]
            high = by_key[high_key]
            expected = (
                f"{low['mean_reported_se']:.6f}",
                f"{high['mean_reported_se']:.6f}",
                f"{low['rmse']:.6f}",
                f"{high['rmse']:.6f}",
            )
            actual = table[(option_type, antithetic)]
            if actual != expected:
                raise EvidenceConsistencyError(
                    "Convergence table mismatch for "
                    f"{(option_type, antithetic)}: expected {expected}, found {actual}"
                )

    convergence_replications = int(convergence["replications"])
    covered = [int(row["covered_replications"]) for row in convergence_rows]
    narrative_range = tuple(
        int(value)
        for value in _require_match(
            r"Observed coverage in\s+this 50-replication study ranged from "
            r"(\d+)/(\d+) to (\d+)/(\d+)",
            markdown,
            "convergence coverage range",
        )
    )
    expected_range = (
        min(covered),
        convergence_replications,
        max(covered),
        convergence_replications,
    )
    if narrative_range != expected_range:
        raise EvidenceConsistencyError(
            f"Convergence coverage mismatch: expected {expected_range}, found {narrative_range}"
        )

    coverage_rows = coverage["rows"]
    if len(coverage_rows) != 36:
        raise EvidenceConsistencyError(
            f"Expected 36 coverage-grid rows, found {len(coverage_rows)}"
        )
    grid_replications = int(coverage["replications"])
    grid_covered = [int(row["covered_replications"]) for row in coverage_rows]
    narrative_grid_range = tuple(
        int(value)
        for value in _require_match(
            r"Observed coverage ranged from (\d+)/(\d+) \(`\d+%`\) to "
            r"(\d+)/(\d+) \(`\d+%`\)",
            markdown,
            "coverage-grid range",
        )
    )
    expected_grid_range = (
        min(grid_covered),
        grid_replications,
        max(grid_covered),
        grid_replications,
    )
    if narrative_grid_range != expected_grid_range:
        raise EvidenceConsistencyError(
            "Coverage-grid range mismatch: "
            f"expected {expected_grid_range}, found {narrative_grid_range}"
        )

    intervals = [row["coverage_binomial_ci95"] for row in coverage_rows]
    if not all(float(lower) <= 0.95 <= float(upper) for lower, upper in intervals):
        raise EvidenceConsistencyError(
            "Case study says every exact interval contains 95%, but an artifact row does not"
        )

    widest = max(intervals, key=lambda interval: float(interval[1]) - float(interval[0]))
    narrative_widest = tuple(
        float(value)
        for value in _require_match(
            r"widest recorded interval was approximately `([0-9.]+)–([0-9.]+)`",
            markdown,
            "widest coverage interval",
        )
    )
    expected_widest = (round(float(widest[0]), 4), round(float(widest[1]), 4))
    if narrative_widest != expected_widest:
        raise EvidenceConsistencyError(
            f"Widest interval mismatch: expected {expected_widest}, found {narrative_widest}"
        )

    return {
        "convergence_rows": len(convergence_rows),
        "coverage_rows": len(coverage_rows),
        "convergence_min_covered": min(covered),
        "convergence_max_covered": max(covered),
        "coverage_min_covered": min(grid_covered),
        "coverage_max_covered": max(grid_covered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project01-case-study", type=Path, default=DEFAULT_PROJECT01_CASE_STUDY
    )
    parser.add_argument("--stability", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--grid-refinement", type=Path, default=DEFAULT_GRID_REFINEMENT)
    parser.add_argument("--peak-refinement", type=Path, default=DEFAULT_PEAK_REFINEMENT)
    parser.add_argument("--case-study", type=Path, default=DEFAULT_CASE_STUDY)
    parser.add_argument("--convergence", type=Path, default=DEFAULT_CONVERGENCE)
    parser.add_argument("--coverage-grid", type=Path, default=DEFAULT_COVERAGE_GRID)
    args = parser.parse_args()
    try:
        project01_summary = validate_project01_case_study(
            args.project01_case_study,
            args.stability,
            args.grid_refinement,
            args.peak_refinement,
        )
        project02_summary = validate_project02_case_study(
            args.case_study, args.convergence, args.coverage_grid
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, EvidenceConsistencyError) as error:
        print(f"Case-study evidence validation failed: {error}", file=sys.stderr)
        return 2
    print(
        "Case-study evidence valid: "
        f"project 01 has {project01_summary['stability_rows']} stability rows, "
        f"{project01_summary['grid_rows']} grid rows and "
        f"{project01_summary['peak_rows']} peak rows; "
        f"project 02 has {project02_summary['convergence_rows']} convergence rows and "
        f"{project02_summary['coverage_rows']} coverage rows, "
        f"coverage counts {project02_summary['coverage_min_covered']}–"
        f"{project02_summary['coverage_max_covered']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
