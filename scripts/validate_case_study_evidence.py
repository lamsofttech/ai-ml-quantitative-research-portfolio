"""Verify Project 02 case-study claims against committed JSON evidence."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--case-study", type=Path, default=DEFAULT_CASE_STUDY)
    parser.add_argument("--convergence", type=Path, default=DEFAULT_CONVERGENCE)
    parser.add_argument("--coverage-grid", type=Path, default=DEFAULT_COVERAGE_GRID)
    args = parser.parse_args()
    try:
        summary = validate_project02_case_study(
            args.case_study, args.convergence, args.coverage_grid
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, EvidenceConsistencyError) as error:
        print(f"Case-study evidence validation failed: {error}", file=sys.stderr)
        return 2
    print(
        "Case-study evidence valid: "
        f"{summary['convergence_rows']} convergence rows, "
        f"{summary['coverage_rows']} coverage rows, "
        f"coverage counts {summary['coverage_min_covered']}–"
        f"{summary['coverage_max_covered']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
