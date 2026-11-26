"""Test machine-checked Project 02 case-study evidence."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_case_study_evidence.py"
CASE_STUDY = ROOT / "projects/02-monte-carlo-option-pricing/CASE_STUDY.md"
CONVERGENCE = (
    ROOT / "projects/02-monte-carlo-option-pricing/results/convergence.json"
)
COVERAGE_GRID = (
    ROOT / "projects/02-monte-carlo-option-pricing/results/coverage-grid.json"
)
SPEC = importlib.util.spec_from_file_location("validate_case_study_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaseStudyEvidenceTests(unittest.TestCase):
    def test_repository_case_study_matches_artifacts(self):
        summary = MODULE.validate_project02_case_study()
        self.assertEqual(summary["convergence_rows"], 16)
        self.assertEqual(summary["coverage_rows"], 36)
        self.assertEqual(summary["coverage_min_covered"], 188)
        self.assertEqual(summary["coverage_max_covered"], 196)

    def test_changed_convergence_value_fails(self):
        markdown = CASE_STUDY.read_text(encoding="utf-8").replace(
            "0.448147", "0.448148", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "CASE_STUDY.md"
            changed.write_text(markdown, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Convergence table mismatch"):
                MODULE.validate_project02_case_study(changed, CONVERGENCE, COVERAGE_GRID)

    def test_changed_coverage_range_fails(self):
        markdown = CASE_STUDY.read_text(encoding="utf-8").replace(
            "188/200 (`94%`)", "189/200 (`94%`)", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "CASE_STUDY.md"
            changed.write_text(markdown, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Coverage-grid range mismatch"):
                MODULE.validate_project02_case_study(changed, CONVERGENCE, COVERAGE_GRID)

    def test_cli_validates_repository_evidence(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("16 convergence rows", result.stdout)
        self.assertIn("36 coverage rows", result.stdout)

    def test_cli_rejects_changed_case_study(self):
        markdown = CASE_STUDY.read_text(encoding="utf-8").replace(
            "0.448147", "0.448148", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "CASE_STUDY.md"
            changed.write_text(markdown, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--case-study",
                    str(changed),
                    "--convergence",
                    str(CONVERGENCE),
                    "--coverage-grid",
                    str(COVERAGE_GRID),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Convergence table mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
