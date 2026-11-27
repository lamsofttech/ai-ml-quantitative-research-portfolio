"""Test machine-checked Project 01 case-study evidence."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_case_study_evidence.py"
CASE_STUDY = ROOT / "projects/01-numerical-methods/CASE_STUDY.md"
STABILITY = ROOT / "projects/01-numerical-methods/results/stability.json"
GRID = ROOT / "projects/01-numerical-methods/results/grid-refinement.json"
PEAK = ROOT / "projects/01-numerical-methods/results/peak-refinement.json"
SPEC = importlib.util.spec_from_file_location("validate_case_study_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Project01CaseStudyEvidenceTests(unittest.TestCase):
    def test_repository_case_study_matches_all_artifacts(self):
        summary = MODULE.validate_project01_case_study()
        self.assertEqual(summary["stability_rows"], 8)
        self.assertEqual(summary["grid_rows"], 6)
        self.assertEqual(summary["peak_rows"], 2)

    def test_changed_stability_value_fails(self):
        markdown = CASE_STUDY.read_text(encoding="utf-8").replace(
            "14.363078", "14.363079", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "CASE_STUDY.md"
            changed.write_text(markdown, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "employer summary mismatch"):
                MODULE.validate_project01_case_study(changed, STABILITY, GRID, PEAK)

    def test_changed_grid_change_fails(self):
        markdown = CASE_STUDY.read_text(encoding="utf-8").replace(
            "0.1425%", "0.1426%", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "CASE_STUDY.md"
            changed.write_text(markdown, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Grid-refinement changes mismatch"):
                MODULE.validate_project01_case_study(changed, STABILITY, GRID, PEAK)

    def test_changed_refined_ratio_fails(self):
        markdown = CASE_STUDY.read_text(encoding="utf-8").replace(
            "refined ratio was `342.98x`", "refined ratio was `342.97x`", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "CASE_STUDY.md"
            changed.write_text(markdown, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Peak-refinement residuals mismatch"):
                MODULE.validate_project01_case_study(changed, STABILITY, GRID, PEAK)

    def test_cli_validates_both_projects(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "project 01 has 8 stability rows, 6 grid rows and 2 peak rows",
            result.stdout,
        )
        self.assertIn(
            "project 02 has 16 convergence rows and 36 coverage rows",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
