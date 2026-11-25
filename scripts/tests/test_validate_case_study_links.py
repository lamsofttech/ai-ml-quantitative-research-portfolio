"""Test the case-study evidence-link validator and its real CLI."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_case_study_links.py"
SPEC = importlib.util.spec_from_file_location("validate_case_study_links", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvidenceLinkValidationTests(unittest.TestCase):
    def test_repository_case_study_links_resolve(self):
        expected_minimums = {
            ROOT / "projects/01-numerical-methods/CASE_STUDY.md": 9,
            ROOT / "projects/02-monte-carlo-option-pricing/CASE_STUDY.md": 11,
        }
        for case_study, expected_minimum in expected_minimums.items():
            with self.subTest(case_study=case_study):
                self.assertGreaterEqual(
                    MODULE.validate_case_study(case_study), expected_minimum
                )

    def test_external_and_anchor_links_are_ignored(self):
        markdown = "[section](#results) [web](https://example.com/x) [file](results.md)"
        self.assertEqual(MODULE.local_link_targets(markdown), ["results.md"])

    def test_image_links_are_ignored(self):
        self.assertEqual(MODULE.local_link_targets("![plot](results/plot.png)"), [])

    def test_missing_link_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_study = root / "CASE_STUDY.md"
            case_study.write_text("[missing](results/missing.json)", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Broken evidence link"):
                MODULE.validate_case_study(case_study, root)

    def test_link_cannot_leave_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_study = root / "CASE_STUDY.md"
            case_study.write_text("[outside](../outside.md)", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Link leaves repository"):
                MODULE.validate_case_study(case_study, root)

    def test_cli_validates_repository_case_study(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("evidence links valid", result.stdout)

    def test_cli_rejects_broken_case_study(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            case_study = Path(directory) / "CASE_STUDY.md"
            case_study.write_text("[missing](missing.json)", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(case_study)],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Broken evidence link", result.stderr)


if __name__ == "__main__":
    unittest.main()
