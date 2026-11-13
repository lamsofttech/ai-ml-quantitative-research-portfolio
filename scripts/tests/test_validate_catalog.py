"""Check malformed metadata and the real CLI with and without optimization."""

import copy
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_catalog.py"
SPEC = importlib.util.spec_from_file_location("validate_catalog", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CatalogValidationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = yaml.safe_load((ROOT / "projects.yml").read_text(encoding="utf-8"))

    def test_repository_catalog_is_valid(self):
        MODULE.validate_catalog(self.catalog)

    def test_malformed_top_level(self):
        for catalog in (None, [], {}, {"portfolio": []}):
            with self.subTest(catalog=catalog), self.assertRaises(ValueError):
                MODULE.validate_catalog(catalog)

    def test_invalid_category_collection(self):
        for categories in (None, "abcdefghi", [], ["x"] * 9, [None] * 9):
            with self.subTest(categories=categories), self.assertRaises(ValueError):
                MODULE.validate_catalog({**self.catalog, "categories": categories})

    def test_invalid_project_collection(self):
        for projects in (None, {}, [], [None] * 5):
            with self.subTest(projects=projects), self.assertRaises(ValueError):
                MODULE.validate_catalog({**self.catalog, "projects": projects})

    def test_duplicate_id(self):
        self.catalog["projects"][1]["id"] = self.catalog["projects"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate project id"):
            MODULE.validate_catalog(self.catalog)

    def test_missing_id(self):
        del self.catalog["projects"][0]["id"]
        with self.assertRaisesRegex(ValueError, "non-empty id"):
            MODULE.validate_catalog(self.catalog)

    def test_duplicate_resolved_path(self):
        self.catalog["projects"][1]["path"] = self.catalog["projects"][0]["path"] + "/."
        with self.assertRaisesRegex(ValueError, "Duplicate project path"):
            MODULE.validate_catalog(self.catalog)

    def test_missing_directory(self):
        self.catalog["projects"][0]["path"] = "projects/does-not-exist"
        with self.assertRaisesRegex(ValueError, "Missing project directory"):
            MODULE.validate_catalog(self.catalog)

    def test_invalid_paths(self):
        for path in (None, "", "..", str(ROOT), ".", "projects.yml"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                catalog = copy.deepcopy(self.catalog)
                catalog["projects"][0]["path"] = path
                MODULE.validate_catalog(catalog)

    def test_invalid_assigned_categories(self):
        for assigned in (None, [], "Numerical Computing", ["unknown"], [None],
                         ["Numerical Computing", "Numerical Computing"]):
            with self.subTest(assigned=assigned), self.assertRaises(ValueError):
                catalog = copy.deepcopy(self.catalog)
                catalog["projects"][0]["categories"] = assigned
                MODULE.validate_catalog(catalog)

    def test_cli_valid_in_normal_and_optimized_python(self):
        for flags in ([], ["-O"]):
            result = subprocess.run(
                [sys.executable, *flags, str(SCRIPT)], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Catalog valid", result.stdout)

    def test_cli_rejects_invalid_catalog_under_optimization(self):
        self.catalog["portfolio"]["repository"] = "other"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yml"
            for text in (yaml.safe_dump(self.catalog), "projects: ["):
                path.write_text(text, encoding="utf-8")
                for flags in ([], ["-O"]):
                    result = subprocess.run(
                        [sys.executable, *flags, str(SCRIPT), "--catalog", str(path)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("Catalog validation failed", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
