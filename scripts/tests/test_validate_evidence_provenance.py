"""Test evidence artifact integrity and provenance validation."""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_evidence_provenance.py"
MANIFEST = ROOT / "portfolio" / "evidence-manifest.json"
SPEC = importlib.util.spec_from_file_location("validate_evidence_provenance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def copy_fixture(destination: Path) -> Path:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    copied_manifest = destination / "portfolio" / "evidence-manifest.json"
    copied_manifest.parent.mkdir(parents=True)
    shutil.copyfile(MANIFEST, copied_manifest)
    for entry in manifest["artifacts"]:
        for key in ("artifact", "report"):
            source = ROOT / entry[key]
            target = destination / entry[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return copied_manifest


class EvidenceProvenanceTests(unittest.TestCase):
    def test_repository_manifest_matches_all_artifacts(self):
        summary = MODULE.validate_evidence_manifest(verify_git_sources=True)
        self.assertEqual(summary["artifacts"], 5)
        self.assertEqual(summary["projects"], 2)
        self.assertEqual(summary["source_hashes"], 14)
        self.assertEqual(summary["historical_sources"], 14)

    def test_changed_artifact_bytes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = copy_fixture(root)
            artifact = root / "projects/01-numerical-methods/results/stability.json"
            artifact.write_bytes(artifact.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.validate_evidence_manifest(manifest, root)

    def test_changed_manifest_code_version_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = copy_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"][0]["code_version"] = "0" * 40
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "code version mismatch"):
                MODULE.validate_evidence_manifest(manifest_path, root)

    def test_changed_source_file_set_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = copy_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"][0]["source_files"] = ["stability.py"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source file set mismatch"):
                MODULE.validate_evidence_manifest(manifest_path, root)

    def test_artifact_path_cannot_leave_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = copy_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"][0]["artifact"] = "../outside.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leaves the repository"):
                MODULE.validate_evidence_manifest(manifest_path, root)

    def test_cli_validates_repository_manifest(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--verify-git-sources"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("5 artifacts across 2 projects", result.stdout)
        self.assertIn("14 embedded source hashes", result.stdout)
        self.assertIn("14 historical source blobs", result.stdout)


if __name__ == "__main__":
    unittest.main()
