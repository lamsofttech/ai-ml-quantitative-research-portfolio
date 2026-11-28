"""Verify evidence artifact integrity and embedded provenance metadata."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "portfolio/evidence-manifest.json"
HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceProvenanceError(ValueError):
    """An evidence artifact does not match its reviewed provenance contract."""


def _load_object(path: Path, description: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EvidenceProvenanceError(f"{description} must be a JSON object: {path}")
    return data


def _resolve_repo_path(root: Path, raw_path: object, description: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceProvenanceError(f"{description} must be a non-empty repository path")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise EvidenceProvenanceError(f"{description} must be relative: {raw_path}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise EvidenceProvenanceError(
            f"{description} leaves the repository: {raw_path}"
        ) from error
    return resolved


def _require_digest(value: object, length: int, description: str) -> str:
    pattern = HEX_40 if length == 40 else HEX_64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise EvidenceProvenanceError(
            f"{description} must be a lowercase {length}-character hexadecimal digest"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_file_at_commit(root: Path, code_version: str, source_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{code_version}:{source_path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise EvidenceProvenanceError(
            f"Cannot read historical source {source_path} at {code_version}: {detail}"
        )
    return result.stdout


def validate_evidence_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    repository_root: Path = ROOT,
    verify_git_sources: bool = False,
) -> dict[str, int]:
    """Validate exact artifact bytes and their embedded provenance contract."""
    manifest = _load_object(manifest_path, "Evidence manifest")
    if manifest.get("schema_version") != 1:
        raise EvidenceProvenanceError("Evidence manifest schema_version must be 1")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise EvidenceProvenanceError("Evidence manifest must contain artifacts")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    projects: set[str] = set()
    source_hash_count = 0
    historical_source_count = 0

    for entry in entries:
        if not isinstance(entry, dict):
            raise EvidenceProvenanceError("Each evidence manifest entry must be an object")
        artifact_id = entry.get("id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise EvidenceProvenanceError("Each evidence artifact needs a non-empty id")
        if artifact_id in seen_ids:
            raise EvidenceProvenanceError(f"Duplicate evidence artifact id: {artifact_id}")
        seen_ids.add(artifact_id)

        project = entry.get("project")
        if not isinstance(project, str) or not project:
            raise EvidenceProvenanceError(f"Artifact {artifact_id} needs a project id")
        projects.add(project)

        raw_artifact_path = entry.get("artifact")
        if not isinstance(raw_artifact_path, str):
            raise EvidenceProvenanceError(f"Artifact {artifact_id} needs an artifact path")
        if raw_artifact_path in seen_paths:
            raise EvidenceProvenanceError(
                f"Duplicate evidence artifact path: {raw_artifact_path}"
            )
        seen_paths.add(raw_artifact_path)
        artifact_path = _resolve_repo_path(
            repository_root, raw_artifact_path, f"Artifact {artifact_id} path"
        )
        report_path = _resolve_repo_path(
            repository_root, entry.get("report"), f"Artifact {artifact_id} report"
        )
        if not artifact_path.is_file():
            raise EvidenceProvenanceError(
                f"Artifact {artifact_id} file does not exist: {raw_artifact_path}"
            )
        if not report_path.is_file():
            raise EvidenceProvenanceError(
                f"Artifact {artifact_id} report does not exist: {entry.get('report')}"
            )

        expected_hash = _require_digest(
            entry.get("sha256"), 64, f"Artifact {artifact_id} sha256"
        )
        actual_hash = _sha256(artifact_path)
        if actual_hash != expected_hash:
            raise EvidenceProvenanceError(
                f"Artifact {artifact_id} SHA-256 mismatch: expected {expected_hash}, "
                f"found {actual_hash}"
            )

        expected_code_version = _require_digest(
            entry.get("code_version"), 40, f"Artifact {artifact_id} code_version"
        )
        artifact = _load_object(artifact_path, f"Artifact {artifact_id}")
        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict):
            raise EvidenceProvenanceError(
                f"Artifact {artifact_id} must contain provenance metadata"
            )
        actual_code_version = provenance.get("code_version")
        if actual_code_version != expected_code_version:
            raise EvidenceProvenanceError(
                f"Artifact {artifact_id} code version mismatch: expected "
                f"{expected_code_version}, found {actual_code_version}"
            )

        expected_sources = entry.get("source_files")
        if (
            not isinstance(expected_sources, list)
            or not expected_sources
            or not all(isinstance(name, str) and name for name in expected_sources)
            or len(set(expected_sources)) != len(expected_sources)
        ):
            raise EvidenceProvenanceError(
                f"Artifact {artifact_id} source_files must be a unique non-empty list"
            )
        source_root = _resolve_repo_path(
            repository_root,
            entry.get("source_root"),
            f"Artifact {artifact_id} source_root",
        )
        if verify_git_sources and not source_root.is_dir():
            raise EvidenceProvenanceError(
                f"Artifact {artifact_id} source_root does not exist: "
                f"{entry.get('source_root')}"
            )
        relative_source_root = source_root.relative_to(repository_root.resolve())
        source_hashes = provenance.get("source_sha256")
        if not isinstance(source_hashes, dict) or set(source_hashes) != set(expected_sources):
            actual_sources = sorted(source_hashes) if isinstance(source_hashes, dict) else []
            raise EvidenceProvenanceError(
                f"Artifact {artifact_id} source file set mismatch: expected "
                f"{sorted(expected_sources)}, found {actual_sources}"
            )
        for source_name, digest in source_hashes.items():
            if Path(source_name).name != source_name:
                raise EvidenceProvenanceError(
                    f"Artifact {artifact_id} source name must be a filename: {source_name}"
                )
            _require_digest(
                digest, 64, f"Artifact {artifact_id} source hash for {source_name}"
            )
            source_hash_count += 1
            if verify_git_sources:
                source_path = (relative_source_root / source_name).as_posix()
                historical_source = _git_file_at_commit(
                    repository_root.resolve(), expected_code_version, source_path
                )
                historical_hash = hashlib.sha256(historical_source).hexdigest()
                if historical_hash != digest:
                    raise EvidenceProvenanceError(
                        f"Artifact {artifact_id} historical source hash mismatch for "
                        f"{source_name}: expected {digest}, found {historical_hash}"
                    )
                historical_source_count += 1

    return {
        "artifacts": len(entries),
        "projects": len(projects),
        "source_hashes": source_hash_count,
        "historical_sources": historical_source_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--verify-git-sources",
        action="store_true",
        help="hash each recorded source file from its historical code-version commit",
    )
    args = parser.parse_args()
    try:
        summary = validate_evidence_manifest(
            args.manifest, args.repository_root, args.verify_git_sources
        )
    except (OSError, UnicodeError, json.JSONDecodeError, EvidenceProvenanceError) as error:
        print(f"Evidence provenance validation failed: {error}", file=sys.stderr)
        return 2
    print(
        "Evidence provenance valid: "
        f"{summary['artifacts']} artifacts across {summary['projects']} projects, "
        f"{summary['source_hashes']} embedded source hashes checked"
        + (
            f", {summary['historical_sources']} historical source blobs verified"
            if args.verify_git_sources
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
