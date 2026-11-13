"""Validate portfolio metadata without optimizable assertions."""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class CatalogValidationError(ValueError):
    """The catalog violates the portfolio metadata contract."""


def validate_catalog(catalog: object, root: Path = ROOT) -> None:
    """Raise CatalogValidationError for malformed metadata, even under python -O."""
    if not isinstance(catalog, dict):
        raise CatalogValidationError("Catalog must be a mapping")
    portfolio = catalog.get("portfolio")
    if not isinstance(portfolio, dict) or portfolio.get("repository") != (
        "ai-ml-quantitative-research-portfolio"
    ):
        raise CatalogValidationError("Invalid portfolio.repository")
    categories = catalog.get("categories")
    if not isinstance(categories, list) or len(categories) != 9:
        raise CatalogValidationError("categories must contain exactly 9 entries")
    if any(not isinstance(c, str) or not c.strip() for c in categories):
        raise CatalogValidationError("categories must be non-empty strings")
    if len(set(categories)) != len(categories):
        raise CatalogValidationError("categories must be unique")
    projects = catalog.get("projects")
    if not isinstance(projects, list) or len(projects) != 5:
        raise CatalogValidationError("projects must contain exactly 5 entries")
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    root = root.resolve()
    for index, project in enumerate(projects, start=1):
        if not isinstance(project, dict):
            raise CatalogValidationError(f"Project {index} must be a mapping")
        project_id = project.get("id")
        if not isinstance(project_id, str) or not project_id.strip():
            raise CatalogValidationError(f"Project {index} needs a non-empty id")
        if project_id in seen_ids:
            raise CatalogValidationError(f"Duplicate project id: {project_id}")
        seen_ids.add(project_id)
        path = project.get("path")
        if not isinstance(path, str) or not path.strip():
            raise CatalogValidationError(f"Missing path for {project_id}")
        directory = (root / path).resolve()
        if Path(path).is_absolute() or not directory.is_relative_to(root) or directory == root:
            raise CatalogValidationError(f"Project path must stay inside the repository: {path}")
        if not directory.is_dir():
            raise CatalogValidationError(f"Missing project directory: {path}")
        if directory in seen_paths:
            raise CatalogValidationError(f"Duplicate project path: {path}")
        seen_paths.add(directory)
        assigned = project.get("categories")
        if not isinstance(assigned, list) or not assigned:
            raise CatalogValidationError(f"Missing category list for {project_id}")
        if any(not isinstance(c, str) or c not in categories for c in assigned):
            raise CatalogValidationError(f"Unknown category for {project_id}")
        if len(set(assigned)) != len(assigned):
            raise CatalogValidationError(f"Duplicate categories for {project_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "projects.yml")
    args = parser.parse_args()
    try:
        validate_catalog(yaml.safe_load(args.catalog.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        print(f"Catalog validation failed: {error}", file=sys.stderr)
        return 2
    print("Catalog valid: 5 projects, 9 categories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

