"""Check that local evidence links in employer-facing case studies resolve."""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_STUDIES = (
    ROOT / "projects/01-numerical-methods/CASE_STUDY.md",
    ROOT / "projects/02-monte-carlo-option-pricing/CASE_STUDY.md",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


class EvidenceLinkError(ValueError):
    """A case study contains a missing or unsafe local evidence link."""


def local_link_targets(markdown: str) -> list[str]:
    """Return local Markdown link targets, ignoring anchors and web links."""
    targets: list[str] = []
    for raw_target in MARKDOWN_LINK.findall(markdown):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        targets.append(unquote(parsed.path))
    return targets


def validate_case_study(path: Path, root: Path = ROOT) -> int:
    """Validate local evidence links and return the number checked."""
    root = root.resolve()
    path = path.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise EvidenceLinkError(f"Case study is missing or outside the repository: {path}")

    targets = local_link_targets(path.read_text(encoding="utf-8"))
    if not targets:
        raise EvidenceLinkError(f"Case study has no local evidence links: {path}")

    for target in targets:
        resolved = (path.parent / target).resolve()
        if not resolved.is_relative_to(root):
            raise EvidenceLinkError(f"Link leaves repository: {path}: {target}")
        if not resolved.is_file():
            raise EvidenceLinkError(f"Broken evidence link: {path}: {target}")
    return len(targets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_CASE_STUDIES))
    args = parser.parse_args()
    try:
        checked = sum(validate_case_study(path) for path in args.paths)
    except (OSError, UnicodeError, EvidenceLinkError) as error:
        print(f"Case-study link validation failed: {error}", file=sys.stderr)
        return 2
    print(f"Case-study evidence links valid: {checked} local links checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
