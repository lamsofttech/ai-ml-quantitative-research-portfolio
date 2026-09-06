from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    catalog = yaml.safe_load((ROOT / "projects.yml").read_text(encoding="utf-8"))
    assert catalog["portfolio"]["repository"] == "ai-ml-quantitative-research-portfolio"
    assert len(catalog["categories"]) == 9
    assert len(catalog["projects"]) == 5
    for project in catalog["projects"]:
        assert (ROOT / project["path"]).is_dir(), f"Missing {project['path']}"
        assert project["categories"], f"Missing categories for {project['id']}"


if __name__ == "__main__":
    main()

