"""Publish the portfolio hub after validation; never print credentials."""
import os
from pathlib import Path


def main():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is missing; Hugging Face deployment is blocked.")
    source = Path(__file__).resolve().parents[1] / "spaces" / "portfolio-hub"
    for name in ("README.md", "app.py", "requirements.txt"):
        if not (source / name).is_file():
            raise SystemExit(f"Required Space file missing: {name}")
    commit = HfApi(token=token).upload_folder(
        repo_id="nyakslamo/ai-ml-quantitative-research-portfolio",
        repo_type="space",
        folder_path=str(source),
        allow_patterns=["README.md", "app.py", "requirements.txt"],
        commit_message="Sync validated portfolio hub from GitHub",
    )
    print(f"Space source uploaded: {commit.commit_url}")
    print("Upload does not confirm Space build or runtime health; verify those separately.")

if __name__ == "__main__":
    main()
