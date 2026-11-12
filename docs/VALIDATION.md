# Validating the portfolio

CI checks that the numerical-methods and option-pricing projects remain usable as independent packages. It does not establish forecasting accuracy, trading profitability, or research novelty.

## Local checks (Python 3.12)
From the repository root, activate a fresh virtual environment, then run:

```bash
python -m pip install -e "projects/01-numerical-methods[dev]" -e "projects/02-monte-carlo-option-pricing[dev]" -r requirements-dev.txt
python -m pytest projects/01-numerical-methods/tests
python -m pytest projects/02-monte-carlo-option-pricing/tests
python -m ruff check projects/01-numerical-methods projects/02-monte-carlo-option-pricing scripts
python scripts/validate_catalog.py
```

The CI Python matrix installs each project separately on Python 3.10, 3.12 and 3.14. Its catalog job checks the five-project catalog and scripts. Its Docker matrix builds the root numerical demo and the project 02 Streamlit image without publishing images. Passing local checks on 3.12 does not verify the other Python versions or Docker builds.

## Security scope
The Security workflow runs Bandit 1.9.4 over production source, app code and scripts, failing on medium/high severity findings. It excludes test directories by scanning explicit production paths. It is not a dependency vulnerability audit or a secret scanner. Dependabot remains configured separately. Five low severity B101 findings in validate_catalog.py identify assertions that Python optimization can remove; review them before using the validator with python -O.

To reproduce the security gate:
```bash
python -m pip install bandit==1.9.4
python -m bandit -r projects/01-numerical-methods/src projects/02-monte-carlo-option-pricing/src projects/01-numerical-methods/app.py projects/02-monte-carlo-option-pricing/app spaces/portfolio-hub scripts --severity-level medium
```

## Deployment evidence
A GitHub commit, successful Actions run, Hugging Face upload, successful Space build and working application are separate checks. A Space reporting RUNNING does not show that it contains the latest GitHub code. When package source changes, test it before updating the Space dependency commit pins.

On November 12, 2026, GitHub's publish job annotation said the account was locked due to a billing issue. No job steps ran. This prevents Actions-based validation and publishing; it does not establish whether HF_TOKEN exists or is valid. Repository owners must resolve that account issue outside this code change. Existing Space runtime metadata reported RUNNING at commit 40efb957fa2f67f3100d16421a1ffc583fd1ebeb; no upload was performed.

## 10-15 minute reading exercise
Read .github/workflows/ci.yml and the two project test directories. Find one mathematical property tested in each project, then explain how the CI matrix rechecks those properties on a new commit.
Question: Why can all 64 local tests pass while the GitHub badge and Hugging Face deployment still fail?
