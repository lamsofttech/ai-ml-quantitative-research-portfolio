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
python -O scripts/validate_catalog.py
python -m unittest discover -s scripts/tests -v
```

The CI Python matrix installs each project separately on Python 3.10, 3.12 and 3.14. Its catalog job checks the five-project catalog and scripts. Its Docker matrix builds the root numerical demo and the project 02 Streamlit image without publishing images. Passing local checks on 3.12 does not verify the other Python versions or Docker builds.

## Security scope
The Security workflow runs Bandit 1.9.4 over production source, app code and scripts, failing on medium/high severity findings. It excludes test directories by scanning explicit production paths. It is not a dependency vulnerability audit or a secret scanner. Dependabot remains configured separately. The catalog validator uses explicit errors that remain active under python -O. Its regression suite verifies malformed metadata, duplicate IDs/paths, invalid category references, missing directories and command-line failures in normal and optimized execution. Catalog errors print a diagnostic and return exit code 2; successful validation returns 0. Test code in scripts/tests is explicitly excluded from the production security scan.

To reproduce the security gate:
```bash
python -m pip install bandit==1.9.4
python -m bandit -r projects/01-numerical-methods/src projects/02-monte-carlo-option-pricing/src projects/01-numerical-methods/app.py projects/02-monte-carlo-option-pricing/app spaces/portfolio-hub scripts --exclude scripts/tests --severity-level medium
```

## Deployment evidence
A GitHub commit, successful Actions run, Hugging Face upload, successful Space build and working application are separate checks. A Space reporting RUNNING does not show that it contains the latest GitHub code. When package source changes, test it before updating the Space dependency commit pins.

On November 12, 2026, GitHub's publish job annotation said the account was locked due to a billing issue. No job steps ran. This prevents Actions-based validation and publishing; it does not establish whether HF_TOKEN exists or is valid. Repository owners must resolve that account issue outside this code change. Existing Space runtime metadata reported RUNNING at commit 40efb957fa2f67f3100d16421a1ffc583fd1ebeb; no upload was performed.

## 10-15 minute reading exercise
Read .github/workflows/ci.yml and the two project test directories. Find one mathematical property tested in each project, then explain how the CI matrix rechecks those properties on a new commit.
Question: Why can all 64 local tests pass while the GitHub badge and Hugging Face deployment still fail?

## Catalog validation under optimization
Run `python scripts/validate_catalog.py --catalog path/to/catalog.yml` to validate another catalog against directories in this repository. Counts remain fixed at five projects and nine categories to preserve the current roadmap. `assert` is appropriate inside tests, but operational validation must use explicit error handling because Python can remove assertions with `-O`.

Read scripts/validate_catalog.py and scripts/tests/test_validate_catalog.py for 10-15 minutes. Trace one malformed catalog through the CLI to its error message and nonzero exit status. Question: Why is a normal-mode success test insufficient to prove that invalid data is rejected under optimization?
