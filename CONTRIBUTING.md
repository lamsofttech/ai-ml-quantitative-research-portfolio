# Contributing

Each project must remain independently installable and must document its data source, assumptions, evaluation design, leakage controls, limitations, and reproducibility instructions. Add only project-specific dependencies. Before opening a pull request, run the relevant tests and formatting checks.

## Adding a new project

CI does not auto-discover projects, so moving one from planned to implemented needs a few manual updates:

1. `projects.yml` — update the project's `status` and `interfaces`.
2. `.github/workflows/ci.yml` — add the project's path to the `python` job's matrix (and to the `docker` job's matrix if it has its own Dockerfile).
3. `.github/dependabot.yml` — add a `pip` entry (and a `docker` entry if it has its own Dockerfile) pointing at the new project directory.
4. Root `README.md` — update the project's row in the roadmap table.

