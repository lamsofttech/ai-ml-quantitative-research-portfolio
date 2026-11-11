# Portfolio vision and daily delivery contract

## Purpose
By the end of Lameck Nyakweba's two-year master's program, demonstrate practical AI/ML and quantitative research skills through reproducible projects that Lameck can explain, run, defend in interviews, and present to employers. Expected graduation is May 2028; confirm the actual submission date before setting a final deadline. Do not reset the program clock whenever an agent runs.

## Preserve the original scope
The source of truth is projects.yml, the root README, and docs/PROJECT_STANDARD.md. Deliver these five projects:
1. Numerical Methods for Quantitative Research.
2. Monte Carlo Option Pricing and Risk Analysis.
3. Financial Time-Series Forecasting.
4. Machine Learning for Credit-Risk Prediction.
5. Financial News and SEC Filing Analysis.

Do not replace this roadmap with unrelated projects or add new tracks without Lameck's direction. Maintain one active delivery milestone at a time, repairing broken foundations before adding features.

## Working delivery sequence
Dates below are planning targets, not claims of completion:
- November-December 2026: validate projects 01/02, restore any missing CI, verify deployed demos, document reproducible baselines and limitations.
- January-April 2027: forecasting data provenance, naive baselines, walk-forward evaluation, leakage controls, ML comparison and demo.
- May-August 2027: credit-risk baseline, calibration, imbalance, explainability, fairness analysis and API.
- September-December 2027: financial NLP baseline, public-data provenance, evaluation and Hugging Face demo.
- January-May 2028: cross-project reproducibility, deployment checks, case studies, interview walkthroughs and portfolio presentation.
Review pacing weekly against observed progress. A project is complete only when it meets its evidence and delivery gates.

## Daily run
1. Read this contract, projects.yml, portfolio/agent-state.json, the latest progress log, repository instructions, recent commits and relevant open PRs/issues.
2. Select the next small deliverable in the active milestone. Implement a useful, bounded improvement, with relevant tests and reproducibility checks. A reminder or plan alone does not count as implementation.
3. Use existing authenticated GitHub access to commit validated routine changes. If branch protection applies, push a branch and open a PR; preserve protections and report that it awaits merge. Never overwrite concurrent work or force-push.
4. Update the state and a dated log with actual evidence: changes, commands/check outcomes, limitations, commit/PR links, deployment status, next task and what Lameck should read.
5. Publish deployment-ready interface files to the existing Hugging Face Space using the authorized sync path. Pin changed project packages to a tested GitHub commit before deploying. Code push, Space upload, build success and running demo are separate states.
6. Notify Lameck in this task conversation after every run, including blocked or failed runs. Include what changed, why it matters to the vision, verification, GitHub and Hugging Face links/status, a 10-15 minute reading exercise, one comprehension question and the next step.
7. On weekly review, summarize milestones, evidence of skills, drift from the vision, repeated blockers, and schedule adjustments. Record learning as assigned or demonstrated; do not claim Lameck understands code without his response.

## Completion gates
Every project needs a research question; licensed/provenanced data with simulated data clearly labeled; simple baseline; meaningful evaluation and uncertainty; leakage and overfitting controls; reproducible run commands and environment; relevant automated checks; usable demo/API where planned; limitations; and an employer-facing case study explaining decisions and failures. No invented results, tests, deployments, profits or publications.

## Execution boundaries
Use available compute and existing authorized access. Do not start paid services, paid APIs, large training jobs, real financial trades, delete data, expose credentials, or change repository/account protections. If blocked, complete independent useful work and report the exact blocker. Do not create filler commits just to maintain a streak.

## Publishing
GitHub: https://github.com/lamsofttech/ai-ml-quantitative-research-portfolio
Hugging Face: https://huggingface.co/spaces/nyakslamo/ai-ml-quantitative-research-portfolio
Space source: spaces/portfolio-hub/
The HF_TOKEN GitHub Actions secret must have write permission to this Space. Never store the token in code, logs or chat. Missing credentials are a deployment blocker, not a successful sync.
