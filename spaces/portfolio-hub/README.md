---
title: AI and Machine Learning for Quantitative Research
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# AI and Machine Learning for Quantitative Research — portfolio hub

A Gradio front door to the [GitHub portfolio](https://github.com/lamsofttech/ai-ml-quantitative-research-portfolio):
an overview of the identity, career objective, and project roadmap, plus a live demo of Project 02
(Monte Carlo option pricing and risk analysis).

This Space does not reimplement any pricing or risk math. `requirements.txt` installs the
`quant-mc-option-pricing` package directly from the GitHub repository (pinned to a specific commit),
so the demo always runs the exact, tested source of truth rather than a duplicated copy. To pick up
changes made to that project later, bump the commit SHA in `requirements.txt` and push.

See the [portfolio README](https://github.com/lamsofttech/ai-ml-quantitative-research-portfolio) and the
[project README](https://github.com/lamsofttech/ai-ml-quantitative-research-portfolio/tree/main/projects/02-monte-carlo-option-pricing)
for methodology, evidence classification, and limitations. Educational/research use only — not investment
advice.
