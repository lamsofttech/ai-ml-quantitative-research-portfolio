# Quantitative project standard

Every project README should answer:

1. What is the research question and why does it matter?
2. Is the data real, public, licensed, private, or simulated?
3. How is data validated and split without leakage?
4. What simple baseline must the proposed model beat?
5. Which statistical, predictive, and financial metrics are reported?
6. How are uncertainty, overfitting, regime changes, and transaction costs addressed?
7. What do the results mean—and what do they not prove?
8. How can another researcher reproduce the result?
9. How is the model served, containerized, monitored, and maintained when applicable?
10. What fairness, explainability, security, and responsible-AI risks apply?

Generated outputs must record the code version, environment, data version, parameters, random seed, and evaluation period.
Employer-facing evidence artifacts must also be listed in `portfolio/evidence-manifest.json`.
CI verifies each artifact's exact SHA-256 digest, generating code version, linked report, and
historical source-file hashes so that evidence changes require explicit review.

