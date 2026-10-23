# Results

Project execution status: **not run on IEEE-CIS competition data**.

| Experiment | Tuning ROC-AUC | Holdout ROC-AUC | Holdout AP | Holdout recall at 5% review |
| --- | --- | --- | --- | --- |
| Logistic Regression baseline | Not run | Not run | Not run | Not run |
| LightGBM baseline | Not run | Not run | Not run | Not run |
| LightGBM with engineered features | Not run | Not run | Not run | Not run |

After execution, copy aggregate results from `artifacts/<run>/tuning_metrics.csv`,
`evaluation/holdout_metrics.csv`, and `evaluation/review_capacity.csv`. Add the run identifier,
date, data hashes, split dates, dependency versions and any convergence warnings. Explain whether
feature engineering improved the same-model comparison; do not assume it will.

The supplied reference notebook's stored CV AUC is approximately **0.9296**, with a reported
fold standard deviation of **0.0095**. This is a reference output under a different validation
design, not this project's result, not a reproduced score, and not a verified leaderboard score.
See [the reference audit](../docs/reference_audit.md).
