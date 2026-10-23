# IEEE-CIS Fraud Detection

A learning project for transaction fraud prediction: compare a linear baseline with LightGBM,
then measure whether a small set of engineered features adds value under chronological validation.

**Status:** implementation prepared; competition data has not been downloaded and no models have
been trained. Results are intentionally unfilled. The supplied reference notebook's stored
CV AUC of about 0.9296 is recorded separately and is not a result of this repository.

## Questions

1. Does LightGBM rank fraudulent transactions better than Logistic Regression?
2. Do amount profiles, category frequencies, email features and relative time cycles improve LightGBM?
3. How much fraud is found when review capacity is limited to 1%, 5% or 10% of transactions?

These are empirical questions to answer after running the project.

## Start by reading

- [Guided notebook](notebooks/01_fraud_detection_walkthrough.ipynb): an English walkthrough, with execution disabled by default.
- [Methodology](docs/methodology.md): data joins, feature definitions, partitions and interpretation.
- [Reference notebook audit](docs/reference_audit.md): what the uploaded example shows and what this implementation changes.
- [Results](reports/results.md): the three planned comparisons, with no invented scores.
- [Data access and GitHub setup](docs/setup.md): manual download, optional Kaggle CLI and publishing instructions.

## Data

The [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection)
uses Vesta e-commerce transaction data. Predict `isFraud`, a binary transaction-level label.
Join transaction and identity tables on `TransactionID` using a left join: identity information
does not exist for every transaction. Many fields are anonymized and should not be given invented meanings.

| File | Role |
| --- | --- |
| `train_transaction.csv` | Labelled transactions, including `isFraud` |
| `train_identity.csv` | Identity/device attributes for a subset of training transactions |
| `test_transaction.csv` | Transactions with hidden labels |
| `test_identity.csv` | Identity/device attributes for a subset of test transactions |
| `sample_submission.csv` | Required test ID order and submission columns |

The supplied notebook's stored merged dimensions are 590,540 training rows and 506,691 test rows.
These are reference facts; the pipeline checks actual files when you choose to supply them.
It normalizes identity names such as `id-01` to `id_01`, verifies unique IDs, and preserves row order.
`TransactionDT` is elapsed seconds from an unspecified origin, not a Unix timestamp.

## Experiments

| Name | Features | Purpose |
| --- | --- | --- |
| `logistic_baseline` | Raw available features, excluding ID, target and raw time | Linear benchmark with imputation, scaling and sparse one-hot encoding |
| `lightgbm_baseline` | Same raw feature set | Nonlinear tabular benchmark with native categorical treatment |
| `lightgbm_engineered` | Raw features plus the documented transformations | Same-model test of incremental feature value |

All models include a `has_identity` indicator from the join. Only training-constant/all-missing
columns are removed. Both LightGBM models use the same parameter settings, with their own
tuning-selected early-stopping iteration. Logistic Regression uses a pre-specified regularization
setting, so this is a practical baseline comparison, not equal-budget hyperparameter optimization.

## Validation

Order transactions by `TransactionDT`. Use the earliest 60% of unique observed relative days for
training, the next 20% for tuning, and the remaining days for holdout evaluation. Percentages refer
to days, not rows. A whole relative day belongs to only one partition.

- Fit group statistics, categorical vocabularies, imputation and feature selection on training rows.
- Use tuning data for LightGBM early stopping and model selection by ROC-AUC.
- Freeze that selection before evaluating the holdout. Report all three planned comparisons as diagnostics.
- Optionally refit the selected recipe on all labelled data at its selected tree count to predict Kaggle test data.

The code never fits preprocessing on Kaggle test data. Training-window group statistics are frozen
historical profiles for later periods; they are not per-transaction, past-only online features.
The same card-related codes can occur across partitions. No claim of performance on entirely
unseen customers, confirmed label availability, or production deployment is made.

## Install when ready

Python 3.11 or 3.12 is recommended. From this repository's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

Windows activation: `.venv\Scripts\activate`. If macOS reports a missing OpenMP library when
importing LightGBM, follow the [official installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).
Dependency ranges are specified in `pyproject.toml`; they are not a frozen environment lock.

## Run only after obtaining data

Place the five extracted CSV files in `data/raw/`. See [setup](docs/setup.md) for download choices.
No project command performs an automatic download.

```bash
# File presence and configuration only
python -m fraud_detection.cli check

# Training-partition data summaries, without model fitting
python -m fraud_detection.cli inspect

# Fit three models and select using the tuning partition only
python -m fraud_detection.cli train
```

The training command prints a new run directory. Replace the example path below with that exact directory:

```bash
# Freeze and assess the later holdout
python -m fraud_detection.cli evaluate --run-dir artifacts/run_YOUR_RUN_ID

# Optional: refit the selected recipe on all labelled rows and create submission.csv
python -m fraud_detection.cli submit --run-dir artifacts/run_YOUR_RUN_ID
```

`submit` writes a local CSV; it does not upload it. Evaluation and submission each create their
own new subdirectory and refuse to overwrite an existing one. If a stage fails, inspect and
remove only its incomplete subdirectory before retrying. Do not repeatedly tune against holdout results.

Use `--config configs/default.toml` on `check`, `inspect` or `train` to specify configuration.
Relative data/output paths are resolved against the parent of the config directory. Later stages
read the saved run configuration and verify the original training-file hashes.

## Outputs and interpretation

| Output | What it answers |
| --- | --- |
| `tuning_metrics.csv` and `selection.json` | Which recipe was selected before holdout evaluation? |
| `evaluation/holdout_metrics.csv` | ROC-AUC, average precision, log loss, Brier score and mean probability |
| `evaluation/review_capacity.csv` | Precision, recall, false positives and amount capture at fixed review capacity |
| `evaluation/roc.png`, `precision_recall.png`, `calibration.png` | Ranking and probability diagnostics |
| `*_feature_gain.csv` | LightGBM split gain |
| `config.json`, `environment.json`, `data_manifest.json`, `split_membership.csv` | Configuration, versions, hashes and exact membership |
| `submission/submission.csv` | Key-aligned hidden-test probabilities in sample-submission order |

AP is **average precision**, not trapezoidal PR-AUC. A random-ranking reference for AP is fraud
prevalence. AUC measures ranking, not the fraction of transactions correctly classified. Review
metrics rank the evaluation batch and select exactly `ceil(n * fraction)` rows, breaking score
ties by ID. They are offline capacity diagnostics, not a deployable fixed threshold.
Fraud amount capture measures the share of labelled fraudulent transaction amounts flagged;
it does not measure recovered money or net savings. Calibration plots diagnose probabilities;
this first version does not fit a separate calibration model or cost-optimal policy.

## Code checks without competition data

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

Tests use tiny invented rows to check joins, temporal boundaries, frozen feature statistics,
unknown categories, metrics and submission alignment. They do not fit classifiers or access
competition files. GitHub Actions runs these checks. Real-data execution, model convergence,
runtime, memory use and leaderboard performance still require a first run.

The full CSV pipeline retains several copies in memory. Start on a machine with ample available
RAM; no memory or runtime requirement has been benchmarked. A GPU is not required by this code.
Training a subset is a later engineering option, not an automatically enabled sampling shortcut.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/fraud_detection/` | Data, splits, feature recipes, models, metrics, reports and CLI |
| `configs/default.toml` | Pre-specified experiments and settings |
| `notebooks/` | Guided reading and opt-in execution |
| `docs/` | Methodology, source audit, setup and roadmap |
| `reports/` | Reference output and aggregate result status |
| `tests/` | Checks using invented rows only |
| `data/raw/`, `artifacts/` | Local data and generated outputs, ignored by Git |

## References and scope

- [Competition data](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
- [scikit-learn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Kaggle official CLI](https://github.com/Kaggle/kaggle-cli)

