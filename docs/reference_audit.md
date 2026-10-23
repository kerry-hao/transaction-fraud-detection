# Uploaded notebook audit

Source supplied for learning: `eda-and-models.ipynb`, 57 cells, Python 3.6.6 metadata.
Associated public reference: [artgor / EDA and models](https://www.kaggle.com/code/artgor/eda-and-models).
This audit concerns the exact uploaded file, not every version at that URL. The original file is
not included in this repository. The new implementation rewrites the workflow rather than
copying the notebook's code or outputs into an executed project.

SHA-256 of the supplied file:
`334b2b263e33b2b9ff626108929637d066730c0d0ff6904ef9c91716302b51b3`

## What its stored outputs establish

| Item | Stored output |
| --- | --- |
| Merged labelled population | 590,540 rows, 434 columns |
| Merged unlabelled population | 506,691 rows, 433 columns |
| Raw columns with missing values in train | 414 |
| Columns removed after engineered features and filtering | 84 |
| LightGBM fold AUCs | 0.919435, 0.931982, 0.926607, 0.946655, 0.923309 |
| Printed mean and standard deviation | 0.9296 and 0.0095 |

These outputs were read, not reproduced. They do not establish a public/private leaderboard score,
competition rank, a deployed fraud system, or the performance of the rewritten project.

## Changes made for this project

Cell numbers below are zero-based notebook cell indices.

| Uploaded workflow | New workflow and reason |
| --- | --- |
| Cell 48 creates `TimeSeriesSplit`, then overwrites it with `KFold(n_splits=5)` | Use separate chronological train/tune/holdout periods. Ordinary unshuffled KFold can still train on observations later than a validation block. |
| Cell 36 computes train group profiles before CV; test group profiles are recomputed separately on the entire test population | Learn group profiles from training rows, then freeze for tuning/holdout/test. The source's design is a batch/transductive choice and is different from a historical-profile exercise. |
| Cells 39–43 use both train and test distributions to drop columns and fit categorical vocabularies | Fit feature selection and encodings on training rows. Handle unknown categories explicitly. |
| Some features are dropped for exceeding 90% missingness or dominance | Keep non-constant predictors; do not assume sparse signals are useless. Evaluate an ablation later. |
| Older LightGBM fit arguments and broad optional imports | Use callback-based early stopping and only required dependencies. |
| XGBoost/blending cells are commented out | Implement only the documented three experiments; no claim of an executed ensemble. |

Training group profiles in the rewrite are still batch profiles within the training window.
They are not a fully lagged online feature store. Chronological validation is a stronger match
to predicting later transactions, but does not by itself resolve customer overlap, label delay
or anonymized field availability. The original CV score is not a like-for-like benchmark.
