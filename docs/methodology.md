# Methodology

## Population and joins

The unit is a transaction, not a borrower or confirmed customer. `isFraud` is the label.
Read categorical codes as strings consistently across files. Normalize `id-XX` identity field
names to `id_XX`. Reject duplicate/missing IDs and overlapping non-key columns. Left join
identity to transaction, preserve transaction order, and add `has_identity`.

`TransactionID` is an alignment key and is excluded from predictors. `TransactionDT` determines
the chronological split and is excluded in raw form. It does not identify actual calendar dates.
Anonymized `C`, `D`, `V` and `id` fields are used as observed features without reverse-engineered
economic labels. Numeric-looking card/address codes are categorical.

## Partition design

1. Sort by relative transaction time and ID.
2. Compute `floor(TransactionDT / 86400)` and sort the unique observed relative days.
3. Assign `floor(0.60 * number_of_days)` to training and `floor(0.20 * number_of_days)` to tuning.
4. Assign the remaining days to holdout. Require each partition to contain both target classes.

No day straddles a boundary, including duplicate timestamps. There is no random resampling or
stratification. The first version uses one chronological holdout, not a rolling cross-validation
study. It has no embargo or explicit fraud-label reporting-delay simulation. Validate actual
label timing before using this design to claim live performance.

## Feature recipes

Both baselines use all available merged raw columns except ID, raw time and target, plus the
identity-availability flag. Only training-constant or entirely missing columns are removed.
This deliberately avoids dropping all fields above an arbitrary missingness threshold.

The engineered model adds:

| Feature | Definition | Fitting information |
| --- | --- | --- |
| `missing_feature_count` | Count missing raw predictors in the current row | Current row only |
| `amount_log1p` | `log(1 + TransactionAmt)` for non-negative amounts | Current row only |
| `relative_hour` | `(TransactionDT // 3600) % 24` | Relative cycle only; not verified clock hour |
| `relative_day_mod7` | `(TransactionDT // 86400) % 7` | Relative cycle only; not actual weekday |
| Email suffixes | Last dot-separated segment in payer/recipient domain | Current row only |
| `email_domain_match` | Equal payer and recipient domains; missing if either absent | Current row only |
| Group frequency | Share of training rows with the same group code | Training rows only |
| Amount to group mean | Current amount divided by training group mean amount | Training rows only |
| Amount to group standard deviation | Current amount divided by training group sample standard deviation | Training rows only |

Default groups are `card1`, `card4`, `addr1` and `P_emaildomain`. These are codes, not verified
unique customer identities. An unseen group gets frequency zero; its amount ratios are missing.
Zero/undefined denominators become missing, and infinities become missing. Missing categories
use a distinct token that cannot collide with a literal observed string.

For training rows, group profiles use the entire training window, including each row itself.
This is a frozen historical-profile baseline for subsequent batches, not an online replay with
strict past-only aggregates for every training row. A later extension should compute lagged
event-time features and handle delayed labels.

## Models and selection

Logistic Regression: training medians plus numeric missingness indicators, numeric scaling,
sparse one-hot encoding with rare-category grouping, L2 regularization and the SAGA solver.
Unknown categories use the trained infrequent group when available, otherwise an all-zero
encoding. Convergence and other fit warnings are saved; inspect them before reporting results.

LightGBM: training-fitted integer category vocabularies, native categorical splits, numeric
missing values retained, and callback-based early stopping on tuning AUC. Unseen categories
are encoded as -1, which LightGBM treats as missing. No oversampling, SMOTE or class weighting
is applied by default. Changing these can affect probability calibration and needs evaluation.

Choose the highest tuning ROC-AUC across the three pre-specified recipes. Evaluate the same
saved models on the later holdout, mark which was already selected, and retain all three
diagnostic comparisons. Early stopping uses tuning data only. This version does not implement
hyperparameter search, confidence intervals, calibration fitting, SHAP or ensembling.

## Metrics

Report ROC-AUC and average precision for ranking, and log loss/Brier for probability quality.
Show prevalence and mean predicted probability together to reveal broad calibration drift.
At each review fraction, rank scores descending, break ties by ID, and review the first
`ceil(n * fraction)` transactions. Precision is the fraud share in reviewed transactions;
recall is the share of all observed fraud cases included there.

The review exercise assumes all batch scores are available simultaneously. A production threshold
would need to be chosen on tuning data and monitored when future score distributions change.
Amount capture ignores investigation cost, recovery rates, customer harm and intervention effects.

## Final test prediction

After holdout reporting, refit the tuning-selected recipe on all labelled data. Refit its feature
builder and encoders too. For LightGBM, fix the number of trees at the tuning-selected iteration;
do not early-stop using hidden test labels. Merge predicted probabilities onto the sample
submission by `TransactionID`, never by assumed row order. The command creates a local file only.

## Next extensions

First run and document the frozen baseline. Then consider multiple forward validation windows,
feature-family ablations, lagged group histories, missing-identity subgroup diagnostics,
probability calibration on a separate period, and a review cost analysis with explicit assumptions.
Change one component at a time and preserve the earlier run's metadata.
