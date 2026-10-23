"""Ranking, probability quality and batch review-capacity diagnostics."""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from .data import ID, TARGET


def metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    if len(y) != len(probability) or len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Metrics need equally sized non-empty arrays with both binary classes.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    return {"roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "brier": float(brier_score_loss(y, probability)),
            "mean_probability": float(probability.mean()), "fraud_rate": float(y.mean()),
            "rows": int(len(y)), "frauds": int(y.sum())}


def review_capacity(frame, probability, fractions=(0.01, 0.05, 0.10)):
    metrics(frame[TARGET], probability)  # Validate arrays before ranking.
    ordered = frame[[ID, TARGET, "TransactionAmt"]].assign(probability=np.asarray(probability))
    ordered = ordered.sort_values(["probability", ID], ascending=[False, True], kind="stable")
    total_fraud = int(ordered[TARGET].sum())
    total_amount = ordered.loc[ordered[TARGET] == 1, "TransactionAmt"].sum()
    rows = []
    for fraction in fractions:
        if not (0 < fraction <= 1):
            raise ValueError("Review fractions must be in (0, 1].")
        k = max(1, math.ceil(len(ordered) * fraction))
        selected = ordered.head(k)
        caught = int(selected[TARGET].sum())
        flagged_amount = selected.loc[selected[TARGET] == 1, "TransactionAmt"].sum()
        rows.append({"requested_review_fraction": fraction, "reviewed": k,
                     "actual_review_fraction": k / len(ordered), "true_positives": caught,
                     "false_positives": k - caught, "precision": caught / k,
                     "recall": caught / total_fraud,
                     "fraud_amount_capture": float(flagged_amount / total_amount) if total_amount > 0 else np.nan})
    return pd.DataFrame(rows)


def make_submission(sample, transaction_ids, probability):
    transaction_ids, probability = np.asarray(transaction_ids), np.asarray(probability)
    if list(sample.columns) != [ID, TARGET] or sample[ID].isna().any() or sample[ID].duplicated().any():
        raise ValueError("Sample submission must have unique IDs and exactly TransactionID,isFraud.")
    if len(transaction_ids) != len(probability) or len(set(transaction_ids)) != len(transaction_ids):
        raise ValueError("Test IDs must be unique and aligned with predictions.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid submission probabilities.")
    if set(sample[ID]) != set(transaction_ids):
        raise ValueError("Sample submission and test IDs do not match.")
    mapping = pd.Series(probability, index=transaction_ids)
    result = sample.copy()
    result[TARGET] = result[ID].map(mapping)
    return result
