"""Feature recipes. Group statistics and feature selection fit on training rows only."""

import numpy as np
import pandas as pd
from .data import ID, TIME, TARGET, CATEGORICAL


def categorical_values(series):
    # Prefix actual values so a literal '<MISSING>' cannot collide with missingness.
    return series.astype("string").map(lambda x: "M:" if pd.isna(x) else "V:" + str(x)).astype(object)


def safe_ratio(numerator, denominator):
    denominator = denominator.where(denominator.abs() > 1e-8)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


class FeatureBuilder:
    def __init__(self, engineered=False, group_columns=None):
        self.engineered = engineered
        self.group_columns = list(group_columns or ["card1", "card4", "addr1", "P_emaildomain"])

    def _base(self, frame):
        out = frame.drop(columns=[ID, TIME, TARGET], errors="ignore").copy()
        if self.engineered:
            out["missing_feature_count"] = out.isna().sum(axis=1).astype("float32")
            amount = pd.to_numeric(frame["TransactionAmt"], errors="coerce")
            out["amount_log1p"] = np.log1p(amount.where(amount >= 0))
            # Relative seconds have an unknown origin; these are cyclical indices, not calendar labels.
            out["relative_hour"] = (frame[TIME] // 3600) % 24
            out["relative_day_mod7"] = (frame[TIME] // 86400) % 7
            for col in ("P_emaildomain", "R_emaildomain"):
                if col in frame:
                    out[f"{col}_suffix"] = frame[col].astype("string").str.rsplit(".", n=1).str[-1]
            if {"P_emaildomain", "R_emaildomain"} <= set(frame):
                left, right = frame["P_emaildomain"], frame["R_emaildomain"]
                out["email_domain_match"] = (left == right).fillna(False).astype("float32")
                out.loc[left.isna() | right.isna(), "email_domain_match"] = np.nan
            for group in self.active_groups_:
                keys = categorical_values(frame[group])
                stats = self.group_stats_[group]
                out[f"{group}_frequency"] = keys.map(stats["frequency"]).fillna(0)
                out[f"amount_to_{group}_mean"] = safe_ratio(amount, keys.map(stats["mean"]))
                out[f"amount_to_{group}_std"] = safe_ratio(amount, keys.map(stats["std"]))
        for col in out:
            if col in CATEGORICAL or col.endswith("_suffix") or not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = categorical_values(out[col])
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
        return out

    def fit(self, frame):
        self.input_columns_ = [c for c in frame if c not in {ID, TIME, TARGET}]
        self.active_groups_ = [c for c in self.group_columns if c in frame] if self.engineered else []
        self.group_stats_ = {}
        if self.engineered:
            amount = pd.to_numeric(frame["TransactionAmt"], errors="coerce")
            for col in self.active_groups_:
                keys = categorical_values(frame[col])
                grouped = pd.DataFrame({"key": keys, "amount": amount}).groupby("key")["amount"]
                stats = grouped.agg(["mean", "std"])
                stats["frequency"] = keys.value_counts() / len(frame)
                self.group_stats_[col] = stats
        raw = self._base(frame)
        self.columns_ = [c for c in raw if raw[c].nunique(dropna=False) > 1]
        self.dropped_columns_ = [c for c in raw if c not in self.columns_]
        if not self.columns_:
            raise ValueError("No non-constant features remain.")
        self.categorical_columns_ = [c for c in self.columns_ if raw[c].dtype == object]
        return self

    def transform(self, frame):
        missing = set(self.input_columns_) - set(frame)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        result = self._base(frame)
        return result.reindex(columns=self.columns_)

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)
