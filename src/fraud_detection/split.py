"""Chronological partitions with whole-day boundaries and reproducible membership."""

import numpy as np
import pandas as pd
from .data import ID, TIME, TARGET


def chronological_split(frame, train_fraction=0.6, tune_fraction=0.2):
    if not (0 < train_fraction < 1 and 0 < tune_fraction < 1 and train_fraction + tune_fraction < 1):
        raise ValueError("Invalid split fractions.")
    if not np.isfinite(frame[TIME]).all() or (frame[TIME] < 0).any():
        raise ValueError("Split times must be finite non-negative seconds.")
    ordered = frame.sort_values([TIME, ID], kind="stable").reset_index(drop=True)
    day = np.floor(ordered[TIME].to_numpy() / 86400).astype("int64")
    days = np.unique(day)
    n_train = int(len(days) * train_fraction)
    n_tune = int(len(days) * tune_fraction)
    if min(n_train, n_tune, len(days) - n_train - n_tune) < 1:
        raise ValueError("Too few relative days for three non-empty partitions.")
    boundary1, boundary2 = days[n_train], days[n_train + n_tune]
    masks = {"train": day < boundary1, "tune": (day >= boundary1) & (day < boundary2),
             "holdout": day >= boundary2}
    parts = {name: ordered.loc[mask].reset_index(drop=True) for name, mask in masks.items()}
    for name, part in parts.items():
        if part[TARGET].nunique() != 2:
            raise ValueError(f"{name} needs both classes to evaluate ROC-AUC.")
    if not (parts["train"][TIME].max() < parts["tune"][TIME].min()
            and parts["tune"][TIME].max() < parts["holdout"][TIME].min()):
        raise ValueError("Chronological separation failed.")
    return parts


def split_membership(parts):
    return pd.concat([part[[ID, TIME]].assign(partition=name) for name, part in parts.items()],
                     ignore_index=True)
