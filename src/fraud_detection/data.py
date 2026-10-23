"""CSV schema normalization and checked transaction-to-identity joins."""

from pathlib import Path
import re
import numpy as np
import pandas as pd

ID, TIME, TARGET = "TransactionID", "TransactionDT", "isFraud"
RAW_FILES = (
    "train_transaction.csv", "train_identity.csv", "test_transaction.csv",
    "test_identity.csv", "sample_submission.csv",
)
CATEGORICAL = {
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
    *[f"M{i}" for i in range(1, 10)], *[f"id_{i:02d}" for i in range(12, 39)],
}


def normalize_identity_columns(frame):
    names = [re.sub(r"^id-(\d+)$", r"id_\1", str(c)) for c in frame.columns]
    if len(set(names)) != len(names):
        raise ValueError("Identity normalization would create duplicate columns.")
    return frame.set_axis(names, axis=1)


def check_files(data_dir, training_only=False):
    names = RAW_FILES[:2] if training_only else RAW_FILES
    missing = [name for name in names if not (Path(data_dir) / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {', '.join(missing)}. "
            "Download/extract them yourself; no command downloads data automatically."
        )
    return [str(Path(data_dir) / name) for name in names]


def read_csv(path):
    # Read known categorical codes as strings in both splits, including numeric-looking IDs.
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {}
    for col in header:
        normalized = re.sub(r"^id-(\d+)$", r"id_\1", col)
        dtypes[col] = ("string" if normalized in CATEGORICAL else
                       "int64" if col == ID else
                       "float64" if col == TIME else
                       "int8" if col == TARGET else "float32")
    return normalize_identity_columns(pd.read_csv(path, dtype=dtypes))


def merge_identity(transaction, identity):
    transaction = normalize_identity_columns(transaction)
    identity = normalize_identity_columns(identity)
    for name, frame in (("transaction", transaction), ("identity", identity)):
        if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError(f"{name} must have unique non-missing {ID} values.")
    overlap = (set(transaction) & set(identity)) - {ID}
    if overlap or "has_identity" in transaction or "has_identity" in identity:
        raise ValueError(f"Unexpected overlapping or reserved columns: {overlap}")
    if not identity[ID].isin(transaction[ID]).all():
        raise ValueError("Identity contains TransactionIDs absent from its transaction table.")
    out = transaction.merge(identity.assign(has_identity=np.int8(1)), on=ID,
                            how="left", validate="one_to_one", sort=False)
    if len(out) != len(transaction) or not out[ID].equals(transaction[ID].reset_index(drop=True)):
        raise ValueError("The join changed transaction order or row count.")
    out["has_identity"] = out["has_identity"].fillna(0).astype("int8")
    return out


def load_data(data_dir, split="train"):
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    base = Path(data_dir)
    frame = merge_identity(read_csv(base / f"{split}_transaction.csv"),
                           read_csv(base / f"{split}_identity.csv"))
    if TIME not in frame or not np.isfinite(frame[TIME]).all() or (frame[TIME] < 0).any():
        raise ValueError("TransactionDT must contain finite non-negative relative seconds.")
    if split == "train":
        if TARGET not in frame or frame[TARGET].isna().any() or not set(frame[TARGET].unique()) <= {0, 1}:
            raise ValueError("Training isFraud must contain only non-missing binary labels.")
    elif TARGET in frame:
        raise ValueError("Unlabelled test data must not contain isFraud.")
    return frame
