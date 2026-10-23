"""Local audit metadata and figures; no external services are contacted."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import uuid
import numpy as np
import pandas as pd


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def new_run(output_dir, prefix="run"):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output_dir) / f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(paths):
    return [{"file": Path(p).name, "bytes": Path(p).stat().st_size, "sha256": sha256(p)} for p in paths]


def environment():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib", "joblib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def plot_diagnostics(frame, probabilities, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    colors = ["#234E70", "#287D70", "#B87930"]
    for kind in ("roc", "precision_recall", "calibration"):
        fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
        for (name, p), color in zip(probabilities.items(), colors):
            if kind == "roc":
                RocCurveDisplay.from_predictions(frame["isFraud"], p, name=name, ax=ax, color=color)
            elif kind == "precision_recall":
                PrecisionRecallDisplay.from_predictions(frame["isFraud"], p, name=name, ax=ax, color=color)
            else:
                actual, predicted = calibration_curve(frame["isFraud"], p, n_bins=10, strategy="quantile")
                ax.plot(predicted, actual, marker="o", label=name, color=color)
        if kind == "calibration":
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            ax.set(xlabel="Mean predicted probability", ylabel="Observed fraud rate")
        if kind == "precision_recall":
            ax.axhline(frame["isFraud"].mean(), color="gray", linestyle="--", label="Fraud prevalence")
        ax.set_title("Chronological holdout " + kind.replace("_", " "))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.savefig(directory / f"{kind}.png", dpi=180)
        plt.close(fig)


def training_profile(frame, directory):
    """Descriptive outputs use the training partition, leaving later labels unexamined."""
    from .data import ID, TIME, TARGET
    directory = Path(directory)
    write_json(directory / "training_summary.json", {
        "rows": len(frame), "columns": len(frame.columns), "frauds": int(frame[TARGET].sum()),
        "fraud_rate": float(frame[TARGET].mean()), "identity_coverage": float(frame["has_identity"].mean()),
        "time_min": float(frame[TIME].min()), "time_max": float(frame[TIME].max()),
    })
    pd.DataFrame({"feature": frame.columns, "missing_fraction": frame.isna().mean().values,
                  "unique_non_missing": frame.nunique().values}).to_csv(directory / "feature_profile.csv", index=False)
    frame.assign(relative_day=np.floor(frame[TIME] / 86400)).groupby("relative_day").agg(
        transactions=(ID, "size"), fraud_rate=(TARGET, "mean"),
        median_amount=("TransactionAmt", "median")).to_csv(directory / "daily_profile.csv")
