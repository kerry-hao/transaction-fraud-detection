"""Small, validated TOML configuration; all relative paths use the config's repo root."""

from pathlib import Path
import tomllib


def load_config(path="configs/default.toml"):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    root = path.parent.parent
    for key in ("data_dir", "output_dir"):
        cfg[key] = str((root / cfg[key]).resolve())
    a, b = cfg["split"]["train_fraction"], cfg["split"]["tune_fraction"]
    if not (0 < a < 1 and 0 < b < 1 and a + b < 1):
        raise ValueError("Split fractions must be positive and sum to less than one.")
    allowed = {"logistic_baseline", "lightgbm_baseline", "lightgbm_engineered"}
    names = cfg["models"]["names"]
    if not names or len(set(names)) != len(names) or set(names) - allowed:
        raise ValueError("Model names must be unique and belong to the documented model set.")
    if cfg["models"]["selection_metric"] != "roc_auc":
        raise ValueError("This version pre-specifies selection by tuning ROC-AUC.")
    return cfg
