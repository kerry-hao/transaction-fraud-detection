"""Explicit stages: check, inspect, train, evaluate, submit. Nothing runs on import."""

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from .config import load_config
from .data import ID, TIME, TARGET, check_files, load_data
from .split import chronological_split, split_membership
from .features import FeatureBuilder
from .models import fit_model, predict
from .evaluation import metrics, review_capacity, make_submission
from .reporting import (new_run, write_json, environment, file_manifest, sha256,
                        training_profile, plot_diagnostics)


def get_parts(cfg, frame):
    return chronological_split(frame, **cfg["split"])


def inspect_data(cfg):
    check_files(cfg["data_dir"], training_only=True)
    data = load_data(cfg["data_dir"])
    parts = get_parts(cfg, data)
    directory = new_run(cfg["output_dir"], "eda")
    training_profile(parts["train"], directory)
    split_membership(parts).to_csv(directory / "split_membership.csv", index=False)
    print(f"Training-partition EDA saved to {directory}")


def train(cfg):
    paths = check_files(cfg["data_dir"], training_only=True)
    directory = new_run(cfg["output_dir"])
    write_json(directory / "status.json", {"status": "training_started"})
    write_json(directory / "config.json", cfg)
    write_json(directory / "environment.json", environment())
    write_json(directory / "data_manifest.json", file_manifest(paths))
    frame = load_data(cfg["data_dir"])
    parts = get_parts(cfg, frame)
    membership_path = directory / "split_membership.csv"
    split_membership(parts).to_csv(membership_path, index=False)
    partition_summary = {name: {"rows": len(part), "time_min": float(part[TIME].min()),
                               "time_max": float(part[TIME].max())} for name, part in parts.items()}
    write_json(directory / "partitions.json", partition_summary)
    training_profile(parts["train"], directory)
    rows, iteration_counts = [], {}
    for name in cfg["models"]["names"]:
        print(f"Fitting {name} on the earliest partition ...", flush=True)
        builder = FeatureBuilder(engineered=name.endswith("_engineered"),
                                 group_columns=cfg["features"]["group_columns"])
        x_train = builder.fit_transform(parts["train"])
        x_tune = builder.transform(parts["tune"])
        fitted = fit_model(name, x_train, parts["train"][TARGET], x_tune, parts["tune"][TARGET],
                           builder.categorical_columns_, cfg)
        bundle = dict(fitted, features=builder, name=name)
        probability = predict(bundle, parts["tune"])
        rows.append({"model": name, **metrics(parts["tune"][TARGET], probability)})
        iteration_counts[name] = fitted["iterations"]
        joblib.dump(bundle, directory / f"{name}.joblib", compress=3)
        parts["tune"][[ID, TARGET]].assign(probability=probability).to_csv(
            directory / f"{name}_tune_predictions.csv", index=False)
        write_json(directory / f"{name}_features.json", {
            "features": builder.columns_, "categorical": builder.categorical_columns_,
            "dropped_constant": builder.dropped_columns_, "warnings": fitted["warnings"],
        })
        if fitted["encoder"] is not None:
            pd.DataFrame({"feature": builder.columns_, "gain": fitted["model"].feature_importances_}).sort_values(
                "gain", ascending=False).to_csv(directory / f"{name}_feature_gain.csv", index=False)
        del x_train, x_tune, fitted, bundle
        gc.collect()
    scores = pd.DataFrame(rows)
    scores.to_csv(directory / "tuning_metrics.csv", index=False)
    # The winner is frozen before any holdout metrics are produced. Ties follow config order.
    winner = str(scores.loc[scores["roc_auc"].idxmax(), "model"])
    write_json(directory / "selection.json", {"selected_model": winner, "selection_partition": "tune",
                                              "selection_metric": "roc_auc", "iterations": iteration_counts})
    write_json(directory / "status.json", {"status": "trained", "holdout_evaluated": False,
                                            "membership_sha256": sha256(membership_path)})
    print(scores.to_string(index=False))
    print(f"Selected on tuning ROC-AUC: {winner}\nRun directory: {directory}")
    return directory


def restore_run(run_dir):
    directory = Path(run_dir).resolve()
    cfg = json.loads((directory / "config.json").read_text())
    state = json.loads((directory / "status.json").read_text())
    if state["status"] != "trained":
        raise ValueError("Training did not complete; inspect the run before proceeding.")
    for record in json.loads((directory / "data_manifest.json").read_text()):
        path = Path(cfg["data_dir"]) / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Data changed since training: {record['file']}")
    if sha256(directory / "split_membership.csv") != state["membership_sha256"]:
        raise ValueError("Saved split membership changed since training.")
    selected = json.loads((directory / "selection.json").read_text())
    return directory, cfg, selected


def evaluate(run_dir):
    directory, cfg, selection = restore_run(run_dir)
    output = directory / "evaluation"
    output.mkdir(exist_ok=False)
    frame = load_data(cfg["data_dir"])
    holdout = get_parts(cfg, frame)["holdout"]
    rows, capacities, probabilities = [], [], {}
    for name in cfg["models"]["names"]:
        bundle = joblib.load(directory / f"{name}.joblib")
        probability = predict(bundle, holdout)
        probabilities[name] = probability
        rows.append({"model": name, "selected_on_tune": name == selection["selected_model"],
                     **metrics(holdout[TARGET], probability)})
        capacities.append(review_capacity(holdout, probability).assign(model=name))
        holdout[[ID, TARGET]].assign(probability=probability).to_csv(
            output / f"{name}_predictions.csv", index=False)
        del bundle
        gc.collect()
    pd.DataFrame(rows).to_csv(output / "holdout_metrics.csv", index=False)
    pd.concat(capacities, ignore_index=True).to_csv(output / "review_capacity.csv", index=False)
    plot_diagnostics(holdout, probabilities, output)
    write_json(output / "complete.json", {"status": "evaluated", "selected_model": selection["selected_model"],
                                          "evaluated_at": datetime.now(timezone.utc).isoformat(),
                                          "selection_unchanged": True})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Holdout results saved to {output}. Keep model selection based on tuning results.")


def submit(run_dir):
    directory, cfg, selection = restore_run(run_dir)
    if not (directory / "evaluation" / "complete.json").is_file():
        raise ValueError("Evaluate the frozen holdout before fitting on all labelled data.")
    paths = check_files(cfg["data_dir"])
    output = directory / "submission"
    output.mkdir(exist_ok=False)
    frame = load_data(cfg["data_dir"])
    name = selection["selected_model"]
    builder = FeatureBuilder(name.endswith("_engineered"), cfg["features"]["group_columns"])
    x = builder.fit_transform(frame)
    # LightGBM uses the tuning-selected tree count; the linear baseline has no early stopping.
    fitted = fit_model(name, x, frame[TARGET], None, None, builder.categorical_columns_, cfg,
                       fixed_iterations=selection["iterations"][name] if name != "logistic_baseline" else 1)
    bundle = dict(fitted, features=builder, name=name)
    del x, frame
    gc.collect()
    test = load_data(cfg["data_dir"], "test")
    probability = predict(bundle, test)
    sample = pd.read_csv(Path(cfg["data_dir"]) / "sample_submission.csv")
    submission = make_submission(sample, test[ID].to_numpy(), probability)
    submission.to_csv(output / "submission.csv", index=False)
    joblib.dump(bundle, output / "final_model.joblib", compress=3)
    write_json(output / "metadata.json", {"model": name, "rows": len(submission),
                                          "training_population": "all labelled training transactions",
                                          "files": file_manifest(paths), "environment": environment(),
                                          "iterations": fitted["iterations"], "warnings": fitted["warnings"]})
    print(f"Created {output / 'submission.csv'}. No file was uploaded to Kaggle.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "inspect", "train"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="configs/default.toml")
    for name in ("evaluate", "submit"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command in {"evaluate", "submit"}:
        return {"evaluate": evaluate, "submit": submit}[args.command](args.run_dir)
    cfg = load_config(args.config)
    if args.command == "check":
        for path in check_files(cfg["data_dir"]):
            print(f"Found: {Path(path).name}")
        print("Configuration and file presence checks passed. No model was fitted.")
    else:
        return {"inspect": inspect_data, "train": train}[args.command](cfg)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
