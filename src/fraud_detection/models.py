"""Linear baseline and native-categorical LightGBM with saved preprocessing."""

import warnings
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TreeEncoder:
    """Training vocabulary only; unseen categories use LightGBM's missing code -1."""

    def fit(self, frame, categorical_columns):
        self.columns_ = list(frame.columns)
        self.categories_ = {c: {v: i for i, v in enumerate(sorted(frame[c].unique()))}
                            for c in categorical_columns}
        return self

    def transform(self, frame):
        result = frame[self.columns_].copy()
        for col, mapping in self.categories_.items():
            result[col] = result[col].map(mapping).fillna(-1).astype("int32")
        return result


def fit_model(name, x_train, y_train, x_tune, y_tune, categorical_columns, cfg, fixed_iterations=None):
    """When fixed_iterations is set, fit a final model without using an evaluation set."""
    notes = []
    if name == "logistic_baseline":
        p = cfg["logistic"]
        numeric = [c for c in x_train if c not in categorical_columns]
        prep = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("category", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=True,
                                       min_frequency=p["min_frequency"], max_categories=p["max_categories"],
                                       dtype=np.float32), categorical_columns),
        ], sparse_threshold=1.0)
        model = Pipeline([
            ("preprocess", prep),
            ("classifier", LogisticRegression(C=p["C"], max_iter=p["max_iter"], solver="saga",
                                              random_state=cfg["seed"])),
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x_train, y_train)
        notes = [f"{w.category.__name__}: {w.message}" for w in caught]
        return {"model": model, "encoder": None, "iterations": None, "warnings": notes}

    # Import lazily: file checks, EDA and code-only tests do not require LightGBM.
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("Install this project with pip install -e . to use LightGBM.") from exc
    params = dict(cfg["lightgbm"])
    stopping_rounds = params.pop("early_stopping_rounds")
    if fixed_iterations is not None:
        params["n_estimators"] = int(fixed_iterations)
    encoder = TreeEncoder().fit(x_train, categorical_columns)
    train_encoded = encoder.transform(x_train)
    model = lgb.LGBMClassifier(**params, objective="binary", metric="auc", random_state=cfg["seed"],
                               deterministic=True, force_col_wise=True, verbosity=-1,
                               importance_type="gain")
    fit_args = {"categorical_feature": list(encoder.categories_)}
    if fixed_iterations is None:
        fit_args.update(eval_set=[(encoder.transform(x_tune), y_tune)], eval_metric="auc",
                        callbacks=[lgb.early_stopping(stopping_rounds, first_metric_only=True, verbose=False)])
    model.fit(train_encoded, y_train, **fit_args)
    return {"model": model, "encoder": encoder,
            "iterations": int(model.best_iteration_ or model.n_estimators_), "warnings": notes}


def predict(bundle, raw):
    features = bundle["features"].transform(raw)
    if bundle["encoder"] is not None:
        features = bundle["encoder"].transform(features)
    return bundle["model"].predict_proba(features)[:, 1]
