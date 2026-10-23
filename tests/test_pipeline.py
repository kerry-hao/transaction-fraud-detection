"""Structural checks with invented rows only. No classifier fitting or competition access."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from fraud_detection.config import load_config
from fraud_detection.data import merge_identity, normalize_identity_columns, read_csv, check_files
from fraud_detection.split import chronological_split, split_membership
from fraud_detection.features import FeatureBuilder, categorical_values
from fraud_detection.models import TreeEncoder
from fraud_detection.evaluation import metrics, review_capacity, make_submission

ROOT = Path(__file__).resolve().parents[1]


def invented_transactions():
    n = 80
    return pd.DataFrame({
        "TransactionID": np.arange(1000, 1000 + n),
        "TransactionDT": np.repeat(np.arange(20) * 86400, 4),
        "isFraud": np.tile([0, 0, 1, 0], 20),
        "TransactionAmt": np.arange(n, dtype=float) + 1,
        "card1": ["a", "a", "b", "b"] * 20,
        "P_emaildomain": ["a.example", None, "b.example", "a.example"] * 20,
        "R_emaildomain": ["a.example", "c.example", None, "b.example"] * 20,
        "has_identity": [0, 1] * 40,
        "all_missing": np.nan,
    })


class JoinChecks(unittest.TestCase):
    def test_left_join_preserves_population_and_order(self):
        tx = pd.DataFrame({"TransactionID": [3, 1, 2], "TransactionAmt": [4., 5., 6.]})
        identity = pd.DataFrame({"TransactionID": [1], "id-01": [7.]})
        merged = merge_identity(tx, identity)
        self.assertEqual(merged.TransactionID.tolist(), [3, 1, 2])
        self.assertEqual(merged.has_identity.tolist(), [0, 1, 0])
        self.assertIn("id_01", merged)

    def test_duplicate_identity_rejected(self):
        tx = pd.DataFrame({"TransactionID": [1, 2]})
        identity = pd.DataFrame({"TransactionID": [1, 1]})
        with self.assertRaises(ValueError):
            merge_identity(tx, identity)

    def test_identity_column_collision_rejected(self):
        with self.assertRaises(ValueError):
            normalize_identity_columns(pd.DataFrame(columns=["id-01", "id_01"]))

    def test_reader_normalizes_schema_and_category_types(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "identity.csv"
            path.write_text("TransactionID,id-01,id-12,DeviceType\n1,-4,Found,mobile\n2,0,,desktop\n")
            result = read_csv(path)
            self.assertIn("id_12", result)
            self.assertEqual(str(result["id_12"].dtype), "string")
            self.assertEqual(str(result["id_01"].dtype), "float32")

    def test_missing_files_raise_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "train_transaction.csv"):
                check_files(temp)


class TemporalChecks(unittest.TestCase):
    def test_whole_days_and_full_membership(self):
        frame = invented_transactions().sample(frac=1, random_state=42)
        parts = chronological_split(frame)
        self.assertEqual([len(x) for x in parts.values()], [48, 16, 16])
        membership = split_membership(parts)
        self.assertEqual(len(membership), len(frame))
        self.assertFalse(membership.TransactionID.duplicated().any())
        day = membership.TransactionDT // 86400
        self.assertTrue((membership.groupby(day).partition.nunique() == 1).all())
        self.assertLess(parts["train"].TransactionDT.max(), parts["tune"].TransactionDT.min())
        self.assertLess(parts["tune"].TransactionDT.max(), parts["holdout"].TransactionDT.min())

    def test_invalid_fraction_rejected(self):
        with self.assertRaises(ValueError):
            chronological_split(invented_transactions(), .8, .3)


class FeatureChecks(unittest.TestCase):
    def test_no_id_target_or_raw_time_and_train_constant_removed(self):
        frame = invented_transactions()
        transformed = FeatureBuilder().fit_transform(frame)
        self.assertFalse({"TransactionID", "isFraud", "TransactionDT", "all_missing"} & set(transformed))

    def test_training_profiles_are_frozen_and_unknown_groups_safe(self):
        training = invented_transactions()
        training.loc[training.index[-1], "card1"] = "a"  # Non-constant group frequencies.
        builder = FeatureBuilder(True, ["card1"]).fit(training)
        query = training.iloc[:2].copy()
        query.loc[query.index[0], "TransactionAmt"] = 1000
        query.loc[query.index[1], "card1"] = "unseen"
        first = builder.transform(query)
        expected_mean = training.loc[training.card1 == "a", "TransactionAmt"].mean()
        self.assertAlmostEqual(first.iloc[0]["amount_to_card1_mean"], 1000 / expected_mean, places=5)
        self.assertEqual(first.iloc[1]["card1_frequency"], 0)
        self.assertTrue(pd.isna(first.iloc[1]["amount_to_card1_mean"]))
        changed = query.copy()
        changed.loc[changed.index[1], "TransactionAmt"] = 999999
        second = builder.transform(changed)
        self.assertAlmostEqual(first.iloc[0]["amount_to_card1_mean"], second.iloc[0]["amount_to_card1_mean"])

    def test_missing_category_does_not_collide_with_literal(self):
        values = categorical_values(pd.Series([None, "<MISSING>", "M:"]))
        self.assertEqual(values.nunique(), 3)

    def test_tree_unknown_category_has_explicit_missing_code(self):
        x = pd.DataFrame({"cat": ["V:a", "V:b"], "num": [1., 2.]})
        encoder = TreeEncoder().fit(x, ["cat"])
        query = pd.DataFrame({"cat": ["V:c", "V:a"], "num": [3., 4.]})
        self.assertEqual(encoder.transform(query)["cat"].tolist(), [-1, 0])

    def test_zero_group_standard_deviation_becomes_missing(self):
        x = invented_transactions()
        x.loc[x.card1 == "a", "TransactionAmt"] = 3
        builder = FeatureBuilder(True, ["card1"]).fit(x)
        transformed = builder.transform(x)
        self.assertTrue(transformed.loc[x.card1 == "a", "amount_to_card1_std"].isna().all())
        numeric = transformed.select_dtypes(include="number")
        self.assertFalse(np.isinf(numeric.to_numpy()).any())


class EvaluationChecks(unittest.TestCase):
    def test_known_ranking_metrics(self):
        result = metrics([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])
        self.assertEqual(result["roc_auc"], 1.)
        self.assertEqual(result["average_precision"], 1.)
        self.assertEqual(result["frauds"], 2)

    def test_invalid_predictions_rejected(self):
        for p in ([np.nan, .4], [-.1, .8], [.2, 1.1]):
            with self.assertRaises(ValueError):
                metrics([0, 1], p)

    def test_review_budget_ties_are_deterministic(self):
        frame = pd.DataFrame({"TransactionID": [3, 1, 2, 4], "isFraud": [0, 1, 1, 0],
                              "TransactionAmt": [10., 20., 30., 40.]})
        row = review_capacity(frame, [.5] * 4, [.5]).iloc[0]
        self.assertEqual(row["reviewed"], 2)
        self.assertEqual(row["precision"], 1)
        self.assertEqual(row["recall"], 1)

    def test_submission_is_key_aligned(self):
        sample = pd.DataFrame({"TransactionID": [3, 1, 2], "isFraud": [0., 0., 0.]})
        result = make_submission(sample, [1, 2, 3], [.1, .2, .3])
        self.assertEqual(result.isFraud.tolist(), [.3, .1, .2])
        with self.assertRaises(ValueError):
            make_submission(sample, [1, 2, 4], [.1, .2, .3])


class PackagingChecks(unittest.TestCase):
    def test_default_configuration(self):
        cfg = load_config(ROOT / "configs" / "default.toml")
        self.assertEqual(len(cfg["models"]["names"]), 3)
        self.assertTrue(Path(cfg["data_dir"]).is_absolute())

    def test_notebook_is_unexecuted_and_code_parses(self):
        path = ROOT / "notebooks" / "01_fraud_detection_walkthrough.ipynb"
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
