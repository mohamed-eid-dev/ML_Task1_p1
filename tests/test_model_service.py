import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from production_app import model_service


class ModelServiceTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({"value": [0.0, 0.2, 0.8, 1.0]})
        self.target = pd.Series([0, 0, 1, 1])
        self.model = Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression())]
        ).fit(self.features, self.target)

    def test_predict_and_predict_batch(self):
        single_prediction = model_service.predict(self.model, self.features.iloc[[0]])
        batch_predictions = model_service.predict_batch(self.model, self.features)

        self.assertEqual(single_prediction, 0)
        self.assertEqual(batch_predictions, [0, 0, 1, 1])
        self.assertTrue(
            all(type(prediction) is int for prediction in batch_predictions)
        )

    def test_predict_rejects_wrong_row_count_and_feature_count(self):
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            model_service.predict(self.model, self.features)
        with self.assertRaisesRegex(ValueError, "needs 1 features"):
            model_service.predict(
                self.model, pd.DataFrame({"first": [0.0], "second": [1.0]})
            )

    def test_predict_batch_rejects_wrong_feature_count(self):
        with self.assertRaisesRegex(ValueError, "needs 1 features"):
            model_service.predict_batch(
                self.model, pd.DataFrame({"first": [0.0], "second": [1.0]})
            )

    def test_save_and_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "nested" / "pipeline.joblib"
            settings = type("Settings", (), {"model_path": model_path})()
            with patch.object(model_service, "load_settings", return_value=settings):
                model_service.save_model(self.model)
                loaded = model_service.load_model()

        self.assertIsNotNone(loaded)
        self.assertEqual(model_service.predict(loaded, self.features.iloc[[0]]), 0)

    def test_load_model_returns_none_when_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = type(
                "Settings", (), {"model_path": Path(directory) / "missing"}
            )()
            with patch.object(model_service, "load_settings", return_value=settings):
                self.assertIsNone(model_service.load_model())
