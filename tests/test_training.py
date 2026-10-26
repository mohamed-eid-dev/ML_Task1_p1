import unittest

import pandas as pd
from sklearn.pipeline import Pipeline

from production_app.training import train_model


class TrainingTests(unittest.TestCase):
    def test_train_model_returns_fitted_pipeline(self):
        features = pd.DataFrame({"value": [0.0, 0.2, 0.8, 1.0]})
        target = pd.Series([0, 0, 1, 1])

        pipeline = train_model(features, target)

        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual([name for name, _ in pipeline.steps], ["scaler", "model"])
        self.assertEqual(pipeline.predict(features).tolist(), target.tolist())
