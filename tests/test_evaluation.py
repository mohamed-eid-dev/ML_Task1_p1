import logging
import unittest

import pandas as pd

from production_app.evaluation import evaluate_model
from production_app.training import train_model


class EvaluationTests(unittest.TestCase):
    def test_evaluate_model_logs_metrics(self):
        features = pd.DataFrame({"value": [0.0, 0.2, 0.8, 1.0]})
        target = pd.Series([0, 0, 1, 1])
        model = train_model(features, target)

        with self.assertLogs(
            "production_app.evaluation", level=logging.INFO
        ) as captured:
            evaluate_model(model, features, target)

        self.assertTrue(any("Accuracy:" in message for message in captured.output))
        self.assertTrue(
            any("Classification report:" in message for message in captured.output)
        )
