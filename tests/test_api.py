import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from production_app import config
from tests.conftest import valid_sample


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("production_app.model_service.load_model", return_value=None):
            from production_app import main

        cls.api = main
        cls.client = TestClient(main.app)

    def setUp(self):
        self.api.model = object()

    def test_health_reports_model_availability(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    def test_single_prediction_maps_feature_names_and_class(self):
        with patch.object(self.api, "predict", return_value=0) as predict:
            response = self.client.post("/predict", json=valid_sample())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": 0, "class_name": "malignant"})
        self.assertEqual(
            tuple(predict.call_args.args[1].columns), config.SELECTED_FEATURES
        )

    def test_batch_prediction_returns_class_for_each_sample(self):
        with patch.object(
            self.api, "predict_batch_service", return_value=[0, 1]
        ) as predict_batch:
            response = self.client.post(
                "/predict/batch",
                json={"samples": [valid_sample(), valid_sample()]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "predictions": [
                    {"prediction": 0, "class_name": "malignant"},
                    {"prediction": 1, "class_name": "benign"},
                ]
            },
        )
        self.assertEqual(
            tuple(predict_batch.call_args.args[1].columns), config.SELECTED_FEATURES
        )

    def test_api_reports_unavailable_model(self):
        self.api.model = None

        response = self.client.post("/predict", json=valid_sample())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Model is not available")

    def test_api_rejects_invalid_prediction_input(self):
        sample = valid_sample()
        sample["mean_radius"] = 0

        response = self.client.post("/predict", json=sample)

        self.assertEqual(response.status_code, 422)
