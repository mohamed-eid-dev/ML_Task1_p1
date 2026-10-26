import unittest

from production_app.schemas import BatchPredictionRequest, BreastCancerInput
from tests.conftest import valid_sample


class SchemaTests(unittest.TestCase):
    def test_input_schema_accepts_positive_features(self):
        sample = BreastCancerInput(**valid_sample())

        self.assertEqual(sample.mean_radius, 1.0)

    def test_input_schema_rejects_non_positive_features(self):
        sample = valid_sample()
        sample["mean_radius"] = 0

        with self.assertRaises(ValueError):
            BreastCancerInput(**sample)

    def test_batch_request_accepts_multiple_samples(self):
        request = BatchPredictionRequest(samples=[valid_sample(), valid_sample()])

        self.assertEqual(len(request.samples), 2)
