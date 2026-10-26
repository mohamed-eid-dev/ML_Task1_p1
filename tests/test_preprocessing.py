import unittest

from sklearn.preprocessing import StandardScaler

from production_app.preprocessing import create_scaler


class PreprocessingTests(unittest.TestCase):
    def test_create_scaler_returns_standard_scaler(self):
        scaler = create_scaler()
        scaled = scaler.fit_transform([[1.0], [3.0], [5.0]])

        self.assertIsInstance(scaler, StandardScaler)
        self.assertAlmostEqual(float(scaled.mean()), 0.0)
        self.assertAlmostEqual(float(scaled.std()), 1.0)
