import unittest
from unittest.mock import patch

from production_app import config, data


class DataTests(unittest.TestCase):
    def test_load_data_returns_breast_cancer_dataset(self):
        features, target = data.load_data()

        self.assertEqual(features.shape, (569, 30))
        self.assertEqual(len(target), len(features))
        self.assertEqual(set(target.unique()), {0, 1})

    def test_split_data_selects_features_and_is_deterministic(self):
        features, target = data.load_data()
        settings = type(
            "Settings",
            (),
            {
                "selected_features": config.SELECTED_FEATURES,
                "test_size": 0.25,
                "random_state": 7,
            },
        )()
        with patch.object(data, "load_settings", return_value=settings):
            first_split = data.split_data(features, target)
            second_split = data.split_data(features, target)

        x_train, x_test, y_train, y_test = first_split
        self.assertEqual(tuple(x_train.columns), config.SELECTED_FEATURES)
        self.assertEqual(tuple(x_test.columns), config.SELECTED_FEATURES)
        self.assertEqual(len(x_test), 143)
        for first, second in zip(first_split, second_split):
            self.assertTrue(first.equals(second))
        self.assertEqual(len(y_train) + len(y_test), len(target))
