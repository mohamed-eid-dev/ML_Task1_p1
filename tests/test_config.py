import unittest
from pathlib import Path
from unittest.mock import patch

from production_app import config


class ConfigurationTests(unittest.TestCase):
    def test_load_settings_reads_environment(self):
        with (
            patch.object(config, "ROOT", Path("project") / "src"),
            patch.dict(
                "os.environ",
                {
                    "MODEL_PATH": "artifacts/model.joblib",
                    "LOG_LEVEL": "debug",
                    "TEST_SIZE": "0.3",
                    "RANDOM_STATE": "9",
                },
            ),
        ):
            settings = config.load_settings()

        self.assertEqual(settings.model_path, Path("project/artifacts/model.joblib"))
        self.assertEqual(settings.log_level, "DEBUG")
        self.assertEqual(settings.test_size, 0.3)
        self.assertEqual(settings.random_state, 9)
        self.assertEqual(settings.selected_features, config.SELECTED_FEATURES)
