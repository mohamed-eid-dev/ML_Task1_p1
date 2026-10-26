import unittest
from unittest.mock import patch

from production_app import logging_config


class LoggingConfigTests(unittest.TestCase):
    def test_configure_logging_uses_configured_level(self):
        with (
            patch.object(
                logging_config,
                "load_settings",
                return_value=type("Settings", (), {"log_level": "DEBUG"})(),
            ),
            patch.object(logging_config.logging, "basicConfig") as basic_config,
        ):
            logging_config.configure_logging()

        self.assertEqual(basic_config.call_args.kwargs["level"], "DEBUG")
