import logging

from production_app.config import load_settings


def configure_logging() -> None:
    """Configure one consistent log format for the application."""
    logging.basicConfig(
        level=load_settings.log_level,
        format="%(levelname)s | %(name)s | %(message)s",
    )
