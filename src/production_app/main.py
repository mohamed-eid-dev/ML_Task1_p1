import logging

from production_app.config import load_settings
from production_app.logging_config import configure_logging
from production_app.model_service import load_model

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the production application."""
    settings = load_settings()
    configure_logging(settings.log_level)

    logger.info(
        "Application started in %s mode",
        settings.app_env,
    )

    model = load_model(settings.model_path)

    logger.info("Model loaded successfully: %s", settings.model_path)


if __name__ == "__main__":
    main()