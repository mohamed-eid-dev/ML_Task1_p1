

import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def load_model(path: str) -> Any:

    model_path = Path(path)

    logger.info("Loading model from %s", model_path)

    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        logger.exception("Model file was not found: %s", model_path)
        raise


def predict(model: Any, features: Any) -> Any:
    
    logger.debug("Running prediction.")

    result = model.predict(features)

    return result