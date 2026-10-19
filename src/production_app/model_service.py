import logging

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from production_app.config import load_settings

logger = logging.getLogger(__name__)


def save_model(model: Pipeline) -> None:
    load_settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, load_settings.model_path)
    logger.info("Model saved: %s", load_settings.model_path)


def load_model() -> Pipeline:
    if not load_settings.model_path.exists():
        raise FileNotFoundError("Model not found. Run 'train' first.")
    model = joblib.load(load_settings.model_path)
    logger.info("Model loaded: %s", load_settings.model_path)
    return model


def predict(model: Pipeline, X: pd.DataFrame) -> int:
    if X.shape[0] != 1 or X.shape[1] != model.n_features_in_:
        raise ValueError("Prediction needs one row with 30 features.")
    prediction = int(model.predict(X)[0])
    logger.info("Predicted class: %d", prediction)
    return prediction
