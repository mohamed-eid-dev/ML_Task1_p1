import logging

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info("Accuracy: %.4f", accuracy)
    logger.info(
        "Classification report:\n%s", classification_report(y_test, predictions)
    )
