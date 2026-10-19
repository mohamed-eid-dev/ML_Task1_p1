import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.preprocessing import create_scaler

logger = logging.getLogger(__name__)


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    # Same idea as the notebook: scale first, then train a classifier.
    pipeline = Pipeline(
        [
            ("scaler", create_scaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    logger.info("Training started")
    pipeline.fit(x_train, y_train)  # Scaler learns from TRAIN data only.
    logger.info("Training finished")
    return pipeline
