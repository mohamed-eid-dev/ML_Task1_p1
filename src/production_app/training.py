import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from production_app.config import load_settings


logger = logging.getLogger(__name__)


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(random_state=load_settings().random_state),
            ),
        ]
    )
    logger.info("Training started")
    pipeline.fit(x_train, y_train)  # Scaler learns from TRAIN data only.
    logger.info("Training finished")
    return pipeline
