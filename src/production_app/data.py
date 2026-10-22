"""Load the data and split it into train and test."""

import logging

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from production_app.config import load_settings

logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data
    y = dataset.target
    logger.info(f"Loaded data: {x.shape[0]} rows, { x.shape[1]} features")
    return x, y


def split_data(
    x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    settings = load_settings()

    x = x.loc[:, settings.selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )

    logger.info(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    return X_train, X_test, y_train, y_test
