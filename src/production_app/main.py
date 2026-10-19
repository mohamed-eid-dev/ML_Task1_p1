import logging

from production_app.data import load_data, split_data
from production_app.evaluation import evaluate_model
from production_app.model_service import load_model, predict, save_model
from production_app.training import train_model

logger = logging.getLogger(__name__)


def train_workflow() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)


def predict_workflow() -> None:
    model = load_model()  # Only loads the saved model: no model.fit() here.
    X, _ = load_data()
    sample = X.iloc[[0]]  # One sample row, just for demonstration.

    result = predict(model, sample)
    class_names = {0: "malignant", 1: "benign"}
    logger.info("Sample 0 result: %s", class_names[result])
