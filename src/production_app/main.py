import pandas as pd
from production_app.config import FEATURE_NAME_MAPPING
from production_app.logging_config import configure_logging
from production_app.model_service import load_model, predict
from production_app.schemas import (
    BreastCancerInput,
    PredictionResponse,
    BatchPredictionResponse,
    BatchPredictionRequest,
)

from fastapi import FastAPI, HTTPException


configure_logging()
app = FastAPI()
model = load_model()


@app.get("/health")
def home():
    return {"status": "healthy" if model is not None else "not ready"}


# @app.get("/model/info")
# def model_info():
#     pass


@app.post("/predict", response_model=PredictionResponse)
def predict_one(data: BreastCancerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    input_data = pd.DataFrame([data.model_dump()])
    input_data = input_data.rename(columns=FEATURE_NAME_MAPPING)

    prediction = predict(input_data)[0]

    class_names = {
        0: "malignant",
        1: "benign",
    }

    return {
        "prediction": int(prediction),
        "class_name": class_names[int(prediction)],
    }


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    input_data = pd.DataFrame([sample.model_dump() for sample in data.samples])
    input_data = input_data.rename(columns=FEATURE_NAME_MAPPING)

    predictions = predict(input_data)

    class_names = {
        0: "malignant",
        1: "benign",
    }

    return {
        "predictions": [
            {
                "prediction": int(prediction),
                "class_name": class_names[int(prediction)],
            }
            for prediction in predictions
        ]
    }
