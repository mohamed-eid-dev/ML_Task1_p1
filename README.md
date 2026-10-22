# ML Task 2 — Production ML Application

A production-oriented Machine Learning classification application for the Breast Cancer Wisconsin Dataset using Python, scikit-learn, FastAPI, Pydantic, Streamlit, Docker, and Docker Compose.

The project starts with Machine Learning experimentation and model training, then exposes the trained model through a production API and a Streamlit user interface.

---

## Overview

This project implements an end-to-end Machine Learning workflow:

```text
Data Loading
     ↓
Data Inspection
     ↓
Data Validation
     ↓
EDA
     ↓
Train / Test Split
     ↓
Feature Selection (RFECV)
     ↓
Selected Features
     ↓
StandardScaler
     ↓
Logistic Regression
     ↓
Evaluation
     ↓
Model Serialization
     ↓
FastAPI
     ↓
Streamlit UI
     ↓
Docker
     ↓
Docker Compose
```

The project separates:

- Machine Learning experimentation
- Model training
- Model evaluation
- Model serialization
- Production inference
- API layer
- User interface
- Containerization

---

## Dataset

The project uses the Breast Cancer Wisconsin Dataset provided by `scikit-learn`.

- Dataset: Breast Cancer Wisconsin Dataset
- Samples: 569
- Original numerical features: 30
- Target: Binary classification
- `0`: Malignant
- `1`: Benign

The dataset is loaded using:

```python
from sklearn.datasets import load_breast_cancer
```

---

## Project Structure

```text
ML_Task1_p1/
│
├── models/
│   └── pipeline.joblib
│
├── notebook/
│   └── 01_exploration.ipynb
│
├── src/
│   └── production_app/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── logging_config.py
│       ├── main.py
│       ├── model_service.py
│       ├── preprocessing.py
│       ├── schemas.py
│       └── training.py
│
├── ui/
│   └── streamlit_app.py
│
├── Dockerfile.api
├── Dockerfile.streamlit
├── compose.yaml
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── model_train.py
├── run.py
├── pyproject.toml
├── requirement.txt
├── README.md
└── uv.lock
```

---

# Machine Learning Workflow

## 1. Data Inspection

The notebook is used for exploring and understanding the dataset before production implementation.

The following pandas operations are used:

- `head()`
- `shape`
- `info()`
- `describe()`
- `dtypes`
- `isna()`
- `duplicated()`
- `value_counts()`

The dataset contains:

```text
569 rows
30 features
```

---

## 2. Data Validation

The dataset is validated before training.

Validation includes:

- Missing values
- Duplicate rows
- Data types
- Invalid values
- Numerical ranges
- Target validity
- Potential outliers

Statistical outliers are not automatically removed because an extreme value does not necessarily represent an invalid observation.

---

## 3. Exploratory Data Analysis

EDA is performed to understand the dataset and identify useful patterns.

The analysis includes:

- Target distribution
- Feature distributions
- Skewness analysis
- Boxplots
- Correlation analysis

---

# Train / Test Split

The dataset is split into training and testing sets.

```text
Total samples: 569

Training: 455
Testing : 114
```

The split uses:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

`stratify=y` keeps the target-class distribution approximately consistent between training and testing sets.

---

# Feature Selection

The original dataset contains 30 features.

Instead of sending all 30 features to the production API, feature selection was performed using:

```text
RFECV
```

`RFECV` stands for:

```text
Recursive Feature Elimination with Cross-Validation
```

The selector uses Logistic Regression as its estimator.

The final model uses 8 selected features.

## Selected Features

```text
mean radius
mean concavity
mean concave points
radius error
worst perimeter
worst concavity
worst concave points
worst symmetry
```

These features are stored centrally in:

```text
src/production_app/config.py
```

This allows the production application to use the same feature definition consistently.

---

# Preprocessing

Logistic Regression is sensitive to feature scale.

Therefore, the selected features are standardized using:

```python
StandardScaler()
```

The scaler is fitted only on the training data.

```text
X_train
   ↓
fit + transform
   ↓
Scaled X_train
```

The test data is only transformed:

```text
X_test
   ↓
transform
   ↓
Scaled X_test
```

This prevents data leakage from the test set.

---

# Model

The classification model used by the project is:

```python
LogisticRegression(random_state=42)
```

Logistic Regression was selected as a straightforward baseline model for binary classification.

---

# ML Pipeline

The final trained pipeline combines preprocessing and the classifier.

The production model receives the selected features:

```text
8 Selected Features
        ↓
StandardScaler
        ↓
LogisticRegression
        ↓
Prediction
```

The trained pipeline is serialized as a Joblib artifact.

---

# Model Serialization

The trained model is saved using `joblib`.

```text
models/pipeline.joblib
```

The saved artifact contains the trained preprocessing and model workflow.

The production application loads this artifact instead of retraining the model when the API starts.

---

# Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

The latest training run produced:

```text
Accuracy: 0.9474
```

Classification results from the latest run:

```text
              precision    recall  f1-score   support

           0       0.93      0.93      0.93        42
           1       0.96      0.96      0.96        72

    accuracy                           0.95       114
   macro avg       0.94      0.94      0.94       114
weighted avg       0.95      0.95      0.95       114
```

Because the model is retrained during development, evaluation results may change when the training workflow or configuration changes.

---

# Production Application

The production application is responsible for model inference.

The production flow is:

```text
Application Start
       ↓
Load Configuration
       ↓
Configure Logging
       ↓
Load Serialized Model
       ↓
Validate Input
       ↓
Prepare Features
       ↓
Model Prediction
       ↓
Return Response
```

The production application does not depend on the notebook at runtime.

---

# Production Modules

## `config.py`

Centralizes application configuration.

It contains settings such as:

```text
MODEL_PATH
LOG_LEVEL
TEST_SIZE
RANDOM_STATE
SELECTED_FEATURES
```

The selected production features are defined centrally:

```python
SELECTED_FEATURES = (
    "mean radius",
    "mean concavity",
    "mean concave points",
    "radius error",
    "worst perimeter",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
)
```

This prevents feature definitions from being duplicated across the application.

---

## `data.py`

Responsible for:

- Loading the dataset
- Splitting data into training and testing sets
- Applying the configured dataset settings

---

## `preprocessing.py`

Contains preprocessing-related functionality used during the ML workflow.

---

## `training.py`

Responsible for model training.

The training workflow uses:

```text
Logistic Regression
        +
RFECV
        +
StandardScaler
```

The trained pipeline is returned as a single artifact.

---

## `evaluation.py`

Responsible for evaluating the trained model.

It calculates and logs:

- Accuracy
- Classification report
- Other evaluation metrics

---

## `model_service.py`

Responsible for model lifecycle operations:

- Loading the serialized model
- Saving the trained model
- Running model predictions
- Logging model operations

The model is loaded from:

```text
models/pipeline.joblib
```

---

## `schemas.py`

Contains the Pydantic models used to validate API requests and responses.

The prediction input schema contains the 8 selected features instead of all 30 original dataset features.

This provides a clear API contract and prevents invalid request structures from reaching the model.

---

## `logging_config.py`

Provides centralized logging configuration.

Example:

```text
INFO | production_app.data | Loaded data: 569 rows, 30 features
INFO | production_app.training | Training started
INFO | production_app.training | Training finished
INFO | production_app.model_service | Model saved
```

---

## `main.py`

The FastAPI application entry point.

It provides the production API endpoints:

```text
GET  /health
POST /predict
POST /predict/batch
```

---

# FastAPI

The trained model is exposed through a FastAPI inference API.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check application/model availability |
| POST | `/predict` | Predict one sample |
| POST | `/predict/batch` | Predict multiple samples |

---

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

The endpoint can also report that the model is not ready if the serialized model cannot be loaded.

---

# Single Prediction

```http
POST /predict
```

The API accepts the 8 selected features.

Example request:

```json
{
  "mean_radius": 17.99,
  "mean_concavity": 0.3001,
  "mean_concave_points": 0.1471,
  "radius_error": 1.095,
  "worst_perimeter": 184.6,
  "worst_concavity": 0.7119,
  "worst_concave_points": 0.2654,
  "worst_symmetry": 0.4601
}
```

Example response:

```json
{
  "prediction": 0,
  "class_name": "malignant"
}
```

---

# Batch Prediction

The API also supports multiple predictions in a single request.

```http
POST /predict/batch
```

The request contains multiple samples.

Example:

```json
{
  "samples": [
    {
      "mean_radius": 17.99,
      "mean_concavity": 0.3001,
      "mean_concave_points": 0.1471,
      "radius_error": 1.095,
      "worst_perimeter": 184.6,
      "worst_concavity": 0.7119,
      "worst_concave_points": 0.2654,
      "worst_symmetry": 0.4601
    },
    {
      "mean_radius": 10.38,
      "mean_concavity": 0.0869,
      "mean_concave_points": 0.0702,
      "radius_error": 0.9053,
      "worst_perimeter": 122.8,
      "worst_concavity": 0.175,
      "worst_concave_points": 0.111,
      "worst_symmetry": 0.2443
    }
  ]
}
```

The API returns a prediction for every sample.

---

# API Documentation

FastAPI automatically provides interactive documentation.

After starting the API, open:

```text
http://localhost:8000/docs
```

Swagger UI can be used to test:

```text
GET  /health
POST /predict
POST /predict/batch
```

---

# Streamlit UI

The project also provides a Streamlit user interface.

The UI is responsible for collecting user input and communicating with the FastAPI service.

The architecture is:

```text
User
 ↓
Streamlit UI
 ↓
FastAPI
 ↓
Pydantic Validation
 ↓
ML Pipeline
 ↓
Prediction
 ↓
FastAPI Response
 ↓
Streamlit UI
```

The Streamlit application does not need to load or retrain the model directly.

---

# Docker

The application is containerized using Docker.

The project contains separate Dockerfiles for the API and Streamlit UI:

```text
Dockerfile.api
Dockerfile.streamlit
```

The containers provide isolated environments containing the required Python dependencies and application code.

---

# Docker Compose

Docker Compose is used to run the API and UI together.

The application consists of two services:

```text
Docker Compose
      │
      ├───────────────┐
      ↓               ↓
    API              UI
  FastAPI         Streamlit
   :8000            :8501
```

The Streamlit service communicates with the API using the Docker Compose service name.

```text
http://api:8000
```

---

# Running with Docker Compose

Build and start the complete application:

```bash
docker compose up --build
```

Run in detached mode:

```bash
docker compose up --build -d
```

Stop the application:

```bash
docker compose down
```

---

# Application URLs

When running locally:

### FastAPI

```text
http://localhost:8000
```

### Swagger Documentation

```text
http://localhost:8000/docs
```

### Streamlit

```text
http://localhost:8501
```

---

# Environment Configuration

The application supports environment-based configuration.

Example:

```text
LOG_LEVEL=INFO
MODEL_PATH=models/pipeline.joblib
TEST_SIZE=0.2
RANDOM_STATE=42
```

The example configuration is provided in:

```text
.env.example
```

The actual `.env` file should not be committed to the repository.

---

# Environment Management

The project uses `uv` for Python environment and dependency management.

Important files:

```text
pyproject.toml
uv.lock
.python-version
```

Install/synchronize the project environment:

```bash
uv sync
```

`uv.lock` provides reproducible dependency versions.

---

# Local Development

## 1. Install dependencies

```bash
uv sync
```

## 2. Train the model

```bash
uv run python run.py train
```

This executes the training workflow and saves:

```text
models/pipeline.joblib
```

## 3. Start FastAPI

Use:

```bash
uv run uvicorn src.production_app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# Reproducing the Model

The original Machine Learning experimentation is available in:

```text
notebook/01_exploration.ipynb
```

The notebook covers:

```text
Data Loading
     ↓
Inspection
     ↓
Validation
     ↓
EDA
     ↓
Train/Test Split
     ↓
Feature Selection
     ↓
Preprocessing
     ↓
Training
     ↓
Evaluation
```

The production application uses Python modules instead of depending on the notebook during runtime.

---

# Reproducibility

The project uses:

```python
random_state=42
```

for deterministic behavior where applicable.

Dependency reproducibility is handled through:

```text
uv.lock
```

---

# Engineering Principles

The project follows several production-oriented principles:

- Separation of experimentation and production
- Separation of training and inference
- No data leakage during preprocessing
- Reproducible training
- Pipeline-based inference
- Centralized configuration
- Centralized logging
- Feature selection
- Input validation with Pydantic
- Model serialization
- API-based inference
- Containerized deployment
- Separation of API and UI
- Clear separation of responsibilities

---

# Architecture

The complete system can be summarized as:

```text
                    ┌─────────────────────┐
                    │       User          │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │    Streamlit UI     │
                    │       :8501         │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │       :8000         │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │ Pydantic Validation │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │  Selected Features │
                    │       8 features    │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │   ML Pipeline       │
                    │                     │
                    │  StandardScaler     │
                    │         ↓           │
                    │ LogisticRegression  │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │     Prediction      │
                    └─────────────────────┘
```

---

# Security and Deployment Considerations

The current project is designed for educational and portfolio purposes.

For a production deployment, additional considerations would include:

- Authentication and authorization
- HTTPS/TLS
- API rate limiting
- Secret management
- Input size limits
- Monitoring
- Centralized logs
- Health checks
- Model versioning
- Container image scanning

---

# Future Improvements

Potential future improvements include:

- Automated testing
- CI/CD pipeline
- Model versioning
- Experiment tracking
- Model monitoring
- Data drift detection
- ROC-AUC analysis
- Precision-Recall analysis
- Hyperparameter tuning
- Model comparison
- API authentication
- API rate limiting
- Production monitoring

---

# Disclaimer

This project is for educational and portfolio purposes.

It is not intended for clinical diagnosis or medical decision-making.

---

# Author

**Mohamed Eid**

GitHub:

```text
https://github.com/mohamed-eid-dev
```