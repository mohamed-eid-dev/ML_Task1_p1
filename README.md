# ML Task 1 — Breast Cancer Classification

A production-oriented Machine Learning classification project using Python, scikit-learn, uv, and Joblib.

## Overview

This project implements a complete Machine Learning workflow for the Breast Cancer Wisconsin Dataset:

```text
Data Inspection
      ↓
Data Cleaning / Validation
      ↓
EDA
      ↓
Train / Test Split
      ↓
Preprocessing
      ↓
Feature Scaling
      ↓
Logistic Regression
      ↓
Evaluation
      ↓
Model Serialization
      ↓
Production Model Loading
```

## Dataset

- Dataset: Breast Cancer Wisconsin Dataset
- Samples: 569
- Numerical features: 30
- Target: binary classification
- `0`: Malignant
- `1`: Benign

## Project Structure

```text
ML_Task1_p1/
│
├── models/
│   └── logistic_regression_pipeline.joblib
│
├── notebook/
│   └── 01_exploration.ipynb
│
├── src/
│   └── production_app/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_config.py
│       ├── model_service.py
│       └── main.py
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── requirement.txt
└── uv.lock
```

## Machine Learning Workflow

### 1. Inspection

The notebook uses pandas operations such as:

- `head()`
- `shape`
- `info()`
- `describe()`
- `dtypes`
- `isna()`
- `duplicated()`
- `value_counts()`

### 2. Cleaning

The dataset is already relatively clean. Therefore, cleaning focuses on deterministic data-quality validation rather than aggressive modification.

Checks include:

- Missing values
- Duplicate rows
- Invalid values
- Numerical ranges
- Potential outliers
- Target validity

Statistical outliers are not automatically deleted because an extreme value is not necessarily an invalid observation.

### 3. EDA

EDA includes:

- Target distribution
- Numerical distributions
- Skewness analysis
- Boxplots
- Correlation analysis

The EDA results are used to guide later preprocessing decisions.

### 4. Train / Test Split

The dataset is split into:

```text
Train: 455 samples
Test : 114 samples
```

The target distribution is checked in both sets.

## Preprocessing

Logistic Regression is sensitive to feature scale, so the project uses:

```python
StandardScaler()
```

The scaler is fitted only on training data and then used to transform both training and test data.

```text
X_train → fit → transform
X_test  → transform
```

This prevents test-set information from being used to learn preprocessing statistics.

## Model

The baseline model is:

```python
LogisticRegression(random_state=42)
```

Logistic Regression was selected as a straightforward baseline for binary classification.

## Pipeline

The preferred trained artifact combines preprocessing and the model:

```text
Input Features
      ↓
StandardScaler
      ↓
LogisticRegression
      ↓
Prediction
```

This ensures that the same preprocessing learned during training is applied during inference.

## Evaluation

The baseline model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Current baseline results:

```text
Accuracy  : 0.9825
Precision : 0.9861
Recall    : 0.9861
F1-Score  : 0.9861
```

Classification report:

```text
              precision    recall  f1-score   support

malignant        0.98      0.98      0.98        42
benign           0.99      0.99      0.99        72

accuracy                             0.98       114
macro avg        0.98      0.98      0.98       114
weighted avg     0.98      0.98      0.98       114
```

These are the results of the current baseline experiment.

## Model Serialization

The trained pipeline is saved as:

```text
models/logistic_regression_pipeline.joblib
```

The artifact contains the preprocessing + model workflow, allowing the production application to load the trained pipeline without retraining.

## Production Application

The notebook is responsible for experimentation and training:

```text
Exploration
   ↓
Cleaning
   ↓
EDA
   ↓
Train/Test Split
   ↓
Training
   ↓
Evaluation
   ↓
Model Serialization
```

The production application is responsible for inference:

```text
Application Start
       ↓
Load Configuration
       ↓
Configure Logging
       ↓
Load Serialized Model
       ↓
Inference
```

The production application does not depend on the notebook at runtime.

## Production Modules

### `config.py`

Handles application configuration:

```text
APP_ENV
MODEL_PATH
LOG_LEVEL
```

Default model path:

```text
models/logistic_regression_pipeline.joblib
```

### `logging_config.py`

Provides centralized logging configuration using a consistent format:

```text
timestamp | level | module | message
```

### `model_service.py`

Responsible for:

- Loading the serialized model
- Running predictions
- Handling model loading errors
- Logging model operations

### `main.py`

Production entry point responsible for:

1. Loading settings
2. Configuring logging
3. Loading the trained model
4. Starting the application flow

## Environment Management

The project uses **uv** for dependency and environment management.

Important files:

```text
pyproject.toml
uv.lock
.python-version
```

`uv.lock` provides reproducible dependency versions.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/mohamed-eid-dev/ML_Task1_p1.git
cd ML_Task1_p1
```

Install the environment and dependencies:

```bash
uv sync
```

Optional: copy the environment settings:

```bash
cp .env.example .env
```

You can leave `.env` missing; the application uses the defaults.

## Run it (2 commands)

**1. Train once and save the model:**

```bash
uv run python run.py train
```
This creates `models/pipeline.joblib`.

**2. Predict using the SAVED model:**

```bash
uv run python run.py predict
```

The model artifact must exist at:

```text
models/pipeline.joblib
```

## Reproducing the Model

Open:

```text
notebook/01_exploration.ipynb
```

Run the notebook from top to bottom.

The notebook covers:

```text
Data Loading
     ↓
Inspection
     ↓
Cleaning / Validation
     ↓
EDA
     ↓
Train/Test Split
     ↓
Preprocessing
     ↓
Scaling
     ↓
Logistic Regression
     ↓
Evaluation
     ↓
Model Serialization
```

## Reproducibility

The model uses:

```python
random_state=42
```

Dependency reproducibility is handled by:

```text
uv.lock
```

## Engineering Principles

- Separation of experimentation and production inference
- No data leakage during preprocessing
- Reproducible training
- Pipeline-based inference
- Centralized configuration
- Centralized logging
- Model serialization
- Clear separation of responsibilities

## Future Improvements

- Cross-validation
- Hyperparameter tuning
- ROC-AUC
- Precision-Recall analysis
- Model comparison
- Feature selection
- Input validation
- Automated testing
- FastAPI inference API
- Dockerization
- CI/CD
- Model versioning
- Experiment tracking
- Model monitoring
- Data drift detection

## Disclaimer

This project is for educational and portfolio purposes. It is not intended for clinical diagnosis or medical decision-making.

## Author

**Mohamed Eid**

GitHub: https://github.com/mohamed-eid-dev
