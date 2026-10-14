
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str
    model_path: str
    log_level: str

def load_settings() -> Settings:

    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        model_path=os.getenv(
            "MODEL_PATH",
            "models/logistic_regression.joblib"
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )