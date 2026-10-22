import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

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


FEATURE_NAME_MAPPING = {
    "mean_radius": "mean radius",
    "mean_concavity": "mean concavity",
    "mean_concave_points": "mean concave points",
    "radius_error": "radius error",
    "worst_perimeter": "worst perimeter",
    "worst_concavity": "worst concavity",
    "worst_concave_points": "worst concave points",
    "worst_symmetry": "worst symmetry",
}


@dataclass(frozen=True)
class Settings:
    model_path: Path
    log_level: str
    test_size: float
    random_state: int
    selected_features: tuple[str, ...]


def load_settings() -> Settings:

    return Settings(
        model_path=ROOT.parent / os.getenv("MODEL_PATH", "models/pipeline.joblib"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        selected_features=SELECTED_FEATURES,
    )
