import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    model_path: Path
    log_level: str
    test_size: float
    random_state: int


def load_settings() -> Settings:

    return Settings(
        model_path=ROOT / os.getenv("MODEL_PATH", "models/pipeline.joblib"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
    )
