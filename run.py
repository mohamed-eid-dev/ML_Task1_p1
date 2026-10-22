import logging
import sys

from production_app.logging_config import configure_logging
from model_train import predict_workflow, train_workflow


def main() -> int:
    configure_logging()
    if len(sys.argv) != 2 or sys.argv[1] not in {"train", "predict"}:
        logging.error("Use: python run.py train OR python run.py predict")
        return 1

    try:
        if sys.argv[1] == "train":
            train_workflow()
        else:
            predict_workflow()
    except (FileNotFoundError, ValueError) as error:
        logging.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
