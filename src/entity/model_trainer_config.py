import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join("artifacts", "model_trainer")

    trained_model_file_path: str = os.path.join(
        model_trainer_dir,
        f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    )

    expected_accuracy: float = 0.70

    model_report_file_path: str = os.path.join(
        model_trainer_dir,
        "model_report.yaml"
    )