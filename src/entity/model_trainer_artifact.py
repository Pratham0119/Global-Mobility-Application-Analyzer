from dataclasses import dataclass


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric: float
    test_metric: float
    