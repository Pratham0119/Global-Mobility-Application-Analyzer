from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class DataTransformationConfig:
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_file_path: str


    def __init__(self, artifact_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        self.transformed_train_file_path = os.path.join(
            artifact_dir, "data_transformation", f"train_{timestamp}.csv"
        )

        self.transformed_test_file_path = os.path.join(
            artifact_dir, "data_transformation", f"test_{timestamp}.csv"
        )

        self.preprocessor_object_file_path = os.path.join(
            artifact_dir, "data_transformation", f"preprocessor_{timestamp}.pkl"
        )