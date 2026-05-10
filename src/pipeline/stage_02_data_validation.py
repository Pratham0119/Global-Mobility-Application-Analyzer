import sys
import os

from src.exception import CustomException
from src.logger import logging

from src.entity.data_validation_artifact import DataValidationConfig
from src.components.data_validation import DataValidation


class DataValidationPipeline:
    def __init__(self):
        self.train_file_path = os.path.join("artifacts", "data_ingestion", "train.csv")
        self.test_file_path = os.path.join("artifacts", "data_ingestion", "test.csv")
        self.schema_file_path = os.path.join("config", "schema.yaml")
        self.report_file_path = os.path.join("artifacts", "data_validation", "report.yaml")

    def initiate_data_validation(self):
        try:
            logging.info("Starting Data Validation Pipeline")

            data_validation_config = DataValidationConfig(
                schema_file_path=self.schema_file_path,
                report_file_path=self.report_file_path
            )

            data_validation = DataValidation(
                train_file_path=self.train_file_path,
                test_file_path=self.test_file_path,
                data_validation_config=data_validation_config
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info(f"Validation Status: {data_validation_artifact.validation_status}")
            logging.info(f"Message: {data_validation_artifact.message}")

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)