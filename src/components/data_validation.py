import os
import sys
import yaml
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.entity.data_validation_artifact import DataValidationConfig, DataValidationArtifact


class DataValidation:
    def __init__(self, train_file_path: str, test_file_path: str, data_validation_config: DataValidationConfig):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.data_validation_config = data_validation_config

    def read_schema(self) -> dict:
        try:
            with open(self.data_validation_config.schema_file_path, "r") as yaml_file:
                schema = yaml.safe_load(yaml_file)
            return schema
        except Exception as e:
            raise CustomException(e, sys)

    def validate_columns(self, dataframe: pd.DataFrame, schema: dict) -> bool:
        try:
            expected_columns = list(schema["columns"].keys())
            actual_columns = list(dataframe.columns)

            missing_columns = []

            for column in expected_columns:
                if column not in actual_columns:
                    missing_columns.append(column)

            if len(missing_columns) > 0:
                logging.info(f"Missing columns: {missing_columns}")
                return False

            return True

        except Exception as e:
            raise CustomException(e, sys)

    def check_missing_values(self, dataframe: pd.DataFrame) -> bool:
        try:
            missing = dataframe.isnull().sum()
            missing_columns = missing[missing > 0].index.tolist()

            if len(missing_columns) > 0:
                logging.info(f"Missing values found in columns: {missing_columns}")
                return False

            return True

        except Exception as e:
            raise CustomException(e, sys)

    def check_duplicates(self, dataframe: pd.DataFrame) -> bool:
        try:
            duplicate_count = dataframe.duplicated().sum()

            if duplicate_count > 0:
                logging.info(f"Duplicate rows found: {duplicate_count}")
                return False

            return True

        except Exception as e:
            raise CustomException(e, sys)

    def validate_target_column(self, dataframe: pd.DataFrame, schema: dict) -> bool:
        try:
            target_column = schema["target_column"]["name"]

            if target_column not in dataframe.columns:
                logging.info(f"Target column missing: {target_column}")
                return False

            if dataframe[target_column].isnull().sum() > 0:
                logging.info(f"Null values found in target column: {target_column}")
                return False

            return True

        except Exception as e:
            raise CustomException(e, sys)

    def write_report(self, report: dict):
        try:
            report_dir = os.path.dirname(self.data_validation_config.report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            with open(self.data_validation_config.report_file_path, "w") as file:
                yaml.dump(report, file)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Data validation started")

            train_df = pd.read_csv(self.train_file_path)
            test_df = pd.read_csv(self.test_file_path)

            schema = self.read_schema()

            train_status = self.validate_columns(train_df, schema)
            test_status = self.validate_columns(test_df, schema)

            train_missing_status = self.check_missing_values(train_df)
            test_missing_status = self.check_missing_values(test_df)

            train_duplicate_status = self.check_duplicates(train_df)
            test_duplicate_status = self.check_duplicates(test_df)

            train_target_status = self.validate_target_column(train_df, schema)
            test_target_status = self.validate_target_column(test_df, schema)

            validation_status = (
                train_status
                and test_status
                and train_missing_status
                and test_missing_status
                and train_duplicate_status
                and test_duplicate_status
                and train_target_status
                and test_target_status
            )

            report = {
                "train_file_valid": train_status,
                "test_file_valid": test_status,
                "train_missing_values": train_missing_status,
                "test_missing_values": test_missing_status,
                "train_duplicates": train_duplicate_status,
                "test_duplicates": test_duplicate_status,
                "train_target_column": train_target_status,
                "test_target_column": test_target_status,
                "validation_status": validation_status
            }

            self.write_report(report)

            if validation_status:
                message = "Data validation completed successfully"
            else:
                message = "Data validation failed."

            logging.info(message)

            return DataValidationArtifact(
                validation_status=validation_status,
                message=message,
                report_file_path=self.data_validation_config.report_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)