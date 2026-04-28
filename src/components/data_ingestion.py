import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Data ingestion started")

            # Read dataset
            df = pd.read_csv(self.data_ingestion_config.dataset_path)
            logging.info("Dataset read successfully")

            # Create folders
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved successfully")

            # Train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )
            logging.info("Train-test split completed")

            # Save train and test
            train_set.to_csv(self.data_ingestion_config.train_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_path, index=False)
            logging.info("Train and test files saved successfully")

            # Create artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_path,
                test_file_path=self.data_ingestion_config.test_path,
                raw_data_file_path=self.data_ingestion_config.raw_data_path
            )

            logging.info("Data ingestion completed successfully")

            return data_ingestion_artifact

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = DataIngestionConfig(
        dataset_path="notebook/EasyVisa.csv",
        train_path="artifacts/data_ingestion/train.csv",
        test_path="artifacts/data_ingestion/test.csv",
        raw_data_path="artifacts/data_ingestion/raw.csv"
    )

    data_ingestion = DataIngestion(data_ingestion_config=config)
    artifact = data_ingestion.initiate_data_ingestion()

    print("Data Ingestion completed successfully!")
    print(artifact)