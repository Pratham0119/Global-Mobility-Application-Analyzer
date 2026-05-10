import sys

from src.logger import logging
from src.exception import CustomException
from src.components.model_trainer import ModelTrainer


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(
        self,
        train_file_path: str,
        test_file_path: str,
        preprocessor_file_path: str
    ):
        try:
            logging.info("Model Trainer Pipeline started")

            model_trainer = ModelTrainer()
            model_trainer_artifact = model_trainer.initiate_model_training(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                preprocessor_file_path=preprocessor_file_path
            )

            logging.info("Model Trainer Pipeline completed successfully")

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)