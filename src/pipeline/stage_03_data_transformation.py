import sys

from src.components.data_transformation import DataTransformation
from src.entity.data_transformation_config import DataTransformationConfig
from src.entity.data_transformation_artifact import DataTransformationArtifact
from src.exception import CustomException
from src.logger import logging


class DataTransformationPipeline:
    def __init__(self):
        pass

    def start_data_transformation(
        self,
        train_file_path: str,
        test_file_path: str,
        artifact_dir: str
    ) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Pipeline started")

            data_transformation_config = DataTransformationConfig(
                artifact_dir=artifact_dir
            )

            data_transformation = DataTransformation(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                data_transformation_config=data_transformation_config
            )

            data_transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info("Data Transformation Pipeline completed successfully")

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)