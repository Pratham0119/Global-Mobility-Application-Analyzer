import os
import sys
import pandas as pd
import numpy as np
import pickle


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.entity.data_transformation_config import DataTransformationConfig
from src.entity.data_transformation_artifact import DataTransformationArtifact
from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(
        self,
        train_file_path: str,
        test_file_path: str,
        data_transformation_config: DataTransformationConfig
    ):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.data_transformation_config = data_transformation_config

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            print("Reading train and test files...")

            train_df = pd.read_csv(self.train_file_path)
            test_df = pd.read_csv(self.test_file_path)

            logging.info("Train and test files loaded successfully")
            print("Train and test files loaded successfully")

            target_column = "case_status"

            # Drop target column and case_id
            drop_columns = [target_column, "case_id"]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Input and target features separated successfully")
            print("Input and target features separated successfully")

            target_feature_train_df = target_feature_train_df.map(
                {"Certified": 1, "Denied": 0}
            )

            target_feature_test_df = target_feature_test_df.map(
                {"Certified": 1, "Denied": 0}
            )

            logging.info("Target column encoded successfully")
            print("Target column encoded successfully")

            categorical_columns = input_feature_train_df.select_dtypes(
                include=["object"]
            ).columns

            numerical_columns = input_feature_train_df.select_dtypes(
                exclude=["object"]
            ).columns

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            print(f"Categorical columns: {list(categorical_columns)}")
            print(f"Numerical columns: {list(numerical_columns)}")

            numerical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessor object created successfully")
            print("Preprocessor object created successfully")

            print("Transforming train and test input features...")

            transformed_input_train = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test = preprocessor.transform(input_feature_test_df)

            # Safe conversion: only convert if result is sparse matrix
            if hasattr(transformed_input_train, "toarray"):
                transformed_input_train = transformed_input_train.toarray()

            if hasattr(transformed_input_test, "toarray"):
                transformed_input_test = transformed_input_test.toarray()

            logging.info("Train and test input features transformed successfully")
            print("Train and test input features transformed successfully")

            print("Combining transformed features with target column...")

            train_arr = np.c_[
                transformed_input_train,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                transformed_input_test,
                np.array(target_feature_test_df)
            ]

            logging.info("Transformed input features combined with target column successfully")
            print("Transformed input features combined with target column successfully")

            os.makedirs(
                os.path.dirname(self.data_transformation_config.transformed_train_file_path),
                exist_ok=True
            )

            print("Saving transformed train file...")

            pd.DataFrame(train_arr).to_csv(
                self.data_transformation_config.transformed_train_file_path,
                index=False,
                header=True
            )

            print("Saving transformed test file...")

            pd.DataFrame(test_arr).to_csv(
                self.data_transformation_config.transformed_test_file_path,
                index=False,
                header=True
            )

            logging.info("Transformed train and test files saved successfully")
            print("Transformed train and test files saved successfully")

            print("Saving preprocessor object...")

            with open(self.data_transformation_config.preprocessor_object_file_path, "wb") as file_obj:
                pickle.dump(preprocessor, file_obj)

            logging.info("Preprocessor object saved successfully")
            print("Preprocessor object saved successfully")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessor_object_file_path=self.data_transformation_config.preprocessor_object_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)