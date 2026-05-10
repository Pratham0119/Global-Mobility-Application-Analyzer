import os
import sys
import yaml
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.logger import logging
from src.exception import CustomException

from src.entity.model_trainer_config import ModelTrainerConfig
from src.entity.model_trainer_artifact import ModelTrainerArtifact


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(
        self,
        train_file_path: str,
        test_file_path: str,
        preprocessor_file_path: str
    ):
        try:
            logging.info("Model training started")

            os.makedirs(
                self.model_trainer_config.model_trainer_dir,
                exist_ok=True
            )

            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            logging.info("Train and test transformed data loaded successfully")

            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]

            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            logging.info("Features and target separated successfully")

            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier()
            }

            model_report = {}

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)

                model_report[model_name] = {
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "train_f1_score": float(train_f1),
                    "test_f1_score": float(test_f1)
                }

            logging.info("All models trained successfully")

            best_model_name = None
            best_model_score = -1
            best_model = None

            for model_name, scores in model_report.items():
                if scores["test_f1_score"] > best_model_score:
                    best_model_score = scores["test_f1_score"]
                    best_model_name = model_name
                    best_model = models[model_name]

            logging.info(f"Best model selected: {best_model_name}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No best model found with acceptable performance")

            with open(self.model_trainer_config.model_report_file_path, "w") as file:
                yaml.dump(model_report, file)

            logging.info("Model report saved successfully")

            with open(preprocessor_file_path, "rb") as file:
                preprocessor = pickle.load(file)

            final_model = {
                "preprocessor": preprocessor,
                "model": best_model
            }

            with open(self.model_trainer_config.trained_model_file_path, "wb") as file:
                pickle.dump(final_model, file)

            logging.info("Best trained model with preprocessor saved successfully")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric=model_report[best_model_name]["train_f1_score"],
                test_metric=model_report[best_model_name]["test_f1_score"]
            )

            logging.info("Model trainer artifact created successfully")

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)