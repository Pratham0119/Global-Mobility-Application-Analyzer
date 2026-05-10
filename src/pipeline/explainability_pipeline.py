import os
import sys
import pickle
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.logger import logging
from src.exception import CustomException

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class ExplainabilityPipelineConfig:
    artifacts_dir: str = os.path.join("artifacts", "explainability")

    model_file_path: str = os.path.join(
        "artifacts", "model_trainer", "model_20260429_130952.pkl"
    )

    train_data_path: str = os.path.join(
        "artifacts", "data_transformation", "train_20260428202047.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts", "data_transformation", "test_20260428202047.csv"
    )

    shap_values_file: str = os.path.join(artifacts_dir, f"shap_values_{timestamp}.pkl")
    shap_summary_plot: str = os.path.join(artifacts_dir, f"shap_summary_{timestamp}.png")
    shap_force_plot: str = os.path.join(artifacts_dir, f"shap_force_{timestamp}.html")

    lime_explanation_file: str = os.path.join(
        artifacts_dir, f"lime_explanation_{timestamp}.html"
    )

    feature_importance_file: str = os.path.join(
        artifacts_dir, f"feature_importance_{timestamp}.csv"
    )


class ExplainabilityPipeline:
    def __init__(self):
        self.config = ExplainabilityPipelineConfig()

        self.model = None
        self.preprocessor = None

        self.train_df = None
        self.test_df = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.X_sample = None
        self.shap_values = None

    def create_artifacts_dir(self):
        try:
            os.makedirs(self.config.artifacts_dir, exist_ok=True)
            print("Explainability folder ready:", self.config.artifacts_dir)

            logging.info(
                f"Explainability artifacts directory created: {self.config.artifacts_dir}"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def load_model_and_preprocessor(self):
        try:
            print("Loading model from:", self.config.model_file_path)

            with open(self.config.model_file_path, "rb") as file:
                model_data = pickle.load(file)

            self.model = model_data["model"]
            self.preprocessor = model_data["preprocessor"]

            print("Model and preprocessor loaded successfully")
            logging.info("Model and preprocessor loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):
        try:
            print("Loading train data from:", self.config.train_data_path)
            print("Loading test data from:", self.config.test_data_path)

            self.train_df = pd.read_csv(self.config.train_data_path)
            self.test_df = pd.read_csv(self.config.test_data_path)

            self.X_train = self.train_df.iloc[:, :-1]
            self.y_train = self.train_df.iloc[:, -1]

            self.X_test = self.test_df.iloc[:, :-1]
            self.y_test = self.test_df.iloc[:, -1]

            print("Train data shape:", self.train_df.shape)
            print("Test data shape:", self.test_df.shape)
            print("X_train shape:", self.X_train.shape)
            print("X_test shape:", self.X_test.shape)

            logging.info("Train and test data loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def generate_shap_values(self):
        try:
            import shap

            print("Generating SHAP values...")

            self.X_sample = self.X_train.sample(100, random_state=42)

            explainer = shap.Explainer(self.model)
            self.shap_values = explainer(self.X_sample)

            print("SHAP values generated successfully")
            print("SHAP sample shape:", self.X_sample.shape)

            logging.info("SHAP values generated successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def save_shap_values(self):
        try:
            with open(self.config.shap_values_file, "wb") as file:
                pickle.dump(self.shap_values, file)

            print("SHAP values saved at:", self.config.shap_values_file)
            logging.info(f"SHAP values saved at: {self.config.shap_values_file}")

        except Exception as e:
            raise CustomException(e, sys)

    def save_shap_summary_plot(self):
        try:
            import shap
            import matplotlib.pyplot as plt

            print("Creating SHAP summary plot...")

            shap.summary_plot(
                self.shap_values,
                self.X_sample,
                show=False
            )

            plt.tight_layout()
            plt.savefig(self.config.shap_summary_plot, bbox_inches="tight")
            plt.close()

            print("SHAP summary plot saved at:", self.config.shap_summary_plot)
            logging.info(
                f"SHAP summary plot saved at: {self.config.shap_summary_plot}"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def save_shap_force_plot(self):
        try:
            import shap

            print("Creating SHAP force plot...")

            shap_force_plot = shap.plots.force(
                self.shap_values[0],
                matplotlib=False
            )

            shap.save_html(
                self.config.shap_force_plot,
                shap_force_plot
            )

            print("SHAP force plot saved at:", self.config.shap_force_plot)
            logging.info(
                f"SHAP force plot saved at: {self.config.shap_force_plot}"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def save_feature_importance(self):
        try:
            import numpy as np

            print("Calculating feature importance...")

            importance = np.abs(self.shap_values.values).mean(axis=0)

            feature_importance_df = pd.DataFrame(
                {
                    "feature": self.X_sample.columns,
                    "importance": importance,
                }
            )

            feature_importance_df = feature_importance_df.sort_values(
                by="importance",
                ascending=False
            )

            feature_importance_df.to_csv(
                self.config.feature_importance_file,
                index=False
            )

            print("Feature importance saved at:", self.config.feature_importance_file)
            logging.info(
                f"Feature importance saved at: {self.config.feature_importance_file}"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def generate_lime_explanation(self):
        try:
            from lime.lime_tabular import LimeTabularExplainer

            print("Generating LIME explanation...")

            explainer = LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=self.X_train.columns.tolist(),
                class_names=["Denied", "Certified"],
                mode="classification",
            )

            sample = self.X_test.iloc[0].values

            def predict_fn(data):
                return self.model.predict_proba(data)

            explanation = explainer.explain_instance(
                data_row=sample,
                predict_fn=predict_fn,
            )

            explanation.save_to_file(self.config.lime_explanation_file)

            print("LIME explanation saved at:", self.config.lime_explanation_file)
            logging.info(
                f"LIME explanation saved at: {self.config.lime_explanation_file}"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Explainability pipeline started")

            self.create_artifacts_dir()
            self.load_model_and_preprocessor()
            self.load_data()
            self.generate_shap_values()
            self.save_shap_values()
            self.save_shap_summary_plot()
            self.save_shap_force_plot()
            self.save_feature_importance()
            self.generate_lime_explanation()

            logging.info("Explainability pipeline completed")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = ExplainabilityPipeline()
    pipeline.run_pipeline()