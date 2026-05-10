import sys
import pickle
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import PredictionPipelineConfig


class CustomData:
    def __init__(
        self,
        continent: str,
        education_of_employee: str,
        has_job_experience: str,
        requires_job_training: str,
        no_of_employees: int,
        yr_of_estab: int,
        region_of_employment: str,
        prevailing_wage: float,
        unit_of_wage: str,
        full_time_position: str,
    ):
        self.continent = continent
        self.education_of_employee = education_of_employee
        self.has_job_experience = has_job_experience
        self.requires_job_training = requires_job_training
        self.no_of_employees = no_of_employees
        self.yr_of_estab = yr_of_estab
        self.region_of_employment = region_of_employment
        self.prevailing_wage = prevailing_wage
        self.unit_of_wage = unit_of_wage
        self.full_time_position = full_time_position

    def get_data_as_dict(self):
        return {
            "continent": self.continent,
            "education_of_employee": self.education_of_employee,
            "has_job_experience": self.has_job_experience,
            "requires_job_training": self.requires_job_training,
            "no_of_employees": self.no_of_employees,
            "yr_of_estab": self.yr_of_estab,
            "region_of_employment": self.region_of_employment,
            "prevailing_wage": self.prevailing_wage,
            "unit_of_wage": self.unit_of_wage,
            "full_time_position": self.full_time_position,
        }


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig(
            model_path="artifacts/model_trainer/model.pkl"
        )

    def load_model(self):
        try:
            with open(self.config.model_path, "rb") as file:
                model_data = pickle.load(file)

            logging.info("Model loaded successfully")
            return model_data

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data: dict):
        try:
            logging.info("Prediction started")

            df = pd.DataFrame([input_data])

            model_data = self.load_model()
            preprocessor = model_data["preprocessor"]
            model = model_data["model"]

            transformed_data = preprocessor.transform(df)

            probability = model.predict_proba(transformed_data)[0][1]
            prediction = model.predict(transformed_data)[0]

            if prediction == 1:
                result = "Certified"
            else:
                result = "Denied"

            logging.info("Prediction completed successfully")

            return {
                "prediction": int(prediction),
                "result": result,
                "certified_probability": float(probability),
                "denied_probability": float(1 - probability),
            }

        except Exception as e:
            raise CustomException(e, sys)