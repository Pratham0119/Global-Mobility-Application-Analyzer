from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception import CustomException
from src.utils.explainability_utils import get_latest_file


app = FastAPI()


class VisaInput(BaseModel):
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    no_of_employees: int
    yr_of_estab: int
    region_of_employment: str
    prevailing_wage: float
    unit_of_wage: str
    full_time_position: str


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Global Mobility Analyzer API is running"
    }


@app.post("/predict")
def predict(data: VisaInput):
    try:
        pipeline = PredictionPipeline()
        result = pipeline.predict(data.dict())

        return {
            "status": "success",
            "message": "Prediction completed successfully",
            "data": result
        }

    except CustomException as ce:
        return {
            "status": "failed",
            "message": "Prediction failed",
            "error": str(ce)
        }

    except Exception as e:
        return {
            "status": "failed",
            "message": "Unexpected error occurred",
            "error": str(e)
        }


@app.get("/explainability/latest")
def latest_explainability():
    try:
        explainability_dir = "artifacts/explainability"

        data = {
            "shap_summary_plot": get_latest_file(explainability_dir, "shap_summary"),
            "shap_force_plot": get_latest_file(explainability_dir, "shap_force"),
            "lime_explanation": get_latest_file(explainability_dir, "lime_explanation"),
            "feature_importance": get_latest_file(explainability_dir, "feature_importance"),
        }

        return {
            "status": "success",
            "message": "Latest explainability files fetched successfully",
            "data": data
        }

    except Exception as e:
        return {
            "status": "failed",
            "message": "Could not fetch explainability files",
            "error": str(e)
        }