from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline


# Step 1: Create sample input
data = CustomData(
    continent="Asia",
    education_of_employee="Bachelor's",
    has_job_experience="Y",
    requires_job_training="N",
    no_of_employees=500,
    yr_of_estab=2000,
    region_of_employment="West",
    prevailing_wage=60000,
    unit_of_wage="Year",
    full_time_position="Y",
)

# Step 2: Convert to dict
input_dict = data.get_data_as_dict()

# Step 3: Predict
pipeline = PredictionPipeline()
result = pipeline.predict(input_dict)

print(result)