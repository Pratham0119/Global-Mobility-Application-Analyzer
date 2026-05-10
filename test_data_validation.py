from src.pipeline.stage_02_data_validation import DataValidationPipeline

if __name__ == "__main__":
    pipeline = DataValidationPipeline()
    artifact = pipeline.initiate_data_validation()

    print("Validation Status:", artifact.validation_status)
    print("Message:", artifact.message)