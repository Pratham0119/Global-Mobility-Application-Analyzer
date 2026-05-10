from src.pipeline.stage_04_model_trainer import ModelTrainerPipeline


train_file_path = r"artifacts\data_transformation\train_20260428202047.csv"
test_file_path = r"artifacts\data_transformation\test_20260428202047.csv"
preprocessor_file_path = r"artifacts\data_transformation\preprocessor_20260428202047.pkl"


if __name__ == "__main__":
    pipeline = ModelTrainerPipeline()

    artifact = pipeline.main(
        train_file_path=train_file_path,
        test_file_path=test_file_path,
        preprocessor_file_path=preprocessor_file_path
    )

    print(artifact)