from src.pipeline.stage_03_data_transformation import DataTransformationPipeline


if __name__ == "__main__":
    train_file_path = "artifacts/data_ingestion/train.csv"
    test_file_path = "artifacts/data_ingestion/test.csv"
    artifact_dir = "artifacts"

    pipeline = DataTransformationPipeline()

    artifact = pipeline.start_data_transformation(
        train_file_path=train_file_path,
        test_file_path=test_file_path,
        artifact_dir=artifact_dir
    )

    print("Data Transformation Completed Successfully")
    print(artifact)