from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str
    raw_data_file_path: str
    