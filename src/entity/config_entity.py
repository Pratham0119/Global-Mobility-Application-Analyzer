from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    dataset_path: str
    train_path: str
    test_path: str
    raw_data_path: str
    