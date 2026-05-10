from dataclasses import dataclass


@dataclass
class DataValidationConfig:
    schema_file_path: str
    report_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    report_file_path: str