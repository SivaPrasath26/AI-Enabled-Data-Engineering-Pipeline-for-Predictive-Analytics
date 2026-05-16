"""Configuration loader and typed accessors."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


class SparkConfig(BaseModel):
    app_name: str = "edu-pipeline"
    master: str = "local[*]"
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    shuffle_partitions: int = 64


class TrainingConfig(BaseModel):
    test_size: float = 0.2
    cv_folds: int = 5
    class_balance: str = "stratified"
    early_stopping_rounds: int = 50


class PathsConfig(BaseModel):
    raw: str
    bronze: str
    silver: str
    gold: str
    artifacts: str
    logs: str


class PipelineConfig(BaseModel):
    project: Dict[str, Any]
    paths: PathsConfig
    spark: SparkConfig
    ingestion: Dict[str, Any]
    preprocessing: Dict[str, Any]
    features: Dict[str, Any]
    training: TrainingConfig
    models: Dict[str, Any]
    evaluation: Dict[str, Any]
    serving: Dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.project.get("random_seed", 42))


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return PipelineConfig(**raw)
