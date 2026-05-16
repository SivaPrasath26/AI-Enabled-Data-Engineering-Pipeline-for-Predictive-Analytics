"""Trainer interface shared by all model implementations."""
from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger("models.base")


@dataclass
class TrainingArtifact:
    model_name: str
    run_id: str
    artifact_dir: Path
    feature_columns: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    training_seconds: float = 0.0
    dataset_hash: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "run_id": self.run_id,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "training_seconds": self.training_seconds,
            "dataset_hash": self.dataset_hash,
        }


class BaseTrainer(ABC):
    model_name: str = "base"

    def __init__(self, hyperparameters: dict[str, Any], seed: int = 42) -> None:
        self.hyperparameters = hyperparameters
        self.seed = seed
        self.model: Any = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, *, X_val=None, y_val=None) -> "BaseTrainer":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, directory: str | Path) -> TrainingArtifact:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "model.pkl").open("wb") as fh:
            pickle.dump(self.model, fh)
        artifact = TrainingArtifact(
            model_name=self.model_name,
            run_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            artifact_dir=directory,
            feature_columns=getattr(self, "feature_columns_", []),
            hyperparameters=self.hyperparameters,
        )
        with (directory / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(artifact.manifest(), fh, indent=2)
        log.info(f"  saved {self.model_name} artifact -> {directory}")
        return artifact

    @classmethod
    def load(cls, directory: str | Path) -> "BaseTrainer":
        directory = Path(directory)
        with (directory / "manifest.json").open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        instance = cls(hyperparameters=manifest["hyperparameters"])
        with (directory / "model.pkl").open("rb") as fh:
            instance.model = pickle.load(fh)
        instance.feature_columns_ = manifest["feature_columns"]
        return instance
