"""LightGBM trainer — alternative gradient boosting baseline."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .base import BaseTrainer

log = get_logger("models.lightgbm")


class LightGBMTrainer(BaseTrainer):
    model_name = "lightgbm"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LightGBMTrainer":
        self.feature_columns_ = list(X.columns)
        params = {
            "objective": self.hyperparameters.get("objective", "binary"),
            "metric": "binary_logloss",
            "learning_rate": self.hyperparameters.get("learning_rate", 0.05),
            "num_leaves": self.hyperparameters.get("num_leaves", 63),
            "feature_fraction": self.hyperparameters.get("feature_fraction", 0.85),
            "bagging_fraction": self.hyperparameters.get("bagging_fraction", 0.85),
            "bagging_freq": 5,
            "verbose": -1,
            "seed": self.seed,
        }
        train_set = lgb.Dataset(X, label=y, feature_name=self.feature_columns_)
        valid_sets = [train_set]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(X_val, label=y_val, reference=train_set))
            valid_names.append("val")
        log.info(f"  fitting LightGBM  n={len(X):,}  d={X.shape[1]}")
        t0 = time.time()
        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.hyperparameters.get("n_estimators", 800),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(
                    self.hyperparameters.get("early_stopping_rounds", 50),
                    verbose=False,
                ),
                lgb.log_evaluation(0),
            ],
        )
        self.training_seconds_ = time.time() - t0
        log.info(f"  done in {self.training_seconds_:.1f}s  best_iter={self.model.best_iteration}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_columns_])
