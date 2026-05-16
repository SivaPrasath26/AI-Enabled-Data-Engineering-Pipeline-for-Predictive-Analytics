"""XGBoost trainer — primary tabular baseline."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from ..utils.logging import get_logger
from .base import BaseTrainer

log = get_logger("models.xgboost")


class XGBoostTrainer(BaseTrainer):
    model_name = "xgboost"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "XGBoostTrainer":
        self.feature_columns_ = list(X.columns)
        params = {
            "objective": self.hyperparameters.get("objective", "binary:logistic"),
            "eval_metric": "logloss",
            "learning_rate": self.hyperparameters.get("learning_rate", 0.05),
            "max_depth": self.hyperparameters.get("max_depth", 7),
            "subsample": self.hyperparameters.get("subsample", 0.85),
            "colsample_bytree": self.hyperparameters.get("colsample_bytree", 0.85),
            "reg_alpha": self.hyperparameters.get("reg_alpha", 0.1),
            "reg_lambda": self.hyperparameters.get("reg_lambda", 1.0),
            "seed": self.seed,
        }
        n_estimators = self.hyperparameters.get("n_estimators", 800)
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_columns_)
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_columns_)
            evals.append((dval, "val"))
        log.info(f"  fitting XGBoost  n={len(X):,}  d={X.shape[1]}")
        t0 = time.time()
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=self.hyperparameters.get("early_stopping_rounds", 50),
            verbose_eval=False,
        )
        self.training_seconds_ = time.time() - t0
        log.info(f"  done in {self.training_seconds_:.1f}s  best_iter={self.model.best_iteration}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(X[self.feature_columns_], feature_names=self.feature_columns_)
        return self.model.predict(d)
