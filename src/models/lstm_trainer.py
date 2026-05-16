"""LSTM trainer over weekly VLE click sequences."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .base import BaseTrainer

log = get_logger("models.lstm")


class LSTMTrainer(BaseTrainer):
    model_name = "lstm"

    def __init__(self, hyperparameters: dict, seed: int = 42) -> None:
        super().__init__(hyperparameters, seed)
        self._tf = None

    def _tf_import(self):
        if self._tf is None:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
            self._tf = tf
        return self._tf

    def _build_model(self, n_features: int):
        tf = self._tf_import()
        inputs = tf.keras.layers.Input(shape=(self.hyperparameters["sequence_length"], n_features))
        x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
        x = tf.keras.layers.LSTM(
            self.hyperparameters.get("hidden_units", 64),
            return_sequences=False,
            dropout=self.hyperparameters.get("dropout", 0.3),
            recurrent_dropout=0.1,
        )(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.hyperparameters.get("dropout", 0.3))(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        m = tf.keras.Model(inputs, out)
        m.compile(
            optimizer=tf.keras.optimizers.Adam(self.hyperparameters.get("learning_rate", 1e-3)),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
        )
        return m

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_val=None,
        y_val=None,
    ) -> "LSTMTrainer":
        tf = self._tf_import()
        self.feature_columns_ = list(X.columns)
        seq_len = self.hyperparameters["sequence_length"]
        n_features = X.shape[1] // seq_len
        log.info(f"  fitting LSTM  seq_len={seq_len}  features/step={n_features}")

        X_arr = X.to_numpy().reshape(-1, seq_len, n_features)
        y_arr = y.to_numpy().astype(float)

        self.model = self._build_model(n_features)
        t0 = time.time()
        val_data = None
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.to_numpy().reshape(-1, seq_len, n_features)
            val_data = (X_val_arr, y_val.to_numpy().astype(float))
        self.model.fit(
            X_arr,
            y_arr,
            batch_size=self.hyperparameters.get("batch_size", 128),
            epochs=self.hyperparameters.get("epochs", 25),
            validation_data=val_data,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, monitor="val_auc" if val_data else "auc",
                    mode="max",
                )
            ],
            verbose=0,
        )
        self.training_seconds_ = time.time() - t0
        log.info(f"  done in {self.training_seconds_:.1f}s")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        seq_len = self.hyperparameters["sequence_length"]
        n_features = X.shape[1] // seq_len
        X_arr = X[self.feature_columns_].to_numpy().reshape(-1, seq_len, n_features)
        return self.model.predict(X_arr, verbose=0).reshape(-1)
