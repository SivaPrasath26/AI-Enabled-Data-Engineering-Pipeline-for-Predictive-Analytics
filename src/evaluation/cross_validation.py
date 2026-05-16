"""Stratified k-fold evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..utils.logging import get_logger
from .metrics import classification_metrics

log = get_logger("evaluation.cv")


def stratified_kfold_eval(
    trainer_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Run k-fold CV and return per-fold metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        log.info(f"  fold {fold}/{n_splits}")
        model = trainer_factory()
        model.fit(X.iloc[tr], y.iloc[tr], X_val=X.iloc[va], y_val=y.iloc[va])
        score = model.predict_proba(X.iloc[va])
        m = classification_metrics(y.iloc[va].to_numpy(), score)
        m["fold"] = fold
        rows.append(m)
    df = pd.DataFrame(rows)
    summary = df.drop(columns=["fold"]).agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    log.info(f"\n{summary}")
    return df
