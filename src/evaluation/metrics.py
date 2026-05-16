"""Classification + fairness metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def fairness_metrics(
    df: pd.DataFrame,
    y_true: str,
    y_score: str,
    slice_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for value, sub in df.groupby(slice_col):
        if len(sub) == 0:
            continue
        m = classification_metrics(sub[y_true].to_numpy(), sub[y_score].to_numpy(), threshold)
        m["slice_value"] = value
        m["n"] = int(len(sub))
        rows.append(m)
    out = pd.DataFrame(rows)
    cols = ["slice_value", "n"] + [c for c in out.columns if c not in ("slice_value", "n")]
    return out[cols]
