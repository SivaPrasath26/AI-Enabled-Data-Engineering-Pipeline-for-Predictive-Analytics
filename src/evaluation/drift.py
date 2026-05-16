"""Population Stability Index (PSI) drift detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(expected, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = e_counts / max(len(expected), 1) + EPS
    a_pct = a_counts / max(len(actual), 1) + EPS
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def detect_drift(
    train: pd.DataFrame, serve: pd.DataFrame, threshold: float = 0.2
) -> pd.DataFrame:
    rows = []
    for col in train.select_dtypes(include="number").columns:
        if col not in serve.columns:
            continue
        psi = population_stability_index(train[col].dropna().to_numpy(), serve[col].dropna().to_numpy())
        rows.append({"feature": col, "psi": psi, "drifted": psi >= threshold})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
