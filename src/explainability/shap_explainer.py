"""SHAP wrapper used by both training reports and the serving layer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LocalExplanation:
    base_value: float
    contributions: list[tuple[str, float]]  # sorted by |value| desc
    prediction: float

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        return self.contributions[:k]


class ShapExplainer:
    """Tree-based SHAP wrapper. Use only with tree ensembles."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        import shap
        self._shap = shap
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        return self.explainer.shap_values(X[self.feature_names])

    def explain_one(self, row: pd.DataFrame, prediction: float) -> LocalExplanation:
        sv = self.explainer.shap_values(row[self.feature_names])
        if isinstance(sv, list):  # binary classifier => [class0, class1]
            sv = sv[1]
        sv = np.asarray(sv).reshape(-1)
        order = np.argsort(np.abs(sv))[::-1]
        contributions = [(self.feature_names[i], float(sv[i])) for i in order]
        base = float(getattr(self.explainer, "expected_value", 0.0))
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).ravel()[-1])
        return LocalExplanation(base_value=base, contributions=contributions, prediction=prediction)

    def global_summary_plot(self, X: pd.DataFrame, out_path: str | Path) -> Path:
        import matplotlib.pyplot as plt
        sv = self.explain(X)
        if isinstance(sv, list):
            sv = sv[1]
        plt.figure(figsize=(8, 6))
        self._shap.summary_plot(sv, X[self.feature_names], show=False, plot_size=None)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
