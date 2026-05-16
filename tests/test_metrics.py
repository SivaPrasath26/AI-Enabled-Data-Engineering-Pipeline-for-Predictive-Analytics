import numpy as np
import pandas as pd

from src.evaluation import classification_metrics, fairness_metrics
from src.evaluation.drift import population_stability_index


def test_classification_metrics_basic():
    y = np.array([0, 1, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
    m = classification_metrics(y, s)
    assert 0.9 <= m["roc_auc"] <= 1.0
    assert 0 <= m["accuracy"] <= 1


def test_fairness_metrics_slices():
    df = pd.DataFrame({
        "y": [0, 1, 1, 0, 1, 0],
        "s": [0.2, 0.8, 0.7, 0.4, 0.6, 0.3],
        "g": ["M", "M", "M", "F", "F", "F"],
    })
    out = fairness_metrics(df, "y", "s", "g")
    assert set(out["slice_value"]) == {"M", "F"}


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    psi = population_stability_index(x, y)
    assert psi < 0.1
