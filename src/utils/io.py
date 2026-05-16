"""IO helpers — parquet read/write, directory utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False, **kwargs)
