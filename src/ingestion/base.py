"""Base ingestion adapter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd


class BaseIngestionAdapter(ABC):
    """Abstract ingestion adapter — concrete implementations write to Bronze."""

    source_name: str = "base"

    def __init__(self, target_dir: str | Path) -> None:
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def read(self, *args, **kwargs) -> pd.DataFrame:
        """Read records from the source and return a DataFrame."""

    def stamp_provenance(self, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        df = df.copy()
        df["_ingest_ts"] = datetime.utcnow().isoformat(timespec="seconds")
        df["_source_file"] = source_file
        df["_source_system"] = self.source_name
        return df

    def write_bronze(self, df: pd.DataFrame, dataset: str) -> Path:
        out = self.target_dir / f"{dataset}.parquet"
        df.to_parquet(out, index=False)
        return out
