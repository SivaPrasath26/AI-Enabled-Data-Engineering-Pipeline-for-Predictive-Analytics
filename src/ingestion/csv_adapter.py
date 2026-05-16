"""CSV ingestion adapter for the OULAD release."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils.logging import get_logger
from .base import BaseIngestionAdapter

log = get_logger("ingestion.csv")


class CSVIngestionAdapter(BaseIngestionAdapter):
    """Read OULAD CSV files and persist them as Bronze parquet."""

    source_name = "oulad_csv"

    def __init__(self, source_dir: str | Path, target_dir: str | Path) -> None:
        super().__init__(target_dir)
        self.source_dir = Path(source_dir)
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory missing: {self.source_dir}")

    def read(self, filename: str) -> pd.DataFrame:
        path = self.source_dir / filename
        log.info(f"Reading {path}")
        df = pd.read_csv(path)
        log.info(f"  rows={len(df):,}  cols={len(df.columns)}")
        return df

    def ingest(self, filenames: Iterable[str]) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        for f in filenames:
            df = self.read(f)
            df = self.stamp_provenance(df, source_file=f)
            dataset = Path(f).stem
            out = self.write_bronze(df, dataset)
            log.info(f"  bronze written -> {out}")
            outputs[dataset] = out
        return outputs
