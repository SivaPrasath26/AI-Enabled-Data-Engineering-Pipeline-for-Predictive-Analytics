"""Deduplication helpers."""
from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger

log = get_logger("preprocessing.dedup")


def deduplicate(
    df: pd.DataFrame, keys: list[str], keep: str = "last"
) -> pd.DataFrame:
    keys = [k for k in keys if k in df.columns]
    if not keys:
        return df
    before = len(df)
    df = df.sort_values(keys).drop_duplicates(subset=keys, keep=keep)
    dropped = before - len(df)
    if dropped > 0:
        log.info(f"  removed {dropped:,} duplicates on keys={keys}")
    return df
