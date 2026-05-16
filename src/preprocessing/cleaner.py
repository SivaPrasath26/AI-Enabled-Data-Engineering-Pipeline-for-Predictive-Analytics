"""Cleaning rules — null handling, whitespace, outlier filtering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger("preprocessing.cleaner")


class Cleaner:
    """Apply standard cleaning rules to a tabular dataset."""

    def __init__(
        self,
        critical_cols: list[str] | None = None,
        fill_strategy: dict[str, str] | None = None,
    ) -> None:
        self.critical_cols = critical_cols or []
        self.fill_strategy = fill_strategy or {}

    def trim_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        obj_cols = df.select_dtypes(include="object").columns
        for c in obj_cols:
            df[c] = df[c].astype(str).str.strip()
        return df

    def drop_critical_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.critical_cols:
            return df
        before = len(df)
        df = df.dropna(subset=[c for c in self.critical_cols if c in df.columns])
        log.info(f"  dropped {before - len(df):,} rows with critical nulls")
        return df

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, strategy in self.fill_strategy.items():
            if col not in df.columns:
                continue
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            elif strategy == "zero":
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(strategy)
        return df

    def remove_outliers_iqr(
        self, df: pd.DataFrame, column: str, multiplier: float = 3.0
    ) -> pd.DataFrame:
        if column not in df.columns:
            return df
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        before = len(df)
        df = df[(df[column] >= lo) & (df[column] <= hi)]
        log.info(f"  IQR filter on '{column}' dropped {before - len(df):,} rows")
        return df

    def standardise_categories(
        self, df: pd.DataFrame, mapping: dict[str, dict[str, str]]
    ) -> pd.DataFrame:
        for col, mp in mapping.items():
            if col in df.columns:
                df[col] = df[col].replace(mp)
        return df

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.trim_whitespace(df)
        df = self.drop_critical_nulls(df)
        df = self.fill_missing(df)
        return df


def encode_final_result(df: pd.DataFrame, col: str = "final_result") -> pd.DataFrame:
    """Map OULAD 4-class outcome to binary at-risk label.

    at_risk = 1 if Fail or Withdrawn, else 0 (Pass / Distinction).
    """
    if col not in df.columns:
        return df
    at_risk = df[col].isin({"Fail", "Withdrawn"}).astype(int)
    df = df.copy()
    df["final_result_binary"] = at_risk
    return df
