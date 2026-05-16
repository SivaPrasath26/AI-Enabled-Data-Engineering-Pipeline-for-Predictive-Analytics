"""Schema definition and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class ColumnSpec:
    name: str
    type: str
    nullable: bool = True
    allowed: list[Any] | None = None
    min: float | int | None = None
    max: float | int | None = None


@dataclass
class Schema:
    name: str
    primary_key: list[str]
    columns: list[ColumnSpec] = field(default_factory=list)

    @property
    def column_map(self) -> dict[str, ColumnSpec]:
        return {c.name: c for c in self.columns}


def load_schema(path: str | Path) -> Schema:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cols = [ColumnSpec(**c) for c in raw.get("columns", [])]
    return Schema(name=raw["name"], primary_key=raw["primary_key"], columns=cols)


class SchemaValidator:
    """Asserts a DataFrame conforms to a schema."""

    _TYPE_MAP = {
        "integer": "Int64",
        "long": "Int64",
        "double": "float64",
        "float": "float64",
        "string": "string",
        "boolean": "boolean",
    }

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        errors: list[str] = []
        for col in self.schema.columns:
            if col.name not in df.columns:
                if not col.nullable:
                    errors.append(f"missing required column '{col.name}'")
                continue
            df[col.name] = self._cast(df[col.name], col.type)
            if not col.nullable and df[col.name].isna().any():
                errors.append(f"column '{col.name}' has nulls but is non-nullable")
            if col.allowed is not None:
                bad = ~df[col.name].dropna().isin(col.allowed)
                if bad.any():
                    errors.append(
                        f"column '{col.name}' has {int(bad.sum())} values "
                        f"outside allowed set {col.allowed}"
                    )
            if col.min is not None and (df[col.name].dropna() < col.min).any():
                errors.append(f"column '{col.name}' violates min={col.min}")
            if col.max is not None and (df[col.name].dropna() > col.max).any():
                errors.append(f"column '{col.name}' violates max={col.max}")
        if errors:
            raise ValueError(
                f"Schema validation failed for {self.schema.name}: "
                + "; ".join(errors)
            )
        return df

    @classmethod
    def _cast(cls, series: pd.Series, dtype: str) -> pd.Series:
        target = cls._TYPE_MAP.get(dtype, dtype)
        try:
            return series.astype(target)
        except (TypeError, ValueError):
            return pd.to_numeric(series, errors="coerce").astype(target, errors="ignore")
