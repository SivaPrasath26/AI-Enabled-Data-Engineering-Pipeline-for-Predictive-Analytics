"""Categorical encoding utilities used by the model trainers."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


CATEGORICAL_COLS = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "code_module",
    "code_presentation",
]


def fit_one_hot(df: pd.DataFrame, columns: list[str] | None = None) -> OneHotEncoder:
    columns = columns or [c for c in CATEGORICAL_COLS if c in df.columns]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(df[columns].astype(str).fillna("UNK"))
    return enc


def apply_one_hot(
    df: pd.DataFrame, enc: OneHotEncoder, columns: list[str]
) -> pd.DataFrame:
    arr = enc.transform(df[columns].astype(str).fillna("UNK"))
    cols = enc.get_feature_names_out(columns).tolist()
    encoded = pd.DataFrame(arr, columns=cols, index=df.index)
    return pd.concat([df.drop(columns=columns), encoded], axis=1)


def fit_ordinal(df: pd.DataFrame, columns: list[str]) -> OrdinalEncoder:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(df[columns].astype(str).fillna("UNK"))
    return enc
