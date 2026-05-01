from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str = "dataframe") -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def validate_unique(df: pd.DataFrame, column: str, context: str = "dataframe") -> None:
    duplicated = df[column].duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, column].head(5).tolist()
        raise ValueError(f"{context} has duplicate {column} values, examples={examples}")


def validate_binary_labels(df: pd.DataFrame, label_cols: Iterable[str]) -> None:
    for col in label_cols:
        values = set(df[col].dropna().unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Label column {col} must be binary 0/1; observed={sorted(values)}")


def parse_csv_list(value: str | None) -> list[str]:
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

