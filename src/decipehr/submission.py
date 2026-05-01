from __future__ import annotations

import pandas as pd

from decipehr.schema import require_columns, validate_unique


def wide_submission(
    predictions: pd.DataFrame,
    patient_col: str,
    disease_col: str = "disease",
    probability_col: str = "y_prob",
) -> pd.DataFrame:
    require_columns(predictions, [patient_col, disease_col, probability_col], "predictions")
    out = predictions.pivot_table(
        index=patient_col,
        columns=disease_col,
        values=probability_col,
        aggfunc="first",
    ).reset_index()
    out.columns.name = None
    validate_unique(out, patient_col, "submission")
    return out


def validate_probability_columns(df: pd.DataFrame, patient_col: str) -> None:
    require_columns(df, [patient_col], "submission")
    for col in df.columns:
        if col == patient_col:
            continue
        if df[col].isna().any():
            raise ValueError(f"Submission column {col} contains missing predictions")
        if ((df[col] < 0) | (df[col] > 1)).any():
            raise ValueError(f"Submission column {col} must be in [0, 1]")

