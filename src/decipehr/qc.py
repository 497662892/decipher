from __future__ import annotations

import pandas as pd


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame(
        {
            "column": df.columns,
            "missing": [int(df[col].isna().sum()) for col in df.columns],
            "missing_rate": [float(df[col].isna().mean()) if n else 0.0 for col in df.columns],
        }
    )


def label_balance(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    rows = []
    for label in label_cols:
        y = df[label].dropna().astype(int)
        rows.append(
            {
                "label": label,
                "n": int(len(y)),
                "cases": int(y.sum()),
                "controls": int((1 - y).sum()),
                "prevalence": float(y.mean()) if len(y) else float("nan"),
            }
        )
    return pd.DataFrame(rows)

