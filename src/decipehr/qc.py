from __future__ import annotations

from collections.abc import Iterable

import numpy as np
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


def _column_role(
    column: str,
    patient_col: str,
    label_cols: set[str],
    covariate_cols: set[str],
) -> str:
    if column == patient_col:
        return "patient_id"
    if column in label_cols:
        return "label"
    if column in covariate_cols:
        return "covariate"
    return "other"


def feature_summary(
    df: pd.DataFrame,
    patient_col: str,
    label_cols: Iterable[str],
    covariate_cols: Iterable[str],
) -> pd.DataFrame:
    label_set = set(label_cols)
    covariate_set = set(covariate_cols)
    n_rows = len(df)
    rows = []
    for column in df.columns:
        series = df[column]
        role = _column_role(column, patient_col, label_set, covariate_set)
        missing = int(series.isna().sum())
        non_missing = int(n_rows - missing)
        row = {
            "column": column,
            "role": role,
            "dtype": str(series.dtype),
            "n_rows": int(n_rows),
            "non_missing": non_missing,
            "missing": missing,
            "missing_rate": float(missing / n_rows) if n_rows else 0.0,
            "unique_non_missing": int(series.nunique(dropna=True)),
            "numeric_non_missing": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            row.update(
                {
                    "numeric_non_missing": int(numeric.notna().sum()),
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std()),
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def feature_count_report(
    df: pd.DataFrame,
    patient_col: str,
    label_cols: Iterable[str],
    covariate_cols: Iterable[str],
) -> pd.DataFrame:
    summary = feature_summary(df, patient_col, label_cols, covariate_cols)
    role_counts = summary["role"].value_counts().to_dict()
    numeric_features = int((summary["numeric_non_missing"] > 0).sum())
    rows = [
        {"metric": "n_rows", "value": int(len(df))},
        {"metric": "n_columns", "value": int(df.shape[1])},
        {"metric": "n_patient_id_columns", "value": int(role_counts.get("patient_id", 0))},
        {"metric": "n_label_columns", "value": int(role_counts.get("label", 0))},
        {"metric": "n_covariate_columns", "value": int(role_counts.get("covariate", 0))},
        {"metric": "n_other_columns", "value": int(role_counts.get("other", 0))},
        {"metric": "n_model_feature_columns", "value": int(role_counts.get("covariate", 0) + role_counts.get("other", 0))},
        {"metric": "n_numeric_columns", "value": numeric_features},
        {"metric": "n_non_numeric_columns", "value": int(len(summary) - numeric_features)},
        {"metric": "n_columns_with_missing", "value": int((summary["missing"] > 0).sum())},
        {"metric": "n_complete_columns", "value": int((summary["missing"] == 0).sum())},
    ]
    return pd.DataFrame(rows)


def parse_maybe_percent(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    is_percent = values.str.endswith("%", na=False)
    numeric = pd.to_numeric(values.str.rstrip("%"), errors="coerce")
    numeric = numeric.where(~is_percent, numeric / 100.0)
    return numeric.replace([np.inf, -np.inf], np.nan)


def filter_info_score_frame(
    df: pd.DataFrame,
    min_info: float,
    min_maf: float,
    variant_col: str = "col0",
    info_col: str = "info_IMPUTE",
    maf_col: str | None = None,
    freq_col: str | None = None,
    freq_is_maf: bool = True,
    max_af: float = 0.99,
) -> pd.Series:
    masks = info_score_filter_masks(
        df,
        min_info=min_info,
        min_maf=min_maf,
        max_af=max_af,
        variant_col=variant_col,
        info_col=info_col,
        maf_col=maf_col,
        freq_col=freq_col,
        freq_is_maf=freq_is_maf,
    )
    return df.loc[masks["pass_all"], variant_col].astype("string").dropna()


def info_score_filter_masks(
    df: pd.DataFrame,
    min_info: float,
    min_maf: float,
    max_af: float = 0.99,
    variant_col: str = "col0",
    info_col: str = "info_IMPUTE",
    maf_col: str | None = None,
    freq_col: str | None = None,
    freq_is_maf: bool = True,
) -> dict[str, pd.Series]:
    effective_freq_col = freq_col or maf_col or "MAF"
    missing = [col for col in [variant_col, info_col, effective_freq_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Info-score table is missing required columns: {missing}")
    info = pd.to_numeric(df[info_col], errors="coerce")
    freq = parse_maybe_percent(df[effective_freq_col])
    pass_info = info >= min_info
    if freq_is_maf:
        maf = freq
        pass_maf = maf >= min_maf
        pass_af = pd.Series(True, index=df.index)
    else:
        af = freq
        maf = pd.Series(np.minimum(af, 1.0 - af), index=df.index)
        pass_maf = maf >= min_maf
        lower_af = 1.0 - max_af
        pass_af = (af >= lower_af) & (af <= max_af)
    pass_all = pass_info & pass_maf & pass_af & df[variant_col].notna()
    return {
        "pass_info": pass_info.fillna(False),
        "pass_maf": pass_maf.fillna(False),
        "pass_af": pass_af.fillna(False),
        "pass_all": pass_all.fillna(False),
    }
