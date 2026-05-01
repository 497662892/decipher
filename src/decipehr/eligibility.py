from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from decipehr.schema import require_columns


def _normalize(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    return str(value).strip().lower()


def load_eligibility_rules(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load disease-specific eligibility rules from YAML.

    Expected shape:

    disease_eligibility:
      breast_cancer:
        column: sex
        allowed_values: ["F", "Female", "2"]
    """
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "disease_eligibility" in data:
        data = data["disease_eligibility"]
    if not isinstance(data, dict):
        raise ValueError("Eligibility config must be a mapping")
    return data


def filter_eligible(
    df: pd.DataFrame,
    disease: str,
    rules: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Return rows eligible for a disease-specific model/evaluation."""
    rule = rules.get(disease)
    if rule is None:
        return df
    column = rule.get("column")
    allowed_values = rule.get("allowed_values", [])
    if not column or not allowed_values:
        raise ValueError(f"Eligibility rule for {disease} must define column and allowed_values")
    require_columns(df, [column], f"eligibility rule for {disease}")
    allowed = {_normalize(value) for value in allowed_values}
    mask = df[column].map(_normalize).isin(allowed)
    return df.loc[mask].copy()


def eligible_counts(
    df: pd.DataFrame,
    diseases: list[str],
    rules: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for disease in diseases:
        rows.append(
            {
                "disease": disease,
                "n_total": int(len(df)),
                "n_eligible": int(len(filter_eligible(df, disease, rules))),
                "has_rule": disease in rules,
            }
        )
    return pd.DataFrame(rows)

