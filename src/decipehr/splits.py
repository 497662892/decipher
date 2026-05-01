from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from decipehr.schema import require_columns, validate_binary_labels


@dataclass(frozen=True)
class SplitSummary:
    n_patients: int
    n_folds: int
    label_counts: dict[str, dict[int, int]]


def _patient_label_frame(df: pd.DataFrame, patient_col: str, label_cols: list[str]) -> pd.DataFrame:
    require_columns(df, [patient_col, *label_cols], "phenotype")
    validate_binary_labels(df, label_cols)
    return df[[patient_col, *label_cols]].groupby(patient_col, as_index=False).max()


def _make_strata(frame: pd.DataFrame, label_cols: list[str], n_folds: int) -> pd.Series:
    if not label_cols:
        return pd.Series(["all"] * len(frame), index=frame.index)
    strata = frame[label_cols].astype(int).astype(str).agg("_".join, axis=1)
    counts = strata.value_counts()
    rare = set(counts[counts < n_folds].index)
    if rare:
        burden = frame[label_cols].astype(int).sum(axis=1).clip(upper=2).astype(str)
        strata = strata.where(~strata.isin(rare), "burden_" + burden)
    if (strata.value_counts() < n_folds).any():
        strata = pd.Series(["all"] * len(frame), index=frame.index)
    return strata


def make_patient_folds(
    phenotype: pd.DataFrame,
    patient_col: str,
    label_cols: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    patients = _patient_label_frame(phenotype, patient_col, label_cols)
    if len(patients) < n_folds:
        raise ValueError(f"Need at least {n_folds} patients, observed {len(patients)}")

    strata = _make_strata(patients, label_cols, n_folds)
    folds = np.full(len(patients), -1, dtype=int)
    if strata.nunique() > 1:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(patients, strata)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(patients)
    for fold, (_, val_idx) in enumerate(split_iter):
        folds[val_idx] = fold
    out = patients[[patient_col]].copy()
    out["fold"] = folds
    return out.sort_values([patient_col]).reset_index(drop=True)


def summarize_folds(
    phenotype: pd.DataFrame,
    folds: pd.DataFrame,
    patient_col: str,
    label_cols: list[str],
) -> SplitSummary:
    require_columns(folds, [patient_col, "fold"], "folds")
    merged = phenotype[[patient_col, *label_cols]].merge(folds, on=patient_col, how="inner")
    label_counts: dict[str, dict[int, int]] = {}
    for label in label_cols:
        label_counts[label] = merged.groupby("fold")[label].sum().astype(int).sort_index().to_dict()
    return SplitSummary(folds[patient_col].nunique(), folds["fold"].nunique(), label_counts)

