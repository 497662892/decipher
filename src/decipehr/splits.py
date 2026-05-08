from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from decipehr.eligibility import filter_eligible
from decipehr.schema import require_columns, validate_binary_labels


@dataclass(frozen=True)
class SplitSummary:
    n_patients: int
    n_folds: int
    label_counts: dict[str, dict[int, int]]


@dataclass(frozen=True)
class DiseaseSplitSummary:
    disease: str
    n_patients: int
    n_folds: int
    label_counts: dict[int, int]
    fold_counts: dict[int, int]


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


def _age_bins(series: pd.Series, n_bins: int = 3) -> pd.Series:
    age = pd.to_numeric(series, errors="coerce")
    if age.isna().any():
        raise ValueError("age column must be numeric and complete for disease-specific folds")
    if age.nunique() < 2:
        return pd.Series(["age_all"] * len(age), index=series.index)
    bins = pd.qcut(age, q=min(n_bins, age.nunique()), labels=False, duplicates="drop")
    return "age_" + bins.astype("Int64").astype(str)


def _combined_strata(frame: pd.DataFrame, strata_cols: list[str], n_folds: int) -> pd.Series:
    for width in range(len(strata_cols), 0, -1):
        strata = frame[strata_cols[:width]].astype(str).agg("_".join, axis=1)
        if strata.nunique() > 1 and (strata.value_counts() >= n_folds).all():
            return strata
    return pd.Series(["all"] * len(frame), index=frame.index)


def _assign_folds(
    frame: pd.DataFrame,
    strata: pd.Series,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    folds = np.full(len(frame), -1, dtype=int)
    if strata.nunique() > 1:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(frame, strata)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(frame)
    for fold, (_, val_idx) in enumerate(split_iter):
        folds[val_idx] = fold
    return folds


def _proportion_balance_score(frame: pd.DataFrame, column: str, n_folds: int) -> float:
    overall = frame[column].astype(str).value_counts(normalize=True)
    score = 0.0
    for fold in range(n_folds):
        fold_frame = frame[frame["fold"] == fold]
        observed = fold_frame[column].astype(str).value_counts(normalize=True)
        score += float((observed.reindex(overall.index, fill_value=0.0) - overall).abs().sum())
    return score / n_folds


def _mean_balance_score(frame: pd.DataFrame, column: str, n_folds: int) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        return 0.0
    overall = float(values.mean())
    score = 0.0
    for fold in range(n_folds):
        fold_mean = float(pd.to_numeric(frame.loc[frame["fold"] == fold, column], errors="coerce").mean())
        score += abs(fold_mean - overall) / sd
    return score / n_folds


def _balance_score(
    frame: pd.DataFrame,
    categorical_cols: list[str],
    mean_cols: list[str],
    n_folds: int,
) -> float:
    score = 0.0
    for column in categorical_cols:
        score += _proportion_balance_score(frame, column, n_folds)
    for column in mean_cols:
        score += _mean_balance_score(frame, column, n_folds)
    return score


def _choose_balanced_folds(
    frame: pd.DataFrame,
    strata_cols: list[str],
    categorical_balance_cols: list[str],
    mean_balance_cols: list[str],
    n_folds: int,
    seed: int,
    n_seed_candidates: int,
) -> np.ndarray:
    best_folds: np.ndarray | None = None
    best_score = float("inf")
    n_candidates = max(1, n_seed_candidates)
    strata = _combined_strata(frame, strata_cols, n_folds)
    for offset in range(n_candidates):
        folds = _assign_folds(frame, strata, n_folds, seed + offset)
        candidate = frame.copy()
        candidate["fold"] = folds
        score = _balance_score(candidate, categorical_balance_cols, mean_balance_cols, n_folds)
        if score < best_score:
            best_score = score
            best_folds = folds
    if best_folds is None:
        raise ValueError("Failed to assign folds")
    return best_folds


def _patient_split_frame(
    df: pd.DataFrame,
    patient_col: str,
    disease: str,
    cluster_col: str,
    age_col: str,
    sex_col: str | None,
) -> pd.DataFrame:
    cols = [patient_col, disease, cluster_col, age_col]
    if sex_col:
        cols.append(sex_col)
    require_columns(df, cols, f"phenotype for {disease}")
    validate_binary_labels(df, [disease])
    aggregations: dict[str, Any] = {disease: "max", cluster_col: "first", age_col: "first"}
    if sex_col:
        aggregations[sex_col] = "first"
    patients = df[cols].groupby(patient_col, as_index=False).agg(aggregations)
    patients["_age_bin"] = _age_bins(patients[age_col])
    return patients


def make_disease_specific_folds(
    phenotype: pd.DataFrame,
    patient_col: str,
    label_cols: list[str],
    cluster_col: str,
    age_col: str,
    sex_col: str | None = None,
    sex_balance_labels: list[str] | None = None,
    eligibility_rules: dict[str, dict[str, Any]] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    n_seed_candidates: int = 100,
) -> pd.DataFrame:
    """Create disease-specific folds balanced by label, ancestry cluster, age, and sex."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    sex_balance_set = set(sex_balance_labels or [])
    eligibility_rules = eligibility_rules or {}
    require_columns(phenotype, [patient_col, *label_cols, cluster_col, age_col], "phenotype")
    if sex_balance_set and sex_col is None:
        raise ValueError("sex_col is required when sex_balance_labels is non-empty")
    if sex_col is not None:
        require_columns(phenotype, [sex_col], "phenotype")

    outputs = []
    for disease in label_cols:
        disease_data = filter_eligible(phenotype, disease, eligibility_rules)
        patients = _patient_split_frame(disease_data, patient_col, disease, cluster_col, age_col, sex_col)
        if len(patients) < n_folds:
            raise ValueError(f"Need at least {n_folds} eligible patients for {disease}")
        strata_cols = [disease, cluster_col, "_age_bin"]
        categorical_balance_cols = [disease, cluster_col, "_age_bin"]
        if sex_col is not None and disease in sex_balance_set:
            categorical_balance_cols.append(sex_col)
        patients["fold"] = _choose_balanced_folds(
            patients,
            strata_cols=strata_cols,
            categorical_balance_cols=categorical_balance_cols,
            mean_balance_cols=[age_col],
            n_folds=n_folds,
            seed=seed,
            n_seed_candidates=n_seed_candidates,
        )
        out = patients[[patient_col, "fold"]].copy()
        out["disease"] = disease
        outputs.append(out)

    return pd.concat(outputs, ignore_index=True)[[patient_col, "disease", "fold"]].sort_values(
        ["disease", patient_col]
    ).reset_index(drop=True)


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


def summarize_disease_folds(
    phenotype: pd.DataFrame,
    folds: pd.DataFrame,
    patient_col: str,
    label_cols: list[str],
) -> list[DiseaseSplitSummary]:
    require_columns(folds, [patient_col, "disease", "fold"], "folds")
    summaries = []
    for disease in label_cols:
        disease_folds = folds[folds["disease"] == disease]
        require_columns(phenotype, [patient_col, disease], "phenotype")
        merged = phenotype[[patient_col, disease]].merge(disease_folds, on=patient_col, how="inner")
        label_counts = merged.groupby("fold")[disease].sum().astype(int).sort_index().to_dict()
        fold_counts = merged.groupby("fold")[patient_col].count().astype(int).sort_index().to_dict()
        summaries.append(
            DiseaseSplitSummary(
                disease=disease,
                n_patients=int(merged[patient_col].nunique()),
                n_folds=int(merged["fold"].nunique()),
                label_counts=label_counts,
                fold_counts=fold_counts,
            )
        )
    return summaries


def disease_fold_balance_report(
    phenotype: pd.DataFrame,
    folds: pd.DataFrame,
    patient_col: str,
    label_cols: list[str],
    cluster_col: str | None = None,
    age_col: str | None = None,
    sex_col: str | None = None,
) -> pd.DataFrame:
    require_columns(folds, [patient_col, "disease", "fold"], "folds")
    optional_cols = [col for col in [cluster_col, age_col, sex_col] if col is not None]
    require_columns(phenotype, [patient_col, *label_cols, *optional_cols], "phenotype")
    rows: list[dict[str, float | int | str]] = []
    cluster_values = (
        sorted(phenotype[cluster_col].dropna().astype(str).unique().tolist()) if cluster_col else []
    )
    sex_values = sorted(phenotype[sex_col].dropna().astype(str).unique().tolist()) if sex_col else []
    for disease in label_cols:
        disease_folds = folds.loc[folds["disease"] == disease, [patient_col, "fold"]]
        merged = phenotype.merge(disease_folds, on=patient_col, how="inner")
        for fold, frame in merged.groupby("fold", sort=True):
            label = pd.to_numeric(frame[disease], errors="coerce")
            row: dict[str, float | int | str] = {
                "disease": disease,
                "fold": int(fold),
                "n": int(len(frame)),
                "cases": int(label.sum()),
                "prevalence": float(label.mean()),
            }
            if age_col:
                row[f"{age_col}_mean"] = float(pd.to_numeric(frame[age_col], errors="coerce").mean())
            if cluster_col:
                counts = frame[cluster_col].astype(str).value_counts()
                for value in cluster_values:
                    count = int(counts.get(value, 0))
                    row[f"{cluster_col}_{value}_n"] = count
                    row[f"{cluster_col}_{value}_prop"] = count / len(frame) if len(frame) else float("nan")
            if sex_col:
                counts = frame[sex_col].astype(str).value_counts()
                for value in sex_values:
                    count = int(counts.get(value, 0))
                    row[f"{sex_col}_{value}_n"] = count
                    row[f"{sex_col}_{value}_prop"] = count / len(frame) if len(frame) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)
