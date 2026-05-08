from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from decipehr.schema import require_columns, validate_binary_labels, validate_unique


def prepare_official_validation_phenotype(
    labelled_phenotype: pd.DataFrame,
    validation_covariates: pd.DataFrame,
    patient_col: str,
    label_cols: list[str],
    covariate_cols: list[str],
) -> pd.DataFrame:
    """Attach newly released labels to the official validation patient table.

    The validation cohort is defined by the covariate-only `Testing_Cov_file.txt`.
    Labels are copied from the labelled holdout-phase phenotype file.
    """
    require_columns(labelled_phenotype, [patient_col, *label_cols], "labelled phenotype")
    require_columns(validation_covariates, [patient_col, *covariate_cols], "validation covariates")
    validate_unique(labelled_phenotype, patient_col, "labelled phenotype")
    validate_unique(validation_covariates, patient_col, "validation covariates")
    validate_binary_labels(labelled_phenotype, label_cols)

    label_frame = labelled_phenotype[[patient_col, *label_cols]].copy()
    covariate_frame = validation_covariates[[patient_col, *covariate_cols]].copy()
    merged = covariate_frame.merge(label_frame, on=patient_col, how="left", validate="one_to_one")
    missing = merged[label_cols].isna().any(axis=1)
    if missing.any():
        examples = merged.loc[missing, patient_col].head(5).tolist()
        raise ValueError(
            "Validation covariate rows are missing released labels; "
            f"n_missing={int(missing.sum())}, examples={examples}"
        )
    merged = merged[[patient_col, *label_cols, *covariate_cols]]
    validate_binary_labels(merged, label_cols)
    return merged


def iter_repeated_holdout_indices(
    frame: pd.DataFrame,
    label_col: str,
    eval_size: int,
    n_repeats: int,
    seed: int = 42,
    stratify: bool = True,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield repeated fitting/prediction indices for a labelled validation frame."""
    require_columns(frame, [label_col], "validation frame")
    if eval_size < 1:
        raise ValueError("eval_size must be at least 1")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if eval_size >= len(frame):
        raise ValueError(f"eval_size={eval_size} must be smaller than n={len(frame)}")
    validate_binary_labels(frame, [label_col])
    if frame[label_col].nunique() < 2:
        raise ValueError(f"Validation frame for {label_col} has fewer than two classes")

    indices = np.arange(len(frame))
    if stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=n_repeats,
            test_size=eval_size,
            random_state=seed,
        )
        for repeat, (fit_idx, eval_idx) in enumerate(splitter.split(frame, frame[label_col].astype(int))):
            yield repeat, fit_idx, eval_idx
    else:
        rng = np.random.default_rng(seed)
        for repeat in range(n_repeats):
            eval_idx = np.sort(rng.choice(indices, size=eval_size, replace=False))
            is_eval = np.zeros(len(frame), dtype=bool)
            is_eval[eval_idx] = True
            fit_idx = indices[~is_eval]
            yield repeat, fit_idx, eval_idx


def iter_bootstrap_indices(
    n: int,
    n_resamples: int,
    seed: int = 42,
    sample_size: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield bootstrap index vectors sampled with replacement."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if sample_size is None:
        sample_size = n
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    rng = np.random.default_rng(seed)
    for repeat in range(n_resamples):
        yield repeat, rng.integers(0, n, size=sample_size)


def summarize_repeated_metrics(
    metrics: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Aggregate repeat-level metrics with mean/std/median/min/max columns."""
    if metrics.empty:
        return pd.DataFrame()
    require_columns(metrics, [*group_cols, "repeat"], "repeat metrics")
    exclude = set(group_cols) | {"repeat"}
    metric_cols = [
        col
        for col in metrics.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(metrics[col])
    ]
    if not metric_cols:
        return metrics[group_cols].drop_duplicates().reset_index(drop=True)
    grouped = metrics.groupby(group_cols, dropna=False, sort=True)
    summary = grouped[metric_cols].agg(["mean", "std", "median", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.insert(len(group_cols), "n_repeats", grouped["repeat"].nunique().to_numpy())
    return summary
