from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from decipehr.schema import require_columns, validate_unique

DEFAULT_1000G_GROUPS = ("AFR", "AMR", "EAS", "EUR", "SAS")


@dataclass(frozen=True)
class NormalizedScores:
    train: pd.DataFrame
    validation: pd.DataFrame
    parameters: pd.DataFrame
    summary: dict[str, Any]


def infer_score_columns(
    frame: pd.DataFrame,
    *,
    patient_col: str,
    explicit: list[str] | None = None,
) -> list[str]:
    if explicit:
        require_columns(frame, explicit, "PRS scores")
        return list(explicit)
    score_cols = []
    for column in frame.columns:
        if column == patient_col:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            score_cols.append(column)
    if not score_cols:
        raise ValueError("No numeric PRS score columns were inferred; pass --score-cols")
    return score_cols


def _sd(values: pd.Series) -> float:
    sd = float(pd.to_numeric(values, errors="coerce").std(ddof=0))
    return sd if np.isfinite(sd) and sd > 0 else float("nan")


def _global_stats(frame: pd.DataFrame, score_cols: list[str]) -> dict[str, tuple[float, float]]:
    stats = {}
    for score_col in score_cols:
        values = pd.to_numeric(frame[score_col], errors="coerce")
        mean = float(values.mean())
        sd = _sd(values)
        if not np.isfinite(mean) or not np.isfinite(sd):
            raise ValueError(f"Score column {score_col!r} does not have finite training mean/sd")
        stats[score_col] = (mean, sd)
    return stats


def _merge_ancestry(
    scores: pd.DataFrame,
    ancestry: pd.DataFrame,
    *,
    patient_col: str,
    columns: list[str],
    context: str,
) -> pd.DataFrame:
    require_columns(scores, [patient_col], f"{context} scores")
    require_columns(ancestry, [patient_col, *columns], f"{context} ancestry")
    validate_unique(scores, patient_col, f"{context} scores")
    validate_unique(ancestry, patient_col, f"{context} ancestry")
    merged = scores.merge(ancestry[[patient_col, *columns]], on=patient_col, how="left", validate="one_to_one")
    missing = merged[columns].isna().any(axis=1)
    if missing.any():
        examples = merged.loc[missing, patient_col].head(5).astype(str).tolist()
        raise ValueError(f"{context} scores are missing ancestry annotations for patients: {examples}")
    return merged


def normalize_by_hard_label(
    *,
    train_scores: pd.DataFrame,
    validation_scores: pd.DataFrame,
    hard_labels: pd.DataFrame,
    patient_col: str,
    score_cols: list[str],
    hard_col: str = "hard_label",
    min_group_size: int = 20,
) -> NormalizedScores:
    require_columns(train_scores, [patient_col, *score_cols], "training PRS scores")
    require_columns(validation_scores, [patient_col, *score_cols], "validation PRS scores")
    train = _merge_ancestry(
        train_scores,
        hard_labels,
        patient_col=patient_col,
        columns=[hard_col],
        context="training",
    )
    validation = _merge_ancestry(
        validation_scores,
        hard_labels,
        patient_col=patient_col,
        columns=[hard_col],
        context="validation",
    )
    global_stats = _global_stats(train, score_cols)
    parameter_rows = []
    lookup: dict[tuple[str, str], tuple[float, float, str]] = {}
    for score_col in score_cols:
        global_mean, global_sd = global_stats[score_col]
        parameter_rows.append(
            {
                "normalization": "hard_z",
                "score_col": score_col,
                "group": "__GLOBAL__",
                "n_effective": int(pd.to_numeric(train[score_col], errors="coerce").notna().sum()),
                "mean": global_mean,
                "sd": global_sd,
                "fallback": False,
            }
        )
        for group, frame in train.groupby(hard_col, dropna=False, sort=True):
            values = pd.to_numeric(frame[score_col], errors="coerce")
            n = int(values.notna().sum())
            mean = float(values.mean()) if n else float("nan")
            sd = _sd(values)
            fallback = n < min_group_size or not np.isfinite(mean) or not np.isfinite(sd)
            if fallback:
                mean, sd = global_mean, global_sd
            group_name = str(group)
            lookup[(score_col, group_name)] = (mean, sd, "hard_group" if not fallback else "global")
            parameter_rows.append(
                {
                    "normalization": "hard_z",
                    "score_col": score_col,
                    "group": group_name,
                    "n_effective": n,
                    "mean": mean,
                    "sd": sd,
                    "fallback": fallback,
                }
            )

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[[patient_col]].copy()
        for score_col in score_cols:
            global_mean, global_sd = global_stats[score_col]
            values = pd.to_numeric(frame[score_col], errors="coerce")
            means = []
            sds = []
            for group in frame[hard_col].astype(str):
                mean, sd, _ = lookup.get((score_col, group), (global_mean, global_sd, "global"))
                means.append(mean)
                sds.append(sd)
            out[score_col] = (values.to_numpy(dtype=float) - np.asarray(means)) / np.asarray(sds)
        return out

    return NormalizedScores(
        train=transform(train),
        validation=transform(validation),
        parameters=pd.DataFrame(parameter_rows),
        summary={
            "normalization": "hard_z",
            "hard_col": hard_col,
            "score_cols": score_cols,
            "min_group_size": int(min_group_size),
            "n_train": int(len(train_scores)),
            "n_validation": int(len(validation_scores)),
        },
    )


def normalize_by_soft_weights(
    *,
    train_scores: pd.DataFrame,
    validation_scores: pd.DataFrame,
    soft_weights: pd.DataFrame,
    patient_col: str,
    score_cols: list[str],
    groups: tuple[str, ...] = DEFAULT_1000G_GROUPS,
    min_effective_weight: float = 20.0,
) -> NormalizedScores:
    weight_cols = [f"w_{group}" for group in groups]
    require_columns(train_scores, [patient_col, *score_cols], "training PRS scores")
    require_columns(validation_scores, [patient_col, *score_cols], "validation PRS scores")
    train = _merge_ancestry(
        train_scores,
        soft_weights,
        patient_col=patient_col,
        columns=weight_cols,
        context="training",
    )
    validation = _merge_ancestry(
        validation_scores,
        soft_weights,
        patient_col=patient_col,
        columns=weight_cols,
        context="validation",
    )
    global_stats = _global_stats(train, score_cols)
    parameter_rows = []
    group_stats: dict[tuple[str, str], tuple[float, float]] = {}
    train_weights = train[weight_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(train_weights).all():
        raise ValueError("Training soft weights contain nonfinite values")
    for score_col in score_cols:
        values = pd.to_numeric(train[score_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError(f"Training score column {score_col!r} has no finite values")
        global_mean, global_sd = global_stats[score_col]
        parameter_rows.append(
            {
                "normalization": "soft_z",
                "score_col": score_col,
                "group": "__GLOBAL__",
                "n_effective": float(finite.sum()),
                "mean": global_mean,
                "sd": global_sd,
                "fallback": False,
            }
        )
        for index, group in enumerate(groups):
            weights = train_weights[finite, index]
            finite_values = values[finite]
            denom = float(weights.sum())
            if denom > 0:
                mean = float(np.sum(weights * finite_values) / denom)
                var = float(np.sum(weights * (finite_values - mean) ** 2) / denom)
                sd = float(np.sqrt(max(var, 0.0)))
            else:
                mean, sd = float("nan"), float("nan")
            fallback = denom < min_effective_weight or not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0
            if fallback:
                mean, sd = global_mean, global_sd
            group_stats[(score_col, group)] = (mean, sd)
            parameter_rows.append(
                {
                    "normalization": "soft_z",
                    "score_col": score_col,
                    "group": group,
                    "n_effective": denom,
                    "mean": mean,
                    "sd": sd,
                    "fallback": fallback,
                }
            )

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        weights = frame[weight_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(weights).all():
            raise ValueError("Soft weights contain nonfinite values")
        row_sums = weights.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Soft weights must have positive row sums")
        weights = weights / row_sums
        out = frame[[patient_col]].copy()
        for score_col in score_cols:
            values = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float)
            means = np.asarray([group_stats[(score_col, group)][0] for group in groups], dtype=float)
            sds = np.asarray([group_stats[(score_col, group)][1] for group in groups], dtype=float)
            expected_mean = weights @ means
            expected_second = weights @ (sds**2 + means**2)
            expected_sd = np.sqrt(np.maximum(expected_second - expected_mean**2, 1e-12))
            out[score_col] = (values - expected_mean) / expected_sd
        return out

    return NormalizedScores(
        train=transform(train),
        validation=transform(validation),
        parameters=pd.DataFrame(parameter_rows),
        summary={
            "normalization": "soft_z",
            "groups": list(groups),
            "score_cols": score_cols,
            "min_effective_weight": float(min_effective_weight),
            "n_train": int(len(train_scores)),
            "n_validation": int(len(validation_scores)),
        },
    )
