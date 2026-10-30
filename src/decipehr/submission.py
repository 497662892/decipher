from __future__ import annotations

import numpy as np
import pandas as pd

from decipehr.schema import require_columns, validate_unique
from decipehr.tabpfn_ensemble import logit_from_probability, scale_validation_logits


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


def standardized_logit_submission(
    oof: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    train_ids: pd.Series,
    holdout_ids: pd.Series,
    folds: pd.DataFrame,
    clip_eps: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Export one disease's direct probabilities on the historical OOF logit scale.

    Callers supply eligibility-filtered cohorts and the shared patient folds.
    The final export transform uses all train+development OOF logits; holdout
    values never determine the transform. This does not refit a predictive model.
    """
    if not 0 < clip_eps < 0.5:
        raise ValueError("clip_eps must be between zero and 0.5")
    require_columns(oof, ["sampleid", "fold", "y_prob"], "OOF predictions")
    require_columns(holdout, ["sampleid", "y_prob"], "holdout predictions")
    require_columns(folds, ["sampleid", "fold"], "shared folds")
    if set(train_ids) & set(holdout_ids):
        raise ValueError("Train+development and holdout patients overlap")
    aligned = []
    for name, frame, ids in [("trainval_oof", oof, train_ids),
                             ("holdout", holdout, holdout_ids),
                             ("folds", folds, train_ids)]:
        if ids.empty or ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"Invalid eligible cohort for {name}")
        if frame.sampleid.isna().any() or frame.sampleid.duplicated().any():
            raise ValueError(f"Missing or duplicate patient IDs in {name}")
        if set(frame.sampleid) != set(ids):
            raise ValueError(f"Incomplete or unexpected eligible patients in {name}")
        aligned.append(frame.set_index("sampleid").loc[ids].reset_index())
    train, query, split = aligned
    if set(split.fold) != set(range(5)) or not np.array_equal(train.fold, split.fold):
        raise ValueError("OOF assignments differ from the shared five folds")
    if set(query.columns) & {"T2D", "cad", "bc", "label", "y_true"}:
        raise ValueError("Holdout outcome columns are not permitted")
    logits = []
    for frame in [train, query]:
        probability = frame.y_prob.to_numpy(dtype=float)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Direct predictions must be finite probabilities in [0, 1]")
        logits.append(logit_from_probability(probability, eps=clip_eps))
    if np.std(logits[0], ddof=0) <= 0:
        raise ValueError("OOF logits have zero variance")
    train_score, query_score, mean, sd = scale_validation_logits(
        train_logits=logits[0], validation_logits=logits[1], mode="train_sd",
    )
    parts = []
    for frame, raw_logit, score, cohort in zip(
        [train, query], logits, [train_score, query_score], ["trainval_oof", "holdout"],
        strict=True,
    ):
        parts.append(frame[["sampleid"]].assign(
            cohort=cohort, fold=frame.fold.to_numpy() if cohort == "trainval_oof" else -1,
            direct_probability=frame.y_prob.to_numpy(dtype=float), direct_logit=raw_logit,
            score=score,
        ))
    return pd.concat(parts, ignore_index=True), {
        "clip_eps": clip_eps, "logit_trainval_oof_mean": mean, "logit_trainval_oof_sd": sd,
    }

