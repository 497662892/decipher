from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from decipehr.schema import require_columns, validate_unique

EnsembleStrategy = Literal["prs_only", "covariate_adjusted"]


@dataclass(frozen=True)
class EnsembleResult:
    train_scores: pd.DataFrame
    validation_scores: pd.DataFrame
    weights: pd.DataFrame
    fit_summary: pd.DataFrame


def pivot_long_prs_scores(
    scores: pd.DataFrame,
    disease: str,
    *,
    patient_col: str,
    disease_col: str = "disease",
    model_col: str = "model",
    score_col: str = "prs",
    context: str = "PRS scores",
) -> pd.DataFrame:
    require_columns(scores, [patient_col, disease_col, model_col, score_col], context)
    disease_scores = scores.loc[scores[disease_col].astype(str) == disease].copy()
    if disease_scores.empty:
        raise ValueError(f"No {context} rows found for disease {disease!r}")
    duplicate_keys = [patient_col, disease_col, model_col]
    if disease_scores.duplicated(duplicate_keys).any():
        examples = disease_scores.loc[disease_scores.duplicated(duplicate_keys), duplicate_keys].head(5)
        raise ValueError(f"Duplicate {context} rows: {examples.to_dict('records')}")
    disease_scores[score_col] = pd.to_numeric(disease_scores[score_col], errors="coerce")
    invalid = disease_scores[score_col].isna()
    if invalid.any():
        examples = disease_scores.loc[invalid, [patient_col, model_col, score_col]].head(5)
        raise ValueError(f"Invalid numeric {context} values: {examples.to_dict('records')}")
    wide = disease_scores.pivot(index=patient_col, columns=model_col, values=score_col).reset_index()
    wide.columns.name = None
    return wide


def align_wide_prs_scores(
    scores: pd.DataFrame,
    *,
    patient_col: str,
    prs_cols: list[str],
    context: str,
) -> pd.DataFrame:
    require_columns(scores, [patient_col, *prs_cols], context)
    validate_unique(scores, patient_col, context)
    out = scores[[patient_col, *prs_cols]].copy()
    for column in prs_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    invalid = out[prs_cols].isna().all(axis=1)
    if invalid.any():
        examples = out.loc[invalid, [patient_col]].head(5).to_dict("records")
        raise ValueError(f"{context} has rows with no usable PRS values: {examples}")
    return out


def _nonconstant_prs_columns(frame: pd.DataFrame, prs_cols: list[str]) -> list[str]:
    usable = []
    for column in prs_cols:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        usable.append(column)
    return usable


def _standardize_from_fit(
    fit_prs: pd.DataFrame,
    train_prs: pd.DataFrame,
    validation_prs: pd.DataFrame,
    prs_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    usable_cols = _nonconstant_prs_columns(fit_prs, prs_cols)
    if not usable_cols:
        raise ValueError("No nonconstant PRS columns are available for ensemble fitting")

    train_std = pd.DataFrame(index=train_prs.index)
    validation_std = pd.DataFrame(index=validation_prs.index)
    scale_rows = []
    standardized_cols = []
    for column in usable_cols:
        fit_values = pd.to_numeric(fit_prs[column], errors="coerce")
        mean = float(fit_values.mean())
        sd = float(fit_values.std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            continue
        std_col = f"prs__{column}"
        train_values = pd.to_numeric(train_prs[column], errors="coerce").fillna(mean)
        validation_values = pd.to_numeric(validation_prs[column], errors="coerce").fillna(mean)
        train_std[std_col] = (train_values - mean) / sd
        validation_std[std_col] = (validation_values - mean) / sd
        standardized_cols.append(std_col)
        scale_rows.append(
            {
                "feature": column,
                "standardized_feature": std_col,
                "mean": mean,
                "sd": sd,
            }
        )
    if not standardized_cols:
        raise ValueError("No PRS columns have finite training mean and standard deviation")
    return train_std, validation_std, pd.DataFrame(scale_rows), standardized_cols


def _standardize_numeric_covariate(train: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(train[column], errors="coerce")
    mean = float(values.mean())
    if not np.isfinite(mean):
        mean = 0.0
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(train)), index=train.index, name=column)
    return ((values.fillna(mean) - mean) / sd).rename(column)


def _covariate_matrix(train: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if not covariates:
        return pd.DataFrame(index=train.index), []
    require_columns(train, covariates, "training phenotype")
    encoded_parts = []
    feature_names: list[str] = []
    for column in covariates:
        if pd.api.types.is_numeric_dtype(train[column]):
            encoded = _standardize_numeric_covariate(train, column).to_frame()
            encoded_parts.append(encoded)
            feature_names.append(column)
        else:
            values = train[column].astype("string")
            mode = values.dropna().mode()
            fill_value = mode.iloc[0] if len(mode) else "__missing__"
            values = values.fillna(fill_value)
            dummies = pd.get_dummies(values, prefix=column, dtype=float)
            encoded_parts.append(dummies)
            feature_names.extend(dummies.columns.tolist())
    if not encoded_parts:
        return pd.DataFrame(index=train.index), []
    matrix = pd.concat(encoded_parts, axis=1)
    return matrix.astype(float), feature_names


def _logistic_model(
    *,
    penalty: str,
    c_value: float,
    l1_ratio: float | None,
    class_weight: str | None,
    seed: int,
) -> LogisticRegression:
    kwargs: dict[str, object] = {
        "max_iter": 5000,
        "solver": "lbfgs",
        "class_weight": class_weight,
        "random_state": seed,
    }
    if penalty == "none":
        kwargs["penalty"] = None
    elif penalty == "l2":
        kwargs["penalty"] = "l2"
        kwargs["C"] = c_value
    elif penalty == "l1":
        kwargs["penalty"] = "l1"
        kwargs["C"] = c_value
        kwargs["solver"] = "saga"
    elif penalty == "elasticnet":
        if l1_ratio is None:
            raise ValueError("l1_ratio is required for elasticnet penalty")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1")
        kwargs["penalty"] = "elasticnet"
        kwargs["C"] = c_value
        kwargs["l1_ratio"] = l1_ratio
        kwargs["solver"] = "saga"
    else:
        raise ValueError(f"Unsupported penalty: {penalty}")
    return LogisticRegression(**kwargs)


def _fit_strategy(
    *,
    strategy: EnsembleStrategy,
    disease: str,
    fit_frame: pd.DataFrame,
    train_prs_std: pd.DataFrame,
    validation_prs_std: pd.DataFrame,
    prs_feature_cols: list[str],
    covariates: list[str],
    penalty: str,
    c_value: float,
    l1_ratio: float | None,
    class_weight: str | None,
    seed: int,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, object]]:
    fit_prs_std = train_prs_std.loc[fit_frame.index, prs_feature_cols]
    feature_matrix = fit_prs_std.copy()
    covariate_features: list[str] = []
    if strategy == "covariate_adjusted":
        cov_matrix, covariate_features = _covariate_matrix(fit_frame, covariates)
        feature_matrix = pd.concat([feature_matrix, cov_matrix], axis=1)

    y = fit_frame[disease].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(f"Training data for {disease} has fewer than two classes")

    model = _logistic_model(
        penalty=penalty,
        c_value=c_value,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        seed=seed,
    )
    model.fit(feature_matrix.to_numpy(dtype=float), y)

    coefs = model.coef_[0]
    feature_names = feature_matrix.columns.tolist()
    coef_map = dict(zip(feature_names, coefs, strict=True))
    prs_weights = np.array([coef_map[col] for col in prs_feature_cols], dtype=float)
    train_score = pd.Series(
        train_prs_std[prs_feature_cols].to_numpy(dtype=float) @ prs_weights,
        index=train_prs_std.index,
        name=f"prs_{disease}",
    )
    validation_score = pd.Series(
        validation_prs_std[prs_feature_cols].to_numpy(dtype=float) @ prs_weights,
        index=validation_prs_std.index,
        name=f"prs_{disease}",
    )
    weight_rows = []
    for feature in feature_names:
        feature_type = "prs" if feature in prs_feature_cols else "covariate"
        raw_feature = feature.removeprefix("prs__") if feature_type == "prs" else feature
        weight_rows.append(
            {
                "disease": disease,
                "strategy": strategy,
                "feature": raw_feature,
                "standardized_feature": feature,
                "feature_type": feature_type,
                "coefficient": float(coef_map[feature]),
            }
        )
    fit_summary = {
        "disease": disease,
        "strategy": strategy,
        "n_fit": int(len(fit_frame)),
        "fit_cases": int(y.sum()),
        "n_prs_features": int(len(prs_feature_cols)),
        "n_covariate_features": int(len(covariate_features)),
        "penalty": penalty,
        "C": float(c_value) if penalty in {"l1", "l2", "elasticnet"} else np.nan,
        "l1_ratio": float(l1_ratio) if penalty == "elasticnet" and l1_ratio is not None else np.nan,
        "class_weight": class_weight or "none",
        "intercept": float(model.intercept_[0]),
    }
    return train_score, validation_score, pd.DataFrame(weight_rows), fit_summary


def fit_ensemble_scores(
    *,
    train_phenotype: pd.DataFrame,
    train_prs: pd.DataFrame,
    validation_prs: pd.DataFrame,
    labels: list[str],
    covariates: list[str],
    strategies: list[EnsembleStrategy],
    patient_col: str,
    eligibility_rules: dict[str, dict] | None = None,
    penalty: str = "l2",
    c_value: float = 1.0,
    l1_ratio: float | None = None,
    class_weight: str | None = None,
    seed: int = 42,
) -> dict[EnsembleStrategy, EnsembleResult]:
    from decipehr.eligibility import filter_eligible

    require_columns(train_phenotype, [patient_col, *labels, *covariates], "training phenotype")
    validate_unique(train_phenotype, patient_col, "training phenotype")
    validate_unique(train_prs, patient_col, "training PRS scores")
    validate_unique(validation_prs, patient_col, "validation PRS scores")

    eligibility_rules = eligibility_rules or {}
    train_patient_order = train_prs[[patient_col]].copy()
    validation_patient_order = validation_prs[[patient_col]].copy()
    results: dict[EnsembleStrategy, dict[str, object]] = {
        strategy: {
            "train_scores": train_patient_order.copy(),
            "validation_scores": validation_patient_order.copy(),
            "weights": [],
            "fit_summary": [],
        }
        for strategy in strategies
    }

    prs_cols = [col for col in train_prs.columns if col != patient_col]
    if not prs_cols:
        raise ValueError("No PRS columns were provided")
    missing_validation_cols = [col for col in prs_cols if col not in validation_prs.columns]
    if missing_validation_cols:
        raise ValueError(f"Validation PRS scores are missing columns: {missing_validation_cols}")

    train_base = train_phenotype.merge(train_prs, on=patient_col, how="inner", validate="one_to_one")
    if train_base.empty:
        raise ValueError("No training patients have PRS scores")
    train_base.index = train_base.index.astype(int)
    train_prs_indexed = train_base[prs_cols].copy()
    train_prs_indexed.index = train_base.index

    validation_prs_indexed = validation_prs[prs_cols].copy()
    validation_prs_indexed.index = validation_prs.index

    for disease in labels:
        fit_frame = filter_eligible(train_base, disease, eligibility_rules).copy()
        if fit_frame.empty:
            raise ValueError(f"No eligible training patients remain for disease {disease}")
        fit_frame.index = fit_frame.index.astype(int)
        fit_prs = train_prs_indexed.loc[fit_frame.index, prs_cols]
        train_std, validation_std, scale_frame, prs_feature_cols = _standardize_from_fit(
            fit_prs=fit_prs,
            train_prs=train_prs_indexed,
            validation_prs=validation_prs_indexed,
            prs_cols=prs_cols,
        )
        train_std.index = train_base.index
        validation_std.index = validation_prs.index

        for strategy in strategies:
            train_score, validation_score, weights, fit_summary = _fit_strategy(
                strategy=strategy,
                disease=disease,
                fit_frame=fit_frame,
                train_prs_std=train_std,
                validation_prs_std=validation_std,
                prs_feature_cols=prs_feature_cols,
                covariates=covariates,
                penalty=penalty,
                c_value=c_value,
                l1_ratio=l1_ratio,
                class_weight=class_weight,
                seed=seed,
            )
            scale_with_context = scale_frame.assign(disease=disease, strategy=strategy)
            prs_weight_rows = weights.merge(
                scale_with_context,
                on=["disease", "strategy", "feature", "standardized_feature"],
                how="left",
            )
            result = results[strategy]
            train_out = result["train_scores"]
            validation_out = result["validation_scores"]
            assert isinstance(train_out, pd.DataFrame)
            assert isinstance(validation_out, pd.DataFrame)
            train_score_by_patient = pd.DataFrame(
                {
                    patient_col: train_base[patient_col].to_numpy(),
                    f"prs_{disease}": train_score.to_numpy(),
                }
            )
            validation_score_by_patient = pd.DataFrame(
                {
                    patient_col: validation_prs[patient_col].to_numpy(),
                    f"prs_{disease}": validation_score.to_numpy(),
                }
            )
            result["train_scores"] = train_out.merge(
                train_score_by_patient,
                on=patient_col,
                how="left",
                validate="one_to_one",
            )
            result["validation_scores"] = validation_out.merge(
                validation_score_by_patient,
                on=patient_col,
                how="left",
                validate="one_to_one",
            )
            result_weights = result["weights"]
            result_summary = result["fit_summary"]
            assert isinstance(result_weights, list)
            assert isinstance(result_summary, list)
            result_weights.append(prs_weight_rows)
            result_summary.append(fit_summary)

    finalized: dict[EnsembleStrategy, EnsembleResult] = {}
    for strategy, result in results.items():
        weights = result["weights"]
        summaries = result["fit_summary"]
        assert isinstance(weights, list)
        assert isinstance(summaries, list)
        finalized[strategy] = EnsembleResult(
            train_scores=result["train_scores"],  # type: ignore[arg-type]
            validation_scores=result["validation_scores"],  # type: ignore[arg-type]
            weights=pd.concat(weights, ignore_index=True) if weights else pd.DataFrame(),
            fit_summary=pd.DataFrame(summaries),
        )
    return finalized
