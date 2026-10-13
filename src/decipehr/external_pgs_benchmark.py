"""Official train/validation and optional OOF evaluation of frozen external PGS."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from decipehr.ancestry_normalization import normalize_by_soft_weights
from decipehr.ancestry_performance import paired_bootstrap_comparisons, prediction_metrics
from decipehr.eligibility import filter_eligible
from decipehr.ensemble import pivot_long_prs_scores
from decipehr.metrics import binary_metrics, challenge_metrics
from decipehr.models import fit_logistic_predict_proba
from decipehr.schema import require_columns, validate_unique
from decipehr.tabpfn_ensemble import (
    TabPFNFeatureTransformer,
    logit_from_probability,
    stratified_oof_splits,
)

DEMOGRAPHICS = ["age_sd", "sex"]
PCS = [f"PC{i}_sd" for i in range(1, 11)]
SOFT = [f"w_{group}" for group in ("AFR", "AMR", "EAS", "EUR", "SAS")]
PRIMARY_METHOD = "tabpfn_prs_pc"
DEFAULT_METHODS = (PRIMARY_METHOD,)
HARD_ANCESTRY_METHOD = "tabpfn_prs_hard_ancestry"
PC_ONLY_LR_METHOD = "multi_prs_pc_lr"
METHODS = (
    "covariates_lr", "covariates_tabpfn", "best_single_lr", "multi_prs_lr",
    "multi_prs_interactions_lr", "tabpfn_prs_demographics", "tabpfn_prs_pc",
    "tabpfn_prs_pc_soft",
)
DISCODIVAS_METHODS = ("matched_multi_prs_lr", "discodivas_lr", "matched_tabpfn_pc")
# Preserve the completed eight-arm experiment's contrasts for sensitivity reporting.
LEGACY_COMPARISONS = (
    ("full_vs_pc", "tabpfn_prs_pc", "tabpfn_prs_pc_soft"),
    ("full_vs_linear_interactions", "multi_prs_interactions_lr", "tabpfn_prs_pc_soft"),
)
C_GRID = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class BenchmarkResult:
    predictions: pd.DataFrame
    fits: pd.DataFrame
    preprocessing: pd.DataFrame


def patient_table(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    frame = frame.copy()
    if "sampleid" not in frame and "IID" in frame:
        frame = frame.rename(columns={"IID": "sampleid"})
    require_columns(frame, ["sampleid"], context)
    if frame["sampleid"].isna().any():
        raise ValueError(f"{context}: missing patient identifiers")
    frame["sampleid"] = frame["sampleid"].astype(str)
    if frame["sampleid"].str.strip().eq("").any():
        raise ValueError(f"{context}: blank patient identifiers")
    return frame


def validate_shared_folds(splits: pd.DataFrame) -> None:
    require_columns(splits, ["sampleid", "disease", "fold"], "shared CV splits")
    if splits[["sampleid", "disease", "fold"]].isna().any().any():
        raise ValueError("Missing shared fold assignments")
    if splits.duplicated(["sampleid", "disease"]).any():
        raise ValueError("Duplicate patient-disease fold assignments")
    fold = pd.to_numeric(splits["fold"], errors="coerce")
    if not fold.isin([0, 1, 2, 3, 4, 5]).all() or set(fold) not in (
        {0, 1, 2, 3, 4}, {1, 2, 3, 4, 5},
    ):
        raise ValueError("Expected exactly five integer-valued shared folds")
    if splits.assign(fold=fold).groupby("sampleid")["fold"].nunique().gt(1).any():
        raise ValueError("A patient has different folds across diseases")


def _prepare_disease_inputs(
    *, phenotype: pd.DataFrame, scores: pd.DataFrame,
    soft_weights: pd.DataFrame | None, hard_labels: pd.DataFrame, disease: str,
    model_ids: list[str], eligibility_rules: dict,
) -> pd.DataFrame:
    phenotype = patient_table(phenotype, "phenotype")
    scores = patient_table(scores, "external PRS")
    hard_labels = patient_table(hard_labels, "1000G hard labels")
    for frame, context in (
        (phenotype, "phenotype"), (hard_labels, "1000G hard labels"),
    ):
        validate_unique(frame, "sampleid", context)
    require_columns(phenotype, [disease, *DEMOGRAPHICS, *PCS], "phenotype")
    soft_columns = []
    if soft_weights is not None:
        soft_weights = patient_table(soft_weights, "1000G soft similarities")
        validate_unique(soft_weights, "sampleid", "1000G soft similarities")
        require_columns(soft_weights, SOFT, "1000G soft similarities")
        soft_columns = SOFT
    require_columns(hard_labels, ["hard_label"], "1000G hard labels")
    if disease == "bc" and "bc" not in eligibility_rules:
        raise ValueError("Breast cancer requires --eligibility-config with the bc rule")
    eligible = filter_eligible(phenotype, disease, eligibility_rules).copy()
    if eligible.empty or not eligible[disease].isin([0, 1]).all():
        raise ValueError("Eligible disease labels must be complete and binary")
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("A unique, nonempty external score bank is required")
    prs = pivot_long_prs_scores(scores, disease, patient_col="sampleid")
    require_columns(prs, model_ids, "locked external score bank")
    data = eligible.merge(prs[["sampleid", *model_ids]], on="sampleid", how="left", validate="one_to_one")
    if soft_weights is not None:
        data = data.merge(soft_weights[["sampleid", *SOFT]], on="sampleid", how="left", validate="one_to_one")
    data = data.merge(hard_labels[["sampleid", "hard_label"]], on="sampleid", how="left", validate="one_to_one")
    if data["hard_label"].isna().any():
        raise ValueError("Every eligible patient must have a hard ancestry label")
    numeric = [*model_ids, *PCS, *soft_columns, "age_sd"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data[[*model_ids, *soft_columns]].to_numpy()).all():
        raise ValueError("Every eligible patient needs finite external scores and ancestry similarities")
    if np.isinf(data[[*PCS, "age_sd"]].to_numpy(dtype=float)).any():
        raise ValueError("Infinite covariates are not supported")
    if soft_columns:
        weights = data[soft_columns].to_numpy()
        if np.any((weights < 0) | (weights > 1)) or not np.allclose(weights.sum(axis=1), 1, atol=1e-6):
            raise ValueError("1000G similarities must be nonnegative and sum to one")
    return data.sort_values("sampleid", kind="stable").reset_index(drop=True)


def prepare_disease_data(
    *, phenotype: pd.DataFrame, scores: pd.DataFrame, splits: pd.DataFrame,
    soft_weights: pd.DataFrame | None, hard_labels: pd.DataFrame, disease: str,
    model_ids: list[str], eligibility_rules: dict,
) -> pd.DataFrame:
    """Prepare the optional shared five-fold development analysis."""
    splits = patient_table(splits, "shared splits")
    validate_shared_folds(splits)
    data = _prepare_disease_inputs(
        phenotype=phenotype, scores=scores, soft_weights=soft_weights, hard_labels=hard_labels,
        disease=disease, model_ids=model_ids, eligibility_rules=eligibility_rules,
    )
    assignments = splits.loc[splits["disease"] == disease, ["sampleid", "fold"]]
    data = data.merge(assignments, on="sampleid", how="left", validate="one_to_one")
    if data["fold"].isna().any():
        raise ValueError("Every eligible patient must have a shared fold")
    if data["fold"].nunique() != 5:
        raise ValueError("All five shared folds must be represented in each eligible disease cohort")
    data["fold"] = data["fold"].astype(int)
    return data.sort_values("sampleid", kind="stable").reset_index(drop=True)


def prepare_official_validation_data(
    *, train_phenotype: pd.DataFrame, validation_phenotype: pd.DataFrame,
    scores: pd.DataFrame, soft_weights: pd.DataFrame | None, hard_labels: pd.DataFrame,
    disease: str, model_ids: list[str], eligibility_rules: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve official membership; reject patient overlap before eligibility filtering."""
    train = patient_table(train_phenotype, "official train")
    validation = patient_table(validation_phenotype, "official validation")
    validate_unique(train, "sampleid", "official train")
    validate_unique(validation, "sampleid", "official validation")
    if set(train["sampleid"]) & set(validation["sampleid"]):
        raise ValueError("Official train and validation patients overlap")
    kwargs = dict(
        scores=scores, soft_weights=soft_weights, hard_labels=hard_labels,
        disease=disease, model_ids=model_ids, eligibility_rules=eligibility_rules,
    )
    return (
        _prepare_disease_inputs(phenotype=train, **kwargs),
        _prepare_disease_inputs(phenotype=validation, **kwargs),
    )


def method_features(method: str, model_ids: list[str]) -> tuple[list[str], list[str], bool]:
    if method not in (*METHODS, HARD_ANCESTRY_METHOD, PC_ONLY_LR_METHOD, *DISCODIVAS_METHODS):
        raise ValueError(f"Unknown external PGS benchmark method: {method}")
    prs = [] if method.startswith("covariates_") else model_ids
    if method == "discodivas_lr":
        prs = ["discodivas_prs"]
    if method == "tabpfn_prs_demographics":
        covariates = DEMOGRAPHICS
    elif method == HARD_ANCESTRY_METHOD:
        covariates = [*DEMOGRAPHICS, "hard_label"]
    elif method in ("tabpfn_prs_pc", PC_ONLY_LR_METHOD, *DISCODIVAS_METHODS):
        covariates = [*DEMOGRAPHICS, *PCS]
    else:
        covariates = [*DEMOGRAPHICS, *PCS, *SOFT]
    return prs, covariates, method == "multi_prs_interactions_lr"


def feature_matrices(
    context: pd.DataFrame, query: pd.DataFrame, *, prs: list[str],
    covariates: list[str], interactions: bool, normalization: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if set(context["sampleid"]) & set(query["sampleid"]):
        raise ValueError("Context and query patients overlap")
    if normalization not in {"none", "soft_z"}:
        raise ValueError("Normalization must be none or soft_z")
    context, query = context.copy(), query.copy()
    normalization_parameters = pd.DataFrame()
    if prs and normalization == "soft_z":
        normalized = normalize_by_soft_weights(
            train_scores=context[["sampleid", *prs]],
            validation_scores=query[["sampleid", *prs]],
            soft_weights=pd.concat([context[["sampleid", *SOFT]], query[["sampleid", *SOFT]]]),
            patient_col="sampleid", score_cols=prs,
        )
        context[prs] = normalized.train[prs].to_numpy()
        query[prs] = normalized.validation[prs].to_numpy()
        normalization_parameters = normalized.parameters.assign(parameter_stage="soft_z")
    transformer = TabPFNFeatureTransformer.fit(
        context, prs_cols=prs, covariates=covariates, include_covariates=True,
    )
    fit_matrix, query_matrix = transformer.transform(context), transformer.transform(query)
    if interactions:
        fit_interactions, query_interactions = {}, {}
        for score in prs:
            score_column = f"prs__{score}"
            if score_column not in fit_matrix:
                continue
            for ancestry in [*PCS, *SOFT]:
                ancestry_column = f"cov__{ancestry}"
                if ancestry_column not in fit_matrix:
                    continue
                name = f"{score_column}*{ancestry_column}"
                fit_interactions[name] = fit_matrix[score_column] * fit_matrix[ancestry_column]
                query_interactions[name] = query_matrix[score_column] * query_matrix[ancestry_column]
        fit_matrix = pd.concat([fit_matrix, pd.DataFrame(fit_interactions, index=fit_matrix.index)], axis=1)
        query_matrix = pd.concat([query_matrix, pd.DataFrame(query_interactions, index=query_matrix.index)], axis=1)
    parameters = pd.concat([
        transformer.summary().assign(parameter_stage="standardization"), normalization_parameters,
    ], ignore_index=True)
    return fit_matrix.to_numpy(dtype=float), query_matrix.to_numpy(dtype=float), parameters


def _predict(
    context: pd.DataFrame, query: pd.DataFrame, *, disease: str, method: str,
    model_ids: list[str], c_value: float, seed: int,
    tabpfn_factory: Callable[[int], Any] | None, normalization: str,
    score_transform: Callable | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    score_parameters = pd.DataFrame()
    if score_transform is not None:
        context, query, score_parameters = score_transform(context, query)
    prs, covariates, interactions = method_features(method, model_ids)
    x_context, x_query, parameters = feature_matrices(
        context, query, prs=prs, covariates=covariates,
        interactions=interactions, normalization=normalization,
    )
    if method.endswith("_lr"):
        model = LogisticRegression(C=c_value, max_iter=5000, random_state=seed)
    else:
        if tabpfn_factory is None:
            raise ValueError("A local TabPFN v3 factory is required for the requested methods")
        model = tabpfn_factory(seed)
    model.fit(x_context, context[disease].astype(int).to_numpy())
    positive = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("Classifier is missing positive class 1")
    probabilities = np.asarray(model.predict_proba(x_query))[:, positive[0]]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Classifier returned invalid probabilities")
    return probabilities, pd.concat([score_parameters, parameters], ignore_index=True)


def _inner_splits(context: pd.DataFrame, disease: str, seed: int) -> list:
    counts = context[disease].value_counts()
    if len(counts) != 2 or counts.min() < 3:
        raise ValueError("Each context needs at least three patients per class for inner three-fold CV")
    return stratified_oof_splits(context[disease], requested_splits=3, seed=seed)


def predict_method(
    context: pd.DataFrame, query: pd.DataFrame, *, disease: str, method: str,
    model_ids: list[str], seed: int, tabpfn_factory: Callable[[int], Any] | None = None,
    normalization: str = "none", c_grid: tuple[float, ...] = C_GRID,
    score_transform: Callable | None = None,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    """Select model/C entirely within context; query labels are never accessed."""
    if method in DISCODIVAS_METHODS and score_transform is None:
        raise ValueError("DiscoDivas comparisons require train-fitted score harmonization")
    chosen_ids, chosen_c, best_auc = list(model_ids), 1.0, -np.inf
    if method.endswith("_lr"):
        if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
            raise ValueError("Ridge C values must be positive and finite")
        inner = _inner_splits(context, disease, seed)
        banks = [[accession] for accession in model_ids] if method == "best_single_lr" else [model_ids]
        for bank in banks:
            for c_value in c_grid:
                probabilities = np.full(len(context), np.nan)
                for train_idx, val_idx in inner:
                    probabilities[val_idx], _ = _predict(
                        context.iloc[train_idx], context.iloc[val_idx], disease=disease,
                        method=method, model_ids=bank, c_value=c_value, seed=seed,
                        tabpfn_factory=tabpfn_factory, normalization=normalization,
                        score_transform=score_transform,
                    )
                auc = roc_auc_score(context[disease], probabilities)
                if auc > best_auc:
                    chosen_ids, chosen_c, best_auc = list(bank), c_value, float(auc)
    probabilities, parameters = _predict(
        context, query, disease=disease, method=method, model_ids=chosen_ids,
        c_value=chosen_c, seed=seed, tabpfn_factory=tabpfn_factory, normalization=normalization,
        score_transform=score_transform,
    )
    return probabilities, {
        "selected_models": json.dumps(method_features(method, chosen_ids)[0]),
        "C": chosen_c if method.endswith("_lr") else np.nan,
        "inner_auc": best_auc if np.isfinite(best_auc) else np.nan,
    }, parameters


def _ids_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(sorted(frame["sampleid"].astype(str))).encode()).hexdigest()


def _evaluate_context_query(
    context: pd.DataFrame, query: pd.DataFrame, *, disease: str, model_ids: list[str],
    methods: tuple[str, ...], fold: int, evaluation_protocol: str, seed: int,
    tabpfn_factory: Callable[[int], Any] | None, normalization: str, c_grid: tuple[float, ...],
    score_transform: Callable | None = None,
) -> BenchmarkResult:
    supported = (*METHODS, HARD_ANCESTRY_METHOD, PC_ONLY_LR_METHOD, *DISCODIVAS_METHODS)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(supported):
        raise ValueError("Specify unique supported benchmark methods")
    for frame, name in ((context, "context"), (query, "query")):
        validate_unique(frame, "sampleid", name)
    if set(context["sampleid"]) & set(query["sampleid"]):
        raise ValueError("Context and query patients overlap")
    context, query = context.reset_index(drop=True), query.reset_index(drop=True)
    # Final validation labels never reach model selection, transforms, or calibration.
    query_features = query.drop(columns=["T2D", "cad", "bc"], errors="ignore")
    fit_scope = "official_train_context" if evaluation_protocol == "official_train_val" else "outer_context"
    predictions, fits, preprocessing = [], [], []
    inner = _inner_splits(context, disease, seed + int(fold))
    for method in methods:
        probability, selection, parameters = predict_method(
            context, query_features, disease=disease, method=method, model_ids=model_ids,
            seed=seed + int(fold), tabpfn_factory=tabpfn_factory,
            normalization=normalization, c_grid=c_grid,
            score_transform=score_transform,
        )
        fits.append({
            "disease": disease, "fold": int(fold), "method": method, "fit_scope": fit_scope,
            "context_ids_sha256": _ids_hash(context), "query_ids_sha256": _ids_hash(query),
            "n_context": len(context), "n_query": len(query), **selection,
        })
        preprocessing.append(parameters.assign(
            disease=disease, fold=int(fold), method=method, fit_scope=fit_scope,
        ))
        calibration_oof = np.full(len(context), np.nan)
        for inner_fold, (train_idx, val_idx) in enumerate(inner):
            # Repeat model/C selection inside the inner training subset. Even the
            # wrapper's training predictions never use their own labels for selection.
            train, val = context.iloc[train_idx], context.iloc[val_idx]
            inner_prob, inner_selection, inner_parameters = predict_method(
                train, val.drop(columns=[disease]), disease=disease, method=method, model_ids=model_ids,
                seed=seed + int(fold) * 10 + inner_fold,
                tabpfn_factory=tabpfn_factory, normalization=normalization, c_grid=c_grid,
                score_transform=score_transform,
            )
            calibration_oof[val_idx] = inner_prob
            fits.append({
                "disease": disease, "fold": int(fold), "method": method,
                "fit_scope": f"wrapper_inner_{inner_fold}",
                "context_ids_sha256": _ids_hash(train), "query_ids_sha256": _ids_hash(val),
                "n_context": len(train), "n_query": len(val), **inner_selection,
            })
            preprocessing.append(inner_parameters.assign(
                disease=disease, fold=int(fold), method=method,
                fit_scope=f"wrapper_inner_{inner_fold}",
            ))
        if not np.isfinite(calibration_oof).all():
            raise ValueError("Incomplete inner OOF predictions for the official wrapper")
        wrapped_train = context.assign(
            external_logit=logit_from_probability(calibration_oof, eps=0.001),
        )
        wrapped_query = query_features.assign(
            external_logit=logit_from_probability(probability, eps=0.001),
        )
        wrapped = fit_logistic_predict_proba(
            train=wrapped_train, val=wrapped_query, label_col=disease,
            feature_cols=[*DEMOGRAPHICS, *PCS, "external_logit"],
            patient_col="sampleid", class_weight=None,
        )
        predictions.append(pd.DataFrame({
            "sampleid": query["sampleid"], "disease": disease, "fold": int(fold),
            "method": method, "y_true": query[disease].astype(int),
            "y_prob": probability, "wrapper_prob": wrapped, "ancestry_group": query["hard_label"],
            "normalization": normalization, "evaluation_protocol": evaluation_protocol,
        }))
    return BenchmarkResult(
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(fits).assign(evaluation_protocol=evaluation_protocol),
        pd.concat(preprocessing, ignore_index=True).assign(evaluation_protocol=evaluation_protocol),
    )


def evaluate_official_validation(
    context: pd.DataFrame, validation: pd.DataFrame, *, disease: str, model_ids: list[str],
    methods: tuple[str, ...] = DEFAULT_METHODS, seed: int = 42,
    tabpfn_factory: Callable[[int], Any] | None = None,
    normalization: str = "none", c_grid: tuple[float, ...] = C_GRID,
    score_transform: Callable | None = None,
) -> BenchmarkResult:
    """Fit on official train, predict official val; inner CV uses train patients only."""
    return _evaluate_context_query(
        context, validation, disease=disease, model_ids=model_ids, methods=methods,
        fold=0, evaluation_protocol="official_train_val", seed=seed,
        tabpfn_factory=tabpfn_factory, normalization=normalization, c_grid=c_grid,
        score_transform=score_transform,
    )


def evaluate_disease(
    data: pd.DataFrame, *, disease: str, model_ids: list[str], methods: tuple[str, ...] = DEFAULT_METHODS,
    seed: int = 42, tabpfn_factory: Callable[[int], Any] | None = None,
    normalization: str = "none", c_grid: tuple[float, ...] = C_GRID,
) -> BenchmarkResult:
    """Optional development CV, separate from the primary official validation."""
    if data["sampleid"].duplicated().any() or data["fold"].nunique() != 5:
        raise ValueError("Evaluation requires one row per patient and five shared folds")
    results = []
    for fold in sorted(data["fold"].unique()):
        results.append(_evaluate_context_query(
            data.loc[data["fold"] != fold], data.loc[data["fold"] == fold],
            disease=disease, model_ids=model_ids, methods=methods, fold=int(fold),
            evaluation_protocol="shared_5fold_cv", seed=seed,
            tabpfn_factory=tabpfn_factory, normalization=normalization, c_grid=c_grid,
        ))
    return BenchmarkResult(
        pd.concat([r.predictions for r in results], ignore_index=True),
        pd.concat([r.fits for r in results], ignore_index=True),
        pd.concat([r.preprocessing for r in results], ignore_index=True),
    )


def summarize_benchmark(
    predictions: pd.DataFrame, *, n_bootstraps: int = 2000, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocols = set(predictions.get("evaluation_protocol", pd.Series(["shared_5fold_cv"])))
    if len(protocols) != 1 or not protocols.issubset({"official_train_val", "shared_5fold_cv"}):
        raise ValueError("Report official validation and development CV separately")
    protocol = next(iter(protocols))
    prediction_kind = "validation" if protocol == "official_train_val" else "OOF"
    require_columns(
        predictions, ["sampleid", "disease", "fold", "method", "y_true", "y_prob",
                      "wrapper_prob", "ancestry_group", "normalization"], f"{prediction_kind} predictions",
    )
    if predictions.duplicated(["sampleid", "disease", "method"]).any():
        raise ValueError(f"Duplicate {prediction_kind} predictions")
    if predictions.groupby("sampleid")["fold"].nunique().gt(1).any():
        raise ValueError(f"Inconsistent {prediction_kind} patient folds")
    if predictions.groupby(["sampleid", "disease"])["y_true"].nunique().gt(1).any():
        raise ValueError("Methods disagree on patient labels")
    if predictions["normalization"].nunique() != 1:
        raise ValueError("Report ancestry-normalization sensitivity analyses separately")
    metrics, comparisons = [], []
    for disease, disease_frame in predictions.groupby("disease", sort=True):
        patient_sets = disease_frame.groupby("method")["sampleid"].agg(set)
        if any(ids != patient_sets.iloc[0] for ids in patient_sets):
            raise ValueError(f"Methods have different {prediction_kind} patient sets")
        for method, method_frame in disease_frame.groupby("method", sort=True):
            groups = [("ALL", method_frame), *method_frame.groupby("ancestry_group", sort=True)]
            for group, frame in groups:
                y = frame["y_true"].to_numpy(dtype=int)
                for scale, column in (("raw", "y_prob"), ("official_wrapper", "wrapper_prob")):
                    p = frame[column].to_numpy(dtype=float)
                    if len(np.unique(y)) == 2:
                        result = prediction_metrics(y, p)
                        result["auc"] = float(roc_auc_score(y, p))
                    else:
                        result = {**vars(binary_metrics(y, p)), **challenge_metrics(y, p),
                                  "cases": int(y.sum()), "controls": int(len(y) - y.sum())}
                    metrics.append({
                        "disease": disease, "method": method, "prediction_scale": scale,
                        "evaluation_protocol": protocol,
                        "ancestry_group": group,
                        "exploratory": min(int(y.sum()), int(len(y) - y.sum())) < 20,
                        **result,
                    })
        available = set(disease_frame["method"])
        pairs = tuple(p for p in LEGACY_COMPARISONS if set(p[1:]).issubset(available))
        if not pairs:
            continue
        wide = disease_frame.pivot(
            index=["sampleid", "disease", "y_true", "ancestry_group"],
            columns="method", values="y_prob",
        ).reset_index()
        if wide.isna().any().any():
            raise ValueError("Paired comparisons require complete predictions and consistent ancestry labels")
        comparison = paired_bootstrap_comparisons(
            wide, n_bootstraps=n_bootstraps, seed=seed, comparisons=pairs,
            metric_names=["auc"], family_size=6,
        )
        if not comparison.empty:
            comparison["controls"] = comparison["n"] - comparison["cases"]
            comparison["exploratory"] = (
                comparison[["cases", "controls"]].min(axis=1).lt(20)
                | comparison["ancestry_group"].ne("ALL")
                | (predictions["normalization"].iloc[0] != "none")
            )
            comparison["inference_scope"] = (
                "conditional_on_frozen_validation_predictions" if protocol == "official_train_val"
                else "conditional_on_frozen_oof_predictions"
            )
            comparison["evaluation_protocol"] = protocol
            comparisons.append(comparison)
    return pd.DataFrame(metrics), pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
