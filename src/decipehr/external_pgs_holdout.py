"""Unlabelled prediction with the current external PGS and OOF-trained wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from decipehr.eligibility import filter_eligible
from decipehr.ensemble import pivot_long_prs_scores
from decipehr.external_pgs_benchmark import (
    DEMOGRAPHICS,
    PCS,
    _ids_hash,
    _inner_splits,
    _prepare_disease_inputs,
    patient_table,
    predict_method,
)
from decipehr.models import fit_logistic_predict_proba
from decipehr.schema import require_columns, validate_unique
from decipehr.tabpfn_ensemble import logit_from_probability


def prepare_holdout_data(
    *, phenotype: pd.DataFrame, holdout_covariates: pd.DataFrame,
    context_scores: pd.DataFrame, holdout_scores: pd.DataFrame,
    hard_labels: pd.DataFrame, disease: str, model_ids: list[str], eligibility_rules: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply female eligibility and require a complete, disjoint frozen score bank."""
    holdout = patient_table(holdout_covariates, "holdout")
    validate_unique(holdout, "sampleid", "holdout")
    if set(holdout.columns) & {"T2D", "cad", "bc"}:
        raise ValueError("Holdout outcomes must remain outside the prediction pipeline")
    require_columns(holdout, [*DEMOGRAPHICS, *PCS], "holdout covariates")
    if set(patient_table(phenotype, "context").sampleid) & set(holdout.sampleid):
        raise ValueError("Context and holdout patients overlap")
    context = _prepare_disease_inputs(
        phenotype=phenotype, scores=context_scores, soft_weights=None,
        hard_labels=hard_labels, disease=disease, model_ids=model_ids,
        eligibility_rules=eligibility_rules,
    )
    query = filter_eligible(holdout, disease, eligibility_rules).copy()
    prs = pivot_long_prs_scores(patient_table(holdout_scores, "holdout scores"), disease,
                                patient_col="sampleid")
    require_columns(prs, model_ids, "holdout frozen scores")
    labels = patient_table(hard_labels, "hard labels")
    validate_unique(labels, "sampleid", "hard labels")
    query = query.merge(prs[["sampleid", *model_ids]], on="sampleid", how="left", validate="one_to_one")
    query = query.merge(labels[["sampleid", "hard_label"]], on="sampleid", how="left", validate="one_to_one")
    if query.empty or query["hard_label"].isna().any():
        raise ValueError("Empty eligible holdout or missing ancestry labels")
    for col in [*model_ids, *PCS, "age_sd"]:
        query[col] = pd.to_numeric(query[col], errors="raise")
    if not np.isfinite(query[model_ids].to_numpy()).all():
        raise ValueError("Missing or nonfinite frozen holdout scores")
    return context.reset_index(drop=True), query.reset_index(drop=True)


def predict_holdout_wrapper(
    context: pd.DataFrame, query: pd.DataFrame, *, disease: str,
    model_ids: list[str], tabpfn_factory: Callable[[int], Any], seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use exactly the benchmark's three-fold OOF logit + metadata wrapper."""
    if set(query.columns) & {"T2D", "cad", "bc"}:
        raise ValueError("Holdout outcomes must remain outside the prediction pipeline")
    for frame, name in [(context, "context"), (query, "holdout")]:
        validate_unique(frame, "sampleid", name)
    if set(context.sampleid) & set(query.sampleid):
        raise ValueError("Context and holdout patients overlap")
    context, query = context.reset_index(drop=True), query.reset_index(drop=True)
    method = "tabpfn_prs_pc"
    options = dict(disease=disease, method=method, model_ids=model_ids,
                   tabpfn_factory=tabpfn_factory, normalization="none")
    probability, selection, parameters = predict_method(context, query, seed=seed, **options)
    fits = [{"fit_scope": "full_trainval_context", "n_context": len(context), "n_query": len(query),
             "context_ids_sha256": _ids_hash(context), "query_ids_sha256": _ids_hash(query), **selection}]
    preprocessing = [parameters.assign(fit_scope="full_trainval_context")]
    oof, folds = np.full(len(context), np.nan), np.full(len(context), -1)
    for fold, (train_idx, val_idx) in enumerate(_inner_splits(context, disease, seed)):
        train, val = context.iloc[train_idx], context.iloc[val_idx]
        val_features = val.drop(columns=["T2D", "cad", "bc"], errors="ignore")
        oof[val_idx], selection, parameters = predict_method(train, val_features, seed=seed + fold, **options)
        folds[val_idx] = fold
        fits.append({"fit_scope": f"wrapper_inner_{fold}", "n_context": len(train), "n_query": len(val),
                     "context_ids_sha256": _ids_hash(train), "query_ids_sha256": _ids_hash(val), **selection})
        preprocessing.append(parameters.assign(fit_scope=f"wrapper_inner_{fold}"))
    if not np.isfinite(oof).all() or (folds < 0).any():
        raise ValueError("Incomplete wrapper OOF predictions")
    wrapped = fit_logistic_predict_proba(
        train=context.assign(external_logit=logit_from_probability(oof, eps=0.001)),
        val=query.assign(external_logit=logit_from_probability(probability, eps=0.001)),
        label_col=disease, feature_cols=[*DEMOGRAPHICS, *PCS, "external_logit"],
        patient_col="sampleid", class_weight=None,
    )
    if not np.isfinite(wrapped).all() or ((wrapped < 0) | (wrapped > 1)).any():
        raise ValueError("Invalid wrapper probabilities")
    predictions = query[["sampleid", "hard_label"]].rename(columns={"hard_label": "ancestry_group"}).assign(
        disease=disease, method=method, y_prob=probability, wrapper_prob=wrapped,
    )
    training_oof = context[["sampleid", disease]].rename(columns={disease: "y_true"}).assign(
        disease=disease, fold=folds, y_prob=oof,
    )
    return (predictions, training_oof, pd.DataFrame(fits).assign(disease=disease),
            pd.concat(preprocessing, ignore_index=True).assign(disease=disease))
