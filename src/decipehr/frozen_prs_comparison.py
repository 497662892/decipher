"""Matched downstream CV for previously selected internal and external PRS banks.

This conditions on the supplied frozen scores. It does not replay their historical
construction or selection and must not be described as end-to-end nested PRS CV.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from decipehr.external_pgs_benchmark import DEMOGRAPHICS, PCS, _ids_hash, _inner_splits
from decipehr.external_pgs_holdout import predict_holdout_wrapper
from decipehr.models import fit_logistic_predict_proba
from decipehr.tabpfn_ensemble import logit_from_probability

ARMS = ("internal", "external")
OUTCOMES = ("T2D", "cad", "bc")


def fold_data(data: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.sampleid.duplicated().any() or set(data.fold) != set(range(5)):
        raise ValueError("Expected unique patients and exactly five shared folds")
    if fold not in range(5):
        raise ValueError("Invalid outer fold")
    return tuple(
        frame.sort_values("sampleid", kind="stable").reset_index(drop=True)
        for frame in [data.loc[data.fold.ne(fold)], data.loc[data.fold.eq(fold)]]
    )


def evaluate_frozen_fold(
    data: pd.DataFrame, *, disease: str, arm: str, fold: int, model_ids: list[str],
    tabpfn_factory: Callable, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if arm not in ARMS or disease not in OUTCOMES:
        raise ValueError("Unknown comparison arm or disease")
    context, query = fold_data(data, fold)
    pred, inner, fits, preprocessing = predict_holdout_wrapper(
        context, query.drop(columns=list(OUTCOMES), errors="ignore"),
        disease=disease, model_ids=model_ids, tabpfn_factory=tabpfn_factory, seed=seed + fold,
    )
    # Promote stored native float32 values before CSV serialization, preserving ties
    # and exact wrapper inputs when they are read back as float64.
    pred["y_prob"] = pred.y_prob.astype(float)
    pred = pred.assign(y_true=query[disease].to_numpy(dtype=int), fold=fold, arm=arm)
    inner = inner.rename(columns={"fold": "inner_fold"}).assign(outer_fold=fold, arm=arm)
    fits["fit_scope"] = fits.fit_scope.replace({"full_trainval_context": "outer_context"})
    fits = fits.assign(fold=fold, arm=arm)
    preprocessing["fit_scope"] = preprocessing.fit_scope.replace(
        {"full_trainval_context": "outer_context"},
    )
    return pred, inner, fits, preprocessing.assign(fold=fold, arm=arm)


def verify_frozen_fold(
    data: pd.DataFrame, predictions: pd.DataFrame, inner_oof: pd.DataFrame,
    fits: pd.DataFrame, *, disease: str, arm: str, fold: int, seed: int = 42,
) -> dict:
    """Check patient sets, inner/outer fit scopes, labels, and independently refit wrapper."""
    context, query = fold_data(data, fold)
    expected_scopes = {"outer_context", *(f"wrapper_inner_{i}" for i in range(3))}
    if len(fits) != 4 or set(fits.fit_scope) != expected_scopes:
        raise ValueError("Incomplete or duplicated fit scopes")
    for frame in [predictions, inner_oof, fits]:
        if not frame.disease.eq(disease).all() or not frame.arm.eq(arm).all():
            raise ValueError("Prediction/fit arm or disease mismatch")
    if not predictions.fold.eq(fold).all() or not fits.fold.eq(fold).all():
        raise ValueError("Outer fold mismatch")
    if not inner_oof.outer_fold.eq(fold).all():
        raise ValueError("Wrapper outer fold mismatch")
    for frame, expected in [(predictions, query), (inner_oof, context)]:
        if frame.sampleid.duplicated().any() or set(frame.sampleid) != set(expected.sampleid):
            raise ValueError("Prediction coverage differs from the eligible patient set")
    pred = predictions.set_index("sampleid").loc[query.sampleid].reset_index()
    inner = inner_oof.set_index("sampleid").loc[context.sampleid].reset_index()
    for frame, expected in [(pred, query), (inner, context)]:
        if not np.array_equal(frame.y_true, expected[disease]):
            raise ValueError("Prediction labels differ from phenotype")
    if not np.array_equal(pred.ancestry_group, query.hard_label):
        raise ValueError("Ancestry labels differ from frozen inputs")
    for values in [pred[["y_prob", "wrapper_prob"]].to_numpy(), inner.y_prob.to_numpy()]:
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Invalid probabilities")

    def check_fit(name, train, val):
        row = fits.loc[fits.fit_scope.eq(name)].iloc[0]
        if (row.context_ids_sha256 != _ids_hash(train)
                or row.query_ids_sha256 != _ids_hash(val)
                or row.n_context != len(train) or row.n_query != len(val)):
            raise ValueError("Fit patient hash/count mismatch")

    check_fit("outer_context", context, query)
    for i, (train_idx, val_idx) in enumerate(_inner_splits(context, disease, seed + fold)):
        check_fit(f"wrapper_inner_{i}", context.iloc[train_idx], context.iloc[val_idx])
        if not inner.iloc[val_idx].inner_fold.eq(i).all():
            raise ValueError("Inner OOF fold assignments differ")
    expected_prob = fit_logistic_predict_proba(
        train=context.assign(external_logit=logit_from_probability(inner.y_prob.to_numpy(), eps=0.001)),
        val=query.drop(columns=list(OUTCOMES), errors="ignore").assign(
            external_logit=logit_from_probability(pred.y_prob.to_numpy(), eps=0.001),
        ),
        label_col=disease, feature_cols=[*DEMOGRAPHICS, *PCS, "external_logit"],
        patient_col="sampleid", class_weight=None,
    )
    error = float(np.max(np.abs(expected_prob - pred.wrapper_prob.to_numpy())))
    if error > 1e-10:
        raise ValueError(f"Independent wrapper replay failed: {error}")
    return {
        "disease": disease, "arm": arm, "fold": fold, "n_context": len(context),
        "n_query": len(query), "query_cases": int(query[disease].sum()),
        "wrapper_replay_max_abs_error": error,
        "outer_context_ids_sha256": _ids_hash(context), "outer_query_ids_sha256": _ids_hash(query),
    }


def paired_predictions(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    if set(predictions.arm) != set(ARMS):
        raise ValueError("Both frozen arms are required")
    keys = ["sampleid", "disease", "fold", "y_true", "ancestry_group"]
    if predictions.duplicated(["sampleid", "disease", "arm"]).any():
        raise ValueError("Duplicate OOF predictions")
    if predictions.groupby("sampleid").fold.nunique().gt(1).any():
        raise ValueError("Patients have different folds across arms or diseases")
    left, right = [predictions.loc[predictions.arm.eq(arm), keys + [column]] for arm in ARMS]
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True,
                        suffixes=("_internal", "_external"))
    if not merged._merge.eq("both").all():
        raise ValueError("Paired patients, labels, folds or ancestry differ")
    for disease, frame in merged.groupby("disease"):
        if set(frame.fold) != set(range(5)) or set(frame.y_true) != {0, 1}:
            raise ValueError(f"Incomplete five-fold binary disease cohort: {disease}")
    return merged.drop(columns="_merge").rename(
        columns={f"{column}_{arm}": arm for arm in ARMS},
    )
