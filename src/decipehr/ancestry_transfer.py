"""Matched PC replacement and train-only single-PRS logistic baselines."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from decipehr.prs_distillation.preprocessing import AGE, PCS, assert_boundary, id_hash
from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer


def replace_pcs(frame, projected):
    """Replace only PC values, joining by ID without reordering or dropping patients."""
    for table in (frame, projected):
        if table.sampleid.isna().any() or not table.sampleid.is_unique:
            raise ValueError("PC replacement requires unique, nonmissing sample IDs")
    columns = [f"refPC{i}" for i in range(1, 11)]
    indexed = projected.set_index("sampleid")
    if not set(frame.sampleid) <= set(indexed.index):
        raise ValueError("Reference PCs do not cover every requested patient")
    values = indexed.loc[frame.sampleid, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Reference PCs must be finite")
    result = frame.copy()
    result[PCS] = values
    pd.testing.assert_frame_equal(result.drop(columns=PCS), frame.drop(columns=PCS))
    return result


def validate_folds(frame, folds):
    folds = np.asarray(folds)
    if folds.shape != (len(frame),) or set(folds) != set(range(5)):
        raise ValueError("Expected exactly five shared folds with complete coverage")
    if frame.sampleid.isna().any() or not frame.sampleid.is_unique:
        raise ValueError("Fold input contains missing or duplicate patient IDs")
    if frame.kinship_group.isna().any():
        raise ValueError("Missing kinship group")
    if pd.DataFrame({"group": frame.kinship_group, "fold": folds}).groupby("group").fold.nunique().max() != 1:
        raise ValueError("A kinship group crosses folds")
    for fold in range(5):
        if frame.loc[folds == fold, "y"].nunique() != 2:
            raise ValueError("Each fold must contain both outcomes")
    return folds


def fit_logistic(context, query, prs, *, allow_family_overlap=False):
    """Unpenalized MLE, with all preprocessing fitted on context only."""
    if any(c in query for c in ("y", "T2D", "cad", "bc")):
        raise ValueError("Query must contain features only")
    assert_boundary(context, query, require_group_disjoint=not allow_family_overlap)
    if not context.y.isin([0, 1]).all() or context.y.nunique() != 2:
        raise ValueError("Training requires both binary outcomes")
    prep = TabPFNFeatureTransformer.fit(
        context, prs_cols=prs, covariates=[AGE, "sex", *PCS], include_covariates=True)
    references = {f"{c.output_prefix}__{c.categories[0]}" for c in prep.categorical_features}
    columns = [c for c in prep.output_columns if c not in references]
    x, xq = [prep.transform(f)[columns].to_numpy() for f in (context, query)]
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=20000, tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings("ignore", message="Setting penalty=None will ignore the C and l1_ratio parameters", category=UserWarning)
        model.fit(x, context.y.to_numpy())
    np.testing.assert_array_equal(model.classes_, [0, 1])
    p = model.predict_proba(xq)[:, 1]
    np.testing.assert_allclose(p, expit(xq @ model.coef_[0]+model.intercept_[0]), atol=1e-15)
    design = np.column_stack([np.ones(len(x)), x])
    gradient = float(np.abs(design.T @ (model.predict_proba(x)[:, 1]-context.y.to_numpy()) / len(x)).max())
    if gradient > 1e-5:
        raise ValueError(f"Logistic MLE did not converge: gradient {gradient}")
    bundle = {"preprocessing": prep, "columns": columns, "model": model}
    record = {"n_context": len(context), "n_query": len(query), "prs": prs,
              "fit_ids_hash": id_hash(context), "query_ids_hash": id_hash(query),
              "iterations": int(model.n_iter_[0]), "gradient_max_abs": gradient,
              "query_labels_read": False, "allow_family_overlap": allow_family_overlap}
    return p, bundle, record


def select_single_prs(context, prs, folds):
    """Choose by pooled internal OOF log loss, never by assessment labels.

    Returned OOF predictions are selection data, not an unbiased performance estimate.
    Nest this routine in an outer fold when reporting training OOF performance.
    """
    folds = validate_folds(context, folds)
    rows, predictions, fits = [], {}, []
    for column in prs:
        q = np.full(len(context), np.nan)
        for fold in range(5):
            held = folds == fold
            q[held], _, record = fit_logistic(
                context.loc[~held], context.loc[held].drop(columns="y"), [column])
            fits.append({"candidate": column, "fold": fold, **record})
        if not np.isfinite(q).all():
            raise ValueError("Incomplete candidate OOF predictions")
        rows.append({"prs": column, "log_loss": log_loss(context.y, q),
                     "auroc": roc_auc_score(context.y, q)})
        predictions[column] = q
    ranking = pd.DataFrame(rows).sort_values(["log_loss", "prs"], kind="stable")
    return ranking.iloc[0].prs, ranking, pd.DataFrame(predictions), fits
