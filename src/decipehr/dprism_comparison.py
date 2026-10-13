"""Compare direct integration with published D-PRISM aggregation of 25 scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from decipehr.discodivas import REFERENCE_PCS, ScoreBank, TrainScoreHarmonizer
from decipehr.internal_prs_benchmark import (
    DISEASES,
    Specification,
    assert_boundary,
    evaluate_partition,
    fit_predict,
    ids_hash,
)
from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer

SOURCE_IDS = tuple(f"PGS{i:06d}" for i in range(5353, 5378))
DIRECT = "tabpfn_dprism25"
EXISTING = ("tabpfn_matched", "disco_linear", "disco_spline", "selected_disco")
METHODS = (DIRECT, *EXISTING)


def component_design_pair(context: pd.DataFrame, query: pd.DataFrame, bank: ScoreBank):
    """Residualize all 25 components in training and verify the 25-to-5 identity.

    OLS projection commutes with the published linear combination. Component
    residuals therefore reconstruct the existing metascores after their original
    training-fitted scaling; both models use the same ancestry correction.
    """
    assert_boundary(context, query)
    if set(DISEASES) & set(query.columns):
        raise ValueError("Query outcomes must be removed before preprocessing")
    if bank.disease != "T2D" or bank.source_ids != SOURCE_IDS:
        raise ValueError("Expected exactly the 25 published D-PRISM components")
    columns = list(SOURCE_IDS)
    raw = context[columns].to_numpy(dtype=float)
    mean, sd = raw.mean(axis=0), raw.std(axis=0, ddof=1)
    if not np.isfinite(raw).all() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("Invalid component scores in training")
    scaled = (raw - mean) / sd
    pc_model = LinearRegression().fit(context[REFERENCE_PCS], scaled)
    residuals = scaled - pc_model.predict(context[REFERENCE_PCS])
    residual_mean = residuals.mean(axis=0)
    residuals -= residual_mean
    meta_residuals = residuals @ bank.combination
    meta_sd = meta_residuals.std(axis=0, ddof=1)
    component_sd = residuals.std(axis=0, ddof=1)
    if (component_sd <= 0).any() or (meta_sd <= 0).any():
        raise ValueError("Degenerate PC-residualized components or metascores")

    reference_train, reference_query, _ = TrainScoreHarmonizer(
        bank, frozenset(context.sampleid),
    )(context, query)
    transformed, checks = [], []
    for label, frame, reference in (
        ("context", context, reference_train), ("query", query, reference_query),
    ):
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Invalid component scores in query")
        residual = (values - mean) / sd - pc_model.predict(frame[REFERENCE_PCS]) - residual_mean
        rebuilt = (residual @ bank.combination) / meta_sd
        expected = reference[bank.model_ids].to_numpy()
        np.testing.assert_allclose(rebuilt, expected, atol=1e-10, rtol=1e-10)
        ancestry_weights = frame[[f"disco_a_{g}" for g in bank.groups]].to_numpy()
        # Expand the actual residualized/scaled Disco score into all 25 components.
        component_weights = (ancestry_weights / meta_sd) @ bank.combination.T
        expanded = np.sum(residual * component_weights, axis=1)
        np.testing.assert_allclose(expanded, reference.discodivas_prs, atol=1e-10, rtol=1e-10)
        checks.append({"scope": label, "n": len(frame),
                       "metascore_max_abs_error": float(np.max(np.abs(rebuilt - expected))),
                       "disco_expansion_max_abs_error": float(np.max(np.abs(expanded - reference.discodivas_prs)))})
        current = frame.copy()
        current[columns] = residual / component_sd
        transformed.append(current)
    train, target = transformed
    transform = TabPFNFeatureTransformer.fit(
        train, prs_cols=columns, covariates=["age_sd", "sex", *REFERENCE_PCS],
        include_covariates=True,
    )
    x, q = transform.transform(train), transform.transform(target)
    if sum(name.startswith("prs__") for name in x) != 25:
        raise ValueError("A D-PRISM component was dropped")
    audit = {"source_ids": columns, "context_ids_sha256": ids_hash(context),
             "source_mean": mean.tolist(), "source_sd": sd.tolist(),
             "pc_intercept": pc_model.intercept_.tolist(), "pc_coefficients": pc_model.coef_.tolist(),
             "residual_mean": residual_mean.tolist(), "component_residual_sd": component_sd.tolist(),
             "metascore_residual_sd": meta_sd.tolist(), "published_combination": bank.combination.tolist(),
             "standardization": transform.summary().to_dict("records"), "equivalence_checks": checks}
    return x.to_numpy(), q.to_numpy(), list(x.columns), audit


def evaluate_source_partition(context, query, *, bank, seed=42, tabpfn_factory=None):
    """Reuse the frozen matched models and add one direct 25-component TabPFN."""
    prediction, selection, fits, parameters = evaluate_partition(
        context, query, disease="T2D", full_ids=list(SOURCE_IDS), bank=bank,
        methods=EXISTING, seed=seed, tabpfn_factory=tabpfn_factory,
    )
    print(f"Fitting T2D {DIRECT}", flush=True)
    spec = Specification(engine="tabpfn", panel="dprism_components", prs=SOURCE_IDS)
    x, q, names, audit = component_design_pair(
        context, query.drop(columns=list(DISEASES), errors="ignore"), bank,
    )
    p, seconds, _, _ = fit_predict(x, context.T2D.to_numpy(dtype=int), q, spec, seed, tabpfn_factory)
    direct = query[["sampleid", "kinship_group", "hard_label", "T2D"]].rename(columns={"T2D": "label"})
    direct = direct.assign(disease="T2D", method=DIRECT, probability=p)
    fits = pd.concat([fits, pd.DataFrame([{
        "method": DIRECT, "stage": "final", "n_context": len(context), "n_query": len(query),
        "n_features": x.shape[1], "context_ids_sha256": ids_hash(context),
        "query_ids_sha256": ids_hash(query), "seconds": seconds,
    }])], ignore_index=True)
    parameters.append({"method": DIRECT, "configuration": spec.__dict__, "features": names,
                       "preprocessing": audit})
    return pd.concat([prediction, direct], ignore_index=True), selection, fits, parameters
