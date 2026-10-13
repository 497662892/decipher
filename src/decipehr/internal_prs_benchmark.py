"""Training-only model selection for the NYU internal PRS benchmark.

All estimators use established libraries. The glue here defines panels, fits
preprocessing within each context, and keeps every query prediction-only.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

from decipehr.discodivas import REFERENCE_PCS, ScoreBank, TrainScoreHarmonizer
from decipehr.external_pgs_benchmark import PCS
from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer

DISEASES = ("T2D", "cad", "bc")
C_GRID = (0.01, 0.1, 1.0, 10.0)
METHODS = (
    "covariates_lr", "full_lr", "best_single_lr", "flex_covariates", "flex_fixed",
    "flex_varying", "catboost", "selected_baseline", "tabpfn_covariates",
    "tabpfn_single", "tabpfn_full", "tabpfn_no_pc", "tabpfn_hard",
    "matched_fixed", "disco_linear", "disco_spline", "selected_disco",
    "tabpfn_matched",
)
BASELINES = ("best_single_lr", "full_lr", "flex_fixed", "catboost")


def ids_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(sorted(frame.sampleid.astype(str))).encode()).hexdigest()


def kinship_groups(ids: list[str], pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Connected components of validated PLINK kinship edges, including singletons."""
    ids = sorted(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient IDs")
    index = pd.Index(ids)
    edges = pairs.loc[pairs.KINSHIP >= threshold]
    i, j = index.get_indexer(edges.IID1), index.get_indexer(edges.IID2)
    if (i < 0).any() or (j < 0).any():
        raise ValueError("Kinship edge refers to an unknown patient")
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(ids), len(ids))).tocsr()
    _, groups = connected_components(graph, directed=False)
    return pd.DataFrame({"sampleid": ids, "kinship_group": groups.astype(str)})


def shared_folds(training: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if not training.sampleid.is_unique or training.kinship_group.isna().any():
        raise ValueError("Patients need unique IDs and complete groups")
    # Joint outcome/sex strata preserve one shared assignment across diseases.
    strata = training[[*DISEASES, "sex"]].astype(str).agg("|".join, axis=1)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    result = training[["sampleid", "kinship_group"]].copy()
    result["fold"] = -1
    for fold, (fit, query) in enumerate(splitter.split(training, strata, training.kinship_group)):
        assert_boundary(training.iloc[fit], training.iloc[query])
        result.iloc[query, result.columns.get_loc("fold")] = fold
    if set(result.fold) != set(range(5)):
        raise ValueError("Could not create all five folds")
    return result


def assert_boundary(context: pd.DataFrame, query: pd.DataFrame) -> None:
    if set(context.sampleid) & set(query.sampleid):
        raise ValueError("Context and query share patients")
    if set(context.kinship_group) & set(query.kinship_group):
        raise ValueError("Context and query share kinship groups")


def inner_splits(frame: pd.DataFrame, disease: str, seed: int):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(splitter.split(frame, frame[disease], frame.kinship_group))
    for a, b in splits:
        assert_boundary(frame.iloc[a], frame.iloc[b])
        if frame.iloc[a][disease].nunique() != 2 or frame.iloc[b][disease].nunique() != 2:
            raise ValueError("Every inner fit and query needs both classes")
    return splits


def add_metascores(context: pd.DataFrame, query: pd.DataFrame, bank: ScoreBank):
    """Published D-PRISM combination; scaling is learned only in this context."""
    if bank.disease != "T2D":
        return context.copy(), query.copy()
    columns = list(bank.source_ids)
    mean, sd = context[columns].mean(), context[columns].std(ddof=1)
    if (sd <= 0).any() or not np.isfinite(sd).all():
        raise ValueError("Degenerate component score")
    output = []
    for frame in (context, query):
        current = frame.copy()
        current[bank.model_ids] = ((frame[columns] - mean) / sd).to_numpy() @ bank.combination
        output.append(current)
    return output


@dataclass(frozen=True)
class Specification:
    engine: str = "lr"
    panel: str = "full"
    basis: str = "linear"
    prs: tuple[str, ...] = ()
    c: float = 1.0
    interaction_scale: float = 1.0
    depth: int = 4
    l2: float = 3.0
    ancestry: str = "pc"


def specifications(method: str, full_ids: list[str], bank: ScoreBank) -> list[Specification]:
    singles = [*full_ids, *(bank.model_ids if bank.disease == "T2D" else [])]
    prs = tuple(full_ids)
    if method in ("best_single_lr", "tabpfn_single"):
        engine = "tabpfn" if method == "tabpfn_single" else "lr"
        return [Specification(engine=engine, prs=(single,), c=c)
                for single in singles for c in (C_GRID if engine == "lr" else (1.0,))]
    if method == "catboost":
        return [Specification(engine="catboost", prs=prs, depth=d, l2=l2)
                for d in (4, 6) for l2 in (3.0, 10.0)]
    if method.startswith("tabpfn_"):
        if method == "tabpfn_covariates":
            prs = ()
        if method == "tabpfn_matched":
            return [Specification(engine="tabpfn", panel="matched", prs=tuple(bank.model_ids))]
        ancestry = {"tabpfn_no_pc": "none", "tabpfn_hard": "hard"}.get(method, "pc")
        return [Specification(engine="tabpfn", prs=prs, ancestry=ancestry)]
    if method in ("covariates_lr", "flex_covariates"):
        prs = ()
    panel = "matched" if method in ("matched_fixed", "disco_linear", "disco_spline") else "full"
    if panel == "matched":
        prs = ("discodivas_prs",) if method.startswith("disco_") else tuple(bank.model_ids)
    basis = "linear" if method in ("covariates_lr", "full_lr") else "spline"
    if method == "flex_varying":
        basis = "varying"
    if method == "disco_spline":
        basis = "score_spline"
    return [Specification(panel=panel, basis=basis, prs=prs, c=c, interaction_scale=s)
            for c in C_GRID for s in ((0.1, 1.0) if basis == "varying" else (1.0,))]


def design_pair(context: pd.DataFrame, query: pd.DataFrame, spec: Specification, bank: ScoreBank):
    """Return context-fitted matrices; query labels are forbidden at this boundary."""
    assert_boundary(context, query)
    if set(DISEASES) & set(query.columns):
        raise ValueError("Query outcomes must be removed before preprocessing")
    if spec.panel == "matched":
        harmonizer = TrainScoreHarmonizer(bank, frozenset(context.sampleid))
        context, query, harmony = harmonizer(context, query)
        pcs = REFERENCE_PCS
    else:
        context, query = add_metascores(context, query, bank)
        harmony = pd.DataFrame()
        pcs = PCS
    covariates = ["age_sd", "sex", *pcs]
    if spec.ancestry == "none":
        covariates = ["age_sd", "sex"]
    elif spec.ancestry == "hard":
        covariates = ["age_sd", "sex", "hard_label"]
    transform = TabPFNFeatureTransformer.fit(
        context, prs_cols=list(spec.prs), covariates=covariates, include_covariates=True,
    )
    x, q = transform.transform(context), transform.transform(query)
    audit = {"standardization": transform.summary().to_dict("records"),
             "harmonization": harmony.to_dict("records")}
    if spec.basis == "linear":
        return x.to_numpy(), q.to_numpy(), list(x.columns), audit
    numeric = [f"cov__{column}" for column in ("age_sd", *pcs) if f"cov__{column}" in x]
    if spec.basis == "score_spline":
        numeric += [column for column in x if column.startswith("prs__")]
    spline = SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False,
                               extrapolation="linear")
    sx, sq = spline.fit_transform(x[numeric]), spline.transform(q[numeric])
    other = [column for column in x if column not in numeric]
    names = [*spline.get_feature_names_out(numeric), *other]
    tx, tq = np.column_stack([sx, x[other]]), np.column_stack([sq, q[other]])
    audit["spline_knots"] = [s.t.tolist() for s in spline.bsplines_]
    if spec.basis == "varying":
        # Only ancestry PCs modify PRS coefficients. Age/sex remain in the base term.
        zcols = [column for column in x if column.startswith("prs__")]
        acols = [f"cov__{pc}" for pc in pcs if f"cov__{pc}" in x]
        for z in zcols:
            for a in acols:
                tx = np.column_stack([tx, x[z].to_numpy() * x[a].to_numpy()])
                tq = np.column_stack([tq, q[z].to_numpy() * q[a].to_numpy()])
                names.append(f"{z}*{a}")
    scaler = StandardScaler().fit(tx)
    tx, tq = scaler.transform(tx), scaler.transform(tq)
    if spec.basis == "varying":
        interaction = np.array(["*" in name for name in names])
        tx[:, interaction] *= spec.interaction_scale
        tq[:, interaction] *= spec.interaction_scale
    audit["design_mean"] = scaler.mean_.tolist()
    audit["design_scale"] = scaler.scale_.tolist()
    return tx, tq, names, audit


def fit_predict(x, y, q, spec: Specification, seed: int, tabpfn_factory: Callable | None):
    if spec.engine == "lr":
        model = LogisticRegression(C=spec.c, max_iter=3000, random_state=seed)
    elif spec.engine == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=500, depth=spec.depth, learning_rate=0.03,
                                   l2_leaf_reg=spec.l2, loss_function="Logloss", verbose=False,
                                   random_seed=seed, thread_count=4, allow_writing_files=False)
    else:
        if tabpfn_factory is None:
            raise ValueError("TabPFN factory is required")
        model = tabpfn_factory(seed)
    start = time.monotonic()
    model.fit(x, y)
    probability = np.asarray(model.predict_proba(q))[:, 1].astype(float)
    elapsed = time.monotonic() - start
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid probability")
    coefficients = model.coef_.ravel().tolist() if spec.engine == "lr" else []
    intercept = float(model.intercept_[0]) if spec.engine == "lr" else None
    del model
    if spec.engine == "tabpfn":
        gc.collect()
    return probability, elapsed, coefficients, intercept


def evaluate_partition(
    context: pd.DataFrame, query: pd.DataFrame, *, disease: str, full_ids: list[str],
    bank: ScoreBank, methods: tuple[str, ...] = METHODS, seed: int = 42,
    tabpfn_factory: Callable | None = None, on_method: Callable | None = None,
):
    """Select configurations by pooled inner OOF AUC, then predict untouched query."""
    assert_boundary(context, query)
    context, query = context.reset_index(drop=True), query.reset_index(drop=True)
    y = context[disease].astype(int).to_numpy()
    query_features = query.drop(columns=list(DISEASES), errors="ignore")
    splits = inner_splits(context, disease, seed)
    prediction = query[["sampleid", "kinship_group", "hard_label", disease]].rename(
        columns={disease: "label"},
    )
    selection, fits, parameters, outputs = [], [], [], []
    inner_best, final_predictions = {}, {}
    for method in methods:
        print(f"Fitting {disease} {method}", flush=True)
        if method in ("selected_baseline", "selected_disco"):
            pool = BASELINES if method == "selected_baseline" else ("disco_linear", "disco_spline")
            chosen_method = sorted(pool, key=lambda name: (-inner_best[name][0], inner_best[name][1], name))[0]
            p = final_predictions[chosen_method]
            inner_best[method] = inner_best[chosen_method]
            selection.append({"method": method, "chosen_method": chosen_method,
                              "inner_auc": inner_best[method][0], "inner_log_loss": inner_best[method][1]})
        else:
            candidates = specifications(method, full_ids, bank)
            best_key, chosen = None, None
            # No selection labels are required for a single fixed TabPFN specification.
            tune = len(candidates) > 1
            # Cache designs across regularization/depth candidates, within this method only.
            designs = {}
            for candidate_index, spec in enumerate(candidates):
                if tune:
                    oof = np.full(len(context), np.nan)
                    for inner_fold, (a, b) in enumerate(splits):
                        key = (inner_fold, spec.panel, spec.basis, spec.prs, spec.ancestry,
                               spec.interaction_scale)
                        if key not in designs:
                            designs[key] = design_pair(
                                context.iloc[a], context.iloc[b].drop(columns=list(DISEASES), errors="ignore"),
                                spec, bank,
                            )[:2]
                        x, q = designs[key]
                        oof[b], seconds, _, _ = fit_predict(x, y[a], q, spec, seed + inner_fold,
                                                          tabpfn_factory)
                        fits.append({"method": method, "candidate": candidate_index, "stage": "inner",
                                     "inner_fold": inner_fold, "n_context": len(a), "n_query": len(b),
                                     "context_ids_sha256": ids_hash(context.iloc[a]),
                                     "query_ids_sha256": ids_hash(context.iloc[b]), "seconds": seconds})
                    auc, loss = float(roc_auc_score(y, oof)), float(log_loss(y, oof))
                else:
                    auc, loss = np.nan, np.nan
                record = {"method": method, "candidate": candidate_index,
                          "configuration": json.dumps(spec.__dict__, sort_keys=True),
                          "inner_auc": auc, "inner_log_loss": loss, "selected": False}
                selection.append(record)
                key = (-auc, loss, candidate_index) if tune else (0, 0, 0)
                if best_key is None or key < best_key:
                    best_key, chosen = key, (spec, record)
            spec, record = chosen
            record["selected"] = True
            inner_best[method] = record["inner_auc"], record["inner_log_loss"]
            x, q, names, audit = design_pair(context, query_features, spec, bank)
            p, seconds, coef, intercept = fit_predict(x, y, q, spec, seed, tabpfn_factory)
            fits.append({"method": method, "stage": "final", "n_context": len(context),
                         "n_query": len(query), "n_features": x.shape[1],
                         "context_ids_sha256": ids_hash(context), "query_ids_sha256": ids_hash(query),
                         "seconds": seconds})
            parameters.append({"method": method, "configuration": spec.__dict__, "features": names,
                               "coefficients": coef, "intercept": intercept, "preprocessing": audit})
        final_predictions[method] = p
        frame = prediction.copy()
        frame["disease"], frame["method"], frame["probability"] = disease, method, p
        outputs.append(frame)
        if on_method:
            on_method(method, frame, selection, fits, parameters)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(selection), pd.DataFrame(fits), parameters
