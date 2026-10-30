"""Local pinned TabPFN V3, context-local transforms and exact-provenance caches."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import digest
from decipehr.prs_distillation.data import code_hashes, verify_files
from decipehr.prs_distillation.preprocessing import (
    AGE,
    PCS,
    Preprocessor,
    assert_boundary,
    frame_hash,
    id_hash,
)
from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer


def make_teacher(cfg):
    for name in ("TABPFN_NO_BROWSER", "TABPFN_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY",
                 "HF_HUB_OFFLINE"):
        os.environ[name] = "1"
    if not Path(cfg.checkpoint).is_file():
        raise FileNotFoundError("Local V3 checkpoint unavailable; downloads are disabled")
    if importlib.metadata.version("tabpfn") != cfg.teacher_package:
        raise ValueError("Teacher package version differs from the resolved contract")
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3, n_estimators=cfg.teacher_estimators, device=cfg.teacher_device,
        random_state=cfg.teacher_seed, model_path=str(Path(cfg.checkpoint).resolve()),
        balance_probabilities=False,
    )


def teacher_contract(cfg):
    classifier = make_teacher(cfg)  # Constructor only; no download, fit or inference.
    parameters = json.loads(json.dumps(classifier.get_params(deep=True), default=str))
    return {"backend": "local_tabpfn", "model_version": "V3", "package": cfg.teacher_package,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("tabpfn", "torch", "scikit-learn", "numpy", "pandas")},
            "checkpoint": str(Path(cfg.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(cfg.checkpoint), "parameters": parameters,
            "positive_label": 1, "seed": cfg.teacher_seed,
            "feature_pipeline": cfg.teacher_preprocessing, "cohort_policy": cfg.cohort_policy,
            "implementation_sha256": code_hashes()}


def positive_probability(classifier, x):
    classes = np.asarray(classifier.classes_)
    index = np.flatnonzero(classes == 1)
    if classes.shape != (2,) or len(index) != 1 or set(classes.tolist()) != {0, 1}:
        raise ValueError("Teacher classes do not match binary disease labels 0/1")
    probability = np.asarray(classifier.predict_proba(x), dtype=np.float64)
    if probability.shape != (len(x), 2) or not np.isfinite(probability).all() or \
            ((probability < 0) | (probability > 1)).any() or \
            not np.allclose(probability.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Invalid teacher probabilities")
    return probability[:, index[0]], {"classes": classes.tolist(), "positive_index": int(index[0])}


def teacher_predict(context, query, prs, *, directory, contract, trait, manifest_hash,
                    fold_id, factory):
    """query must contain features only, including on cache hits."""
    if "y" in query or any(c in query for c in ("T2D", "cad", "bc")):
        raise ValueError("Teacher query must be prediction-only, with outcome columns removed")
    allow_family_overlap = (contract.get("cohort_policy") == "original_full_sensitivity"
                            and str(fold_id) == "D_to_E")
    assert_boundary(context, query, require_group_disjoint=not allow_family_overlap)
    if not context.y.isin([0, 1]).all() or context.y.nunique() != 2:
        raise ValueError("Teacher context needs both binary classes")
    prep = Preprocessor.fit(context, prs)
    tx, qx = prep.transform(context), prep.transform(query)
    historical = None
    if contract.get("feature_pipeline") == "historical_external":
        historical = TabPFNFeatureTransformer.fit(
            context, prs_cols=prs, covariates=[AGE, "sex", *PCS], include_covariates=True,
        )
        tx["teacher"] = historical.transform(context).to_numpy()
        qx["teacher"] = historical.transform(query).to_numpy()
    fingerprint = {
        "trait": trait, "feature_manifest_hash": manifest_hash, "prs_columns": prs,
        "fit_ids_hash": id_hash(context), "query_ids_hash": id_hash(query),
        "context_data_hash": frame_hash(context), "query_data_hash": frame_hash(query),
        "preprocessing_hash": digest(prep.metadata()), "teacher": contract,
        "fold_id": str(fold_id), "positive_label": 1,
        "require_group_disjoint": not allow_family_overlap,
        "shared_kinship_groups": len(set(context.kinship_group) & set(query.kinship_group)),
        "historical_teacher_columns": list(historical.output_columns) if historical else None,
    }
    root = Path(directory)
    record = root / "cache.json"
    if record.exists():
        saved = json.loads(record.read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError(f"Teacher cache scope/provenance mismatch: {root}")
        verify_files(root, saved["output_sha256"])
        cached = pd.read_parquet(root / "predictions.parquet")
        if not cached.sampleid.equals(query.sampleid.reset_index(drop=True)):
            raise ValueError("Teacher cache order/coverage mismatch")
        if not cached.q_teacher.between(0, 1).all():
            raise ValueError("Invalid cached probabilities")
        return cached.q_teacher.to_numpy()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Incomplete teacher cache must be quarantined before retry: {root}")
    start = time.monotonic()
    print(f"Teacher {trait}/{fold_id}: context={len(context)}, query={len(query)}", flush=True)
    classifier = factory()
    classifier.fit(tx["teacher"], context.y.to_numpy())
    q, mapping = positive_probability(classifier, qx["teacher"])
    root.mkdir(parents=True, exist_ok=True)
    result = query[["sampleid"]].reset_index(drop=True).assign(fold_id=str(fold_id), q_teacher=q)
    result.to_parquet(root / "predictions.parquet", index=False)
    pd.concat([context[["sampleid", "kinship_group"]].assign(role="context"),
               query[["sampleid", "kinship_group"]].assign(role="query")],
              ignore_index=True).to_parquet(root / "scope.parquet", index=False)
    joblib.dump(prep, root / "preprocessing.joblib")
    if historical is not None:
        joblib.dump(historical, root / "historical_teacher_preprocessing.joblib")
    write_json(record, {"fingerprint": fingerprint, "mapping": mapping,
                        "query_support_counts": {key: int(qx[key].sum()) for key in
                                                 ("age_out_of_support", "sex_out_of_support")},
                        "preprocessing": prep.metadata(), "elapsed_seconds": time.monotonic() - start,
                        "output_sha256": {p.name: sha256_file(p) for p in root.iterdir()}})
    del classifier
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return q


def teacher_oof(frame, folds, prs, *, directory, contract, trait, manifest_hash, factory):
    folds = np.asarray(folds)
    if len(folds) != len(frame) or set(folds) != set(range(5)):
        raise ValueError("OOF requires one fixed five-fold assignment per patient")
    output = np.full(len(frame), np.nan)
    coverage = np.zeros(len(frame), dtype=int)
    for fold in range(5):
        fit, query = frame.loc[folds != fold], frame.loc[folds == fold]
        output[folds == fold] = teacher_predict(
            fit, query.drop(columns="y"), prs, directory=Path(directory) / str(fold),
            contract=contract, trait=trait, manifest_hash=manifest_hash, fold_id=fold, factory=factory,
        )
        coverage[folds == fold] += 1
    if not np.isfinite(output).all() or not (coverage == 1).all():
        raise ValueError("OOF coverage is incomplete or duplicated")
    return output
