"""Restore the agreed frozen external-PRS, official train/development comparison.

Audit precedes fitting. The experimental heterozygosity/kinship exclusions are
not inputs to this run. Existing predictions and score files are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_external_pgs_cv import _tabpfn_factory
from sklearn.metrics import roc_auc_score

from decipehr.discodivas import REFERENCE_PCS, comparison_bank
from decipehr.eligibility import load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_pgs_benchmark import (
    feature_matrices,
    method_features,
    prepare_official_validation_data,
)
from decipehr.internal_prs_benchmark import (
    DISEASES,
    METHODS,
    Specification,
    design_pair,
    evaluate_partition,
    ids_hash,
)

LEGACY = Path("outputs/external_pgs/run_20261010/official_validation")
PREVIOUS = Path("outputs/internal_prs/nyu_20261011")
CODE = (
    "scripts/correct_external_prs_analysis.py", "scripts/evaluate_external_pgs_cv.py",
    "src/decipehr/internal_prs_benchmark.py", "src/decipehr/external_pgs_benchmark.py",
    "src/decipehr/tabpfn_ensemble.py", "src/decipehr/discodivas.py",
)


def load_frames(contract, disease):
    paths = contract["source_paths"]
    ids = contract["panels"][disease]["full_ids"]
    context, query = prepare_official_validation_data(
        train_phenotype=pd.read_csv(paths["train"], dtype={"sampleid": str}),
        validation_phenotype=pd.read_csv(paths["development"], dtype={"sampleid": str}),
        scores=pd.read_csv(paths["scores"], dtype={"sampleid": str}), soft_weights=None,
        hard_labels=pd.read_csv(paths["ancestry"], sep="\t", dtype={"IID": str}),
        disease=disease, model_ids=ids, eligibility_rules=load_eligibility_rules(paths["eligibility"]),
    )
    pcs = pd.read_csv(paths["reference_pcs"], sep="\t", dtype={"sampleid": str})
    geometry = pd.read_csv(paths[f"geometry_{disease}"], dtype={"sampleid": str})
    frames = []
    for original in (context, query):
        frame = original.merge(pcs[["sampleid", *REFERENCE_PCS]], on="sampleid", validate="one_to_one")
        frame = frame.merge(geometry, on="sampleid", validate="one_to_one")
        if not frame.sampleid.equals(original.sampleid):
            raise ValueError("Reference inputs changed official patient membership/order")
        frame["kinship_group"] = "patient_only_" + frame.sampleid
        frames.append(frame)
    return context, query, *frames


def verify_contract(root):
    contract = json.loads((root / "CONTRACT.json").read_text())
    for key, digest in contract["source_sha256"].items():
        if sha256_file(contract["source_paths"][key]) != digest:
            raise ValueError(f"Frozen input changed: {key}")
    for path, digest in contract["code_sha256"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen implementation changed: {path}")
    return contract


def audit(root, eligibility):
    previous = json.loads((PREVIOUS / "prepared/manifest.json").read_text())
    excluded = {"hardcall_counts", "hardcall_missingness", "kinship"}
    paths = {k: v for k, v in previous["source_paths"].items() if k not in excluded}
    if sha256_file(eligibility) != previous["source_sha256"]["eligibility"]:
        raise ValueError("Disease eligibility changed")
    digests = {k: sha256_file(v) for k, v in paths.items()}
    for key, digest in digests.items():
        if digest != previous["source_sha256"][key]:
            raise ValueError(f"External source changed: {key}")
    contract = {
        "protocol": "external_prs_original_official_development_correction_v1",
        "evaluation": "official_train_to_previously_used_development",
        "selection": "three-fold patient-only CV inside official training only",
        "cohort_basis": "restore agreed official membership, not selection by performance",
        "new_genotype_qc_exclusions": False, "verified_family_separation": False,
        "external_snp_weights_retrained": False, "independent_external_validation": False,
        "source_paths": paths, "source_sha256": digests, "panels": previous["panels"],
        "code_sha256": {p: sha256_file(p) for p in CODE},
        "seed": 42, "n_estimators": 16, "methods": list(METHODS),
        "required_runtime": {"gpu": "L40S", "threads": 8},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "tabpfn", "torch", "catboost")},
        "cohorts": {}, "legacy_artifact_sha256": {},
    }
    rows = []
    for disease in DISEASES:
        legacy = LEGACY / disease
        run = json.loads((legacy / "run.json").read_text())
        for current, original in (("scores", "scores"), ("weight_lock", "weight_lock"),
                                  ("train", "train_phenotype"), ("development", "validation_phenotype"),
                                  ("eligibility", "eligibility_config"), ("ancestry", "ancestry_labels")):
            if digests[current] != run["input_sha256"][original]:
                raise ValueError(f"Original input mismatch: {disease}/{current}")
        ids = contract["panels"][disease]["full_ids"]
        if ids != run["score_banks"][disease] or digests["checkpoint"] != run["tabpfn_checkpoint_sha256"]:
            raise ValueError("Original PRS order or checkpoint changed")
        context, query, current, target = load_frames(contract, disease)
        bank = comparison_bank(disease, pd.read_csv(paths["mapping"]))
        prs, covariates, interactions = method_features("tabpfn_prs_pc", ids)
        x, q, _ = feature_matrices(context, query.drop(columns=list(DISEASES)), prs=prs,
                                   covariates=covariates, interactions=interactions, normalization="none")
        nx, nq, names, _ = design_pair(current, target.drop(columns=list(DISEASES)),
                                      Specification(engine="tabpfn", prs=tuple(ids)), bank)
        np.testing.assert_array_equal(x, nx)
        np.testing.assert_array_equal(q, nq)
        old = pd.read_csv(legacy / "validation_predictions.csv",
                          dtype={"sampleid": str, "y_prob": "float32"})
        old = old.loc[old.method == "tabpfn_prs_pc"].set_index("sampleid").sort_index()
        truth = query.set_index("sampleid")[disease].sort_index()
        if not old.index.equals(truth.index):
            raise ValueError("Original prediction membership differs")
        np.testing.assert_array_equal(old.y_true.to_numpy(), truth.to_numpy())
        auc = float(roc_auc_score(old.y_true, old.y_prob))
        report = pd.read_csv(legacy / "summary_metrics.csv")
        reported = report.query("method == 'tabpfn_prs_pc' and prediction_scale == 'raw' and ancestry_group == 'ALL'").iloc[0].auc
        if abs(auc - reported) > 1e-14:
            raise ValueError("Original published AUC did not reproduce")
        contract["cohorts"][disease] = {
            "n_context": len(context), "n_query": len(query), "query_cases": int(truth.sum()),
            "context_ids_sha256": ids_hash(context), "query_ids_sha256": ids_hash(query),
            "features": names, "context_matrix_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "query_matrix_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
        }
        for name in ("run.json", "validation_predictions.csv", "summary_metrics.csv"):
            path = legacy / name
            contract["legacy_artifact_sha256"][str(path)] = sha256_file(path)
        rows.append({"disease": disease, "n_train": len(context), "n_development": len(query),
                     "cases": int(truth.sum()), "n_external_prs": len(ids), "n_encoded_features": x.shape[1],
                     "feature_matrices_exactly_equal": True, "original_auc_recomputed": auc})
    root.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(root / "INPUT_AUDIT.csv", index=False)
    write_json(root / "CONTRACT.json", contract)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def fit(root, disease, anchor):
    import torch

    contract = verify_contract(root)
    gpu = torch.cuda.get_device_name(0)
    if "L40S" not in gpu or os.environ.get("OMP_NUM_THREADS") != "8":
        raise ValueError("Correction requires original L40S / 8-thread runtime")
    _, _, context, query = load_frames(contract, disease)
    counts = contract["cohorts"][disease]
    if ids_hash(context) != counts["context_ids_sha256"] or ids_hash(query) != counts["query_ids_sha256"]:
        raise ValueError("Official patient membership changed")
    out = root / ("anchor" if anchor else "development") / disease
    out.mkdir(parents=True, exist_ok=False)
    methods = ("tabpfn_full",) if anchor else METHODS

    def save_method(method, frame, selection, fits, parameters):
        frame.to_csv(out / f"{method}.predictions.csv", index=False, float_format="%.17g")
        pd.DataFrame(selection).to_csv(out / "selection.partial.csv", index=False)

    paths = contract["source_paths"]
    predictions, selection, fits, parameters = evaluate_partition(
        context, query, disease=disease, full_ids=contract["panels"][disease]["full_ids"],
        bank=comparison_bank(disease, pd.read_csv(paths["mapping"])), methods=methods,
        seed=contract["seed"], tabpfn_factory=_tabpfn_factory("cuda", paths["checkpoint"]),
        on_method=save_method,
    )
    predictions.to_csv(out / "predictions.csv", index=False, float_format="%.17g")
    selection.to_csv(out / "selection.csv", index=False)
    fits.to_csv(out / "fits.csv", index=False)
    write_json(out / "parameters.json", parameters)
    write_json(out / "COMPLETE.json", {
        "status": "complete", "disease": disease, "methods": list(methods),
        "n_context": len(context), "n_query": len(query), "gpu": gpu,
        "threads": os.environ["OMP_NUM_THREADS"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "contract_sha256": sha256_file(root / "CONTRACT.json"),
        "output_sha256": {name: sha256_file(out / name) for name in
                          ("predictions.csv", "selection.csv", "fits.csv", "parameters.json")},
    })
    print(f"Complete original-cohort external PRS: {disease}; methods={len(methods)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["audit", "anchor", "fit"], required=True)
    parser.add_argument("--run-dir", default="outputs/external_pgs/correction_20261011")
    parser.add_argument("--disease", choices=DISEASES)
    parser.add_argument("--eligibility-config", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.run_dir)
    if args.phase == "audit":
        audit(root, args.eligibility_config)
    else:
        contract = json.loads((root / "CONTRACT.json").read_text())
        if sha256_file(args.eligibility_config) != contract["source_sha256"]["eligibility"]:
            raise ValueError("Eligibility config changed")
        if args.disease is None:
            parser.error("--disease is required for fitting")
        fit(root, args.disease, args.phase == "anchor")


if __name__ == "__main__":
    main()
