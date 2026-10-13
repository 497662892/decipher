"""Freeze the NYU-only benchmark inputs and one shared kinship-group split."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.discodivas import REFERENCE_PCS, comparison_bank
from decipehr.eligibility import load_eligibility_rules
from decipehr.external_pgs import load_weight_lock, sha256_file, write_json
from decipehr.external_pgs_benchmark import prepare_official_validation_data
from decipehr.internal_prs_benchmark import (
    BASELINES,
    C_GRID,
    DISEASES,
    METHODS,
    kinship_groups,
    shared_folds,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/internal_prs/nyu_20261011")
    parser.add_argument("--eligibility-config", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.out_dir)
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=False)
    scoring = Path("outputs/external_pgs/run_20261010/scoring_v2")
    projection = Path("outputs/external_pgs/discodivas/projection_1000g_refit_v2_20261010")
    resources = Path("outputs/external_pgs/discodivas_resource_audit_20261010")
    paths = {
        "train": Path("data/processed/phenotype.csv"),
        "development": Path("data/processed/official_validation_phenotype.csv"),
        "scores": scoring / "scores.long.csv", "weight_lock": scoring / "weights.scored.lock.json",
        "scoring_run": scoring / "scoring_run.json", "projection": projection / "projection.json",
        "reference_pcs": projection / "target_pcs.tsv", "mapping": resources / "dprism_component_mapping.csv",
        "ancestry": Path("outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv"),
        "eligibility": Path(args.eligibility_config),
        "kinship": root / "kinship/rehard.kin0", "hardcall_counts": root / "kinship/rehard_qc.scount",
        "hardcall_missingness": root / "kinship/rehard_qc.smiss",
        "checkpoint": Path("outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt"),
    }
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    scored = json.loads(paths["scoring_run"].read_text())
    if scored["scores_sha256"] != hashes["scores"] or scored["weight_lock_sha256"] != hashes["weight_lock"]:
        raise ValueError("Frozen scores or weights changed")
    projection_run = json.loads(paths["projection"].read_text())
    if projection_run["status"] != "complete":
        raise ValueError("Projection is incomplete")
    for name, digest in projection_run["output_sha256"].items():
        if sha256_file(projection / name) != digest:
            raise ValueError(f"Projection output changed: {name}")
    resource_audit = json.loads((resources / "AUDIT.json").read_text())
    if hashes["mapping"] != resource_audit["outputs"]["dprism_component_mapping.csv"]["sha256"]:
        raise ValueError("D-PRISM mapping changed")
    train, dev, scores = [pd.read_csv(paths[name], dtype={"sampleid": str})
                          for name in ("train", "development", "scores")]
    ancestry = pd.read_csv(paths["ancestry"], sep="\t", dtype={"IID": str})
    qc = pd.read_csv(paths["hardcall_counts"], sep="\t", dtype={"#IID": str}).rename(columns={"#IID": "sampleid"})
    qc = qc.merge(ancestry[["IID", "hard_label"]].rename(columns={"IID": "sampleid"}),
                  on="sampleid", validate="one_to_one")
    qc["heterozygosity"] = qc.HET_SNP_CT / (qc.HOM_REF_CT + qc.HOM_ALT_SNP_CT + qc.HET_SNP_CT)
    qc["exclude_high_heterozygosity"] = False
    limits = []
    # Outcome-blind, conservative robust QC; thresholds are recorded before any new model fits.
    for label, group in qc.groupby("hard_label", observed=True):
        median = float(group.heterozygosity.median())
        robust_sd = float(1.4826 * np.median(np.abs(group.heterozygosity - median)))
        threshold = median + 6 * robust_sd
        flagged = group.heterozygosity > threshold
        qc.loc[group.index, "exclude_high_heterozygosity"] = flagged
        limits.append({"hard_label": label, "n": len(group), "median": median,
                       "robust_sd": robust_sd, "threshold": threshold, "n_flagged": int(flagged.sum())})
    ids = set(train.sampleid) | set(dev.sampleid)
    if len(ids) != len(train) + len(dev) or ids != set(qc.sampleid):
        raise ValueError("Cohort/QC mismatch or overlapping official patients")
    retained = set(qc.loc[~qc.exclude_high_heterozygosity, "sampleid"])
    pairs = pd.read_csv(paths["kinship"], sep="\t", dtype={"#IID1": str, "IID2": str}).rename(columns={"#IID1": "IID1"})
    pairs = pairs.loc[pairs.IID1.isin(retained) & pairs.IID2.isin(retained)].copy()
    groups = kinship_groups(list(retained), pairs, 0.0884)
    grouped_train = train.loc[train.sampleid.isin(retained)].merge(groups, on="sampleid", validate="one_to_one")
    folds = shared_folds(grouped_train)
    dev_groups = set(groups.loc[groups.sampleid.isin(dev.sampleid), "kinship_group"])
    dev_context_ids = set(grouped_train.loc[~grouped_train.kinship_group.isin(dev_groups), "sampleid"])
    qc.to_csv(out / "sample_qc.csv", index=False)
    pd.DataFrame(limits).to_csv(out / "qc_thresholds.csv", index=False)
    groups.to_csv(out / "kinship_groups.csv", index=False)
    folds.to_csv(out / "shared_training_folds.csv", index=False)
    pairs.loc[pairs.KINSHIP >= 0.0884].to_csv(out / "retained_kinship_edges.csv", index=False)
    rules = load_eligibility_rules(args.eligibility_config)
    if "bc" not in rules:
        raise ValueError("Female BC eligibility rule is mandatory")
    lock = load_weight_lock(paths["weight_lock"])
    mapping = pd.read_csv(paths["mapping"])
    pcs = pd.read_csv(paths["reference_pcs"], sep="\t", dtype={"sampleid": str})
    cohort_rows, panel_metadata = [], {}
    for disease in DISEASES:
        full_ids = [r["id"] for r in lock["resources"] if r["disease"] == disease]
        bank = comparison_bank(disease, mapping)
        context, query = prepare_official_validation_data(
            train_phenotype=train, validation_phenotype=dev, scores=scores, soft_weights=None,
            hard_labels=ancestry, disease=disease, model_ids=full_ids, eligibility_rules=rules,
        )
        geometry_dir = Path("outputs/external_pgs/discodivas/validation_20261010") / disease
        geometry_path = geometry_dir / "interpolation_coefficients.csv"
        geometry_run = json.loads((geometry_dir / "run.json").read_text())
        if geometry_run["input_sha256"]["pcs"] != hashes["reference_pcs"]:
            raise ValueError("Disco geometry uses different coordinates")
        if sha256_file(geometry_path) != geometry_run["output_sha256"][geometry_path.name]:
            raise ValueError("Disco interpolation coefficients changed")
        paths[f"geometry_{disease}"] = geometry_path
        hashes[f"geometry_{disease}"] = sha256_file(geometry_path)
        geometry = pd.read_csv(geometry_path, dtype={"sampleid": str})
        for role, frame in (("training", context), ("development", query)):
            original_n = len(frame)
            frame = frame.loc[frame.sampleid.isin(retained)].copy()
            frame = frame.merge(groups, on="sampleid", validate="one_to_one")
            frame = frame.merge(pcs[["sampleid", *REFERENCE_PCS]], on="sampleid", validate="one_to_one")
            frame = frame.merge(geometry, on="sampleid", validate="one_to_one")
            if role == "training":
                frame = frame.merge(folds[["sampleid", "fold"]], on="sampleid", validate="one_to_one")
                frame["development_context"] = frame.sampleid.isin(dev_context_ids)
            if frame.isna().any().any():
                raise ValueError("Prepared data contain missing entries")
            frame.to_parquet(out / f"{disease}.{role}.parquet", index=False)
            cohort_rows.append({"disease": disease, "role": role, "original_n": original_n,
                                "n": len(frame), "cases": int(frame[disease].sum()),
                                "n_development_context": int(frame.development_context.sum()) if role == "training" else None})
            if role == "training":
                frame.groupby("fold", observed=True).agg(n=(disease, "size"), cases=(disease, "sum")).reset_index().to_csv(
                    out / f"{disease}.fold_counts.csv", index=False)
        panel_metadata[disease] = {"full_ids": full_ids, "matched_ids": bank.model_ids,
                                  "matched_sources": list(bank.source_ids),
                                  "single_candidates": [*full_ids, *(bank.model_ids if disease == "T2D" else [])],
                                  "published_combination": bank.combination.tolist()}
    pd.DataFrame(cohort_rows).to_csv(out / "cohort_counts.csv", index=False)
    source_files = ["src/decipehr/internal_prs_benchmark.py", "scripts/prepare_internal_prs_benchmark.py",
                    "scripts/evaluate_internal_prs_benchmark.py", "src/decipehr/discodivas.py",
                    "src/decipehr/tabpfn_ensemble.py", "src/decipehr/external_pgs_benchmark.py",
                    "scripts/evaluate_external_pgs_cv.py", "pyproject.toml", "uv.lock"]
    manifest = {
        "status": "prepared", "protocol": "NYU_internal_20261011_v1", "seed": 42,
        "evaluation": "training-only shared grouped 5-fold OOF plus previously-used development",
        "external_validation": "not available", "methods": list(METHODS), "panels": panel_metadata,
        "inner_folds": 3, "selection": "pooled inner OOF AUC, then log loss, then fixed candidate order",
        "C_grid": list(C_GRID), "baseline_pool": list(BASELINES), "tabpfn_n_estimators": 16,
        "kinship_threshold": 0.0884, "n_genotype_markers": 53571,
        "kinship_limitation": "KING hardcall estimates after heterozygosity QC; not verified pedigrees; admixture and genotype uncertainty remain",
        "heterozygosity_qc": "Upper median + 6*1.4826*MAD within frozen 1000G similarity groups; outcome-blind",
        "qc_excluded": int(qc.exclude_high_heterozygosity.sum()),
        "largest_retained_kinship_group": int(groups.kinship_group.value_counts().max()),
        "full_pc_input": "provided clinical PC1_sd-PC10_sd, context scaling refitted",
        "matched_pc_input": "same externally fitted 1000G refPC1-refPC10 for every matched method",
        "source_paths": {key: str(value) for key, value in paths.items()}, "source_sha256": hashes,
        "code_sha256": {name: sha256_file(name) for name in source_files},
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "catboost", "tabpfn", "torch")},
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
    }
    write_json(out / "manifest.json", manifest)
    print(pd.DataFrame(cohort_rows).to_string(index=False))
    print(f"Prepared: {out}; excluded high heterozygosity={manifest['qc_excluded']}")


if __name__ == "__main__":
    main()
