"""Matched holdout predictions from frozen-train and train+validation refit models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_ancestry_transfer import BIO, PROJECTION, SOURCE, TRAITS, hashes, read, refit_student

from decipehr.ancestry_transfer import replace_pcs, validate_folds
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.ensemble import pivot_long_prs_scores
from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import load_config
from decipehr.prs_distillation.data import grouped_folds, verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import assert_boundary, frame_hash
from decipehr.prs_distillation.teacher import make_teacher, teacher_oof, teacher_predict
from decipehr.prs_distillation_bio import load, predict
from decipehr.prs_monotone_boosting import predict as predict_hgb

BASELINE = Path("outputs/ancestry_transfer_20261104")
HOLDOUT = Path("outputs/external_pgs/holdout_wrapper_20261012")
COVARIATES = Path("data/processed/final/holdout_test_covariates.csv")
ANCESTRY = Path("outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.holdout_test.tsv")
SCOPES = ["train_only", "trainval_refit"]


def saved_student(directory, query, arm):
    if arm.startswith("HGB"):
        bundle = joblib.load(directory / "model.joblib")
        return predict_hgb(bundle, bundle["features"].transform(query))
    model, prep, _ = load(directory)
    return predict(model, prep.transform(query))["probability"]


def prepare(root, eligibility):
    if (root / "PREPARED.json").exists():
        verify_files(Path.cwd(), read(root / "PREPARED.json")["input_sha256"])
        return
    verify_files(Path.cwd(), read(BASELINE / "PREPARED.json")["input_sha256"])
    selection = read(BASELINE / "FINAL_PC_SELECTION.json")
    if selection["selected_mode"] != "reference_pc" or selection["oadp_enabled"]:
        raise ValueError("The selected standard reference PC scheme changed")
    verify_files(root / "projection", read(root / "projection/COMPLETE.json")["output_sha256"])
    rules = load_eligibility_rules(eligibility)
    if "bc" not in rules:
        raise ValueError("Female BC eligibility is required")
    cov = pd.read_csv(COVARIATES, dtype={"sampleid": str})
    if set(cov) & {"y", *TRAITS} or not cov.sampleid.is_unique:
        raise ValueError("Holdout must contain unique patients and no outcome columns")
    pcs = pd.read_csv(root / "projection/holdout_pcs.tsv", sep="\t", dtype={"sampleid": str})
    context_pcs = pd.read_csv(PROJECTION / "target_pcs.tsv", sep="\t", dtype={"sampleid": str})
    ancestry = pd.read_csv(ANCESTRY, sep="\t", dtype={"sampleid": str})[["sampleid", "hard_label"]]
    if not ancestry.sampleid.is_unique or set(ancestry.sampleid) != set(cov.sampleid):
        raise ValueError("Holdout ancestry coverage changed")
    scoring = read(HOLDOUT / "scoring/VERIFIED.json")
    verify_files(HOLDOUT / "scoring", {"scores.long.csv": scoring["scores_sha256"]})
    scores = pd.read_csv(HOLDOUT / "scoring/scores.long.csv", dtype={"sampleid": str})
    chosen = read(HOLDOUT / "SELECTION.json")
    audit = read(SOURCE / "audit_report.json")
    label_paths = [Path(audit["paths"][k]) for k in ["training", "evaluation"]]
    verify_files(Path.cwd(), {str(p): audit["source_sha256"][str(p)] for p in label_paths})
    labels = pd.concat([pd.read_csv(p, dtype={"sampleid": str}, usecols=["sampleid", *TRAITS])
                        for p in label_paths], ignore_index=True).set_index("sampleid")
    if not labels.index.is_unique or not labels.isin([0, 1]).all().all():
        raise ValueError("Combined training outcomes are incomplete or duplicated")
    if set(labels.index) & set(cov.sampleid):
        raise ValueError("Holdout overlaps combined training")
    source_d, source_e, _ = load_inputs(SOURCE, "T2D")
    joint = pd.concat([source_d, source_e], ignore_index=True)[["sampleid", "kinship_group", "hard_label"]]
    if set(joint.sampleid) != set(labels.index):
        raise ValueError("Combined training cohort differs from frozen train/validation union")
    for trait in TRAITS:
        joint[trait] = labels.loc[joint.sampleid, trait].to_numpy()
    cfg = load_config(SOURCE / "resolved_config.yaml")
    folds, stratification = grouped_folds(joint, cfg.split_seed, TRAITS)
    assignments = joint[["sampleid", "kinship_group"]].assign(fold=folds)
    private = root / "private"
    private.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(private / "trainval_folds.parquet", index=False)
    inputs = dict(read(BASELINE / "PREPARED.json")["input_sha256"])
    files = [Path(__file__), Path("scripts/run_ancestry_transfer.py"), Path(eligibility),
             BASELINE / "FINAL_PC_SELECTION.json", COVARIATES, ANCESTRY, *label_paths,
             HOLDOUT / "SELECTION.json", HOLDOUT / "scoring/VERIFIED.json",
             HOLDOUT / "scoring/scores.long.csv", root / "projection/COMPLETE.json",
             root / "projection/holdout_pcs.tsv", private / "trainval_folds.parquet"]
    availability = []
    for trait in TRAITS:
        d0, e0, manifest = load_inputs(SOURCE, trait)
        prs = manifest.column.tolist()
        if prs != chosen["diseases"][trait]["model_ids"]:
            raise ValueError("Holdout and frozen model PRS panels differ")
        for receipt_name in manifest.scoring_receipt.unique():
            scoring_run = read(receipt_name)
            if scoring_run["reference_frequency_sha256"] != scoring["reference_frequency_sha256"]:
                raise ValueError("Holdout and model PRS genotype imputation frequencies differ")
            files.append(Path(receipt_name))
        d, e = replace_pcs(d0, context_pcs), replace_pcs(e0, context_pcs)
        np.testing.assert_array_equal(d.y.to_numpy(), labels.loc[d.sampleid, trait].to_numpy())
        merged = pd.concat([d, e.assign(y=labels.loc[e.sampleid, trait].to_numpy(dtype=d.y.dtype))], ignore_index=True)
        if len(filter_eligible(merged, trait, rules)) != len(merged):
            raise ValueError("Combined training eligibility changed")
        validate_folds(merged, assignments.set_index("sampleid").loc[merged.sampleid, "fold"].to_numpy())
        query = filter_eligible(cov, trait, rules).copy()
        wide = pivot_long_prs_scores(scores, trait, patient_col="sampleid")
        if set(wide.columns) != {"sampleid", *prs} or set(wide.sampleid) != set(cov.sampleid):
            raise ValueError("Incomplete or expanded holdout PRS panel")
        query = query.merge(wide, on="sampleid", validate="one_to_one", how="left")
        query = query.merge(ancestry, on="sampleid", validate="one_to_one", how="left")
        query["kinship_group"] = "UNRESOLVED_HOLDOUT::" + query.sampleid
        query = replace_pcs(query, pcs)[e.columns].reset_index(drop=True)
        if query.isna().any().any() or not np.isfinite(query[[*prs, "age_sd", *[f"PC{i}_sd" for i in range(1, 11)]]].to_numpy()).all():
            raise ValueError("Missing/nonfinite holdout features")
        assert_boundary(merged, query, require_group_disjoint=False)
        previous = BASELINE / "private" / trait
        verify_files(previous, read(previous / "COMPLETE.json")["output_sha256"])
        final = previous / "reference_pc/final"
        scope = read(final / "scope.json")
        if frame_hash(d) != scope["context_hash"] or frame_hash(e) != scope["query_hash"]:
            raise ValueError("Frozen model context or validation features changed")
        selected = scope["student_policy_arm"]
        old_predictions = pd.read_parquet(previous / "predictions.parquet")
        old_predictions = old_predictions.loc[old_predictions.fold.eq(-1) & old_predictions["mode"].eq("reference_pc")]
        replay = {}
        for label, arm in [("M0_KD", "M0_KD"), ("H_SELECTED", selected)]:
            p = saved_student(final / "models" / arm, e, arm)
            old = old_predictions.loc[old_predictions.arm.eq(label)].set_index("sampleid").loc[e.sampleid, "probability"].to_numpy()
            np.testing.assert_allclose(p, old, rtol=0, atol=1e-12)
            replay[label] = float(np.max(np.abs(p - old)))
        dest = private / trait
        dest.mkdir(exist_ok=True)
        for name, frame in [("train", d), ("trainval", merged), ("holdout", query)]:
            frame.to_parquet(dest / f"{name}.parquet", index=False)
            files.append(dest / f"{name}.parquet")
        write_json(dest / "INPUTS.json", {"prs": prs, "selected_student": selected,
            "frozen_student_validation_replay_max_error": replay,
            "context_train_hash": frame_hash(d), "context_trainval_hash": frame_hash(merged),
            "holdout_hash": frame_hash(query), "holdout_ids_disjoint": True})
        files += [dest / "INPUTS.json", previous / "COMPLETE.json", final / "scope.json"]
        files += [p for p in (final / "models").rglob("*") if p.is_file()]
        files += [final / "teacher_query/cache.json"]
        availability.append({"trait": trait, "n_train": len(d), "n_trainval": len(merged),
            "n_holdout": len(query), "n_prs": len(prs), "selected_student": selected,
            "holdout_unknown_sex": int(query.sex.eq("Unknown").sum())})
    pd.DataFrame(availability).to_csv(root / "availability.csv", index=False)
    protocol = {"purpose": "Holdout student/TabPFN prediction agreement using standard reference PCs",
        "scopes": SCOPES, "holdout_outcomes_available": False, "holdout_outcomes_used": False,
        "validation_outcomes_use": "Training labels only in trainval_refit; frozen train_only models unchanged",
        "pc_mode": "reference_pc", "oadp": False, "projection_basis_refitted": False,
        "teacher": "Frozen TabPFN v3 checkpoint; package 8.0.1; 16 estimators; seed 42",
        "teacher_output": "Direct positive-class probability without an official-PC wrapper",
        "student_arms": ["M0_KD", "H_SELECTED"], "student_selection": "Original full-training policy, hyperparameters and iteration counts frozen; no new search",
        "trainval_targets": "Five-fold OOF teacher predictions generated inside combined training",
        "combined_folds": {"seed": cfg.split_seed, "stratification": stratification,
            "shared_across_traits": True, "known_kinship_groups_disjoint": True,
            "unresolved_kinship": "Existing patient placeholders retained"},
        "holdout_kinship": "Not established; patient placeholders are not evidence of unrelatedness",
        "statistics": "Pearson/Spearman agreement, probability differences, top-decile overlap; no outcome accuracy metrics",
        "uncertainty": "1000 paired patient bootstrap draws conditional on fitted predictions; unadjusted; holdout kinship dependence not modeled",
        "cohorts": availability, "eligibility": str(eligibility)}
    write_json(root / "PROTOCOL.json", protocol)
    files.append(root / "PROTOCOL.json")
    inputs.update({str(p): sha256_file(p) for p in files})
    write_json(root / "PREPARED.json", {"input_sha256": inputs, "holdout_outcomes_used": False})
    snapshot = root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for p in [Path(__file__), Path("scripts/run_ancestry_transfer.py"), Path("scripts/project_holdout_reference_pcs.py")]:
        (snapshot / p.name).write_bytes(p.read_bytes())
    print(pd.DataFrame(availability).to_string(index=False), flush=True)


def run(root, trait):
    torch.set_num_threads(1)
    verify_files(Path.cwd(), read(root / "PREPARED.json")["input_sha256"])
    private = root / "private" / trait
    info = read(private / "INPUTS.json")
    query = pd.read_parquet(private / "holdout.parquet")
    cfg = load_config(SOURCE / "resolved_config.yaml")
    contract = read(SOURCE / "core" / trait / "teacher_contract.json")
    verify_files(Path.cwd(), contract["implementation_sha256"])
    verify_files(Path.cwd(), {contract["checkpoint"]: contract["checkpoint_sha256"]})
    common = dict(contract=contract, trait=trait,
        manifest_hash=sha256_file(SOURCE / trait / "feature_manifest.tsv"), factory=lambda: make_teacher(cfg))
    selected, prs = info["selected_student"], info["prs"]
    for scope in SCOPES:
        out = private / scope
        if (out / "COMPLETE.json").exists():
            verify_files(out, read(out / "COMPLETE.json")["output_sha256"])
            continue
        out.mkdir(exist_ok=True)
        context = pd.read_parquet(private / ("train.parquet" if scope == "train_only" else "trainval.parquet"))
        assert_boundary(context, query, require_group_disjoint=False)
        q = teacher_predict(context, query, prs, directory=out / "teacher_holdout", fold_id="D_to_E", **common)
        if scope == "train_only":
            frozen = BASELINE / "private" / trait / "reference_pc/final"
            old = read(frozen / "teacher_query/cache.json")["fingerprint"]
            new = read(out / "teacher_holdout/cache.json")["fingerprint"]
            for key in ["teacher", "context_data_hash", "fit_ids_hash", "preprocessing_hash", "prs_columns", "historical_teacher_columns"]:
                if old[key] != new[key]:
                    raise ValueError(f"Frozen train teacher contract changed: {key}")
            fitted = {arm: saved_student(frozen / "models" / arm, query, arm)
                      for arm in dict.fromkeys(["M0_KD", selected])}
        else:
            split = pd.read_parquet(root / "private/trainval_folds.parquet").set_index("sampleid")
            folds = validate_folds(context, split.loc[context.sampleid, "fold"].to_numpy())
            targets = teacher_oof(context, folds, prs, directory=out / "teacher_oof", **common)
            fitted = {arm: refit_student(context, query, prs, targets, BIO, trait, arm, out / "models" / arm)
                      for arm in dict.fromkeys(["M0_KD", selected])}
        arrays = {"TabPFN_v3": q, "M0_KD": fitted["M0_KD"], "H_SELECTED": fitted[selected]}
        frames = []
        for arm, probability in arrays.items():
            if len(probability) != len(query) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError("Invalid or incomplete holdout predictions")
            frames.append(query[["sampleid", "hard_label"]].assign(
                trait=trait, scope=scope, arm=arm, probability=probability))
        pd.concat(frames, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
        write_json(out / "COMPLETE.json", {"status": "complete", "trait": trait, "scope": scope,
            "n_context": len(context), "n_holdout": len(query), "selected_student": selected,
            "holdout_outcomes_used": False, "new_hyperparameter_search": False,
            "student_weights_refitted": scope == "trainval_refit",
            "context_hash": frame_hash(context), "query_hash": frame_hash(query),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(),
            "output_sha256": hashes(out)})
        print(f"Complete {trait}/{scope}: context={len(context)}, holdout={len(query)}", flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "predict"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eligibility-config", type=Path, required=True)
    parser.add_argument("--trait", choices=TRAITS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        prepare(args.out_dir, args.eligibility_config)
    else:
        if args.trait is None:
            parser.error("--trait required")
        if str(args.eligibility_config) != read(args.out_dir / "PROTOCOL.json")["eligibility"]:
            raise ValueError("Eligibility configuration differs")
        run(args.out_dir, args.trait)


if __name__ == "__main__":
    main()
