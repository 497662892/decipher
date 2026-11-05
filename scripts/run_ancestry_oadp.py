"""Refit the frozen six-model comparison on OADP-corrected reference PCs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_ancestry_transfer import (
    BIO,
    SOURCE,
    TRAITS,
    hashes,
    original_predictions,
    read,
    refit_student,
)

from decipehr.ancestry_transfer import fit_logistic, replace_pcs, select_single_prs, validate_folds
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import load_config
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import assert_boundary, frame_hash
from decipehr.prs_distillation.teacher import make_teacher, teacher_oof, teacher_predict

BASELINE = Path("outputs/ancestry_transfer_20261104")
MODE = "oadp_pc"


def prepare(root, eligibility):
    if (root / "PREPARED.json").exists():
        verify_files(Path.cwd(), read(root / "PREPARED.json")["input_sha256"])
        return
    verify_files(Path.cwd(), read(BASELINE / "PREPARED.json")["input_sha256"])
    verify_files(BASELINE, read(BASELINE / "VERIFIED.json")["output_sha256"])
    projection = root / "projection"
    seal = read(projection / "COMPLETE.json")
    verify_files(projection, seal["output_sha256"])
    if not seal["reference_basis_unchanged"] or seal["phenotypes_used"]:
        raise ValueError("OADP projection violates frozen reference contract")
    pcs = pd.read_csv(projection / "target_pcs.tsv", sep="\t", dtype={"sampleid": str})
    rules = load_eligibility_rules(eligibility)
    if "bc" not in rules:
        raise ValueError("Female breast-cancer eligibility is required")
    folds = pd.read_parquet(SOURCE / "split_manifest.parquet").set_index("sampleid")
    inputs = dict(read(BASELINE / "PREPARED.json")["input_sha256"])
    extra = [Path(__file__), Path("scripts/run_ancestry_transfer.py"), Path(eligibility),
             projection / "COMPLETE.json", projection / "target_pcs.tsv", BASELINE / "VERIFIED.json"]
    availability = []
    for trait in TRAITS:
        d, e, manifest = load_inputs(SOURCE, trait)
        for frame in [d, e]:
            if len(filter_eligible(frame, trait, rules)) != len(frame):
                raise ValueError("Eligibility differs from frozen cohort")
            replace_pcs(frame, pcs)
        validate_folds(d, folds.loc[d.sampleid, "final_teacher_fold"].to_numpy())
        prior = BASELINE / "private" / trait
        receipt = read(prior / "COMPLETE.json")
        verify_files(prior, {name: receipt["output_sha256"][name] for name in ["predictions.parquet", "selections.csv"]})
        extra += [prior / n for n in ["COMPLETE.json", "predictions.parquet", "selections.csv"]]
        availability.append({"trait": trait, "n_train": len(d), "n_validation": len(e),
                             "n_prs": len(manifest), "n_projected": len(d) + len(e)})
    protocol = read(BASELINE / "PROTOCOL.json")
    protocol.update({"purpose": "OADP shrinkage correction sensitivity on the frozen reference PCA",
        "baseline": str(BASELINE), "projection": str(projection), "new_mode": MODE,
        "modes": ["official_pc", "reference_pc", MODE], "cohorts": availability,
        "primary_comparison": "OADP minus uncorrected reference PCs; official PCs are a secondary reference",
        "projection_basis_refitted": False, "tuning_on_validation": False,
        "student_selection": "Reuse the previous fold-local and full-training policies; no family or hyperparameter search",
        "scope": "Post-hoc development sensitivity; no All of Us participant data",
        "eligibility": str(eligibility), "correction_selection_uses_outcomes": False})
    write_json(root / "PROTOCOL.json", protocol)
    extra += [root / "PROTOCOL.json"]
    inputs.update({str(p): sha256_file(p) for p in extra})
    write_json(root / "PREPARED.json", {"input_sha256": inputs, "validation_labels_read": False})
    pd.DataFrame(availability).to_csv(root / "availability.csv", index=False)
    snapshot = root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for p in [Path(__file__), Path("scripts/run_ancestry_transfer.py")]:
        (snapshot / p.name).write_bytes(p.read_bytes())
    print(json.dumps(availability), flush=True)


def train(root, trait):
    torch.set_num_threads(1)
    verify_files(Path.cwd(), read(root / "PREPARED.json")["input_sha256"])
    dest = root / "private" / trait
    if (dest / "COMPLETE.json").exists():
        verify_files(dest, read(dest / "COMPLETE.json")["output_sha256"])
        return
    dest.mkdir(parents=True, exist_ok=True)
    pcs = pd.read_csv(root / "projection/target_pcs.tsv", sep="\t", dtype={"sampleid": str})
    d0, e0, manifest = load_inputs(SOURCE, trait)
    d, e = replace_pcs(d0, pcs), replace_pcs(e0, pcs)
    prs = manifest.column.tolist()
    split = pd.read_parquet(SOURCE / "split_manifest.parquet").set_index("sampleid")
    folds = validate_folds(d, split.loc[d.sampleid, "final_teacher_fold"].to_numpy())
    teacher_cfg = load_config(SOURCE / "resolved_config.yaml")
    common = dict(contract=read(SOURCE / "core" / trait / "teacher_contract.json"), trait=trait,
                  manifest_hash=sha256_file(SOURCE / trait / "feature_manifest.tsv"),
                  factory=lambda: make_teacher(teacher_cfg))
    predictions, selections, fits = [], [], []
    for fold in [-1, *range(5)]:
        run = dest / MODE / ("final" if fold == -1 else f"outer{fold}")
        run.mkdir(parents=True, exist_ok=True)
        scope = BIO if fold == -1 else BIO / "nested" / f"outer{fold}"
        context = d if fold == -1 else d.loc[folds != fold]
        query = e if fold == -1 else d.loc[folds == fold].drop(columns="y")
        assert_boundary(context, query, require_group_disjoint=fold != -1)
        if fold == -1:
            inner = folds
        else:
            split = pd.read_parquet(scope / "source/split_manifest.parquet").set_index("sampleid")
            if set(split.loc[split.split.eq("official_train")].index) & set(query.sampleid):
                raise ValueError("Outer query contaminates inner training")
            inner = split.loc[context.sampleid, "final_teacher_fold"].to_numpy()
        best, ranking, selection_oof, fit_records = select_single_prs(context, prs, inner)
        fits.extend({"mode": MODE, "outer_fold": fold, "stage": "selection", **r} for r in fit_records)
        ranking.to_csv(run / "single_prs_ranking.csv", index=False)
        selection_oof.insert(0, "sampleid", context.sampleid.to_numpy())
        selection_oof.to_parquet(run / "selection_oof.parquet", index=False)
        selections.append({"trait": trait, "mode": MODE, "outer_fold": fold,
                           "selected_prs": best, "selection_log_loss": float(ranking.iloc[0].log_loss)})
        arrays = {}
        for arm, columns in [("LR_SINGLE_BEST", [best]), ("LR_ALL", prs)]:
            arrays[arm], bundle, record = fit_logistic(context, query, columns, allow_family_overlap=fold == -1)
            joblib.dump(bundle, run / f"{arm}.joblib")
            fits.append({"mode": MODE, "outer_fold": fold, "stage": arm, **record})
        prior, selected = original_predictions(trait, fold)
        if set(prior.index) != set(query.sampleid):
            raise ValueError("Frozen assessment patients differ")
        q = teacher_oof(context, inner, prs, directory=run / "teacher_oof", **common)
        arrays["TabPFN_v3"] = teacher_predict(context, query, prs, directory=run / "teacher_query",
                                               fold_id="D_to_E" if fold == -1 else fold, **common)
        fitted = {arm: refit_student(context, query, prs, q, scope, trait, arm, run / "models" / arm)
                  for arm in dict.fromkeys(["M0_KD", "M0_SUP", selected])}
        arrays.update({k: fitted[k] for k in ["M0_KD", "M0_SUP"]})
        arrays["H_SELECTED"] = fitted[selected]
        result = query[["sampleid", "kinship_group", "hard_label"]].copy()
        for arm, p in arrays.items():
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError("Invalid probabilities")
            predictions.append(result.assign(trait=trait, mode=MODE, fold=fold, arm=arm, probability=p))
        write_json(run / "scope.json", {"context_hash": frame_hash(context), "query_hash": frame_hash(query),
            "student_policy_arm": selected, "query_labels_read": False, "preprocessing_training_only": True})
        print(f"{trait}/{MODE}/fold={fold}: n_fit={len(context)}, n_query={len(query)}, best={best}, student={selected}", flush=True)
    combined = pd.concat(predictions, ignore_index=True)
    for arm in combined.arm.unique():
        oof = combined.loc[combined.fold.ge(0) & combined.arm.eq(arm)]
        if not oof.sampleid.is_unique or set(oof.sampleid) != set(d0.sampleid):
            raise ValueError("OOF coverage differs")
    combined.to_parquet(dest / "predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(dest / "selections.csv", index=False)
    pd.DataFrame(fits).to_json(dest / "logistic_fits.json", orient="records", indent=2)
    write_json(dest / "COMPLETE.json", {"trait": trait, "validation_labels_read": False,
        "nested_single_prs_selection": True, "fold_local_teacher_targets": True,
        "all_oof_patients_exactly_once": True, "logistic_fits": len(fits), "output_sha256": hashes(dest)})


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trait", choices=TRAITS)
    parser.add_argument("--eligibility-config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out_dir, args.eligibility_config)
    else:
        if args.trait is None:
            parser.error("--trait required")
        if str(args.eligibility_config) != read(args.out_dir / "PROTOCOL.json")["eligibility"]:
            raise ValueError("Eligibility configuration differs")
        train(args.out_dir, args.trait)


if __name__ == "__main__":
    main()
