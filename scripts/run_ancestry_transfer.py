"""Frozen-coordinate sensitivity and nested single-best PRS LR comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from decipehr.ancestry_transfer import (
    fit_logistic,
    replace_pcs,
    select_single_prs,
    validate_folds,
)
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import load_config
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import PCS, assert_boundary, frame_hash
from decipehr.prs_distillation.teacher import make_teacher, teacher_oof, teacher_predict
from decipehr.prs_distillation_bio import BioPreprocessor, fit, load, predict, save
from decipehr.prs_monotone_boosting import Features, bundle_model, fit_booster
from decipehr.prs_monotone_boosting import predict as predict_booster

SOURCE = Path("outputs/prs_distillation_v1/phase_a_fulltrain_20261026_3d4334d2_0161ad36d950")
BIO = Path("outputs/prs_distillation_bio/exploration_20261026_3569648f99a8")
PROJECTION = Path("outputs/external_pgs/discodivas/projection_1000g_refit_v2_20261010")
PLAIN = Path("outputs/prs_distillation_bio/plain_logistic_20261027")
FILTERS = Path("outputs/prs_distillation_bio/validation_kinship_filters_20261027")
TRAITS = ["T2D", "cad", "bc"]
MODES = ["official_pc", "reference_pc"]


def read(path):
    return json.loads(Path(path).read_text())


def hashes(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def original_predictions(trait, fold):
    scope = BIO if fold == -1 else BIO / "nested" / f"outer{fold}"
    policy = read(BIO / "report" / ("val" if fold == -1 else "nested") / "POLICY_LOCK.json")
    selected = policy["choices"][trait][str(fold)]
    core = scope / "core" / trait
    receipt = read(core / "PREDICTIONS_COMPLETE.json")
    verify_files(core, {"direct_predictions.parquet": receipt["output_sha256"]["direct_predictions.parquet"]})
    data = pd.read_parquet(core / "direct_predictions.parquet", filters=[("seed", "=", 17)])
    base = data.loc[data.arm.eq("M0_KD")].set_index("sampleid")
    result = base[["probability", "p_teacher"]].rename(columns={"probability": "M0_KD", "p_teacher": "TabPFN_v3"})
    result["M0_SUP"] = data.loc[data.arm.eq("M0_SUP")].set_index("sampleid").probability
    if selected.startswith("HGB"):
        low = scope / "low_order" / trait
        verify_files(low, {"predictions.parquet": read(low / "COMPLETE.json")["output_sha256"]["predictions.parquet"]})
        data = pd.read_parquet(low / "predictions.parquet")
    result["H_SELECTED"] = data.loc[data.arm.eq(selected)].set_index("sampleid").probability
    if not result.index.is_unique or not np.isfinite(result.to_numpy()).all():
        raise ValueError("Incomplete frozen prediction coverage")
    return result, selected


def prepare(root, eligibility):
    if (root / "PREPARED.json").exists():
        verify_files(Path.cwd(), read(root / "PREPARED.json")["input_sha256"])
        return
    root.mkdir(parents=True, exist_ok=False)
    projection = read(PROJECTION / "projection.json")
    verify_files(PROJECTION, projection["output_sha256"])
    if projection["projection_mode"] != "1000G_reference_refit" or projection["n_scored_variants"] != 53571:
        raise ValueError("Unexpected reference-PC contract")
    pcs = pd.read_csv(PROJECTION / "target_pcs.tsv", sep="\t", dtype={"sampleid": str})
    scores = pd.read_csv(PROJECTION / "target.sscore", sep=r"\s+", dtype={"#IID": str}).set_index("#IID")
    np.testing.assert_allclose(pcs.set_index("sampleid").loc[scores.index].to_numpy(),
                               scores[[f"PC{i}_SUM" for i in range(1, 11)]].to_numpy(), atol=1e-12, rtol=1e-12)
    rules = load_eligibility_rules(eligibility)
    if "bc" not in rules:
        raise ValueError("Female BC eligibility rule required")
    audit = read(SOURCE / "audit_report.json")
    verify_files(Path.cwd(), {str(eligibility): audit["source_sha256"][str(eligibility)]})
    split = pd.read_parquet(SOURCE / "split_manifest.parquet").set_index("sampleid")
    verify_files(SOURCE, {"split_manifest.parquet": read(SOURCE / "split_receipt.json")["sha256"]})
    files = [Path(__file__), Path("src/decipehr/ancestry_transfer.py"), Path(eligibility),
             SOURCE / "audit_report.json", SOURCE / "split_manifest.parquet", SOURCE / "resolved_config.yaml",
             PROJECTION / "projection.json", PROJECTION / "target_pcs.tsv", BIO / "PROTOCOL.json",
             BIO / "report/val/POLICY_LOCK.json", BIO / "report/nested/POLICY_LOCK.json",
             FILTERS / "private/cohort_membership.parquet", FILTERS / "VERIFIED.json"]
    rows = []
    for trait in TRAITS:
        d, e, manifest = load_inputs(SOURCE, trait)
        for frame in (d, e):
            if len(filter_eligible(frame, trait, rules)) != len(frame):
                raise ValueError("Eligibility differs from frozen cohort")
            replaced = replace_pcs(frame, pcs)
            if np.array_equal(frame[PCS].to_numpy(), replaced[PCS].to_numpy()):
                raise ValueError("Reference and official PCs unexpectedly identical")
        validate_folds(d, split.loc[d.sampleid, "final_teacher_fold"].to_numpy())
        contract = read(SOURCE / "core" / trait / "teacher_contract.json")
        verify_files(Path.cwd(), contract["implementation_sha256"])
        verify_files(Path.cwd(), {contract["checkpoint"]: contract["checkpoint_sha256"]})
        files += [SOURCE / trait / n for n in ["training.parquet", "evaluation_features.parquet", "feature_manifest.tsv", "input_receipt.json"]]
        files += [SOURCE / "core" / trait / "teacher_contract.json"]
        for fold in [-1, *range(5)]:
            scope = BIO if fold == -1 else BIO / "nested" / f"outer{fold}"
            original_predictions(trait, fold)
            files += [scope / "PROTOCOL.json"]
            files += [scope / "core" / trait / n for n in ["SELECTION_LOCK.json", "direct_predictions.parquet", "PREDICTIONS_COMPLETE.json"]]
            files += [scope / "low_order" / trait / n for n in ["SELECTION_LOCK.json", "predictions.parquet", "COMPLETE.json"]]
            if fold != -1:
                source = scope / "source"
                verify_files(source, {"split_manifest.parquet": read(source / "split_receipt.json")["sha256"]})
                files += [source / "split_manifest.parquet"]
        rows.append({"trait": trait, "n_train": len(d), "n_validation": len(e), "n_prs": len(manifest),
                     "official_training_hash": frame_hash(d), "reference_training_hash": frame_hash(replace_pcs(d, pcs)),
                     "reference_coverage": len(d)+len(e)})
    protocol = {"purpose": "Reference-PC sensitivity plus single-best PRS logistic baseline",
        "source": str(SOURCE), "student_source": str(BIO), "projection": str(PROJECTION),
        "eligibility": str(eligibility), "cohorts": rows, "modes": MODES,
        "primary_student": "M0_KD; H_SELECTED also includes the previously chosen disease-specific student",
        "ancestry_change": "Replace all official PC1_sd-PC10_sd with fixed 1000G refPC1-refPC10 before every model/preprocessor fit",
        "projection_contract": "GRCh38, 2503 reference samples, 53571 fixed SNPs, reference AF and PLINK SUM scores; PCA never fitted on target patients",
        "student_selection": "Keep original train-only family, hyperparameters and iteration count; use corresponding outer-fold locks for OOF; refit weights on new PCs and new fold-local teacher targets",
        "prs_policy": "Keep the frozen 26/7/5 score banks exactly, including previously frozen score corrections",
        "single_best": "Unpenalized LR(single PRS + linear age_sd + sex + 10 PCs); choose lowest pooled training-CV log loss, lexicographic PRS ID tie break",
        "single_best_oof": "Nested: select PRS inside each outer-training fold; candidate-selection OOF is not reported as unbiased performance",
        "primary_seed": 17, "teacher_estimators": 16, "teacher_seed": 42,
        "cohort_policy": "Original train and validation; group-disjoint train CV; existing validation-only kinship filters reused",
        "metrics": ["auroc", "log_loss", "brier", "nagelkerke_r2", "average_precision", "calibration"],
        "uncertainty": "1000 paired available-kinship-group bootstrap draws; conditional on fitted models; unadjusted descriptive intervals",
        "scope": "Post-hoc development sensitivity on decipEHR, not an All of Us external validation",
        "aoU_access": False, "validation_labels_read": False}
    write_json(root / "PROTOCOL.json", protocol)
    files += [root / "PROTOCOL.json"]
    inputs = {str(p): sha256_file(p) for p in files}
    write_json(root / "PREPARED.json", {"input_sha256": inputs, "validation_labels_read": False})
    pd.DataFrame(rows).to_csv(root / "availability.csv", index=False)
    print(json.dumps(rows, indent=2), flush=True)


def refit_student(context, query, prs, q, scope, trait, arm, out):
    if "y" in query:
        raise ValueError("Student query must be prediction-only")
    if arm.startswith("HGB"):
        choice = read(scope / "low_order" / trait / "SELECTION_LOCK.json")["choices"][arm]
        features = Features.fit(context, prs)
        target = (1-choice["lambda"])*context.y.to_numpy()+choice["lambda"]*q
        model = fit_booster(features.transform(context), target, features, family=choice["family"],
                            leaves=choice["leaves"], l2=choice["l2"], iterations=choice["iterations"])
        bundle = bundle_model(model, features, choice["family"], choice["iterations"], target.mean(), {"choice": choice})
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out / "model.joblib")
        p = predict_booster(bundle, features.transform(query))
        replay = joblib.load(out / "model.joblib")
        np.testing.assert_array_equal(p, predict_booster(replay, features.transform(query)))
        return p
    lock = read(scope / "core" / trait / "SELECTION_LOCK.json")
    choice = lock["choices"][arm]
    if (out / "HASHES.json").exists():
        restored, prep, metadata = load(out)
        if metadata["choice"] != choice:
            raise ValueError("Cached student selection differs")
        return predict(restored, prep.transform(query))["probability"]
    warm = None
    if choice["parent"] is not None:
        parent_path = out.parent / choice["parent"]
        refit_student(context, query, prs, q, scope, trait, choice["parent"], parent_path)
        warm, _, _ = load(parent_path)
    protocol = read(scope / "PROTOCOL.json")
    cfg = protocol["identity"]["config"]
    prep = BioPreprocessor.fit(context, prs, age_segments=cfg["families"][choice["family"]]["age_segments"])
    model, info = fit(prep.transform(context), context.y.to_numpy(), q, **choice["settings"],
                      rho=protocol["priors"][trait]["common_gate_ridge"],
                      kappa_shape=cfg["shape_penalty"], baseline_ridge=cfg["baseline_ridge"],
                      seed=17, fixed_steps=choice["steps"], warm=warm)
    save(out, model, prep, {"choice": choice, "training": info, "original_scope": str(scope)})
    p = predict(model, prep.transform(query))["probability"]
    restored, prep2, _ = load(out)
    np.testing.assert_array_equal(p, predict(restored, prep2.transform(query))["probability"])
    return p


def train(root, trait):
    torch.set_num_threads(1)
    verify_files(Path.cwd(), read(root / "PREPARED.json")["input_sha256"])
    dest = root / "private" / trait
    if (dest / "COMPLETE.json").exists():
        verify_files(dest, read(dest / "COMPLETE.json")["output_sha256"])
        return
    dest.mkdir(parents=True, exist_ok=True)
    d0, e0, manifest = load_inputs(SOURCE, trait)
    prs = manifest.column.tolist()
    pcs = pd.read_csv(PROJECTION / "target_pcs.tsv", sep="\t", dtype={"sampleid": str})
    assignments = pd.read_parquet(SOURCE / "split_manifest.parquet").set_index("sampleid")
    folds = validate_folds(d0, assignments.loc[d0.sampleid, "final_teacher_fold"].to_numpy())
    teacher_cfg = load_config(SOURCE / "resolved_config.yaml")
    contract = read(SOURCE / "core" / trait / "teacher_contract.json")
    common = dict(contract=contract, trait=trait, manifest_hash=sha256_file(SOURCE / trait / "feature_manifest.tsv"),
                  factory=lambda: make_teacher(teacher_cfg))
    predictions, selections, fits = [], [], []
    for mode in MODES:
        d, e = (d0, e0) if mode == "official_pc" else (replace_pcs(d0, pcs), replace_pcs(e0, pcs))
        for fold in [-1, *range(5)]:
            run = dest / mode / ("final" if fold == -1 else f"outer{fold}")
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
            fits.extend({"mode": mode, "outer_fold": fold, "stage": "selection", **r} for r in fit_records)
            ranking.to_csv(run / "single_prs_ranking.csv", index=False)
            selection_oof.insert(0, "sampleid", context.sampleid.to_numpy())
            selection_oof.to_parquet(run / "selection_oof.parquet", index=False)
            selections.append({"trait": trait, "mode": mode, "outer_fold": fold, "selected_prs": best,
                               "selection_log_loss": float(ranking.iloc[0].log_loss)})
            arrays = {}
            for arm, columns in [("LR_SINGLE_BEST", [best]), ("LR_ALL", prs)]:
                p, bundle, record = fit_logistic(context, query, columns, allow_family_overlap=fold == -1)
                arrays[arm] = p
                joblib.dump(bundle, run / f"{arm}.joblib")
                fits.append({"mode": mode, "outer_fold": fold, "stage": arm, **record})
            old, selected = original_predictions(trait, fold)
            if set(old.index) != set(query.sampleid):
                raise ValueError("Frozen comparison cohort mismatch")
            if mode == "official_pc":
                for arm in old:
                    arrays[arm] = old.loc[query.sampleid, arm].to_numpy()
            else:
                q = teacher_oof(context, inner, prs, directory=run / "teacher_oof", **common)
                arrays["TabPFN_v3"] = teacher_predict(context, query, prs,
                    directory=run / "teacher_query", fold_id="D_to_E" if fold == -1 else fold, **common)
                fitted = {}
                for arm in dict.fromkeys(["M0_KD", "M0_SUP", selected]):
                    fitted[arm] = refit_student(context, query, prs, q, scope, trait, arm, run / "models" / arm)
                arrays.update({k: fitted[k] for k in ["M0_KD", "M0_SUP"]})
                arrays["H_SELECTED"] = fitted[selected]
            result = query[["sampleid", "kinship_group", "hard_label"]].copy()
            for arm, p in arrays.items():
                if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError("Invalid model probabilities")
                predictions.append(result.assign(trait=trait, mode=mode, fold=fold, arm=arm, probability=p))
            write_json(run / "scope.json", {"context_hash": frame_hash(context), "query_hash": frame_hash(query),
                "student_policy_arm": selected, "query_labels_read": False, "preprocessing_training_only": True})
            print(f"{trait}/{mode}/fold={fold}: n_fit={len(context)}, n_query={len(query)}, best={best}, student={selected}", flush=True)
    combined = pd.concat(predictions, ignore_index=True)
    combined.to_parquet(dest / "predictions.parquet", index=False)
    pd.DataFrame(selections).to_csv(dest / "selections.csv", index=False)
    pd.DataFrame(fits).to_json(dest / "logistic_fits.json", orient="records", indent=2)
    # Independently reproduce the previously sealed all-PRS unpenalized baseline.
    verify_files(PLAIN, {"private/"+trait+"/validation_predictions.parquet": read(PLAIN / "VERIFIED.json")["output_sha256"]["private/"+trait+"/validation_predictions.parquet"]})
    old_lr = pd.read_parquet(PLAIN / "private" / trait / "validation_predictions.parquet").set_index("sampleid")
    new_lr = combined.loc[(combined["mode"] == "official_pc") & combined.fold.eq(-1) & combined.arm.eq("LR_ALL")].set_index("sampleid")
    np.testing.assert_allclose(new_lr.loc[old_lr.index, "probability"], old_lr.LR_UNPENALIZED, atol=1e-12, rtol=0)
    for mode in MODES:
        for arm in combined.arm.unique():
            oof = combined.loc[(combined["mode"] == mode) & combined.fold.ge(0) & combined.arm.eq(arm)]
            if not oof.sampleid.is_unique or set(oof.sampleid) != set(d0.sampleid):
                raise ValueError("OOF coverage mismatch")
    write_json(dest / "COMPLETE.json", {"trait": trait, "validation_labels_read": False,
        "nested_single_prs_selection": True, "fold_local_teacher_targets": True,
        "official_all_prs_lr_replayed": True, "all_oof_patients_exactly_once": True,
        "logistic_fits": len(fits), "output_sha256": hashes(dest)})


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
            parser.error("--trait required for train")
        if str(args.eligibility_config) != read(args.out_dir / "PROTOCOL.json")["eligibility"]:
            raise ValueError("Eligibility configuration mismatch")
        train(args.out_dir, args.trait)


if __name__ == "__main__":
    main()
