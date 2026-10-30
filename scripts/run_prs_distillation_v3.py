"""Prepare, smoke-test and train the specified sign-aware student using audited caches."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import Optimization, digest
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import AGE, assert_boundary, frame_hash, id_hash
from decipehr.prs_distillation.teacher import teacher_oof, teacher_predict
from decipehr.prs_distillation_v3 import (
    SignAwarePreprocessor,
    fit_student,
    load_bundle,
    numpy_forward,
    predict,
    save_bundle,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def forbidden_teacher_fit():
    raise RuntimeError("v3 is cache-only: a missing/mismatched teacher cache must fail")


def checked_targets(source, trait, eligibility):
    d, e, manifest = load_inputs(source, trait)
    rules = load_eligibility_rules(eligibility)
    if trait == "bc" and "bc" not in rules:
        raise ValueError("Female BC eligibility rule is required")
    for frame in [d, e]:
        if len(filter_eligible(frame, trait, rules)) != len(frame):
            raise ValueError("Frozen cohort differs from disease eligibility")
    split = pd.read_parquet(source / "split_manifest.parquet").set_index("sampleid").loc[d.sampleid]
    u = d.loc[split.internal_role.eq("U").to_numpy()]
    h = d.loc[split.internal_role.eq("H").to_numpy()]
    assert_boundary(u, h)
    core = source / "core" / trait
    contract = read_json(core / "teacher_contract.json")
    if contract["model_version"] != "V3" or contract["positive_label"] != 1 or contract["seed"] != 42:
        raise ValueError("Unexpected historical teacher identity")
    for package, version in contract["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Teacher cache runtime differs: {package}")
    verify_files(Path.cwd(), contract["implementation_sha256"])
    if sha256_file(contract["checkpoint"]) != contract["checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint changed")
    common = dict(contract=contract, trait=trait, manifest_hash=sha256_file(source / trait / "feature_manifest.tsv"),
                  factory=forbidden_teacher_fit)
    prs = manifest.column.tolist()
    q_u = teacher_oof(u, split.loc[u.sampleid, "u_teacher_fold"].to_numpy(), prs,
                      directory=core / "teacher_cache/U_oof", **common)
    q_h = teacher_predict(u, h.drop(columns="y"), prs, directory=core / "teacher_cache/U_to_H",
                          fold_id="U_to_H", **common)
    q_d = teacher_oof(d, split.final_teacher_fold.to_numpy(), prs, directory=core / "teacher_cache/D_oof", **common)
    q_e = teacher_predict(d, e, prs, directory=core / "teacher_cache/D_to_E", fold_id="D_to_E", **common)
    for scope, frame, q in [("U", u, q_u), ("D", d, q_d)]:
        saved = pd.read_parquet(core / "teacher_oof_predictions.parquet")
        saved = saved.loc[saved.scope.eq(scope)].set_index("sampleid").loc[frame.sampleid]
        np.testing.assert_array_equal(saved.q_teacher, q)
        np.testing.assert_array_equal(saved.y, frame.y)
    return (d, e, u, h, prs), dict(q_U=q_u, q_H=q_h, q_D=q_d, q_E=q_e)


def prepare(config_path, eligibility):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg["schema_version"] != 3 or cfg["phase_b"] or cfg["age_unit"] != 1:
        raise ValueError("Unsupported v3 protocol")
    source = Path(cfg["source_run"])
    complete = read_json(source / "COMPLETE.json")
    verify_files(source, complete["output_sha256"])
    audit = read_json(source / "audit_report.json")
    if str(eligibility) != audit["paths"]["eligibility"]:
        raise ValueError("Eligibility path differs from the approved source")
    verify_files(Path.cwd(), audit["source_sha256"])
    files = [Path(config_path), Path(eligibility), Path(cfg["specification"]), Path(__file__).resolve(),
             Path("src/decipehr/prs_distillation_v3.py"), Path("tests/test_prs_distillation_v3.py"),
             Path("slurm/prs_distillation_v3.sbatch")]
    code = {str(p.relative_to(Path.cwd()) if p.is_absolute() else p): sha256_file(p) for p in files}
    identity = {"config": cfg, "source_complete_sha256": sha256_file(source / "COMPLETE.json"), "code_sha256": code}
    root = Path(cfg["output_root"]) / f"signaware_20261026_{digest(identity)[:12]}"
    if (root / "PROTOCOL.json").exists():
        protocol = read_json(root / "PROTOCOL.json")
        if protocol["identity"] != identity:
            raise ValueError("Existing v3 identity differs")
        verify_files(root, protocol["input_sha256"])
        return root
    root.mkdir(parents=True, exist_ok=True)
    rows, priors, inputs = [], {}, {}
    for trait in cfg["traits"]:
        (d, e, u, h, _), targets = checked_targets(source, trait, eligibility)
        directory = root / "inputs" / trait
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "teacher_targets.npz", **targets)
        inputs[str((directory / "teacher_targets.npz").relative_to(root))] = sha256_file(directory / "teacher_targets.npz")
        lock = read_json(source / "core" / trait / "selection_lock.json")
        priors[trait] = {"lambda_KD": lock["arms"]["S_KD_MIX"]["lambda"],
                         "common_gate_ridge": lock["arms"]["S_KD_MIX"]["gate_ridge"]}
        rows.append({"trait": trait, "train_n": len(d), "u_n": len(u), "h_n": len(h), "val_n": len(e),
                     "teacher_scopes_verified": 12, "training_ids_hash": id_hash(d), "evaluation_ids_hash": id_hash(e),
                     "training_features_labels_hash": frame_hash(d), "evaluation_features_hash": frame_hash(e),
                     "U_ids_hash": id_hash(u), "H_ids_hash": id_hash(h), "val_labels_read": False})
        print(f"Verified {trait}: all 12 cached teacher scopes, {len(d)} train / {len(e)} val", flush=True)
    for relative in code:
        target = root / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(relative).read_bytes())
    pd.DataFrame(rows).to_csv(root / "scope_audit.csv", index=False)
    write_json(root / "PROTOCOL.json", {"identity": identity, "source_run": str(source), "priors": priors,
        "input_sha256": inputs, "source_sealed_files_verified": len(complete["output_sha256"]),
        "cohort_policy": audit["cohort_policy"], "known_cross_groups_retained": audit["known_cross_groups_retained"],
        "unresolved_kinship_retained": audit["unresolved_kinship_retained"], "age_units": "age_sd, age_unit=1",
        "phase_b": False, "val_labels_read": False, "new_teacher_fits": False,
        "gate_ridge_choice": "Source mixed-KD rho used in all v3 structures and both losses within a disease",
        "warm_start_budget": "Same 3-kappa/3-seed grid; M1/M2 have a required M0 fit plus their own stage"})
    return root


def verify_run(root):
    protocol = read_json(root / "PROTOCOL.json")
    verify_files(Path.cwd(), protocol["identity"]["code_sha256"])
    verify_files(root, protocol["input_sha256"])
    source = Path(protocol["source_run"])
    if sha256_file(source / "COMPLETE.json") != protocol["identity"]["source_complete_sha256"]:
        raise ValueError("Source run completion receipt changed")
    return protocol, protocol["identity"]["config"], source


def options_from(cfg, smoke):
    return Optimization(max_epochs=30 if smoke else cfg["max_steps"], evaluate_every=cfg["evaluate_every"],
                        patience=20 if smoke else cfg["patience"], min_delta=cfg["min_delta"],
                        learning_rate=cfg["learning_rate"], retry_learning_rate=cfg["retry_learning_rate"],
                        gradient_clip=cfg["gradient_clip"], baseline_ridge=cfg["baseline_ridge"])


def verify_model(model, prep, frame):
    x = prep.transform(frame)
    p, independent = predict(model, x), numpy_forward(model, x)
    errors = {}
    for key in independent:
        np.testing.assert_allclose(p[key], independent[key], atol=2e-11, rtol=1e-11)
        errors[key] = float(np.max(np.abs(p[key]-independent[key])))
    assert (model.d >= 0).all() and (model.age_slopes >= 0).all()
    np.testing.assert_allclose(p["pi"].sum(1), 1, atol=1e-12)
    assert (p["r"] > 0).all() and (p["pi"] >= 0).all()
    assert (p["age_delta"][x["t"] >= 0] >= -1e-10).all()
    assert (p["age_delta"][x["t"] <= 0] <= 1e-10).all()
    at_ref = copy.deepcopy(x)
    at_ref["t"][:] = 0
    at_ref["age_basis"][:] = 0
    np.testing.assert_allclose(predict(model, at_ref)["logit"], p["reference_age_logit"], atol=1e-11)
    # Fixed observed covariates, all knots, both extrapolation sides, and a dense age grid.
    ages = np.unique(np.r_[np.linspace(prep.age_boundaries[0], prep.age_boundaries[-1], 41),
                           prep.age_boundaries, prep.reference_age, prep.age_boundaries[0]-1, prep.age_boundaries[-1]+1])
    fixed = frame.iloc[:min(32, len(frame))].copy()
    previous, smallest = None, float("inf")
    for age in ages:
        fixed[AGE] = age
        current = predict(model, prep.transform(fixed))["logit"]
        if previous is not None:
            smallest = min(smallest, float(np.min(current-previous)))
        previous = current
    if smallest < -1e-10:
        raise ValueError("Observed-covariate age monotonicity failed")
    return p, {"independent_max_absolute_errors": errors, "monotonicity_min_logit_step": smallest,
               "monotonicity_profiles": len(fixed), "age_grid_points": len(ages),
               "direct_constraints_verified": True, "all_prediction_rows_verified": len(frame)}


def export_prediction(frame, x, result, prs, arm, seed, q):
    columns = {c: frame[c].to_numpy() for c in ["sampleid", "kinship_group", "hard_label", AGE]}
    columns.update(arm=arm, seed=seed, p_teacher=q)
    for key, values in result.items():
        if values.ndim == 1:
            columns[key] = values
        else:
            for k, name in enumerate(prs):
                columns[f"{key}__{name}"] = values[:, k]
    for name in prs:
        columns[f"raw_prs__{name}"] = frame[name].to_numpy()
    for key in ["age_out_of_support", "sex_out_of_support"]:
        columns[key] = x[key]
    return pd.DataFrame(columns)


def run(root, trait, smoke):
    torch.set_num_threads(1)
    protocol, cfg, source = verify_run(root)
    directory = root / ("smoke" if smoke else "core") / trait
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "PREDICTIONS_COMPLETE.json"
    if complete.exists():
        verify_files(directory, read_json(complete)["output_sha256"])
        return
    d, e, manifest = load_inputs(source, trait)
    prs = manifest.column.tolist()
    split = pd.read_parquet(source / "split_manifest.parquet").set_index("sampleid").loc[d.sampleid]
    mask = split.internal_role.eq("U").to_numpy()
    u, h = d.loc[mask], d.loc[~mask]
    assert_boundary(u, h)
    q = np.load(root / "inputs" / trait / "teacher_targets.npz")
    prep_u, prep_d = SignAwarePreprocessor.fit(u, prs), SignAwarePreprocessor.fit(d, prs)
    tx, hx, dx, ex = prep_u.transform(u), prep_u.transform(h), prep_d.transform(d), prep_d.transform(e)
    # The PRS/PC/sex transforms are exactly those used by v1 in the same fit scope.
    assert prep_d.fixed.metadata() == read_json(source / "core" / trait / "preprocessing.json")
    seeds, kappas = ([17], [.001]) if smoke else (cfg["seeds"], cfg["kappas"])
    opt = options_from(cfg, smoke)
    prior = protocol["priors"][trait]
    fingerprint = digest({"protocol": sha256_file(root / "PROTOCOL.json"), "trait": trait, "smoke": smoke,
                          "U": frame_hash(u), "H": frame_hash(h), "preprocessing": prep_u.metadata()})
    rows = []
    for mode in cfg["supervision"]:
        lam = 0 if mode == "SUP" else prior["lambda_KD"]
        for kappa in kappas:
            for seed in seeds:
                warm = None
                for variant in cfg["variants"]:
                    arm = f"{variant}_{mode}"
                    path = directory / "tuning" / f"{arm}_k{kappa:g}_seed{seed}"
                    if path.exists():
                        model, _, meta = load_bundle(path)
                        if meta["fingerprint"] != fingerprint:
                            raise ValueError("Tuning cache provenance mismatch")
                        info = meta["training"]
                    else:
                        print(f"{trait} tune {arm} kappa={kappa:g} seed={seed}", flush=True)
                        model, info = fit_student(tx, u.y.to_numpy(), q["q_U"], variant=variant, lam=lam,
                            rho=prior["common_gate_ridge"], kappa=kappa, seed=seed, options=opt,
                            tune=hx, tune_y=h.y.to_numpy(), warm=warm if variant != "M0" else None)
                        save_bundle(path, model, prep_u, {"fingerprint": fingerprint, "training": info,
                            "arm": arm, "seed": seed, "kappa": kappa, "lambda": lam,
                            "scope": "U_fit_H_selection", "fit_ids_hash": id_hash(u)})
                    if variant == "M0":
                        warm = model
                    rows.append({"arm": arm, "variant": variant, "supervision": mode, "lambda": lam,
                                 "kappa": kappa, "seed": seed, "best_step": info["best_step"],
                                 "H_log_loss": info["selection_metric"], "learning_rate": info["learning_rate"]})
    tuning = pd.DataFrame(rows)
    tuning.to_csv(directory / "tuning_results.csv", index=False)
    grouped = tuning.groupby(["arm", "variant", "supervision", "lambda", "kappa"], as_index=False).agg(
        H_log_loss=("H_log_loss", "mean"), steps=("best_step", "median"), n_seeds=("seed", "nunique"))
    assert grouped.n_seeds.eq(len(seeds)).all()
    choices = {}
    for arm, part in grouped.groupby("arm"):
        best = part.loc[part.H_log_loss <= part.H_log_loss.min()+cfg["selection_tolerance"]].sort_values("kappa", ascending=False).iloc[0]
        choices[arm] = {"kappa": float(best.kappa), "lambda": float(best["lambda"]), "steps": int(best.steps),
                        "H_log_loss": float(best.H_log_loss), "rho": prior["common_gate_ridge"]}
    policies = {}
    for mode in cfg["supervision"]:
        scores = {a: v["H_log_loss"] for a, v in choices.items() if a.endswith(mode)}
        policies[mode] = sorted(a for a, v in scores.items() if v <= min(scores.values())+cfg["selection_tolerance"])[0]
    warm_steps = {f"{r.supervision}_k{r.kappa:g}": int(r.steps) for r in grouped.loc[grouped.variant.eq("M0")].itertuples()}
    lock = {"fingerprint": fingerprint, "choices": choices, "selected_structure_by_H": policies,
            "warm_M0_steps": warm_steps, "primary_seed": 17, "assessment_labels_used": False}
    lock_path = directory / "SELECTION_LOCK.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError("Pre-existing selection lock changed")
    write_json(lock_path, lock)
    frames, audits, parameters = [], [], []
    for arm, config in choices.items():
        variant, mode = arm.split("_")
        kappa, lam = config["kappa"], config["lambda"]
        for seed in seeds:
            warm_path = directory / "warm_refits" / f"{mode}_k{kappa:g}_seed{seed}"
            wsteps = warm_steps[f"{mode}_k{kappa:g}"]
            if warm_path.exists():
                warm, _, warm_meta = load_bundle(warm_path)
            else:
                warm, info = fit_student(dx, d.y.to_numpy(), q["q_D"], variant="M0", lam=lam,
                    rho=config["rho"], kappa=kappa, seed=seed, options=opt, epochs=wsteps)
                save_bundle(warm_path, warm, prep_d, {"fingerprint": fingerprint, "training": info,
                    "scope": "D_refit", "fit_ids_hash": id_hash(d), "lambda": lam, "kappa": kappa, "seed": seed})
                warm_meta = {"training": info}
            path = directory / "models" / arm / f"seed{seed}"
            if path.exists():
                model, _, meta = load_bundle(path)
                if meta["selection_lock_sha256"] != sha256_file(lock_path):
                    raise ValueError("Refit selection lock mismatch")
            else:
                print(f"{trait} refit {arm}/seed{seed}: M0 {wsteps} + extra {0 if variant == 'M0' else config['steps']} steps", flush=True)
                model, info = (warm, warm_meta["training"]) if variant == "M0" else fit_student(
                    dx, d.y.to_numpy(), q["q_D"], variant=variant, lam=lam, rho=config["rho"],
                    kappa=kappa, seed=seed, options=opt, epochs=config["steps"], warm=warm)
                save_bundle(path, model, prep_d, {"fingerprint": fingerprint, "arm": arm, "seed": seed,
                    "training": info, "hyperparameters": config, "scope": "D_refit_E_prediction_only",
                    "fit_ids_hash": id_hash(d), "selection_lock_sha256": sha256_file(lock_path),
                    "warm_steps": wsteps, "additional_steps": 0 if variant == "M0" else config["steps"]})
            result, checked = verify_model(model, prep_d, e)
            loaded, saved_prep, _ = load_bundle(path)
            replay = predict(loaded, saved_prep.transform(e))
            for key in result:
                np.testing.assert_allclose(result[key], replay[key], atol=1e-12, rtol=1e-12)
            frames.append(export_prediction(e, ex, result, prep_d.fixed.kept_prs, arm, seed, q["q_E"]))
            audits.append({"arm": arm, "seed": seed, **checked, "save_load_parity": True})
            for k, name in enumerate(prep_d.fixed.kept_prs):
                parameters.append({"arm": arm, "seed": seed, "prs": name,
                    "r": float(model.log_r.detach().exp()[k]), "d": float(model.effective_d().detach()[k])})
    pd.concat(frames, ignore_index=True).to_parquet(directory / "direct_predictions.parquet", index=False)
    pd.DataFrame(parameters).to_csv(directory / "parameters.csv", index=False)
    write_json(directory / "MODEL_VERIFICATION.json", {"models": audits, "patient_level_outputs_local_only": True})
    write_json(directory / "PREPROCESSING.json", prep_d.metadata())
    write_json(complete, {"status": "predictions_complete_not_evaluated", "smoke": smoke,
        "n_training": len(d), "n_evaluation": len(e), "models": len(audits), "phase_b": False,
        "evaluation_labels_read": False, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_sha256": {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob("*") if p.is_file()}})
    print(f"Predictions complete: {trait}; models={len(audits)}", flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "train"])
    parser.add_argument("--config", default="configs/prs_distillation_v3.yaml")
    parser.add_argument("--eligibility-config", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--trait", choices=["T2D", "cad", "bc"])
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"run_dir": str(prepare(args.config, args.eligibility_config))}))
    else:
        if args.run_dir is None or args.trait is None:
            parser.error("Training/smoke requires --run-dir and --trait")
        protocol, _, source = verify_run(args.run_dir)
        if args.eligibility_config != read_json(source / "audit_report.json")["paths"]["eligibility"]:
            parser.error("Eligibility must match the source audit")
        if args.command == "train":
            smoke = args.run_dir / "smoke" / args.trait / "PREDICTIONS_COMPLETE.json"
            if not smoke.exists():
                parser.error("Run and verify the real-data smoke before full training")
            verify_files(smoke.parent, read_json(smoke)["output_sha256"])
        run(args.run_dir, args.trait, args.command == "smoke")


if __name__ == "__main__":
    main()
