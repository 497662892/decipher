"""Split first, cross-predict teachers, lock selection, then refit and predict E."""

from __future__ import annotations

import copy
import importlib.metadata
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import digest
from decipehr.prs_distillation.data import code_hashes, make_split_manifest, verify_files
from decipehr.prs_distillation.preprocessing import (
    Preprocessor,
    assert_boundary,
    frame_hash,
    id_hash,
)
from decipehr.prs_distillation.student import (
    StudentBundle,
    fit_student,
    load_student_bundle,
    predict,
)
from decipehr.prs_distillation.teacher import (
    make_teacher,
    teacher_contract,
    teacher_oof,
    teacher_predict,
)

CORE_ARMS = ("META", "S_SUP", "S_KD_PURE", "S_KD_MIX")
ABLATIONS = {"S_STATIC_GATE": "static_gate", "S_CONSTANT_SCALE": "constant_scale"}


def select_candidates(results, options):
    """Aggregate seeds first; fidelity selection for pure KD, true LL for other arms."""
    groups = results.groupby(["lambda", "gate_ridge"], as_index=False).agg(
        metric=("selection_metric", "mean"), true_log_loss=("true_label_log_loss", "mean"),
        refit_epochs=("best_epoch", "median"), n_seeds=("seed", "nunique"))
    if not groups.n_seeds.eq(len(options.seeds)).all():
        raise ValueError("Candidate lacks all predeclared initialization seeds")
    def choose(frame, metric):
        pool = frame.loc[frame[metric] <= frame[metric].min() + options.tie_tolerance]
        row = pool.sort_values(["lambda", "gate_ridge"], ascending=[True, False]).iloc[0]
        return {"lambda": float(row["lambda"]), "gate_ridge": float(row.gate_ridge),
                "refit_epochs": int(row.refit_epochs), "internal_mean_metric": float(row.metric),
                "internal_true_log_loss": float(row.true_log_loss)}
    choices = {
        "S_SUP": choose(groups.loc[groups["lambda"] == 0], "metric"),
        "S_KD_PURE": choose(groups.loc[groups["lambda"] == 1], "metric"),
        "S_KD_MIX": choose(groups.loc[groups["lambda"].between(0, 1, inclusive="neither")], "metric"),
    }
    # Pure KD has already been selected on its own fidelity criterion.
    policy = pd.DataFrame([dict(arm=k, **v) for k, v in choices.items()])
    pool = policy.loc[policy.internal_true_log_loss <= policy.internal_true_log_loss.min() +
                      options.tie_tolerance]
    selected = pool.sort_values(["lambda", "gate_ridge"], ascending=[True, False]).iloc[0].arm
    return choices, selected


def tune_models(train, tune, q_train, q_tune, prep, options, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tx, vx = prep.transform(train), prep.transform(tune)
    fingerprint = digest({"training": frame_hash(train), "tuning": frame_hash(tune),
                          "q_training": q_train.tolist(), "q_tuning": q_tune.tolist(),
                          "preprocessing": prep.metadata(), "options": vars(options),
                          "code": code_hashes()})
    rows, meta_rows = [], []
    candidates = [(lam, rho, seed, "full") for lam, rho, seed in itertools.product(
        options.lambdas, options.gate_ridges, options.seeds)]
    candidates += [(0, 0, seed, "metadata") for seed in options.seeds]
    for index, (lam, rho, seed, variant) in enumerate(candidates):
        name = f"{variant}_lambda{lam:g}_ridge{rho:g}_seed{seed}"
        path = out / f"{name}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Tuning cache provenance differs")
            info = saved["info"]
        else:
            print(f"Student selection {index + 1}/{len(candidates)}: {name}", flush=True)
            _, info = fit_student(tx, train.y.to_numpy(), q_train, lam=lam, gate_ridge=rho,
                                  seed=seed, options=options, tune=vx, tune_y=tune.y.to_numpy(),
                                  tune_q=q_tune, variant=variant)
            write_json(path, {"fingerprint": fingerprint, "info": info})
        row = {"lambda": lam, "gate_ridge": rho, "seed": seed,
               **{k: info[k] for k in ("best_epoch", "selection_metric", "true_label_log_loss",
                                      "metric_type", "learning_rate")}}
        (meta_rows if variant == "metadata" else rows).append(row)
    choices, selected = select_candidates(pd.DataFrame(rows), options)
    choices["META"] = {"lambda": 0, "gate_ridge": 0,
                       "refit_epochs": int(np.median([r["best_epoch"] for r in meta_rows]))}
    return choices, selected, pd.DataFrame(rows), pd.DataFrame(meta_rows)


def verify_run(root, cfg):
    import yaml

    root = Path(root)
    saved_cfg = yaml.safe_load((root / "resolved_config.yaml").read_text())
    if cfg.to_dict() != saved_cfg:
        raise ValueError("Configuration differs from frozen resolved_config")
    audit = json.loads((root / "audit_report.json").read_text())
    if audit["status"] != "prepared":
        raise ValueError(f"Real-data run blocked: {audit['blockers']}")
    env = json.loads((root / "environment.json").read_text())
    if any(importlib.metadata.version(p) != version for p, version in env["packages"].items()):
        raise ValueError("Software environment changed since preparation")
    if env["code_sha256"] != code_hashes():
        raise ValueError("Implementation changed since preparation; prepare a fresh run")
    verify_files(Path.cwd(), audit["source_sha256"])
    saved = json.loads((root / "split_receipt.json").read_text())
    if sha256_file(root / "split_manifest.parquet") != saved["sha256"]:
        raise ValueError("Frozen split manifest changed")
    return audit, env


def load_inputs(root, trait):
    out = Path(root) / trait
    receipt = json.loads((out / "input_receipt.json").read_text())
    verify_files(out, {p: h for p, h in receipt.items() if p.endswith((".parquet", ".tsv"))})
    d, e = pd.read_parquet(out / "training.parquet"), pd.read_parquet(out / "evaluation_features.parquet")
    if "y" in e:
        raise ValueError("Evaluation features must not include outcomes")
    audit = json.loads((Path(root) / "audit_report.json").read_text())
    assert_boundary(d, e, require_group_disjoint=audit["cohort_policy"] != "original_full_sensitivity")
    manifest = pd.read_csv(out / "feature_manifest.tsv", sep="\t", keep_default_na=False)
    return d, e, manifest


def group_subset(frame, target, seed):
    rng = np.random.default_rng(seed)
    groups = rng.permutation(frame.kinship_group.unique())
    sizes = frame.groupby("kinship_group").size()
    selected, total = [], 0
    for group in groups:
        selected.append(group)
        total += sizes[group]
        if total >= target:
            break
    return frame.loc[frame.kinship_group.isin(selected)].reset_index(drop=True)


def export_prediction(frame, result, prep, arm, seed, teacher_q):
    output = frame[["sampleid", "kinship_group", "hard_label", "age_sd"]].copy()
    output = output.assign(arm=arm, seed=seed, p_teacher=teacher_q)
    for key, values in result.items():
        if values.ndim == 1:
            output[key] = values
    for key, prefix in (("weights", "pi"), ("effective_coefficients", "beta"),
                        ("contributions", "contribution")):
        if key in result:
            for k, name in enumerate(prep.kept_prs):
                output[f"{prefix}__{name}"] = result[key][:, k]
    transformed = prep.transform(frame)
    output["age_out_of_support"] = transformed["age_out_of_support"]
    output["sex_out_of_support"] = transformed["sex_out_of_support"]
    return output


def refit_arms(out, d, e, q_d, q_e, prs, lock, options, arms, manifest):
    prep = Preprocessor.fit(d, prs)
    tx, ex = prep.transform(d), prep.transform(e)
    frames = []
    for arm in arms:
        config = lock["arms"]["S_KD_MIX" if arm in ABLATIONS else arm]
        variant = ABLATIONS.get(arm, "metadata" if arm == "META" else "full")
        for seed in options.seeds:
            model_dir = out / "models" / arm / f"seed{seed}"
            metadata = {"arm": arm, "seed": seed, "selection_lock_hash": digest(lock),
                        "hyperparameters": config, "feature_manifest": manifest.to_dict("records"),
                        "training_ids_hash": id_hash(d), "training_features_hash": frame_hash(d),
                        "phase": "A_direct", "primary": seed == 17}
            if model_dir.exists():
                bundle = load_student_bundle(model_dir)
                if any(bundle.metadata.get(k) != v for k, v in metadata.items()):
                    raise ValueError("Existing refit bundle provenance differs")
                model = bundle.model
            else:
                print(f"Refit {arm}/seed{seed}: {config['refit_epochs']} locked epochs", flush=True)
                model, info = fit_student(
                    tx, d.y.to_numpy(), q_d, lam=config["lambda"], gate_ridge=config["gate_ridge"],
                    seed=seed, options=options, epochs=config["refit_epochs"], variant=variant,
                )
                StudentBundle(model, prep, {**metadata, "training": info}).save(model_dir)
            result = predict(model, ex)
            # Verify serialized inference, including every exact component.
            replay = load_student_bundle(model_dir).predict(e)
            for key in result:
                np.testing.assert_allclose(result[key], replay[key], atol=1e-12, rtol=1e-12)
            frames.append(export_prediction(e, result, prep, arm, seed, q_e))
    return pd.concat(frames, ignore_index=True), prep


def run_core(root, cfg, trait, *, smoke=False):
    """This function never reads E.y. Reporting is a separate command/function."""
    torch.set_num_threads(1)
    audit, env = verify_run(root, cfg)
    d, e, manifest = load_inputs(root, trait)
    prs = manifest.column.tolist()
    out = Path(root) / ("smoke" if smoke else "core") / trait
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PREDICTIONS_COMPLETE.json").exists():
        complete = json.loads((out / "PREDICTIONS_COMPLETE.json").read_text())
        verify_files(out, complete["output_sha256"])
        return out
    options = copy.deepcopy(cfg.optimization)
    if smoke:
        d, e = group_subset(d, 256, 1717), group_subset(e, 64, 2929)
        splits, _ = make_split_manifest(
            d, e, cfg.split_seed,
            allow_evaluation_kinship_overlap=cfg.cohort_policy == "original_full_sensitivity",
        )
        options.max_epochs, options.patience = 40, 20
        splits.to_parquet(out / "split_manifest.parquet", index=False)
    else:
        splits = pd.read_parquet(Path(root) / "split_manifest.parquet")
    by_id = splits.set_index("sampleid").loc[d.sampleid]
    u_mask = by_id.internal_role.eq("U").to_numpy()
    u, h = d.loc[u_mask], d.loc[~u_mask]
    assert_boundary(u, h)
    contract = teacher_contract(cfg)
    write_json(out / "teacher_contract.json", contract)
    manifest_hash = sha256_file(Path(root) / trait / "feature_manifest.tsv")
    common = dict(contract=contract, trait=trait, manifest_hash=manifest_hash,
                  factory=lambda: make_teacher(cfg))
    q_u = teacher_oof(u, by_id.loc[u.sampleid, "u_teacher_fold"].to_numpy(), prs,
                      directory=out / "teacher_cache" / "U_oof", **common)
    q_h = teacher_predict(u, h.drop(columns="y"), prs, directory=out / "teacher_cache" / "U_to_H",
                          fold_id="U_to_H", **common)
    prep_u = Preprocessor.fit(u, prs)
    lock_path = out / "selection_lock.json"
    lock_base = {"trait": trait, "feature_manifest_hash": manifest_hash,
                 "input_scaling_hash": digest(prep_u.metadata()), "split_hash": frame_hash(splits),
                 "student_structure": "b(age,A,C)+exp(u(age))*sum(softmax(WA+c)*Z)",
                 "final_seed": 17, "teacher_config_hash": digest(contract), "smoke": smoke,
                 "age_basis": "3 quantile knots, degree2, centered, constant extrapolation",
                 "subgroups": "existing ancestry labels; D age quartiles; min N100/events20/non-events20",
                 "code_commit": env["git_commit"], "code_sha256": env["code_sha256"],
                 "options": vars(options), "cohort_policy": cfg.cohort_policy}
    if lock_path.exists():
        if (out / "selection_lock.sha256").read_text().strip() != sha256_file(lock_path):
            raise ValueError("Selection lock checksum changed")
        lock = json.loads(lock_path.read_text())
        if any(lock.get(k) != v for k, v in lock_base.items()):
            raise ValueError("Selection lock scope changed")
    else:
        choices, policy, tuning, meta_tuning = tune_models(u, h, q_u, q_h, prep_u, options,
                                                        out / "training_logs")
        tuning.to_csv(out / "tuning_results.csv", index=False)
        meta_tuning.to_csv(out / "metadata_tuning_results.csv", index=False)
        lock = {**lock_base, "arms": choices, "SELECTED_POLICY": policy}
        write_json(lock_path, lock)
        (out / "selection_lock.sha256").write_text(sha256_file(lock_path) + "\n")
    # D soft targets are regenerated in D, never spliced from U/H caches.
    q_d = teacher_oof(d, by_id.final_teacher_fold.to_numpy(), prs,
                      directory=out / "teacher_cache" / "D_oof", **common)
    q_e = teacher_predict(d, e, prs, directory=out / "teacher_cache" / "D_to_E",
                          fold_id="D_to_E", **common)
    oof_frames = []
    oof_metrics = []
    from decipehr.prs_distillation.reporting import basic_metrics
    for scope, frame, q, fold in [
        ("U", u, q_u, by_id.loc[u.sampleid, "u_teacher_fold"].to_numpy()),
        ("D", d, q_d, by_id.final_teacher_fold.to_numpy()),
    ]:
        oof_frames.append(frame[["sampleid", "kinship_group", "y"]].assign(
            scope=scope, fold_id=fold, q_teacher=q))
        oof_metrics.append({"scope": scope, **basic_metrics(frame.y.to_numpy(), q),
                            "use": "training-only pooled teacher OOF diagnostic; not selection"})
    pd.concat(oof_frames, ignore_index=True).to_parquet(out / "teacher_oof_predictions.parquet", index=False)
    pd.DataFrame(oof_metrics).to_csv(out / "teacher_oof_metrics.csv", index=False)
    predictions, prep = refit_arms(out, d, e, q_d, q_e, prs, lock, options, CORE_ARMS, manifest)
    teacher = e[["sampleid", "kinship_group", "hard_label", "age_sd"]].copy().assign(
        arm="T_MATCHED", seed=17, probability=q_e, p_teacher=q_e)
    predictions = pd.concat([predictions, teacher], ignore_index=True)
    predictions.to_parquet(out / "direct_predictions.parquet", index=False)
    pd.DataFrame(np.atleast_2d(np.corrcoef(d[prep.kept_prs].to_numpy().T)), index=prep.kept_prs,
                 columns=prep.kept_prs).to_csv(out / "prs_correlation.csv")
    write_json(out / "preprocessing.json", prep.metadata())
    write_json(out / "teacher_cache_manifest.json", {
        str(p.relative_to(out)): sha256_file(p) for p in sorted((out / "teacher_cache").rglob("cache.json"))})
    write_json(out / "PREDICTIONS_COMPLETE.json", {
        "status": "predictions_complete_not_yet_evaluated", "smoke": smoke,
        "n_training": len(d), "n_evaluation": len(e), "n_prs": len(prs),
        "evaluation_labels_read": False, "blind_labels_read": False, "phase_b_executed": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
    })
    return out


def run_ablations(root, cfg, trait):
    torch.set_num_threads(1)
    verify_run(root, cfg)
    out = Path(root) / "core" / trait
    complete = json.loads((out / "PREDICTIONS_COMPLETE.json").read_text())
    verify_files(out, complete["output_sha256"])
    if (out / "ABLATIONS_COMPLETE.json").exists():
        verify_files(out, json.loads((out / "ABLATIONS_COMPLETE.json").read_text())["output_sha256"])
        return out
    d, e, manifest = load_inputs(root, trait)
    lock = json.loads((out / "selection_lock.json").read_text())
    cache_manifest = json.loads((out / "teacher_cache_manifest.json").read_text())
    verify_files(out, cache_manifest)
    for relative in cache_manifest:
        record = json.loads((out / relative).read_text())
        verify_files((out / relative).parent, record["output_sha256"])
    saved_q = pd.concat([pd.read_parquet(p) for p in sorted(
        (out / "teacher_cache" / "D_oof").glob("*/predictions.parquet"))])
    q_d = saved_q.set_index("sampleid").loc[d.sampleid].q_teacher.to_numpy()
    q_e = pd.read_parquet(out / "teacher_cache" / "D_to_E" / "predictions.parquet").q_teacher.to_numpy()
    predictions, _ = refit_arms(out, d, e, q_d, q_e, manifest.column.tolist(), lock,
                               cfg.optimization, ABLATIONS, manifest)
    predictions.to_parquet(out / "ablation_predictions.parquet", index=False)
    write_json(out / "ABLATIONS_COMPLETE.json", {
        "status": "complete", "configuration": "locked S_KD_MIX hyperparameters and epochs",
        "output_sha256": {"ablation_predictions.parquet": sha256_file(out / "ablation_predictions.parquet")},
    })
    return out
