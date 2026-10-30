"""Frozen-scope biological-prior exploration; select inside train, lock, then predict."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from run_prs_distillation_v3 import checked_targets, read_json
from run_prs_distillation_v3 import verify_run as verify_v3

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import digest
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import frame_hash, id_hash
from decipehr.prs_distillation_bio import BioPreprocessor, fit, load, numpy_predict, predict, save


def prepare(config_path, eligibility):
    cfg = yaml.safe_load(Path(config_path).read_text())
    v3 = Path(cfg["v3_run"])
    verify_v3(v3)
    verify_files(v3, read_json(v3 / "COMPLETE.json")["output_sha256"])
    source = Path(cfg["source_run"])
    verify_files(source, read_json(source / "COMPLETE.json")["output_sha256"])
    files = [Path(config_path), Path(eligibility), Path(__file__), Path("src/decipehr/prs_distillation_bio.py"),
             Path("tests/test_prs_distillation_bio.py"), Path("slurm/prs_distillation_bio.sbatch")]
    code = {str(p.resolve().relative_to(Path.cwd())): sha256_file(p) for p in files}
    identity = {"config": cfg, "code_sha256": code, "v3_complete_sha256": sha256_file(v3 / "COMPLETE.json")}
    root = Path(cfg["output_root"]) / f"exploration_20261026_{digest(identity)[:12]}"
    if (root / "PROTOCOL.json").exists():
        verify(root)
        return root
    root.mkdir(parents=True, exist_ok=True)
    inputs, scopes = {}, []
    for trait in cfg["traits"]:
        (d, e, u, h, _), targets = checked_targets(source, trait, eligibility)
        out = root / "inputs" / trait
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "targets.npz", **targets)
        inputs[str((out / "targets.npz").relative_to(root))] = sha256_file(out / "targets.npz")
        scopes.append({"trait": trait, "D_n": len(d), "U_n": len(u), "H_n": len(h), "E_n": len(e),
                       "D_hash": frame_hash(d), "E_hash": frame_hash(e), "U_ids_hash": id_hash(u),
                       "H_ids_hash": id_hash(h), "teacher_scopes_verified": 12})
        print(f"Verified {trait} frozen teacher targets", flush=True)
    for filename in code:
        dest = root / "source_snapshot" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(filename).read_bytes())
    pd.DataFrame(scopes).to_csv(root / "scope_audit.csv", index=False)
    source_protocol = read_json(v3 / "PROTOCOL.json")
    write_json(root / "PROTOCOL.json", {"identity": identity, "input_sha256": inputs,
        "eligibility": eligibility, "priors": source_protocol["priors"], "evaluation_labels_read": False,
        "cohort_policy": source_protocol["cohort_policy"], "known_cross_groups_retained": source_protocol["known_cross_groups_retained"],
        "unresolved_kinship_retained": source_protocol["unresolved_kinship_retained"],
        "stage1_candidates_per_trait": 180, "stage2_candidates_per_trait": 120,
        "scope": "train-only development; existing repeatedly used val is developmental",
        "new_teacher_fits": False, "age_units": "age_sd, not years", "phase_b": False})
    return root


def verify(root):
    protocol = read_json(root / "PROTOCOL.json")
    verify_files(Path.cwd(), protocol["identity"]["code_sha256"])
    verify_files(root, protocol["input_sha256"])
    cfg = protocol["identity"]["config"]
    if sha256_file(Path(cfg["v3_run"]) / "COMPLETE.json") != protocol["identity"]["v3_complete_sha256"]:
        raise ValueError("Frozen v3 reference changed")
    return protocol, cfg


def choose(rows, tol):
    groups = pd.DataFrame(rows).groupby("candidate", as_index=False).agg(
        loss=("loss", "mean"), steps=("steps", "median"), seeds=("seed", "nunique"))
    if not groups.seeds.eq(3).all():
        raise ValueError("Incomplete three-seed candidate")
    eligible = groups.loc[groups.loss <= groups.loss.min()+tol]
    definitions = {r["candidate"]: r["settings"] for r in rows}
    def rank(name):
        x = definitions[name]
        return (x["lam"], -x["kappa_r"], -x["kappa_d"], name)
    name = min(eligible.candidate, key=rank)
    score = groups.set_index("candidate").loc[name]
    return {"candidate": name, "settings": definitions[name], "H_loss": float(score.loss), "steps": int(score.steps)}


def verification(model, prep, frame):
    x = prep.transform(frame)
    result = predict(model, x)
    independent = numpy_predict(model, x)
    for key, value in independent.items():
        np.testing.assert_allclose(result[key], value, atol=2e-10, rtol=1e-10)
    assert (result["d"] >= 0).all() and (result["r"] > 0).all()
    assert (model.shape_slopes() > 0).all() and (model.age_slopes >= 0).all()
    np.testing.assert_allclose(result["pi"].sum(1), 1, atol=1e-12)
    np.testing.assert_allclose(result["reference_age_logit"]+result["age_delta"], result["logit"], atol=1e-11)
    assert (result["age_delta"][x["t"] >= 0] >= -1e-10).all()
    assert (result["age_delta"][x["t"] <= 0] <= 1e-10).all()
    grid = np.unique(np.r_[prep.age.age_boundaries[0]-1, prep.age.age_boundaries,
                           prep.age.reference_age, np.linspace(*prep.age.age_boundaries[[0, -1]], 41),
                           prep.age.age_boundaries[-1]+1])
    profiles = frame.iloc[np.linspace(0, len(frame)-1, 24).astype(int)]
    sweep = profiles.loc[profiles.index.repeat(len(grid))].reset_index(drop=True)
    sweep.age_sd = np.tile(grid, len(profiles))
    check = predict(model, prep.transform(sweep))
    age_diff = np.diff(check["logit"].reshape(len(profiles), -1), axis=1)
    assert age_diff.min() >= -1e-10
    return result, {"independent_probability_max_error": float(np.max(np.abs(result["probability"]-independent["probability"]))),
                    "minimum_age_logit_difference": float(age_diff.min()), "age_profiles": len(profiles),
                    "age_grid_points": len(grid), "constraints_passed": True}


def run(root, trait, smoke=False):
    protocol, cfg = verify(root)
    torch.set_num_threads(1)
    source = Path(cfg["source_run"])
    d, e, manifest = load_inputs(source, trait)
    split = pd.read_parquet(source / "split_manifest.parquet").set_index("sampleid").loc[d.sampleid]
    u, h = d.loc[split.internal_role.eq("U").to_numpy()], d.loc[split.internal_role.eq("H").to_numpy()]
    q = np.load(root / "inputs" / trait / "targets.npz")
    prs, prior = manifest.column.tolist(), protocol["priors"][trait]
    out = root / ("smoke" if smoke else "core") / trait
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PREDICTIONS_COMPLETE.json").exists():
        verify_files(out, read_json(out / "PREDICTIONS_COMPLETE.json")["output_sha256"])
        return
    preps = {j: BioPreprocessor.fit(u, prs, j) for j in [3, 6]}
    ux, hx = {j: p.transform(u) for j, p in preps.items()}, {j: p.transform(h) for j, p in preps.items()}
    fixed = dict(rho=prior["common_gate_ridge"], kappa_shape=cfg["shape_penalty"], baseline_ridge=cfg["baseline_ridge"],
                 max_steps=20 if smoke else cfg["max_steps"], patience=10 if smoke else cfg["patience"],
                 evaluate_every=cfg["evaluate_every"])
    lambdas = [0, prior["lambda_KD"]] if smoke else cfg["lambdas"]
    kappas = [.001] if smoke else cfg["kappas_r"]
    rows, choices, candidate_defs = [], {}, {}
    started = time.monotonic()

    def tuned(candidate, settings, family, seed, warm=None):
        path = out / "tuning" / f"{candidate}_seed{seed}"
        segments = cfg["families"][family]["age_segments"]
        signature = digest({"settings": settings, "family": family, "seed": seed,
                            "U": frame_hash(u), "H": frame_hash(h), "protocol": protocol["identity"]})
        if path.exists():
            model, _, meta = load(path)
            if meta["fingerprint"] != signature:
                raise ValueError("Tuning cache identity mismatch")
        else:
            model, info = fit(ux[segments], u.y.to_numpy(), q["q_U"], **fixed, **settings,
                seed=seed, tune=hx[segments], tune_y=h.y.to_numpy(), warm=warm)
            meta = {"fingerprint": signature, "settings": settings, "family": family, "seed": seed, "training": info}
            save(path, model, preps[segments], meta)
        row = {"candidate": candidate, "family": family, "seed": seed, "loss": meta["training"]["selection_loss"],
               "steps": meta["training"]["best_step"], "settings": settings}
        rows.append(row)
        candidate_defs[candidate] = dict(family=family, settings=settings)
        return model, row

    for family, definition in cfg["families"].items():
        family_rows = []
        for lam in lambdas:
            for kr in kappas:
                name = f"{family}_l{lam:g}_r{kr:g}"
                settings = dict(variant="M0", shape=definition["shape"], lam=lam, kappa_r=kr,
                                kappa_d=0., learning_rate=cfg["initial_learning_rate"])
                for seed in cfg["seeds"]:
                    _, row = tuned(name, settings, family, seed)
                    family_rows.append(row)
                print(f"{trait} tuned {name}; elapsed={time.monotonic()-started:.0f}s", flush=True)
        for mode in ["SUP", "KD"]:
            selected = [r for r in family_rows if (r["settings"]["lam"] == 0) == (mode == "SUP")]
            choices[f"{family}_{mode}"] = {**choose(selected, cfg["selection_tolerance"]), "family": family, "parent": None}

    for family in cfg["interaction_parents"]:
        for mode in ["SUP", "KD"]:
            parent = choices[f"{family}_{mode}"]
            for kind, definition in cfg["interactions"].items():
                interaction_rows = []
                for kd in [0.] if kind == "CONTINUE" else ([.001] if smoke else cfg["kappas_d"]):
                    name = f"{family}_{mode}_{kind}_d{kd:g}"
                    settings = {**parent["settings"], **definition, "kappa_d": kd, "learning_rate": cfg["interaction_learning_rate"]}
                    for seed in cfg["seeds"]:
                        warm, _, _ = load(out / "tuning" / f"{parent['candidate']}_seed{seed}")
                        _, row = tuned(name, settings, family, seed, warm)
                        interaction_rows.append(row)
                    print(f"{trait} tuned {name}; elapsed={time.monotonic()-started:.0f}s", flush=True)
                choices[f"{family}_{mode}_{kind}"] = {**choose(interaction_rows, cfg["selection_tolerance"]),
                                                       "family": family, "parent": f"{family}_{mode}"}
    table = pd.DataFrame([{k: v for k, v in row.items() if k != "settings"} | row["settings"] for row in rows])
    table.to_csv(out / "tuning_results.csv", index=False)
    # Complexity preference is applied only inside the predeclared loss tolerance.
    def complexity(arm):
        c = choices[arm]
        s, f = c["settings"], cfg["families"][c["family"]]
        extra_parameters = 4*len(prs)*int(f["shape"])+f["age_segments"]
        extra_parameters += {"M0": 0, "M1": 1, "M2": len(prs)}[s["variant"]]
        return (extra_parameters, s["lam"], -s["kappa_r"], -s["kappa_d"], arm)
    best = min(x["H_loss"] for x in choices.values())
    selected = min([a for a, c in choices.items() if c["H_loss"] <= best+cfg["selection_tolerance"]], key=complexity)
    lock = {"choices": choices, "selected_arm": selected, "selection_uses_val": False, "primary_seed": 17,
            "candidate_count": len(candidate_defs), "tuning_fit_count": len(rows), "step_zero_eligible": True}
    lock_path = out / "SELECTION_LOCK.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError("Selection lock changed")
    write_json(lock_path, lock)
    dpreps = {j: BioPreprocessor.fit(d, prs, j) for j in [3, 6]}
    dx = {j: p.transform(d) for j, p in dpreps.items()}
    predictions, checks, parameters = [], [], []
    for arm, choice in choices.items():
        segments = cfg["families"][choice["family"]]["age_segments"]
        prep = dpreps[segments]
        for seed in cfg["seeds"]:
            path = out / "models" / arm / f"seed{seed}"
            if path.exists():
                model, _, meta = load(path)
                if meta["selection_lock_sha256"] != sha256_file(lock_path):
                    raise ValueError("Final model lock mismatch")
            else:
                warm = None
                if choice["parent"]:
                    warm, _, _ = load(out / "models" / choice["parent"] / f"seed{seed}")
                model, info = fit(dx[segments], d.y.to_numpy(), q["q_D"], **fixed, **choice["settings"],
                                   seed=seed, fixed_steps=choice["steps"], warm=warm)
                meta = {"arm": arm, "seed": seed, "choice": choice, "training": info,
                        "selection_lock_sha256": sha256_file(lock_path), "D_ids_hash": id_hash(d)}
                save(path, model, prep, meta)
            result, checked = verification(model, prep, e)
            restored, prep2, _ = load(path)
            np.testing.assert_array_equal(result["probability"], predict(restored, prep2.transform(e))["probability"])
            frame = e[["sampleid", "kinship_group", "age_sd", "hard_label"]].copy()
            frame = frame.assign(arm=arm, seed=seed, p_teacher=q["q_E"])
            for key in ["probability", "logit", "base_reference_logit", "baseline_age", "reference_age_logit", "age_delta"]:
                frame[key] = result[key]
            for key in ["pi", "r", "d", "fixed_prs", "shaped_prs", "transformed_prs", "age_multiplier", "contributions", "reference_contributions"]:
                block = pd.DataFrame(result[key], columns=[f"{key}__{p}" for p in prep.age.fixed.kept_prs], index=frame.index)
                frame = pd.concat([frame, block], axis=1)
            frame["age_out_of_support"] = prep.transform(e)["age_out_of_support"]
            predictions.append(frame)
            checks.append({"arm": arm, "seed": seed, **checked, "save_load_exact": True})
            for k, prs_name in enumerate(prep.age.fixed.kept_prs):
                parameters.append({"arm": arm, "seed": seed, "prs": prs_name,
                    "r": float(model.log_r.detach().exp()[k]), "d": float(model.effective_d().detach()[k]),
                    **{f"slope_{j}": float(model.shape_slopes().detach()[k, j]) for j in range(5)}})
        print(f"{trait} refit {arm} complete", flush=True)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "direct_predictions.parquet", index=False)
    pd.DataFrame(parameters).to_csv(out / "parameters.csv", index=False)
    write_json(out / "MODEL_VERIFICATION.json", {"models": checks})
    write_json(out / "PREDICTIONS_COMPLETE.json", {"status": "predictions_complete_not_evaluated", "smoke": smoke,
        "models": len(checks), "selected_arm": selected, "evaluation_labels_read": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "elapsed_seconds": time.monotonic()-started,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*") if p.is_file()}})


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "smoke", "train"])
    p.add_argument("--config", default="configs/prs_distillation_bio.yaml")
    p.add_argument("--eligibility-config", required=True)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--trait", choices=["T2D", "cad", "bc"])
    a = p.parse_args()
    if a.command == "prepare":
        print(json.dumps({"run_dir": str(prepare(a.config, a.eligibility_config))}))
    else:
        if a.run_dir is None or a.trait is None:
            p.error("run-dir and trait are required")
        protocol, _ = verify(a.run_dir)
        if protocol["eligibility"] != a.eligibility_config:
            p.error("Eligibility mismatch")
        if a.command == "train":
            smoke = a.run_dir / "smoke" / a.trait / "PREDICTIONS_COMPLETE.json"
            verify_files(smoke.parent, read_json(smoke)["output_sha256"])
        run(a.run_dir, a.trait, a.command == "smoke")


if __name__ == "__main__":
    main()
