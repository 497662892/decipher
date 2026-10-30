"""Secondary, matched train-only ablation of age versus per-PRS monotonicity."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from run_prs_distillation_bio import verify as verify_parent

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_constraint_ablation import AblationFeatures, verify_constraints
from decipehr.prs_distillation.config import digest
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import assert_boundary, frame_hash
from decipehr.prs_monotone_boosting import (
    bundle_model,
    explain,
    fit_booster,
    predict,
    raw_predict,
    select_iteration,
)


def read(path):
    return json.loads(Path(path).read_text())


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trait", choices=["T2D", "cad", "bc"], required=True)
    parser.add_argument("--constraint", choices=["AGE", "PRS", "FREE"], required=True)
    parser.add_argument("--eligibility-config", required=True)
    args = parser.parse_args()
    protocol, cfg = verify_parent(args.run_dir)
    assert args.eligibility_config == protocol["eligibility"]
    source, trait = Path(cfg["source_run"]), args.trait
    d, e, manifest = load_inputs(source, trait)
    rules = load_eligibility_rules(args.eligibility_config)
    for frame in [d, e]:
        assert len(filter_eligible(frame, trait, rules)) == len(frame)
    split = pd.read_parquet(source / "split_manifest.parquet").set_index("sampleid").loc[d.sampleid]
    u, h = d.loc[split.internal_role.eq("U").to_numpy()], d.loc[split.internal_role.eq("H").to_numpy()]
    assert_boundary(u, h)
    targets_path = args.run_dir / "inputs" / trait / "targets.npz"
    targets = np.load(targets_path)
    out = args.run_dir / "constraint_ablation" / args.constraint / trait
    out.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__).resolve().relative_to(Path.cwd()),
        Path("src/decipehr/prs_constraint_ablation.py"), Path("src/decipehr/prs_monotone_boosting.py")]
    contract = {"parent_protocol_sha256": sha256_file(args.run_dir / "PROTOCOL.json"),
        "constraint": args.constraint, "code_sha256": {str(p): sha256_file(p) for p in source_files},
        "targets_sha256": sha256_file(targets_path), "train_hash": frame_hash(d),
        "evaluation_features_hash": frame_hash(e), "evaluation_labels_read": False,
        "phase": "secondary exploratory ablation motivated after primary val and train-internal teacher probes",
        "matched_recipe": "same HGB GAM/PAIR graph, preprocessing, soft BCE, 40-candidate grid and step selection as primary strict HGB",
        "primary_BIO_POLICY_unchanged": True}
    if (out / "CONTRACT.json").exists():
        assert read(out / "CONTRACT.json") == contract
    write_json(out / "CONTRACT.json", contract)
    if (out / "COMPLETE.json").exists():
        verify_files(out, read(out / "COMPLETE.json")["output_sha256"])
        return
    features = AblationFeatures.fit(u, manifest.column.tolist())
    features.constraint = args.constraint
    x, hx = features.transform(u), features.transform(h)
    rows, started = [], time.monotonic()
    for family, lam, leaves, l2 in itertools.product(["GAM", "PAIR"], [0, .25, .5, .75, 1], [4, 8], [1., 10.]):
        target = (1-lam)*u.y.to_numpy()+lam*targets["q_U"]
        name = f"{family}_lambda{lam:g}_leaves{leaves}_l2{l2:g}"
        path = out / "tuning" / f"{name}.json"
        if path.exists():
            record = read(path)
            assert record["contract_digest"] == digest(contract)
        else:
            model = fit_booster(x, target, features, family=family, leaves=leaves, l2=l2)
            chosen, curve = select_iteration(model, hx, h.y.to_numpy(), target.mean())
            record = {"contract_digest": digest(contract), "family": family, "lambda": lam,
                      "leaves": leaves, "l2": l2, **chosen, "curve": curve}
            write_json(path, record)
        rows.append({k: v for k, v in record.items() if k not in ["curve", "contract_digest"]} | {"candidate": name})
    table = pd.DataFrame(rows)
    table.to_csv(out / "tuning_results.csv", index=False)
    choices = {}
    for family, mode in itertools.product(["GAM", "PAIR"], ["SUP", "KD"]):
        pool = table.loc[table.family.eq(family) & (table["lambda"].eq(0) if mode == "SUP" else table["lambda"].gt(0))]
        pool = pool.loc[pool.H_log_loss <= pool.H_log_loss.min()+1e-4]
        chosen = pool.sort_values(["leaves", "iterations", "lambda", "l2"], ascending=[True, True, True, False]).iloc[0].to_dict()
        choices[f"HGB_{args.constraint}_{family}_{mode}"] = chosen
    best = min(c["H_log_loss"] for c in choices.values())
    pool = [a for a, c in choices.items() if c["H_log_loss"] <= best+1e-4]
    selected = min(pool, key=lambda a: (choices[a]["family"] != "GAM", choices[a]["leaves"], choices[a]["iterations"], choices[a]["lambda"]))
    lock = {"constraint": args.constraint, "choices": choices, "selected_arm": selected,
        "assessment_used_for_selection": False, "patient_n_U": len(u), "patient_n_D": len(d),
        "binary_rows_D": 2*len(d), "seed": 17, "candidate_count": 40}
    lock_path = out / "SELECTION_LOCK.json"
    if lock_path.exists():
        assert read(lock_path) == lock
    write_json(lock_path, lock)
    features = AblationFeatures.fit(d, manifest.column.tolist())
    features.constraint = args.constraint
    dx, ex = features.transform(d), features.transform(e)
    predictions, checks = [], []
    for arm, choice in choices.items():
        path = out / "models" / arm
        path.mkdir(parents=True, exist_ok=True)
        metadata = {"choice": choice, "lock_sha256": sha256_file(lock_path)}
        target = (1-choice["lambda"])*d.y.to_numpy()+choice["lambda"]*targets["q_D"]
        if (path / "model.joblib").exists():
            bundle = joblib.load(path / "model.joblib")
            assert bundle["metadata"] == metadata
        else:
            model = fit_booster(dx, target, features, family=choice["family"], leaves=choice["leaves"],
                               l2=choice["l2"], iterations=choice["iterations"])
            bundle = bundle_model(model, features, choice["family"], choice["iterations"], target.mean(), metadata)
            joblib.dump(bundle, path / "model.joblib")
        audit = verify_constraints(bundle, ex)
        probability, eta = predict(bundle, ex), raw_predict(bundle, ex)
        np.testing.assert_array_equal(probability, predict(joblib.load(path / "model.joblib"), ex))
        ref = ex.copy()
        ref[:, features.k] = 0
        age_delta = eta-raw_predict(bundle, ref)
        if args.constraint == "AGE":
            assert (age_delta[ex[:, features.k] >= 0] >= -1e-10).all()
            assert (age_delta[ex[:, features.k] <= 0] <= 1e-10).all()
        predictions.append(e[["sampleid", "kinship_group", "age_sd", "hard_label"]].assign(
            arm=arm, seed=17, probability=probability, logit=eta, age_delta=age_delta, p_teacher=targets["q_E"]))
        if arm == selected:
            intercept, first, second = explain(bundle, ex)
            names = [f"main__{n}" for n in features.names]
            names += [f"pair__{features.names[i]}__{features.names[j]}" for i, j in bundle["active_pairs"]]
            explanation = pd.DataFrame(np.column_stack([first, second]), columns=names)
            explanation.insert(0, "sampleid", e.sampleid.to_numpy())
            explanation["reference_logit"], explanation["logit"] = intercept, eta
            explanation.to_parquet(out / "selected_exact_explanations.parquet", index=False)
        checks.append({"arm": arm, **audit, "saved_replay_exact": True})
        write_json(path / "FIT.json", {**metadata, **audit, "model_sha256": sha256_file(path / "model.joblib")})
    pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    write_json(out / "COMPLETE.json", {"status": "locked_secondary_predictions_complete", "models": 4,
        "evaluation_labels_read": False, "model_verification": checks, "elapsed_seconds": time.monotonic()-started,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*") if p.is_file()}})
    print(f"{args.run_dir.name}/{trait}/{args.constraint}: selected {selected}; H={choices[selected]['H_log_loss']:.6f}", flush=True)


if __name__ == "__main__":
    main()
