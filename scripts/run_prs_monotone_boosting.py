"""Train-only selection of monotone GAM / biological-pair distillation students."""

from __future__ import annotations

import argparse
import importlib.metadata
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
from decipehr.prs_distillation.config import digest
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import assert_boundary, frame_hash
from decipehr.prs_monotone_boosting import (
    Features,
    bundle_model,
    explain,
    fit_booster,
    predict,
    raw_predict,
    select_iteration,
    verify,
)


def read(path):
    return json.loads(Path(path).read_text())


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trait", choices=["T2D", "cad", "bc"], required=True)
    parser.add_argument("--eligibility-config", required=True)
    args = parser.parse_args()
    protocol, cfg = verify_parent(args.run_dir)
    if args.eligibility_config != protocol["eligibility"]:
        raise ValueError("Eligibility mismatch")
    source, trait = Path(cfg["source_run"]), args.trait
    d, e, manifest = load_inputs(source, trait)
    rules = load_eligibility_rules(args.eligibility_config)
    for frame in [d, e]:
        if len(filter_eligible(frame, trait, rules)) != len(frame):
            raise ValueError("Cohort eligibility mismatch")
    split = pd.read_parquet(source / "split_manifest.parquet").set_index("sampleid").loc[d.sampleid]
    u, h = d.loc[split.internal_role.eq("U").to_numpy()], d.loc[split.internal_role.eq("H").to_numpy()]
    assert_boundary(u, h)
    targets_path = args.run_dir / "inputs" / trait / "targets.npz"
    targets = np.load(targets_path)
    out = args.run_dir / "low_order" / trait
    out.mkdir(parents=True, exist_ok=True)
    settings = {"families": ["GAM", "PAIR"], "lambdas": [0, .25, .5, .75, 1], "leaves": [4, 8],
                "l2": [1., 10.], "max_iterations": 400, "learning_rate": .05, "max_depth": 3,
                "max_bins": 32, "min_samples_leaf_binary_rows": 80, "seed": 17,
                "tolerance": 1e-4, "soft_labels": "paired binary rows with weights 1-q and q, total weight one per patient",
                "early_stopping": "explicit H staged log loss, step zero eligible", "max_interaction_order": 2}
    code = {str(p): sha256_file(p) for p in [Path(__file__).resolve().relative_to(Path.cwd()),
        Path("src/decipehr/prs_monotone_boosting.py"), Path("tests/test_prs_monotone_boosting.py")]}
    contract = {"parent_protocol_sha256": sha256_file(args.run_dir / "PROTOCOL.json"), "settings": settings,
        "code_sha256": code, "targets_sha256": sha256_file(targets_path), "train_hash": frame_hash(d),
        "evaluation_features_hash": frame_hash(e), "sklearn": importlib.metadata.version("scikit-learn"),
        "evaluation_labels_read": False, "predictive_priors_not_causal_effects": True}
    if (out / "CONTRACT.json").exists() and read(out / "CONTRACT.json") != contract:
        raise ValueError("Existing low-order run contract differs")
    write_json(out / "CONTRACT.json", contract)
    if (out / "COMPLETE.json").exists():
        verify_files(out, read(out / "COMPLETE.json")["output_sha256"])
        return
    features = Features.fit(u, manifest.column.tolist())
    x, hx = features.transform(u), features.transform(h)
    rows, started = [], time.monotonic()
    for family, lam, leaves, l2 in itertools.product(settings["families"], settings["lambdas"], settings["leaves"], settings["l2"]):
        target = (1-lam)*u.y.to_numpy()+lam*targets["q_U"]
        candidate = f"{family}_lambda{lam:g}_leaves{leaves}_l2{l2:g}"
        path = out / "tuning" / f"{candidate}.json"
        if path.exists():
            record = read(path)
            if record["contract_digest"] != digest(contract):
                raise ValueError("Stale low-order candidate")
        else:
            model = fit_booster(x, target, features, family=family, leaves=leaves, l2=l2)
            chosen, curve = select_iteration(model, hx, h.y.to_numpy(), target.mean())
            record = {"contract_digest": digest(contract), "family": family, "lambda": lam,
                      "leaves": leaves, "l2": l2, **chosen, "curve": curve}
            write_json(path, record)
        rows.append({k: v for k, v in record.items() if k not in ["curve", "contract_digest"]} | {"candidate": candidate})
        print(f"{trait} {candidate}: H={record['H_log_loss']:.6f}, iterations={record['iterations']}, elapsed={time.monotonic()-started:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "tuning_results.csv", index=False)
    choices = {}
    for family in settings["families"]:
        for mode in ["SUP", "KD"]:
            candidates = table.loc[table.family.eq(family) & (table["lambda"].eq(0) if mode == "SUP" else table["lambda"].gt(0))]
            pool = candidates.loc[candidates.H_log_loss <= candidates.H_log_loss.min()+1e-4]
            chosen = pool.sort_values(["leaves", "iterations", "lambda", "l2"], ascending=[True, True, True, False]).iloc[0].to_dict()
            choices[f"HGB_{family}_{mode}"] = chosen
    best = min(c["H_log_loss"] for c in choices.values())
    pool = [a for a, c in choices.items() if c["H_log_loss"] <= best+1e-4]
    selected = min(pool, key=lambda a: (choices[a]["family"] != "GAM", choices[a]["leaves"], choices[a]["iterations"], choices[a]["lambda"]))
    lock = {"choices": choices, "selected_arm": selected, "assessment_used_for_selection": False,
            "patient_n_U": len(u), "effective_weight_U": len(u), "binary_rows_U": 2*len(u),
            "patient_n_D": len(d), "effective_weight_D": len(d), "binary_rows_D": 2*len(d),
            "seed": 17, "stochastic_subsampling": False}
    lock_path = out / "SELECTION_LOCK.json"
    if lock_path.exists() and read(lock_path) != lock:
        raise ValueError("Existing low-order lock changed")
    write_json(lock_path, lock)
    features = Features.fit(d, manifest.column.tolist())
    dx, ex = features.transform(d), features.transform(e)
    predictions, checks = [], []
    for arm, choice in choices.items():
        path = out / "models" / arm
        path.mkdir(parents=True, exist_ok=True)
        metadata = {"choice": choice, "lock_sha256": sha256_file(lock_path)}
        target = (1-choice["lambda"])*d.y.to_numpy()+choice["lambda"]*targets["q_D"]
        if (path / "model.joblib").exists():
            bundle = joblib.load(path / "model.joblib")
            if bundle["metadata"] != metadata:
                raise ValueError("Saved low-order model mismatch")
        else:
            model = fit_booster(dx, target, features, family=choice["family"], leaves=choice["leaves"],
                l2=choice["l2"], iterations=choice["iterations"])
            bundle = bundle_model(model, features, choice["family"], choice["iterations"], target.mean(), metadata)
            joblib.dump(bundle, path / "model.joblib")
        checked = verify(bundle, ex)
        probability, logit = predict(bundle, ex), raw_predict(bundle, ex)
        np.testing.assert_array_equal(probability, predict(joblib.load(path / "model.joblib"), ex))
        reference_age = ex.copy()
        reference_age[:, features.k] = 0
        eta_ref = raw_predict(bundle, reference_age)
        age_delta = logit-eta_ref
        assert (age_delta[ex[:, features.k] >= 0] >= -1e-10).all()
        assert (age_delta[ex[:, features.k] <= 0] <= 1e-10).all()
        predictions.append(e[["sampleid", "kinship_group", "age_sd", "hard_label"]].assign(
            arm=arm, seed=17, probability=probability, logit=logit, reference_age_logit=eta_ref,
            age_delta=age_delta, p_teacher=targets["q_E"]))
        if arm == selected:
            intercept, first, second = explain(bundle, ex)
            columns = [f"main__{name}" for name in features.names]
            columns += [f"pair__{features.names[i]}__{features.names[j]}" for i, j in bundle["active_pairs"]]
            explanation = pd.DataFrame(np.column_stack([first, second]), columns=columns)
            explanation.insert(0, "sampleid", e.sampleid.to_numpy())
            explanation["reference_logit"] = intercept
            explanation["logit"] = logit
            explanation.to_parquet(out / "selected_exact_explanations.parquet", index=False)
        checks.append({"arm": arm, **checked, "saved_replay_exact": True})
        write_json(path / "FIT.json", {**metadata, **checked, "features": features.names,
            "reference": features.reference.tolist(), "model_sha256": sha256_file(path / "model.joblib")})
    pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    write_json(out / "COMPLETE.json", {"status": "locked_predictions_complete", "models": 4,
        "model_verification": checks, "evaluation_labels_read": False, "selected_arm": selected,
        "elapsed_seconds": time.monotonic()-started,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    main()
