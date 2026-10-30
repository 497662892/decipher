"""Compare one prespecified centroid initialization with frozen M0_KD random starts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_prs_distillation_bio import verification, verify

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_centroid_init import centroid_parameters, initialized_model, source_mapping
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import frame_hash, id_hash
from decipehr.prs_distillation_bio import BioPreprocessor, fit, predict

BIO = Path("outputs/prs_distillation_bio/exploration_20261026_3569648f99a8")
TRAITS = ["T2D", "cad", "bc"]
SCOPES = ["val", *[f"outer{i}" for i in range(5)]]
SEEDS = [17, 29, 43]


def read(path):
    return json.loads(Path(path).read_text())


def source_scope(scope):
    return BIO if scope == "val" else BIO / "nested" / scope


def locked_inputs(scope, trait, eligibility):
    root = source_scope(scope)
    protocol, cfg = verify(root)
    source = Path(cfg["source_run"])
    d, e, manifest = load_inputs(source, trait)
    rules = load_eligibility_rules(eligibility)
    assert "bc" in rules
    for frame in [d, e]:
        assert set(filter_eligible(frame, trait, rules).sampleid) == set(frame.sampleid)
    core = root / "core" / trait
    receipt = read(core / "PREDICTIONS_COMPLETE.json")
    parent = read(BIO / "COMPLETE.json")
    assert sha256_file(core / "PREDICTIONS_COMPLETE.json") == parent["upstream_receipt_sha256"][str(core / "PREDICTIONS_COMPLETE.json")]
    assert not receipt["evaluation_labels_read"]
    verify_files(core, {name: receipt["output_sha256"][name] for name in
                        ["direct_predictions.parquet", "SELECTION_LOCK.json"]})
    choice = read(core / "SELECTION_LOCK.json")["choices"]["M0_KD"]
    assert choice["settings"]["variant"] == "M0" and not choice["settings"]["shape"]
    prep = BioPreprocessor.fit(d, manifest.column.tolist(), 3)
    baseline = pd.read_parquet(core / "direct_predictions.parquet", filters=[("arm", "=", "M0_KD")])
    for seed in SEEDS:
        path = core / "models/M0_KD" / f"seed{seed}" / "metadata.json"
        assert sha256_file(path) == receipt["output_sha256"][str(path.relative_to(core))]
        meta = read(path)
        assert meta["D_ids_hash"] == id_hash(d) and meta["choice"] == choice
        assert meta["preprocessing"] == prep.metadata()
        rows = baseline.loc[baseline.seed.eq(seed)]
        assert rows.sampleid.is_unique and set(rows.sampleid) == set(e.sampleid)
    q_path = root / "inputs" / trait / "targets.npz"
    assert sha256_file(q_path) == protocol["input_sha256"][str(q_path.relative_to(root))]
    q = np.load(q_path, allow_pickle=False)
    assert q["q_D"].shape == (len(d),) and q["q_E"].shape == (len(e),)
    used = {str(p): sha256_file(p) for p in [root / "PROTOCOL.json", q_path,
        core / "PREDICTIONS_COMPLETE.json", core / "SELECTION_LOCK.json", core / "direct_predictions.parquet",
        source / trait / "input_receipt.json", source / trait / "feature_manifest.tsv"]}
    return d, e, manifest, prep, q, baseline, choice, protocol, cfg, used


def prepare(out, eligibility):
    out.mkdir(parents=True, exist_ok=False)
    source = Path(read(BIO / "PROTOCOL.json")["identity"]["config"]["source_run"])
    audit = read(source / "audit_report.json")
    assert sha256_file(eligibility) == audit["source_sha256"][str(eligibility)]
    mappings, rows, used = [], [], {str(BIO / "COMPLETE.json"): sha256_file(BIO / "COMPLETE.json"),
                                  str(eligibility): sha256_file(eligibility)}
    for trait in TRAITS:
        _, _, manifest = load_inputs(source, trait)
        mappings.append(source_mapping(manifest))
    mapping = pd.concat(mappings, ignore_index=True)
    mapping.to_csv(out / "prs_ancestry_mapping.csv", index=False)
    for scope in SCOPES:
        for trait in TRAITS:
            d, e, _, prep, _, _, choice, protocol, _, hashes = locked_inputs(scope, trait, eligibility)
            x = prep.transform(d)
            w, c, info = centroid_parameters(x["a"], d.hard_label.to_numpy(), prep.age.fixed.kept_prs,
                                             mapping.loc[mapping.trait.eq(trait)])
            path = out / "initialization" / scope / trait
            path.mkdir(parents=True)
            np.savez_compressed(path / "gate.npz", w=w, c=c)
            write_json(path / "CENTERS.json", {**info, "scope": scope, "trait": trait,
                "fit_ids_hash": id_hash(d), "fit_frame_hash": frame_hash(d),
                "preprocessing": prep.metadata(), "held_out_features_used": False,
                "held_out_labels_used": False})
            rows.append({"scope": scope, "trait": trait, "train_n": len(d), "query_n": len(e),
                "prs_n": len(w), "mapped_n": sum(a["center_used"] for a in info["mapped_prs"]),
                "tau_squared": info["tau_squared"], "lambda": choice["settings"]["lam"],
                "steps": choice["steps"], "gate_ridge": protocol["priors"][trait]["common_gate_ridge"],
                "kappa_r": choice["settings"]["kappa_r"], "fit_ids_hash": id_hash(d),
                "query_ids_hash": id_hash(e)})
            used.update(hashes)
            for filename in ["gate.npz", "CENTERS.json"]:
                used[str(path / filename)] = sha256_file(path / filename)
    pd.DataFrame(rows).to_csv(out / "scope_budgets.csv", index=False)
    code = [Path(__file__), Path("src/decipehr/prs_centroid_init.py"), Path("tests/test_prs_centroid_init.py"),
            Path("slurm/prs_centroid_init.sbatch"), Path("src/decipehr/prs_distillation_bio.py"),
            Path("src/decipehr/prs_distillation_v3.py"), Path("src/decipehr/prs_distillation/student.py"),
            Path("src/decipehr/prs_distillation/preprocessing.py")]
    hashes = {str(p): sha256_file(p) for p in code}
    for p in code:
        dst = out / "source_snapshot" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    protocol = {"source_run": str(BIO), "traits": TRAITS, "scopes": SCOPES, "seeds": SEEDS, "primary_seed": 17,
        "arms": ["M0_RANDOM", "M0_CENTROID"], "mapped_ancestry": "single-ancestry GWAS or explicit local component",
        "dprism_mapping": "GWAS weight-source ancestry; intended trainedfor population does not override this",
        "unmapped": "multiple/unspecified ancestry, including broad ASN; retain original random W and zero c",
        "centers": "means of standardized clinical PC1-PC10 within fit-context ancestry-like groups; n>=10",
        "centers_are_proxies": "training-cohort representatives, not the original GWAS population centers",
        "temperature": "tau^2 = median nonzero pairwise squared centroid distance / 2; no temperature search",
        "gate": "W_k = mu_g / tau^2 + same original Gaussian(0,0.01) row; c_k = -||mu_g||^2/(2*tau^2)",
        "per_prs_prior": "equal, as in the baseline; no ancestry-count reweighting",
        "all_parameters_trainable": True, "changed_parameters_at_initialization": ["W", "c"],
        "fixed_training": "frozen M0_KD hyperparameters and optimizer steps within each matching scope; no retuning",
        "reference_replay": "retrain random seed17 in every scope and compare to frozen probabilities; reuse other reference seeds",
        "new_student_fits": 72, "new_teacher_fits": 0, "evaluation_labels_read": False,
        "original_val_status": "previously inspected development validation; not independent confirmation",
        "nested_status": "shared five outer patient/available-kinship folds; reuse fold-local teacher OOF targets and train-selected settings",
        "eligibility_config": str(eligibility), "age_units": "age_sd, not years",
        "input_sha256": used, "mapping_sha256": sha256_file(out / "prs_ancestry_mapping.csv"), "code_sha256": hashes}
    write_json(out / "PROTOCOL.json", protocol)
    print(f"Prepared {out}: 54 centroid fits + 18 reference replays; no new teacher fits", flush=True)


def train(out, scope, trait, eligibility):
    torch.set_num_threads(1)
    protocol = read(out / "PROTOCOL.json")
    verify_files(Path.cwd(), protocol["code_sha256"])
    assert str(eligibility) == protocol["eligibility_config"]
    d, e, _, prep, q, baseline, choice, prior, cfg, used = locked_inputs(scope, trait, eligibility)
    for p, h in used.items():
        assert protocol["input_sha256"][p] == h
    start_path = out / "initialization" / scope / trait
    for name in ["CENTERS.json", "gate.npz"]:
        assert sha256_file(start_path / name) == protocol["input_sha256"][str(start_path / name)]
    initial = np.load(start_path / "gate.npz", allow_pickle=False)
    center_info = read(start_path / "CENTERS.json")
    assert center_info["fit_ids_hash"] == id_hash(d) and center_info["fit_frame_hash"] == frame_hash(d)
    assert center_info["preprocessing"] == prep.metadata()
    dest = out / "fits" / scope / trait
    if (dest / "COMPLETE.json").exists():
        verify_files(dest, read(dest / "COMPLETE.json")["output_sha256"])
        print(f"Already verified: {scope}/{trait}", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=False)
    x, xe = prep.transform(d), prep.transform(e)
    fixed = dict(rho=prior["priors"][trait]["common_gate_ridge"], kappa_shape=cfg["shape_penalty"],
                 baseline_ridge=cfg["baseline_ridge"], max_steps=cfg["max_steps"], patience=cfg["patience"],
                 evaluate_every=cfg["evaluate_every"], fixed_steps=choice["steps"])
    predictions, gates, checks = [], [], []
    started = time.monotonic()
    for seed in SEEDS:
        old = baseline.loc[baseline.seed.eq(seed)].set_index("sampleid").loc[e.sampleid]
        np.testing.assert_array_equal(old.kinship_group, e.kinship_group)
        np.testing.assert_array_equal(old.p_teacher, q["q_E"])
        if seed == 17:
            control, _ = fit(x, d.y.to_numpy(), q["q_D"], seed=seed, **fixed, **choice["settings"])
            replay = predict(control, xe)["probability"]
            error = float(np.max(np.abs(replay - old.probability.to_numpy())))
            np.testing.assert_allclose(replay, old.probability.to_numpy(), atol=1e-8, rtol=0)
            checks.append({"seed": seed, "check": "random_reference_replay", "max_probability_error": error})
        reference = e[["sampleid", "kinship_group"]].assign(scope=scope, trait=trait, arm="M0_RANDOM", seed=seed,
            probability=old.probability.to_numpy(), p_teacher=q["q_E"])
        predictions.append(reference)
        warm = initialized_model(x, d.y.to_numpy(), q["q_D"], lam=choice["settings"]["lam"], seed=seed,
                                 w=initial["w"], c=initial["c"])
        initial_probability = predict(warm, x)["pi"]
        model, info = fit(x, d.y.to_numpy(), q["q_D"], seed=seed, warm=warm, **fixed, **choice["settings"])
        result, checked = verification(model, prep, e)
        checks.append({"seed": seed, "check": "centroid_constraints_and_numpy", **checked})
        directory = dest / f"seed{seed}"
        directory.mkdir()
        # Numeric arrays and JSON avoid executable preprocessing pickles.
        np.savez_compressed(directory / "state.npz", **{k: v.detach().numpy() for k, v in model.state_dict().items()})
        write_json(directory / "MODEL.json", {"dimensions": model.dimensions, "variant": model.variant,
            "shape": False, "seed": seed, "training": info, "choice": choice, "settings": fixed,
            "preprocessing": prep.metadata(), "initializer_sha256": sha256_file(start_path / "gate.npz"),
            "fit_ids_hash": id_hash(d), "evaluation_labels_used": False})
        trained_probability = predict(model, x)["pi"]
        for label in sorted(d.hard_label.unique()):
            mask = d.hard_label.eq(label).to_numpy()
            for stage, values in [("initial", initial_probability), ("trained", trained_probability)]:
                for k, prs in enumerate(prep.age.fixed.kept_prs):
                    gates.append({"scope": scope, "trait": trait, "seed": seed, "stage": stage,
                                  "patient_group": label, "n": int(mask.sum()), "prs": prs,
                                  "mean_gate": float(values[mask, k].mean())})
        predictions.append(reference.assign(arm="M0_CENTROID", probability=result["probability"]))
        print(f"{scope}/{trait}/seed{seed}: {choice['steps']} steps; elapsed={time.monotonic()-started:.1f}s", flush=True)
    pd.concat(predictions, ignore_index=True).to_parquet(dest / "predictions.parquet", index=False)
    pd.DataFrame(gates).to_csv(dest / "gate_by_ancestry.csv", index=False)
    write_json(dest / "COMPLETE.json", {"status": "predictions_complete_not_evaluated", "scope": scope,
        "trait": trait, "new_models": 4, "seeds": SEEDS, "training_n": len(d), "query_n": len(e),
        "evaluation_labels_read": False, "checks": checks, "input_sha256": used,
        "protocol_sha256": sha256_file(out / "PROTOCOL.json"),
        "output_sha256": {str(p.relative_to(dest)): sha256_file(p) for p in dest.rglob("*") if p.is_file()}})


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eligibility-config", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES)
    parser.add_argument("--trait", choices=TRAITS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out_dir, args.eligibility_config)
    else:
        if args.scope is None or args.trait is None:
            parser.error("train requires scope and trait")
        train(args.out_dir, args.scope, args.trait, args.eligibility_config)


if __name__ == "__main__":
    main()
