"""Exploratory train-internal response probes of the biological monotonicity priors."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from benchmark_prs_foundation_models import predict as foundation_predict

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import load_config
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import AGE, PCS, assert_boundary, frame_hash
from decipehr.prs_distillation.teacher import make_teacher, positive_probability
from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer


def read(path):
    return json.loads(Path(path).read_text())


def build_probes(u, h, prs):
    """Equal maximum representation per observed ancestry; no outcome-based selection."""
    chosen = []
    for _, group in h.groupby("hard_label", sort=True):
        chosen.append(group.sample(n=min(3, len(group)), random_state=20261029))
    anchors = pd.concat(chosen).drop(columns="y").reset_index(drop=True)
    blocks = []
    ages = np.unique(np.quantile(u.age_sd, np.linspace(.05, .95, 21)))
    for i, anchor in anchors.iterrows():
        for feature in [AGE, *prs]:
            grid = ages if feature == AGE else np.clip(
                anchor[feature]+np.linspace(-.5, .5, 5)*u[feature].std(ddof=0),
                u[feature].min(), u[feature].max())
            grid = np.unique(grid)
            block = pd.DataFrame([anchor.to_dict()]*len(grid), columns=anchors.columns)
            block[feature] = grid
            block["sampleid"] = [f"synthetic_probe_{i}_{feature}_{j}" for j in range(len(grid))]
            block["probe_profile"] = i
            block["probe_feature"] = feature
            block["probe_step"] = np.arange(len(grid))
            block["probe_value"] = grid
            blocks.append(block)
    probes = pd.concat(blocks, ignore_index=True)
    assert_boundary(u, probes)
    return probes, len(anchors)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["tabpfn3", "tabpfn35", "kumo"], required=True)
    parser.add_argument("--eligibility-config", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = Path("outputs/prs_distillation_v1/phase_a_fulltrain_20261026_3d4334d2_0161ad36d950")
    cfg = load_config(source / "resolved_config.yaml")
    rules = load_eligibility_rules(args.eligibility_config)
    out = args.output_root / args.backend
    out.mkdir(parents=True, exist_ok=True)
    if args.backend == "tabpfn3":
        teacher = read(source / "core/T2D/teacher_contract.json")
        checkpoint, checkpoint_hash = Path(teacher["checkpoint"]), teacher["checkpoint_sha256"]
        verify_files(Path.cwd(), teacher["implementation_sha256"])
    else:
        contract = read(Path("outputs/foundation_model_probe_20261026/benchmark") / args.backend / "CONTRACT.json")
        checkpoint, checkpoint_hash = Path(contract["checkpoint"]["path"]), contract["checkpoint"]["sha256"]
        assert sha256_file("scripts/benchmark_prs_foundation_models.py") == contract["script_sha256"]
    assert sha256_file(checkpoint) == checkpoint_hash
    probe_contract = {"backend": args.backend, "checkpoint_sha256": checkpoint_hash,
        "script_sha256": sha256_file(__file__), "estimators": 16, "seed": 42,
        "context": "U only", "anchors": "up to three H feature profiles per ancestry; seed 20261029",
        "age_grid": "21 U quantiles from 0.05 to 0.95", "prs_grid": "anchor +/-0.5 U SD in five steps, clipped to U support",
        "tolerance_probability": 1e-4, "H_labels_used": False, "official_validation_used": False,
        "exploratory_after_main_val_results": True, "private_data_sent": False,
        "interpretation": "Model response probes; not causal interventions; joint-support violations remain possible"}
    if (out / "CONTRACT.json").exists():
        assert read(out / "CONTRACT.json") == probe_contract
    write_json(out / "CONTRACT.json", probe_contract)
    if (out / "COMPLETE.json").exists():
        verify_files(out, read(out / "COMPLETE.json")["output_sha256"])
        return
    rows = []
    for trait in ["T2D", "cad", "bc"]:
        d, _, manifest = load_inputs(source, trait)
        assert len(filter_eligible(d, trait, rules)) == len(d)
        roles = pd.read_parquet(source / "split_manifest.parquet").set_index("sampleid").loc[d.sampleid]
        u, h = d.loc[roles.internal_role.eq("U").to_numpy()], d.loc[roles.internal_role.eq("H").to_numpy()]
        assert_boundary(u, h)
        probes, n_profiles = build_probes(u, h, manifest.column.tolist())
        prep = TabPFNFeatureTransformer.fit(u, prs_cols=manifest.column.tolist(),
                                          covariates=[AGE, "sex", *PCS], include_covariates=True)
        x, xq = prep.transform(u).to_numpy(), prep.transform(probes).to_numpy()
        print(f"{args.backend}/{trait}: {len(u)} context, {n_profiles} profiles, {len(probes)} queries", flush=True)
        if args.backend == "tabpfn3":
            model = make_teacher(cfg)
            model.fit(x, u.y.to_numpy())
            probability, _ = positive_probability(model, xq)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            probability, _ = foundation_predict(args.backend, x, u.y.to_numpy(), xq, checkpoint)
        result = probes[["probe_profile", "probe_feature", "probe_step", "probe_value"]].assign(probability=probability)
        result.to_parquet(out / f"{trait}_private_probes.parquet", index=False)
        for feature, group in result.groupby("probe_feature", sort=False):
            curves = [p.sort_values("probe_step").probability.to_numpy() for _, p in group.groupby("probe_profile")]
            differences = [np.diff(p) for p in curves]
            steps = np.concatenate(differences)
            drawdowns = [float(np.max(np.maximum.accumulate(p)-p)) for p in curves]
            rows.append({"backend": args.backend, "trait": trait, "feature": feature,
                "dimension": "age" if feature == AGE else "PRS", "profiles": len(differences),
                "steps": len(steps), "negative_steps": int((steps < -1e-4).sum()),
                "fraction_negative_steps": float((steps < -1e-4).mean()),
                "fraction_steps_drop_over_0p1pp": float((steps < -.001).mean()),
                "fraction_steps_drop_over_1pp": float((steps < -.01).mean()),
                "profiles_with_decrease": sum(bool((delta < -1e-4).any()) for delta in differences),
                "profiles_with_drawdown_over_1pp": sum(x > .01 for x in drawdowns),
                "maximum_profile_drawdown": max(drawdowns),
                "maximum_probability_drop": float(max(0., -steps.min()))})
        write_json(out / f"{trait}_SCOPE.json", {"train_feature_hash": frame_hash(u), "query_hash": frame_hash(probes),
            "context_n": len(u), "anchor_n": n_profiles, "probe_n": len(probes), "H_labels_used": False})
    pd.DataFrame(rows).to_csv(out / "diagnostics.csv", index=False)
    write_json(out / "COMPLETE.json", {"status": "completed", "contract": probe_contract,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    main()
