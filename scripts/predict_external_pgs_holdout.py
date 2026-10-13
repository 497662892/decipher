"""Fit the selected external PRS wrapper on labelled train+development; predict holdout."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import pandas as pd
from evaluate_external_pgs_cv import _tabpfn_factory

from decipehr.eligibility import load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_pgs_holdout import predict_holdout_wrapper, prepare_holdout_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/external_pgs/holdout_wrapper_20261012")
    parser.add_argument("--disease", choices=["T2D", "cad", "bc"], required=True)
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--model-path", default="outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_dir)
    selection = json.loads((root / "SELECTION.json").read_text())
    selected = selection["diseases"][args.disease]
    bank = Path(selected["bank"])
    for path, digest in selection["input_sha256"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen selection input changed: {path}")
    scoring = json.loads((root / "scoring/VERIFIED.json").read_text())
    if sha256_file(root / "scoring/scores.long.csv") != scoring["scores_sha256"]:
        raise ValueError("Changed holdout scores")
    scoring_run = json.loads((bank / "scoring_run.json").read_text())
    for filename, key in [("scores.long.csv", "scores_sha256"), ("weights.scored.lock.json", "weight_lock_sha256")]:
        if sha256_file(bank / filename) != scoring_run[key]:
            raise ValueError("Changed context score bank")
    if scoring_run["reference_frequency_sha256"] != scoring["reference_frequency_sha256"]:
        raise ValueError("Context and holdout used different genotype imputation frequencies")
    paths = {
        "phenotype": Path("data/processed/final/train_val_phenotype.csv"),
        "original_train": Path("data/processed/phenotype.csv"),
        "original_development": Path("data/processed/official_validation_phenotype.csv"),
        "holdout": Path("data/processed/final/holdout_test_covariates.csv"),
        "context_ancestry": Path("outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv"),
        "holdout_ancestry": Path("outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.holdout_test.tsv"),
        "context_scores": bank / "scores.long.csv", "holdout_scores": root / "scoring/scores.long.csv",
        "selection": root / "SELECTION.json", "checkpoint": Path(args.model_path),
        "eligibility": Path(args.eligibility_config), "code": Path("src/decipehr/external_pgs_holdout.py"),
        "benchmark_code": Path("src/decipehr/external_pgs_benchmark.py"), "script": Path(__file__),
    }
    hashes = {key: sha256_file(path) for key, path in paths.items()}

    def read(path):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", dtype={"sampleid": str, "IID": str})

    phenotype = read(paths["phenotype"])
    old_parts = [read(paths[key]) for key in ["original_train", "original_development"]]
    original = pd.concat(old_parts, ignore_index=True)
    if original.sampleid.duplicated().any():
        raise ValueError("Original training and development cohorts overlap")
    pd.testing.assert_frame_equal(
        original[phenotype.columns].sort_values("sampleid").reset_index(drop=True),
        phenotype.sort_values("sampleid").reset_index(drop=True), check_dtype=False,
    )
    context, query = prepare_holdout_data(
        phenotype=phenotype, holdout_covariates=read(paths["holdout"]),
        context_scores=read(paths["context_scores"]), holdout_scores=read(paths["holdout_scores"]),
        hard_labels=pd.concat([read(paths["context_ancestry"]), read(paths["holdout_ancestry"])]),
        disease=args.disease, model_ids=selected["model_ids"],
        eligibility_rules=load_eligibility_rules(args.eligibility_config),
    )
    out = root / ("preflight" if args.preflight_only else "predictions") / args.disease
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "arguments": vars(args), "disease": args.disease, "model_ids": selected["model_ids"],
        "n_context": len(context), "context_cases": int(context[args.disease].sum()), "n_holdout": len(query),
        "metrics_available": False, "holdout_labels_available": False, "patient_overlap": 0,
        "original_train_and_development_union_verified": True,
        "prediction_method": "tabpfn_prs_pc", "tabpfn_version": importlib.metadata.version("tabpfn"),
        "n_estimators": 16, "wrapper_inner_folds": 3, "wrapper_clip_eps": 0.001,
        "input_paths": {key: str(path) for key, path in paths.items()}, "input_sha256": hashes,
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "torch"]},
        "label_use": "Only labelled train+development context; holdout outcome columns rejected",
    }
    write_json(out / "PREFLIGHT.json", receipt)
    if args.preflight_only:
        print(f"Preflight passed: {args.disease}, context {len(context)}, holdout {len(query)}", flush=True)
        return
    print(f"Predicting {args.disease}: {len(context)} context, {len(query)} holdout, {len(selected['model_ids'])} PRS", flush=True)
    results = predict_holdout_wrapper(context, query, disease=args.disease, model_ids=selected["model_ids"],
                                      tabpfn_factory=_tabpfn_factory(args.device, args.model_path), seed=args.seed)
    files = ["holdout_predictions.csv", "wrapper_training_oof.csv", "fit_audit.csv", "preprocessing.csv"]
    for filename, frame in zip(files, results, strict=True):
        frame.to_csv(out / filename, index=False)
    for key, path in paths.items():
        if sha256_file(path) != hashes[key]:
            raise ValueError(f"Input changed during prediction: {key}")
    receipt["output_sha256"] = {filename: sha256_file(out / filename) for filename in files}
    write_json(out / "VERIFIED.json", receipt)
    print(f"Verified {args.disease} holdout predictions", flush=True)


if __name__ == "__main__":
    main()
