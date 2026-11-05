"""Portable frozen PRS/PC preparation, TabPFN/student inference, and separate evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from decipehr.external_bundle import export_bundle, load_bundle, read_json
from decipehr.external_cohort import (
    file_identity,
    genotype_files,
    load_cohort_config,
    prepare_cohort,
    read_covariates,
    run_lock,
)
from decipehr.external_pgs import sha256_file, write_json


def evaluate(args):
    from dataclasses import asdict

    import numpy as np
    import pandas as pd
    from sklearn.metrics import log_loss

    from decipehr.metrics import binary_metrics

    root = args.prediction_dir.resolve()
    receipt = read_json(root / "COMPLETE.json")
    if receipt["status"] != "prediction_complete" or sha256_file(root / "predictions.csv") != receipt["output_sha256"]["predictions.csv"]:
        raise ValueError("Evaluation requires verified completed predictions")
    predictions = pd.read_csv(root / "predictions.csv", dtype={"sampleid": str})
    labels = pd.read_csv(args.outcomes, dtype={"sampleid": str})
    if "sampleid" not in labels or labels.sampleid.isna().any() or not labels.sampleid.is_unique:
        raise ValueError("Outcomes require unique, nonmissing patient IDs")
    if not args.phenotype_note.strip():
        raise ValueError("Document the case/control definition and observation period")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for (trait, model), frame in predictions.groupby(["trait", "model"], sort=False):
        if trait not in labels:
            raise ValueError(f"Missing outcome column: {trait}")
        target = pd.to_numeric(labels[trait], errors="raise")
        if not target.dropna().isin([0, 1]).all():
            raise ValueError("Outcomes must be 0/1 or missing; unknown is not a control")
        paired = frame.merge(labels[["sampleid", trait]].assign(**{trait: target}), on="sampleid", how="left", validate="one_to_one")
        known = paired.loc[paired[trait].notna()]
        if known.empty:
            rows.append({"trait": trait, "model": model, "n_predicted": len(frame), "n_labeled": 0,
                         "n_missing_labels": len(frame), "status": "no_labels"})
            continue
        y, p = known[trait].to_numpy(int), known.probability.to_numpy(float)
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("Invalid prediction probabilities")
        rows.append({"trait": trait, "model": model, "n_predicted": len(frame), "n_labeled": len(y),
            "n_missing_labels": len(frame)-len(y), "n_cases": int(y.sum()), "n_controls": int((y == 0).sum()),
            "status": "both_classes" if len(set(y)) == 2 else "one_class",
            **asdict(binary_metrics(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))})
    pd.DataFrame(rows).to_csv(args.out_dir / "metrics.csv", index=False)
    result = {"status": "label_metrics_computed", "phenotype_definition": args.phenotype_note,
        "prediction_receipt_sha256": sha256_file(root / "COMPLETE.json"), "outcomes_sha256": sha256_file(args.outcomes),
        "metrics_sha256": sha256_file(args.out_dir / "metrics.csv"), "model_or_calibration_fits": 0,
        "intervals": "not computed; descriptive point estimates",
        "external_independence": "Not inferred from labels or IDs; audit discovery/training/cohort overlap separately"}
    write_json(args.out_dir / "EVALUATED.json", result)
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Maintainer: export a portable private model/resource bundle")
    export.add_argument("--sources", type=Path, default=Path("configs/external_model_sources.yaml"))
    export.add_argument("--out-dir", type=Path, required=True)
    export.add_argument("--scope", choices=["train_only", "trainval_refit", "both"], default="trainval_refit")
    for name in ["preflight", "prepare", "predict", "run"]:
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    inspect = sub.add_parser("verify-bundle")
    inspect.add_argument("--bundle", type=Path, required=True)
    evaluation = sub.add_parser("evaluate", help="Optional: read outcomes only after inference")
    evaluation.add_argument("--prediction-dir", type=Path, required=True)
    evaluation.add_argument("--outcomes", type=Path, required=True)
    evaluation.add_argument("--phenotype-note", required=True)
    evaluation.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "export":
            scopes = ["train_only", "trainval_refit"] if args.scope == "both" else [args.scope]
            bundle = export_bundle(args.sources, args.out_dir, scopes=scopes)
            result = {"status": "bundle_exported", "directory": str(args.out_dir), "scopes": bundle["scopes"],
                      "n_required_variants": bundle["n_required_variants"], "contains_private_training_context": True}
        elif args.command == "verify-bundle":
            bundle = load_bundle(args.bundle)
            result = {"status": "bundle_verified", "files": len(bundle["file_sha256"]), "scopes": bundle["scopes"]}
        elif args.command == "evaluate":
            result = evaluate(args)
        else:
            config = load_cohort_config(args.config)
            bundle = load_bundle(config["bundle"])
            if args.command == "preflight":
                covariates = read_covariates(config)
                result = {"status": "input_files_checked", "n_patients": len(covariates),
                    "n_female_bc_eligible": int(covariates.sex.eq("Female").sum()),
                    "genotypes": {str(p): file_identity(p) for source in config["genotypes"] for p in genotype_files(source)},
                    "age_mode": config["age"]["mode"], "genetic_coverage_checked": False,
                    "next": "prepare checks coordinate/allele coverage and computes raw PRS/reference PCs"}
            else:
                with run_lock(config["run_dir"]):
                    if args.command == "prepare":
                        result = prepare_cohort(config, bundle)
                    elif args.command == "run" and config["age"]["mode"] == "deferred":
                        result = prepare_cohort(config, bundle)
                        result = {"status": "preparation_complete_prediction_deferred_age", "n_patients": result["n_patients"],
                                  "next": "Supply documented compatible age_sd, then rerun; genetic preparation will be reused"}
                    else:
                        from decipehr.external_prediction import predict_cohort
                        result = predict_cohort(config, bundle)
                    write_json(Path(config["run_dir"]) / "RUN_STATUS.json", {"command": args.command, "status": result["status"], "config_sha256": sha256_file(args.config)})
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
