"""Audit, fit and report a shared-source T2D comparison without changing old runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from correct_external_prs_analysis import load_frames, verify_contract
from evaluate_external_pgs_cv import _tabpfn_factory
from report_external_prs_correction import read_verified
from report_internal_prs_qc_sensitivity import metrics

from decipehr.ancestry_performance import paired_bootstrap_comparisons, prediction_metrics
from decipehr.discodivas import comparison_bank
from decipehr.dprism_comparison import (
    DIRECT,
    EXISTING,
    METHODS,
    SOURCE_IDS,
    component_design_pair,
    evaluate_source_partition,
)
from decipehr.external_pgs import sha256_file, write_json
from decipehr.internal_prs_benchmark import DISEASES, ids_hash, inner_splits, shared_folds

PARTITIONS = ("0", "1", "2", "3", "4", "development")
CODE = ("src/decipehr/dprism_comparison.py", "scripts/evaluate_dprism_source_comparison.py",
        "slurm/dprism_source_comparison.sbatch", "src/decipehr/ancestry_performance.py")
PAIRS = (("same_25_sources", "selected_disco", DIRECT),
         ("direct25_vs_meta5", "tabpfn_matched", DIRECT),
         ("same_5_metascores", "selected_disco", "tabpfn_matched"))
PROTOCOL = Path("docs/1011/T2D_DPRISM25_Comparison_20261011.md")


def verify_reference(root):
    contract = verify_contract(root)
    final = json.loads((root / "FINAL_VERIFICATION.json").read_text())
    for path, digest in final["delivery_sha256"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen historical delivery changed: {path}")
    return contract


def prepare(root, reference, eligibility):
    old = verify_reference(reference)
    if sha256_file(eligibility) != old["source_sha256"]["eligibility"]:
        raise ValueError("Eligibility config differs from the frozen cohort")
    _, _, train, development = load_frames(old, "T2D")
    expected = old["cohorts"]["T2D"]
    if ids_hash(train) != expected["context_ids_sha256"] or ids_hash(development) != expected["query_ids_sha256"]:
        raise ValueError("Original patient membership changed")
    bank = comparison_bank("T2D", pd.read_csv(old["source_paths"]["mapping"]))
    if bank.source_ids != SOURCE_IDS:
        raise ValueError("Wrong source panel")
    # The unrelated 26th score is physically absent from every prepared partition.
    train, development = [f.drop(columns=["PGS002308"]) for f in (train, development)]
    folds = shared_folds(train, seed=old["seed"])
    train = train.merge(folds[["sampleid", "fold"]], on="sampleid", validate="one_to_one")
    x, q, names, audit = component_design_pair(train, development.drop(columns=list(DISEASES)), bank)
    read_verified(reference, "development", "T2D")
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=False)
    train.to_parquet(out / "training.parquet", index=False)
    development.to_parquet(out / "development.parquet", index=False)
    folds.to_csv(out / "shared_folds.csv", index=False)
    (out / "PROTOCOL.md").write_text(PROTOCOL.read_text())
    manifest = {"protocol": "t2d_dprism25_same_sources_v1", "reference_run": str(reference),
                "reference_contract_sha256": sha256_file(reference / "CONTRACT.json"),
                "reference_final_sha256": sha256_file(reference / "FINAL_VERIFICATION.json"),
                "source_ids": list(SOURCE_IDS), "n_source_prs": 25, "methods": list(METHODS),
                "seed": old["seed"], "n_estimators": old["n_estimators"], "versions": old["versions"],
                "required_runtime": old["required_runtime"], "checkpoint": old["source_paths"]["checkpoint"],
                "mapping": old["source_paths"]["mapping"], "cohorts": expected,
                "outer_cv": "shared five-fold patient split on original training membership",
                "selection": "original three-fold training-only selection nested inside each fit context",
                "development": "previously used official development; exploratory evidence",
                "preprocessing": "training-only standardization and OLS PC residualization; shared refPC1-10",
                "snp_weights_retrained": False, "new_qc_or_kinship_exclusions": False,
                "independent_external_validation": False, "bootstrap_family": list(PAIRS),
                "code_sha256": {p: sha256_file(p) for p in CODE},
                "prepared_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
                "preflight": {"n_train": len(train), "n_development": len(development),
                              "train_cases": int(train.T2D.sum()), "development_cases": int(development.T2D.sum()),
                              "direct_feature_count": x.shape[1], "query_feature_count": q.shape[1],
                              "features": names, "equivalence_checks": audit["equivalence_checks"]}}
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest["preflight"], indent=2), flush=True)


def load_prepared(root):
    path = root / "prepared/manifest.json"
    manifest = json.loads(path.read_text())
    reference = Path(manifest["reference_run"])
    verify_reference(reference)
    for path, digest in ((reference / "CONTRACT.json", manifest["reference_contract_sha256"]),
                         (reference / "FINAL_VERIFICATION.json", manifest["reference_final_sha256"])):
        if sha256_file(path) != digest:
            raise ValueError("Frozen reference receipt changed")
    for name, digest in manifest["code_sha256"].items():
        if sha256_file(name) != digest:
            raise ValueError(f"New comparison code changed after freeze: {name}")
    for name, digest in manifest["prepared_sha256"].items():
        if sha256_file(root / "prepared" / name) != digest:
            raise ValueError(f"Prepared data changed: {name}")
    train = pd.read_parquet(root / "prepared/training.parquet")
    development = pd.read_parquet(root / "prepared/development.parquet")
    return manifest, train, development


def partition_frames(train, development, partition):
    if partition == "development":
        return train.copy(), development.copy()
    return train.loc[train.fold != int(partition)].copy(), train.loc[train.fold == int(partition)].copy()


def fit(root, partition):
    import torch

    manifest, train, development = load_prepared(root)
    gpu = torch.cuda.get_device_name(0)
    if "L40S" not in gpu or os.environ.get("OMP_NUM_THREADS") != "8":
        raise ValueError("Use the frozen L40S / 8-thread runtime")
    for package, version in manifest["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Frozen package version changed: {package}")
    context, query = partition_frames(train, development, partition)
    out = root / "partitions" / partition
    out.mkdir(parents=True, exist_ok=False)
    bank = comparison_bank("T2D", pd.read_csv(manifest["mapping"]))
    predictions, selection, fits, parameters = evaluate_source_partition(
        context, query, bank=bank, seed=manifest["seed"],
        tabpfn_factory=_tabpfn_factory("cuda", manifest["checkpoint"]),
    )
    predictions.assign(partition=partition).to_csv(out / "predictions.csv", index=False, float_format="%.17g")
    selection.to_csv(out / "selection.csv", index=False)
    fits.to_csv(out / "fits.csv", index=False)
    write_json(out / "parameters.json", parameters)
    write_json(out / "COMPLETE.json", {
        "status": "complete", "manifest_sha256": sha256_file(root / "prepared/manifest.json"),
        "n_context": len(context), "n_query": len(query), "gpu": gpu,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_sha256": {name: sha256_file(out / name) for name in
                          ("predictions.csv", "selection.csv", "fits.csv", "parameters.json")},
    })
    print(f"Completed T2D partition {partition}", flush=True)


def report(root, n_bootstraps):
    manifest, train, development = load_prepared(root)
    frames, fit_rows, selection_rows, checks = [], [], [], []
    for partition in PARTITIONS:
        out = root / "partitions" / partition
        receipt = json.loads((out / "COMPLETE.json").read_text())
        if receipt["status"] != "complete" or receipt["manifest_sha256"] != sha256_file(root / "prepared/manifest.json"):
            raise ValueError("Incomplete or incompatible partition")
        for name, digest in receipt["output_sha256"].items():
            if sha256_file(out / name) != digest:
                raise ValueError(f"Changed output: {out / name}")
        context, query = partition_frames(train, development, partition)
        frame = pd.read_csv(out / "predictions.csv", dtype={"sampleid": str, "partition": str}, float_precision="round_trip")
        if set(frame.method) != set(METHODS) or frame.duplicated(["sampleid", "method"]).any():
            raise ValueError("Incomplete method coverage or duplicate predictions")
        truth = query.set_index("sampleid").sort_index()
        for _, current in frame.groupby("method", observed=True):
            current = current.set_index("sampleid").sort_index()
            if not current.index.equals(truth.index):
                raise ValueError("Prediction patients differ from the frozen partition")
            np.testing.assert_array_equal(current.label, truth.T2D)
            np.testing.assert_array_equal(current.hard_label, truth.hard_label)
            if not np.isfinite(current.probability).all() or not current.probability.between(0, 1).all():
                raise ValueError("Invalid probabilities")
        bounds = {"final": (ids_hash(context), ids_hash(query), len(context), len(query))}
        for i, (a, b) in enumerate(inner_splits(context, "T2D", manifest["seed"])):
            bounds[i] = (ids_hash(context.iloc[a]), ids_hash(context.iloc[b]), len(a), len(b))
        fitted = pd.read_csv(out / "fits.csv")
        for row in fitted.itertuples():
            key = "final" if row.stage == "final" else int(row.inner_fold)
            if (row.context_ids_sha256, row.query_ids_sha256, row.n_context, row.n_query) != bounds[key]:
                raise ValueError("Fitting or tuning crossed the training boundary")
        fit_rows.append(fitted.assign(partition=partition))
        selection_rows.append(pd.read_csv(out / "selection.csv").assign(partition=partition))
        parameters = json.loads((out / "parameters.json").read_text())
        audit = next(p["preprocessing"] for p in parameters if p["method"] == DIRECT)
        checks.extend({"partition": partition, **c} for c in audit["equivalence_checks"])
        frames.append(frame.assign(evaluation="development" if partition == "development" else "training_oof"))
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.duplicated(["evaluation", "sampleid", "method"]).any():
        raise ValueError("A patient appears in multiple outer query folds")
    previous = read_verified(Path(manifest["reference_run"]), "development", "T2D")
    reproduction = []
    for method in EXISTING:
        original = previous.loc[previous.method == method].set_index("sampleid").sort_index()
        current = predictions.loc[(predictions.evaluation == "development") & (predictions.method == method)].set_index("sampleid").sort_index()
        if not current.index.equals(original.index):
            raise ValueError("Historical comparison patients changed")
        np.testing.assert_array_equal(current.probability, original.probability)
        reproduction.append({"method": method, "n": len(current), "identical_probabilities": True})
    overall, subgroup, paired = [], [], []
    for (evaluation, method), frame in predictions.groupby(["evaluation", "method"], observed=True):
        overall.append({"evaluation": evaluation, "method": method, "n_source_prs": 25, **metrics(frame)})
        for group, subset in frame.groupby("hard_label", observed=True):
            values = prediction_metrics(subset.label.to_numpy(), subset.probability.to_numpy())
            if subset.label.nunique() == 2:
                values = metrics(subset)
            subgroup.append({"evaluation": evaluation, "method": method, "ancestry_group": group, **values})
    for evaluation, frame in predictions.groupby("evaluation", observed=True):
        wide = frame.pivot(index=["sampleid", "disease", "label", "hard_label"], columns="method", values="probability").reset_index()
        wide = wide.rename(columns={"label": "y_true", "hard_label": "ancestry_group"})
        paired.append(paired_bootstrap_comparisons(
            wide, n_bootstraps=n_bootstraps, seed=20261011, comparisons=PAIRS,
            metric_names=("auc",), family_size=3,
        ).assign(evaluation=evaluation))
    out = root / "report"
    out.mkdir(exist_ok=False)
    tables = {"overall_metrics.csv": pd.DataFrame(overall), "metrics_by_ancestry.csv": pd.DataFrame(subgroup),
              "paired_auc.csv": pd.concat(paired, ignore_index=True), "equivalence.csv": pd.DataFrame(checks),
              "reproduction.csv": pd.DataFrame(reproduction), "fit_audit.csv": pd.concat(fit_rows, ignore_index=True),
              "selection.csv": pd.concat(selection_rows, ignore_index=True)}
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    write_json(out / "VERIFIED.json", {
        "status": "complete", "manifest_sha256": sha256_file(root / "prepared/manifest.json"),
        "methods": list(METHODS), "partitions": list(PARTITIONS), "n_source_prs": 25,
        "n_bootstraps": n_bootstraps, "bootstrap_unit": "outcome-stratified paired patient",
        "family_size": 3, "familywise_scope": "three overall AUC comparisons within each evaluation; subgroups exploratory",
        "inference": "conditional on saved predictions; development previously used; training OOF uses overlapping fit sets; family dependence unresolved",
        "historical_delivery_unchanged": True, "previous_matched_probabilities_exactly_reproduced": True,
        "partition_receipt_sha256": {p: sha256_file(root / "partitions" / p / "COMPLETE.json") for p in PARTITIONS},
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
    })
    print(tables["overall_metrics.csv"][["evaluation", "method", "n", "auc", "log_loss"]].to_string(index=False), flush=True)
    print(tables["paired_auc.csv"].query("evaluation == 'development' and ancestry_group == 'ALL'")
          [["comparison", "candidate_minus_reference", "ci_low", "ci_high", "familywise_ci_low", "familywise_ci_high"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "fit", "report"), required=True)
    parser.add_argument("--run-dir", default="outputs/external_pgs/dprism25_comparison_20261011")
    parser.add_argument("--reference-run", default="outputs/external_pgs/correction_20261011")
    parser.add_argument("--eligibility-config", required=True)
    parser.add_argument("--partition", choices=PARTITIONS)
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.run_dir)
    reference = Path(args.reference_run)
    original = json.loads((reference / "CONTRACT.json").read_text())
    if sha256_file(args.eligibility_config) != original["source_sha256"]["eligibility"]:
        raise ValueError("Eligibility configuration changed")
    if args.phase == "audit":
        prepare(root, reference, args.eligibility_config)
    elif args.phase == "fit":
        if args.partition is None:
            parser.error("--partition is required for fitting")
        fit(root, args.partition)
    else:
        report(root, args.n_bootstraps)


if __name__ == "__main__":
    main()
