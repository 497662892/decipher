"""Verify and compare corrected runs on the original official development cohort."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from correct_external_prs_analysis import LEGACY, PREVIOUS, load_frames, verify_contract
from report_internal_prs_benchmark import cluster_comparisons
from report_internal_prs_qc_sensitivity import metrics
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from decipehr.external_pgs import sha256_file, write_json
from decipehr.internal_prs_benchmark import DISEASES, METHODS, ids_hash, inner_splits


def read_verified(root, kind, disease):
    out = root / kind / disease
    receipt = json.loads((out / "COMPLETE.json").read_text())
    if receipt["status"] != "complete" or receipt["contract_sha256"] != sha256_file(root / "CONTRACT.json"):
        raise ValueError("Incomplete or incompatible run")
    for name, digest in receipt["output_sha256"].items():
        if sha256_file(out / name) != digest:
            raise ValueError(f"Changed output: {out / name}")
    return pd.read_csv(out / "predictions.csv", dtype={"sampleid": str}, float_precision="round_trip")


def reproduction_rows(root, disease, frame):
    old = pd.read_csv(LEGACY / disease / "validation_predictions.csv",
                      dtype={"sampleid": str, "y_prob": "float32"})
    old = old.loc[old.method == "tabpfn_prs_pc"].set_index("sampleid").sort_index()
    anchor = read_verified(root, "anchor", disease)
    previous_path = PREVIOUS / "original_cohort_sensitivity" / disease
    receipt = json.loads((previous_path / "COMPLETE.json").read_text())
    if sha256_file(previous_path / "predictions.csv") != receipt["output_sha256"]["predictions.csv"]:
        raise ValueError("Previous A100 prediction artifact changed")
    previous = pd.read_csv(previous_path / "predictions.csv",
                           dtype={"sampleid": str}, float_precision="round_trip")
    rows = []
    for name, current in (("l40s_anchor", anchor), ("l40s_all_methods", frame), ("previous_a100", previous)):
        current = current.loc[current.method == "tabpfn_full"].set_index("sampleid").sort_index()
        if not current.index.equals(old.index):
            raise ValueError("Reproduction patients differ")
        np.testing.assert_array_equal(current.label.to_numpy(), old.y_true.to_numpy())
        delta = current.probability.to_numpy() - old.y_prob.to_numpy()
        old_auc = roc_auc_score(old.y_true, old.y_prob)
        new_auc = roc_auc_score(current.label, current.probability)
        rows.append({"disease": disease, "run": name, "n": len(current), "original_auc": old_auc,
                     "rerun_auc": new_auc, "auc_delta": new_auc - old_auc,
                     "identical_probabilities": bool(np.all(delta == 0)),
                     "mean_abs_probability_delta": float(np.abs(delta).mean()),
                     "max_abs_probability_delta": float(np.abs(delta).max())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/external_pgs/correction_20261011")
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.run_dir)
    contract = verify_contract(root)
    for name, digest in contract["legacy_artifact_sha256"].items():
        if sha256_file(name) != digest:
            raise ValueError(f"Original reference changed: {name}")
    frames, summaries, subgroups, reproduction, fits_all, selections = [], [], [], [], [], []
    for disease in DISEASES:
        frame = read_verified(root, "development", disease)
        if set(frame.method) != set(METHODS) or frame.duplicated(["sampleid", "method"]).any():
            raise ValueError("Incomplete method coverage or duplicate predictions")
        _, _, context, query = load_frames(contract, disease)
        truth = query.set_index("sampleid").sort_index()
        for method, current in frame.groupby("method", observed=True):
            current = current.set_index("sampleid").sort_index()
            if not current.index.equals(truth.index):
                raise ValueError("Official patients were filtered")
            np.testing.assert_array_equal(current.label.to_numpy(), truth[disease].to_numpy())
            np.testing.assert_array_equal(current.hard_label.to_numpy(), truth.hard_label.to_numpy())
            np.testing.assert_array_equal(current.kinship_group.to_numpy(), truth.kinship_group.to_numpy())
            if not np.isfinite(current.probability).all() or not current.probability.between(0, 1).all():
                raise ValueError("Invalid predicted probabilities")
            summaries.append({"evaluation": contract["evaluation"], "prs_source": "frozen_external_pgs",
                              "disease": disease, "method": method, **metrics(current)})
            for group, subset in current.groupby("hard_label", observed=True):
                subgroups.append({"evaluation": contract["evaluation"],
                                  "disease": disease, "method": method, "ancestry_group": group,
                                  **metrics(subset)})
        fitted = pd.read_csv(root / "development" / disease / "fits.csv")
        bounds = {"final": (ids_hash(context), ids_hash(query), len(context), len(query))}
        for i, (a, b) in enumerate(inner_splits(context, disease, contract["seed"])):
            bounds[i] = (ids_hash(context.iloc[a]), ids_hash(context.iloc[b]), len(a), len(b))
        for row in fitted.itertuples():
            key = "final" if row.stage == "final" else int(row.inner_fold)
            if (row.context_ids_sha256, row.query_ids_sha256, row.n_context, row.n_query) != bounds[key]:
                raise ValueError("Model fitting or selection left the allowed patient boundary")
        fits_all.append(fitted.assign(disease=disease))
        selections.append(pd.read_csv(root / "development" / disease / "selection.csv").assign(disease=disease))
        reproduction.extend(reproduction_rows(root, disease, frame))
        frames.append(frame)
    reproduced = pd.DataFrame(reproduction)
    if not reproduced.loc[reproduced.run.str.startswith("l40s_"), "identical_probabilities"].all():
        raise ValueError("The original full teacher was not reproduced exactly; review before reporting")
    print("Verified all official patients, external sources and training-only fit boundaries", flush=True)
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(cluster_comparisons, f, args.n_bootstraps, 20261011 + i)
                   for i, f in enumerate(frames)]
        comparisons = pd.concat([future.result().assign(disease=d)
                                 for d, future in zip(DISEASES, futures, strict=True)],
                                ignore_index=True)
    comparisons["holm_p"] = np.nan
    comparisons["evaluation"] = contract["evaluation"]
    primary = comparisons.primary.astype(bool)
    if int(primary.sum()) != 6:
        raise ValueError("Expected exactly six primary AUC comparisons")
    comparisons.loc[primary, "holm_p"] = multipletests(comparisons.loc[primary, "p_centered_bootstrap"],
                                                       method="holm")[1]
    out = root / "report"
    out.mkdir(exist_ok=False)
    tables = {"overall_metrics.csv": pd.DataFrame(summaries), "metrics_by_ancestry.csv": pd.DataFrame(subgroups),
              "paired_patient_comparisons.csv": comparisons, "reproduction.csv": reproduced,
              "fit_audit.csv": pd.concat(fits_all), "selection.csv": pd.concat(selections)}
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    write_json(out / "VERIFIED.json", {
        "status": "complete", "evaluation": contract["evaluation"], "methods": list(METHODS),
        "contract_sha256": sha256_file(root / "CONTRACT.json"),
        "primary_comparisons": 6, "bootstrap_unit": "patient", "n_bootstraps": args.n_bootstraps,
        "inference": "conditional on saved development predictions; family structure and adaptive history unresolved",
        "source_code_sha256": {p: sha256_file(p) for p in
                               (__file__, "scripts/report_internal_prs_benchmark.py",
                                "scripts/report_internal_prs_qc_sensitivity.py")},
        "output_sha256": {name: sha256_file(out / name) for name in tables},
    })
    print(tables["reproduction.csv"].to_string(index=False), flush=True)
    print(comparisons.loc[primary, ["disease", "comparison", "delta", "ci_low", "ci_high", "holm_p"]]
          .to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
