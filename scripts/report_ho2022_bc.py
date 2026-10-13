"""Verify and report paired Ho2022 expansion results on fixed female cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.ancestry_performance import paired_bootstrap_comparisons, prediction_metrics
from decipehr.external_pgs import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/external_pgs/bc_ho2022_preparation_20261012")
    args = parser.parse_args()
    root = Path(args.run_dir)
    out = root / "report"
    out.mkdir(exist_ok=False)
    methods = ["multi_prs_pc_lr", "tabpfn_prs_pc"]
    frames, hashes, overall, pairs, ancestry = {}, {}, [], [], []
    for scope in ["development", "oof"]:
        filename = "validation_predictions.csv" if scope == "development" else "oof_predictions.csv"
        for panel in ["baseline", "expanded"]:
            directory = root / "evaluation" / scope / panel
            run = json.loads((directory / "run.json").read_text())
            if run["status"] != "complete":
                raise ValueError("Incomplete benchmark")
            for key, digest in run["input_sha256"].items():
                if sha256_file(run["arguments"][key]) != digest:
                    raise ValueError(f"Changed evaluation input: {key}")
            path = directory / filename
            frame = pd.read_csv(path, dtype={"sampleid": str}, float_precision="round_trip")
            if frame.duplicated(["sampleid", "method"]).any() or set(frame.method) != set(methods):
                raise ValueError("Incomplete or duplicated prediction roster")
            frames[scope, panel] = frame
            metrics = pd.read_csv(directory / "summary_metrics.csv")
            ancestry.append(metrics.assign(scope=scope, panel=panel))
            for method in methods:
                part = frame.loc[frame.method.eq(method)]
                for scale, col in [("raw", "y_prob"), ("official_wrapper", "wrapper_prob")]:
                    prob = part[col].to_numpy()
                    if scale == "raw" and method == "tabpfn_prs_pc":
                        prob = prob.astype(np.float32)
                    measured = prediction_metrics(part.y_true.to_numpy(), prob)
                    stored = metrics.loc[metrics.method.eq(method) & metrics.prediction_scale.eq(scale)
                                         & metrics.ancestry_group.eq("ALL")].iloc[0]
                    for name in ["auc", "nagelkerke_r2", "brier", "log_loss"]:
                        if not np.isclose(measured[name], stored[name], rtol=0, atol=1e-6):
                            raise ValueError(f"Stored metric mismatch: {scope}/{panel}/{method}/{scale}/{name}")
                    overall.append({"scope": scope, "panel": panel, "method": method,
                                    "prediction_scale": scale, **measured})
            for file in [path, directory / "summary_metrics.csv", directory / "run.json", directory / "bc/fit_audit.csv"]:
                hashes[str(file)] = sha256_file(file)
        for scale, col in [("raw", "y_prob"), ("official_wrapper", "wrapper_prob")]:
            wide = None
            for panel in ["baseline", "expanded"]:
                for method in methods:
                    panel_frame = frames[scope, panel]
                    part = panel_frame.loc[panel_frame.method.eq(method)]
                    keys = ["sampleid", "disease", "fold", "y_true", "ancestry_group"]
                    field = f"{panel}__{method}"
                    part = part[keys + [col]].rename(columns={col: field})
                    if wide is None:
                        wide = part
                    else:
                        if set(wide.sampleid) != set(part.sampleid):
                            raise ValueError("Unmatched paired cohorts")
                        merged = wide.merge(part, on=keys, validate="one_to_one")
                        if len(merged) != len(wide):
                            raise ValueError("Paired outcomes, folds or ancestry differ")
                        wide = merged
            # The frozen contrast families concern whole cohorts. Subgroup point
            # estimates remain in ancestry_metrics.csv; do not add unplanned tests.
            result = paired_bootstrap_comparisons(
                wide.assign(ancestry_group=pd.NA), n_bootstraps=2000, seed=42, family_size=8,
                comparisons=[(method, f"baseline__{method}", f"expanded__{method}") for method in methods],
                metric_names=["auc", "nagelkerke_r2"],
            ).assign(scope=scope, prediction_scale=scale)
            pairs.append(result)
            print(f"Verified paired {scope}/{scale}", flush=True)
    pd.DataFrame(overall).to_csv(out / "overall_metrics.csv", index=False)
    pd.concat(ancestry, ignore_index=True).to_csv(out / "ancestry_metrics.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(out / "paired_differences.csv", index=False)
    coverage = root / "scoring_new/scoring_qc.csv"
    pd.read_csv(coverage).to_csv(out / "new_score_coverage.csv", index=False)
    hashes[str(coverage)] = sha256_file(coverage)
    write_json(out / "VERIFIED.json", {
        "input_sha256": hashes,
        "output_sha256": {p.name: sha256_file(p) for p in out.glob("*.csv")},
        "bootstrap": "2000 outcome-stratified paired patient replicates, seed42; separate eight-contrast families for original raw and supplementary wrapper results",
        "subgroups": "Descriptive point estimates only; paired intervals cover whole-cohort contrasts",
        "limitations": ["Development has been used previously", "Intervals condition on saved predictions and omit refitting/selection uncertainty", "Family separation not verified"],
        "female_eligibility": True, "status": "complete",
    })


if __name__ == "__main__":
    main()
