"""Recompute full-model versus training-selected Disco from frozen predictions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.ancestry_performance import paired_bootstrap_comparisons, prediction_metrics
from decipehr.external_pgs import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/external_pgs/progress_review_20261012"
DATA = BASE / "academic_revision_20261012"
OUT = BASE / "primary_disco_revision_20261012"


def main():
    OUT.mkdir(exist_ok=True)
    receipt = json.loads((DATA / "VERIFIED.json").read_text())
    for name, digest in receipt["output_sha256"].items():
        assert sha256_file(DATA / name) == digest
    sources = [DATA / "predictions.private.parquet", DATA / "paired_comparisons.csv",
               DATA / "development_metrics.csv", OUT / "ANALYSIS_PLAN.json"]
    predictions = pd.read_parquet(sources[0])
    metrics = pd.read_csv(sources[2], float_precision="round_trip")
    results, checks = [], []
    for disease in ("T2D", "cad", "bc"):
        frame = predictions.loc[predictions.disease.eq(disease) &
                                predictions.method.isin(["tabpfn_full", "selected_disco"])].copy()
        wide = frame.pivot(index=["sampleid", "label", "hard_label"],
                           columns="method", values="probability").reset_index()
        assert not wide.isna().any().any() and not wide.sampleid.duplicated().any()
        for method in ("tabpfn_full", "selected_disco"):
            actual = prediction_metrics(wide.label, wide[method])
            expected = metrics.loc[metrics.disease.eq(disease) & metrics.method.eq(method) &
                                   metrics.ancestry_group.eq("ALL")].iloc[0]
            for key in ("auc", "nagelkerke_r2", "log_loss", "brier"):
                assert abs(actual[key] - expected[key]) < 1e-12
        choice_path = ROOT / f"outputs/external_pgs/correction_20261011/development/{disease}/selection.csv"
        sources.append(choice_path)
        choice = pd.read_csv(choice_path, float_precision="round_trip")
        selected = choice.loc[choice.method.eq("selected_disco")]
        assert len(selected) == 1
        candidates = choice.loc[choice.method.isin(["disco_linear", "disco_spline"])]
        assert selected.iloc[0].inner_auc == candidates.inner_auc.max()
        checks.append({"disease": disease, "n": len(wide), "cases": int(wide.label.sum()),
                       "selection": selected.iloc[0].chosen_method,
                       "identical_patient_sets_labels_and_groups": True})
        wide = wide.rename(columns={"label": "y_true", "hard_label": "ancestry_group"})
        wide["disease"] = disease
        for metric in ("auc", "nagelkerke_r2"):
            print(f"Bootstrap {disease}: {metric}", flush=True)
            paired = paired_bootstrap_comparisons(
                wide, n_bootstraps=2000, seed=20261012,
                comparisons=(("full_vs_disco", "selected_disco", "tabpfn_full"),),
                metric_names=(metric,), family_size=12,
            )
            subgroup = paired.ancestry_group.ne("ALL")
            paired.loc[subgroup, ["familywise_ci_low", "familywise_ci_high", "family_size"]] = np.nan
            paired["inference"] = np.where(subgroup, "exploratory nominal 95% CI",
                                            "post-hoc 12-comparison Bonferroni")
            results.append(paired)
    new = pd.concat(results, ignore_index=True)
    assert len(new) == 42
    original = pd.read_csv(DATA / "paired_comparisons.csv", float_precision="round_trip")
    combined = pd.concat([original.loc[original.comparison.eq("full_vs_single_lr")], new], ignore_index=True)
    assert len(combined) == 84
    combined.to_csv(OUT / "primary_paired_comparisons.csv", index=False)
    write_json(OUT / "DISCO_PRIMARY_VERIFIED.json", {
        "status": "verified", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_checks": checks, "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
        "new_model_fits": 0, "primary_inputs": {"T2D": 26, "cad": 7, "bc": 3},
        "overall_family_size": 12, "n_bootstraps": 2000,
        "output_sha256": {"primary_paired_comparisons.csv": sha256_file(OUT / "primary_paired_comparisons.csv")},
        "script_sha256": sha256_file(Path(__file__)),
    })
    print(new.loc[new.ancestry_group.eq("ALL"), ["disease", "metric", "candidate_minus_reference",
          "ci_low", "ci_high", "familywise_ci_low", "familywise_ci_high"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
