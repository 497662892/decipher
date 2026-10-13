"""Supplement the frozen PRS review with R² checks, paired intervals and calibration.

No model fitting. Patient-level inputs are read locally; only aggregates are exported.
Run with: uv run --no-sync python scripts/supplement_external_prs_review_r2.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from decipehr.ancestry_performance import paired_bootstrap_comparisons
from decipehr.metrics import nagelkerke_r2
from decipehr.prediction_diagnostics import decile_table

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/external_pgs/progress_review_20261012"
OUT = BASE / "html"
DISEASES = ("T2D", "cad", "bc")
N_BOOTSTRAPS = 2000
SEED = 20261012
FAMILY_SIZE = 29
SOURCES: dict[str, str] = {}
CHECKS: list[dict] = []


def tracked(relative):
    path = ROOT / relative
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(relative, **kwargs):
    return pd.read_csv(tracked(relative), float_precision="round_trip", **kwargs)


def verify_receipt(relative, key, parent=ROOT):
    receipt = json.loads(tracked(relative).read_text())
    for name, expected in receipt[key].items():
        path = parent / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name


def check(frame, expected, disease, method, scope, probability="probability", label="label"):
    assert frame.sampleid.is_unique
    assert len(frame) == expected["n"] and frame[label].sum() == expected["cases"]
    p = frame[probability].to_numpy(dtype=float)
    y = frame[label].to_numpy(dtype=int)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    # This is the convention already used by the frozen reports and bootstrap helper.
    p = np.clip(p, 1e-6, 1 - 1e-6)
    r2 = nagelkerke_r2(y, p)
    auc = roc_auc_score(y, p)
    assert abs(r2 - expected.nagelkerke_r2) < 1e-12, (scope, disease, method, r2)
    assert abs(auc - expected.auc) < 1e-12, (scope, disease, method, auc)
    # Independent identity from saved log loss and the observed case fraction.
    prevalence = y.mean()
    null_loss = -(prevalence * np.log(prevalence) + (1 - prevalence) * np.log1p(-prevalence))
    from_loss = -np.expm1(2 * (expected.log_loss - null_loss)) / -np.expm1(-2 * null_loss)
    assert abs(r2 - from_loss) < 1e-12
    CHECKS.append({"scope": scope, "disease": disease, "method": method,
                   "n": len(y), "cases": int(y.sum()), "auc": auc,
                   "nagelkerke_r2": r2, "r2_from_saved_log_loss": from_loss})


def wide(frame, disease, probability="probability", label="label"):
    assert not frame.duplicated(["sampleid", "method"]).any()
    assert frame.groupby("sampleid")[label].nunique().eq(1).all()
    result = frame.pivot(index=["sampleid", label], columns="method", values=probability).reset_index()
    assert not result.isna().any().any()
    result = result.rename(columns={label: "y_true"})
    result["disease"] = disease
    # Overall intervals only; do not expand the comparison family with subgroup tests.
    result["ancestry_group"] = np.nan
    return result


def bootstrap(frame, comparisons, scope):
    print(f"R² bootstrap: {scope}, {frame.disease.iloc[0]}, {len(comparisons)} comparisons", flush=True)
    result = paired_bootstrap_comparisons(
        frame, n_bootstraps=N_BOOTSTRAPS, seed=SEED, comparisons=comparisons,
        metric_names=("nagelkerke_r2",), family_size=FAMILY_SIZE,
    )
    assert len(result) == len(comparisons)
    return result.assign(scope=scope, exploratory=True, seed=SEED)


def main():
    OUT.mkdir(exist_ok=True)
    verify_receipt(BASE.relative_to(ROOT) / "REPORT_VERIFICATION.json", "files_sha256")
    tracked("src/decipehr/metrics.py")
    tracked("src/decipehr/ancestry_performance.py")
    tracked("src/decipehr/prediction_diagnostics.py")
    metrics = read(BASE.relative_to(ROOT) / "current_development_metrics.csv")
    pairs, calibration = [], []
    current = {}
    common_pairs = (
        ("full_vs_covariates", "tabpfn_covariates", "tabpfn_full"),
        ("full_vs_single_tabpfn", "tabpfn_single", "tabpfn_full"),
        ("continuous_vs_no_pc", "tabpfn_no_pc", "tabpfn_full"),
        ("varying_vs_fixed", "flex_fixed", "flex_varying"),
        ("hard_vs_continuous", "tabpfn_full", "tabpfn_hard"),
        ("full_vs_selected_baseline", "selected_baseline", "tabpfn_full"),
    )
    for disease in DISEASES:
        frame = read(f"outputs/external_pgs/correction_20261011/development/{disease}/predictions.csv",
                     dtype={"sampleid": str})
        current[disease] = wide(frame, disease)
        for method, subset in frame.groupby("method"):
            expected = metrics.loc[(metrics.disease == disease) & (metrics.method == method)].iloc[0]
            check(subset, expected, disease, method, "current_development")
        pair_list = common_pairs
        if disease != "T2D":
            pair_list += (("matched_vs_selected_disco", "selected_disco", "tabpfn_matched"),)
        pairs.append(bootstrap(current[disease], pair_list, "current_development"))
        f = frame.loc[frame.method == "tabpfn_full"].sort_values("sampleid")
        values = decile_table(f.label.to_numpy(), f.probability.to_numpy(),
                              f.probability.to_numpy(), np.full(len(f), f.label.mean()))
        calibration.append(values.assign(disease=disease, method="tabpfn_full"))

    m25 = read(BASE.relative_to(ROOT) / "dprism25_metrics.csv")
    predictions25 = pd.read_parquet(tracked("outputs/external_pgs/dprism25_comparison_20261011/report/predictions.parquet"))
    for (scope, method), subset in predictions25.groupby(["evaluation", "method"]):
        expected = m25.loc[(m25.evaluation == scope) & (m25.method == method)].iloc[0]
        check(subset, expected, "T2D", method, "dprism25_" + scope)
    pairs.append(bootstrap(wide(predictions25.loc[predictions25.evaluation == "development"], "T2D"), (
        ("same_25_sources", "selected_disco", "tabpfn_dprism25"),
        ("direct25_vs_meta5", "tabpfn_matched", "tabpfn_dprism25"),
        ("same_5_metascores", "selected_disco", "tabpfn_matched"),
    ), "dprism25_development"))

    soft = read("outputs/external_pgs/pc_default_update/historical_report_check/validation_predictions.csv",
                dtype={"sampleid": str})
    soft_metrics = read("outputs/external_pgs/pc_default_update/historical_report_check/summary_metrics.csv")
    soft_points = []
    for disease in DISEASES:
        subset = soft.loc[(soft.disease == disease) & soft.method.isin(["tabpfn_prs_pc", "tabpfn_prs_pc_soft"])].copy()
        for method, frame in subset.groupby("method"):
            expected = soft_metrics.loc[(soft_metrics.disease == disease) & (soft_metrics.method == method)
                                        & (soft_metrics.prediction_scale == "raw") & (soft_metrics.ancestry_group == "ALL")].iloc[0]
            check(frame, expected, disease, method, "soft_ancestry", "y_prob", "y_true")
            soft_points.append(expected.to_dict())
        matrix = wide(subset, disease, "y_prob", "y_true")
        reference = current[disease].set_index("sampleid").sort_index()
        matrix = matrix.set_index("sampleid").sort_index()
        assert matrix.index.equals(reference.index)
        np.testing.assert_array_equal(matrix.y_true, reference.y_true)
        np.testing.assert_allclose(matrix.tabpfn_prs_pc, reference.tabpfn_full, rtol=0, atol=1e-12)
        pairs.append(bootstrap(matrix.reset_index(), (("full_vs_pc", "tabpfn_prs_pc", "tabpfn_prs_pc_soft"),), "soft_ancestry"))

    hist_dir = ROOT / "outputs/external_pgs/run_20261010/comparison_with_previous_official_validation"
    verify_receipt(hist_dir.relative_to(ROOT) / "VERIFIED.json", "source_sha256")
    hist = read(BASE.relative_to(ROOT) / "historical_wrapper_metrics.csv")
    old = read("outputs/ancestry_stratified_best_single_vs_tabpfn3/patient_predictions.csv", dtype={"sampleid": str})
    new = read("outputs/external_pgs/run_20261010/report/validation_predictions.csv", dtype={"sampleid": str})
    for disease in DISEASES:
        a = old.loc[old.disease == disease, ["sampleid", "y_true", "tabpfn3_full"]].copy()
        a = a.rename(columns={"tabpfn3_full": "probability", "y_true": "label"}).assign(method="previous_wrapper")
        b = new.loc[(new.disease == disease) & (new.method == "tabpfn_prs_pc"), ["sampleid", "y_true", "wrapper_prob"]].copy()
        b = b.rename(columns={"wrapper_prob": "probability", "y_true": "label"}).assign(method="external_pc_wrapper")
        for frame in (a, b):
            method = frame.method.iloc[0]
            expected = hist.loc[(hist.disease == disease) & (hist.model == method)].iloc[0]
            check(frame, expected, disease, method, "historical_wrapper")
        pairs.append(bootstrap(wide(pd.concat([a, b], ignore_index=True), disease), (
            ("external_pc_wrapper_vs_previous", "previous_wrapper", "external_pc_wrapper"),
        ), "historical_wrapper"))

    results = pd.concat(pairs, ignore_index=True)
    assert len(results) == FAMILY_SIZE
    results.to_csv(OUT / "paired_r2.csv", index=False)
    pd.DataFrame(CHECKS).to_csv(OUT / "r2_recalculation.csv", index=False)
    pd.concat(calibration, ignore_index=True).to_csv(OUT / "calibration_deciles.csv", index=False)
    pd.DataFrame(soft_points).to_csv(OUT / "soft_ancestry_metrics.csv", index=False)
    for name, digest in SOURCES.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    receipt = {
        "status": "verified", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "point_rows_verified": len(CHECKS), "source_sha256": SOURCES,
        "bootstrap": {"resamples": N_BOOTSTRAPS, "seed": SEED, "unit": "outcome-stratified paired patient",
                      "ordinary_ci": 0.95, "family_size": FAMILY_SIZE,
                      "family_scope": "All 29 new overall R² comparisons; Bonferroni percentile intervals; exploratory post hoc.",
                      "inference": "Conditional on frozen predictions, not training, adaptive selection or family-dependence uncertainty."},
        "new_model_fits": 0, "patient_data_exported": False,
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.csv")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "R2_VERIFICATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "verified", "point_rows": len(CHECKS), "r2_comparisons": len(results)}), flush=True)


if __name__ == "__main__":
    main()
