#!/usr/bin/env python3
"""Explain OR90 versus R2 using May final configurations on original validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, roc_curve

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.models import fit_logistic_predict_proba
from decipehr.prediction_diagnostics import (
    decile_table,
    log_likelihood_terms,
    metric_row,
    paired_bootstrap,
    risk_deciles,
    threshold_table,
    validate_predictions,
)

PACKAGE = Path("submissions/final_phase/Go Gene!_audited_pipeline_with_trainval_5cv")
DEFAULT_OUT = "outputs/may_final_validation_prediction_analysis"
COVARIATES = ["sex", "age_sd", *[f"PC{i}_sd" for i in range(1, 11)]]
CONFIGURATIONS = {"cad": 3, "bc": 2, "T2D": 3}
FRACTIONS = [.01, .02, .03, .05, .075, .1, .125, .15, .2, .3, .4, .5]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility-config", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    os.umask(0o077)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rules = load_eligibility_rules(args.eligibility_config)
    if "bc" not in rules:
        raise ValueError("Breast cancer eligibility rule is required")
    train_path = Path("data/processed/phenotype.csv")
    val_path = Path("data/processed/official_validation_phenotype.csv")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    assert train.sampleid.is_unique and val.sampleid.is_unique
    assert not set(train.sampleid) & set(val.sampleid)
    final_record = json.loads((PACKAGE / "run_record.json").read_text())
    sources = [train_path, val_path, Path(args.eligibility_config), PACKAGE / "run_record.json",
               PACKAGE / "tabpfn_fit_summary.csv", Path(__file__),
               Path("src/decipehr/prediction_diagnostics.py")]
    metrics, audits, probability_frames = [], [], []
    bc = None
    for disease, drop in CONFIGURATIONS.items():
        root = Path(f"outputs/tabpfn3_trainval_drop{drop}_disease_specific/clip1e-3_ne16")
        selection_paths = [
            Path("outputs/tabpfn3_final_trainval_selected_drop3_drop2_drop3_inputs")
            / f"{disease}.selected_final_models.csv",
            Path(f"outputs/tabpfn3_trainval_drop{drop}_disease_specific_inputs")
            / f"{disease}.selected_final_models.csv",
        ]
        pd.testing.assert_frame_equal(*(pd.read_csv(p) for p in selection_paths))
        sources.extend(selection_paths)
        fit_path = root / "tabpfn_fit_summary.csv"
        fit = pd.read_csv(fit_path).set_index("disease").loc[disease]
        final_fit = next(x for x in final_record["fit_summaries"] if x["disease"] == disease)
        assert int(fit.n_prs_features) == final_fit["n_prs_features"]
        assert fit.tabpfn_model_version == final_fit["tabpfn_model_version"] == "v3"
        assert fit.score_scale == "train_sd"
        tr = filter_eligible(train, disease, rules)
        va = filter_eligible(val, disease, rules)
        if disease == "bc":
            assert tr.sex.eq("Female").all() and va.sex.eq("Female").all()
        sizes = len(tr), len(va)
        score_col = f"prs_{disease}"
        score_paths = [root / "covariate_adjusted" / name
                       for name in ["train_prs_scores.csv", "validation_prs_scores.csv"]]
        tr = tr.merge(pd.read_csv(score_paths[0])[["sampleid", score_col]],
                      on="sampleid", how="left", validate="one_to_one")
        va = va.merge(pd.read_csv(score_paths[1])[["sampleid", score_col]],
                      on="sampleid", how="left", validate="one_to_one")
        assert (len(tr), len(va)) == sizes
        assert sizes == (int(fit.n_context), int(fit.n_validation_scored))
        assert np.isfinite(tr[score_col]).all() and np.isfinite(va[score_col]).all()
        y = va[disease].to_numpy(dtype=int)
        base = fit_logistic_predict_proba(tr, va, disease, COVARIATES,
                                          patient_col="sampleid", class_weight=None)
        adjusted = fit_logistic_predict_proba(tr, va, disease, COVARIATES + [score_col],
                                              patient_col="sampleid", class_weight=None)
        direct = expit(va[score_col].to_numpy() * fit.logit_train_sd + fit.logit_train_mean)
        validate_predictions(va.assign(disease=disease, label=y, score=va[score_col],
                                       direct_probability=direct))
        point_path = root / "official_validation_covariate_adjusted/point_metrics.csv"
        expected = pd.read_csv(point_path)
        expected = expected.loc[expected.disease.eq(disease)].set_index("method")
        errors = []
        for model, p in {"covariates": base, "direct": direct, "adjusted": adjusted}.items():
            row = metric_row(y, p)
            metrics.append({"disease": disease, "model": model, **row})
            if model != "direct":
                reference = expected.loc["covariate_only" if model == "covariates"
                                         else f"prs_covariate_{disease}"]
                errors.extend(abs(row[k] - reference[k])
                              for k in ["auc", "brier", "nagelkerke_r2", "top10_vs_lower90_or"])
            probability_frames.append(pd.DataFrame({"sampleid": va.sampleid, "disease": disease,
                                                   "model": model, "label": y, "probability": p}))
        assert max(errors) < 1e-10, (disease, errors)
        assert np.array_equal(risk_deciles(direct) == 10,
                              risk_deciles(va[score_col].to_numpy()) == 10)
        audits.append({"disease": disease, "train_n": len(tr), "validation_n": len(va),
                       "validation_cases": int(y.sum()), "n_prs": int(fit.n_prs_features),
                       "configuration": str(root), "max_saved_metric_abs_error": max(errors),
                       "train_validation_overlap": 0, "female_eligibility_pass": True,
                       "direct_score_and_probability_same_top10": True,
                       "exact_final_prs_selection_manifest_match": True})
        sources.extend([fit_path, point_path, *score_paths])
        if disease == "bc":
            bc = va.assign(label=y, covariates=base, direct=direct, adjusted=adjusted,
                           raw_score=va[score_col])
    assert bc is not None
    all_metrics = pd.DataFrame(metrics)
    all_metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "cohort_and_provenance_audit.csv", index=False)
    # Protected local audit output, excluded from the aggregate HTML and figures.
    pd.concat(probability_frames).to_csv(out / "validation_predictions_private.csv", index=False)
    y = bc.label.to_numpy()
    probabilities = {m: bc[m].to_numpy() for m in ["covariates", "direct", "adjusted"]}
    deciles, thresholds, curves, histograms, distribution_summary = [], [], [], [], []
    for model, p in probabilities.items():
        deciles.append(decile_table(y, p, p, probabilities["covariates"]).assign(model=model))
        thresholds.append(threshold_table(y, p, FRACTIONS).assign(model=model))
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        curves.append(pd.DataFrame({"x": fpr, "y": tpr, "curve": "roc", "model": model}))
        curves.append(pd.DataFrame({"x": recall, "y": precision, "curve": "pr", "model": model}))
        for label, name in [(0, "Control"), (1, "Case")]:
            values = p[y == label]
            counts, bins = np.histogram(values, bins=np.linspace(0, 1, 101))
            histograms.append(pd.DataFrame({"left": bins[:-1], "right": bins[1:],
                                            "count": counts, "density": counts / len(values) / np.diff(bins),
                                            "model": model, "label": name}))
            distribution_summary.append({"model": model, "label": name, "n": len(values),
                                         "mean": values.mean(), "sd": values.std(),
                                         **{f"p{q}": np.percentile(values, q)
                                            for q in [0, 10, 25, 50, 75, 90, 95, 99, 100]},
                                         "above_0p4": int((values > .4).sum())})
    decile_data = pd.concat(deciles, ignore_index=True)
    decile_data.to_csv(out / "bc_risk_deciles.csv", index=False)
    pd.concat(thresholds).to_csv(out / "bc_threshold_sensitivity.csv", index=False)
    pd.concat(curves).to_csv(out / "bc_roc_pr_curves.csv", index=False)
    pd.concat(histograms).to_csv(out / "bc_probability_histograms.csv", index=False)
    pd.DataFrame(distribution_summary).to_csv(out / "bc_distribution_summary.csv", index=False)
    print("Cohorts, frozen-score reconstruction and saved-metric reproduction passed.", flush=True)
    draws, intervals = paired_bootstrap(y, probabilities, args.n_bootstrap, args.seed)
    draws.to_csv(out / "bc_bootstrap_draws.csv", index=False)
    intervals.to_csv(out / "bc_metric_intervals.csv", index=False)
    deltas = []
    points = all_metrics.loc[all_metrics.disease.eq("bc")].set_index("model")
    for before, after in [("covariates", "adjusted"), ("covariates", "direct"), ("direct", "adjusted")]:
        a = draws.loc[draws.model.eq(after)].set_index("repeat")
        b = draws.loc[draws.model.eq(before)].set_index("repeat")
        for metric in ["auc", "nagelkerke_r2", "top10_vs_lower90_or", "brier", "log_loss"]:
            diff = a[metric] - b[metric]
            low, high = diff.quantile([.025, .975])
            deltas.append({"before": before, "after": after, "metric": metric,
                           "delta": points.loc[after, metric] - points.loc[before, metric],
                           "low": low, "high": high, "n_valid": len(diff)})
    pd.DataFrame(deltas).to_csv(out / "bc_paired_deltas.csv", index=False)
    # Membership movements explain the ranking change caused by additional covariates.
    bc["direct_top10"] = risk_deciles(probabilities["direct"]) == 10
    bc["adjusted_top10"] = risk_deciles(probabilities["adjusted"]) == 10
    moves = bc.groupby(["direct_top10", "adjusted_top10"]).agg(
        n=("label", "size"), cases=("label", "sum"), case_rate=("label", "mean"),
        age_sd_mean=("age_sd", "mean"), direct_p_mean=("direct", "mean"),
        adjusted_p_mean=("adjusted", "mean"),
    ).reset_index()
    moves.to_csv(out / "bc_top10_membership_changes.csv", index=False)
    # Paired likelihood decomposition in the fixed adjusted-model deciles.
    bc["decile"] = risk_deciles(probabilities["adjusted"])
    bc["ll_adjusted_minus_direct"] = (log_likelihood_terms(y, probabilities["adjusted"])
                                       - log_likelihood_terms(y, probabilities["direct"]))
    bc.groupby("decile").ll_adjusted_minus_direct.sum().reset_index().to_csv(
        out / "bc_wrapper_likelihood_change.csv", index=False)
    bc["age_quintile"] = pd.qcut(bc.age_sd, 5, labels=False, duplicates="drop") + 1
    age = bc.groupby("age_quintile").agg(
        n=("label", "size"), cases=("label", "sum"), observed=("label", "mean"),
        direct_predicted=("direct", "mean"), adjusted_predicted=("adjusted", "mean"),
        age_sd_min=("age_sd", "min"), age_sd_max=("age_sd", "max"),
    )
    age.to_csv(out / "bc_age_quintiles.csv")
    bc["risk_region"] = np.select([bc.decile.le(2), bc.decile.eq(10)],
                                  ["Lowest 20%", "Highest 10%"], default="Middle 70%")
    bc.groupby("risk_region").agg(n=("label", "size"), cases=("label", "sum"),
                                 age_sd_mean=("age_sd", "mean"), age_sd_median=("age_sd", "median"),
                                 observed=("label", "mean")).to_csv(out / "bc_risk_region_age.csv")
    # Fixed 10% group size: exchange one case with one control, keeping total cases fixed.
    m = points.loc["adjusted"]
    sensitivity = []
    for change in [-10, -5, -1, 0, 1, 5, 10]:
        a = int(m.top10_cases) + change
        b = int(m.top10_n) - a
        c = int(m.cases) - a
        d = int(m.lower90_n) - c
        sensitivity.append({"net_case_exchange": change, "cases_top": a,
                            "or90": a * d / (b * c)})
    pd.DataFrame(sensitivity).to_csv(out / "bc_boundary_exchange_illustration.csv", index=False)
    # Published rule uses OR90 then R2. It does not specify the mapping from z-logit to probability.
    record = {
        "analysis_scope": "Original official validation only; BC restricted to eligible women",
        "user_confirmed_package": str(PACKAGE),
        "validation_model_relation": "Same May final v3 PRS panels before final train-val refit; saved original-training-context scores",
        "score_profiles": {"direct": "Saved standardized logits inverted with training mean/sd; clipped TabPFN probability; final submission ranking style",
                           "adjusted": "Existing evaluation: training-fit logistic(score + sex + age + PC1-PC10)",
                           "covariates": "Existing training-fit logistic(sex + age + PC1-PC10)"},
        "bootstrap": {"n": args.n_bootstrap, "seed": args.seed,
                      "method": "Paired non-stratified patient bootstrap, percentile 95% CI; rerank within each resample; no refitting"},
        "deciles": "Stable descending sort; cumulative ceil(n*k/10), so highest decile matches repository OR90",
        "limitations": ["Retrospective selected model; validation influenced PRS/configuration choices",
                        "No training uncertainty or model-selection uncertainty in frozen-score bootstrap",
                        "Direct R2 uses probabilities recovered from clipped logits, not an asserted reproduction of the organizer's final scoring implementation",
                        "Final full-trainval fitted model cannot be evaluated independently on validation patients included in its context",
                        "No holdout or train-val OOF performance in this report"],
        "source_hashes": {str(p): sha256(p) for p in dict.fromkeys(sources)},
        "package_versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels"]},
        "checks": {"original_train_validation_disjoint": True, "saved_metrics_reproduced_at_1e10": True,
                   "same_patients_for_all_bc_models": True, "final_prs_counts_and_version_match": True,
                   "exact_final_prs_selection_manifests_match": True},
    }
    (out / "analysis_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(all_metrics.loc[all_metrics.disease.eq("bc"),
                          ["model", "n", "cases", "auc", "top10_vs_lower90_or", "nagelkerke_r2"]].to_string(index=False), flush=True)
    print(f"Wrote aggregate tables and protected audit predictions to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
