#!/usr/bin/env python3
"""Analyze the exact May submission's saved OOF scores, without fitting a new model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from statsmodels.stats.proportion import proportion_confint

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.metrics import nagelkerke_r2, top_decile_odds_ratio
from decipehr.prediction_diagnostics import (
    decile_table,
    metric_row,
    paired_bootstrap,
    risk_deciles,
    threshold_table,
    validate_predictions,
)

PACKAGE = Path("submissions/final_phase/Go Gene!_audited_pipeline_with_trainval_5cv")
BASELINE = Path("outputs/final_stage_single_prs_performance/trainval_final_score_5fold_covariate_oof_predictions.csv")
PHENOTYPE = Path("data/processed/final/train_val_phenotype.csv")
SPLITS = Path("data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv")
DEFAULT_OUT = "outputs/may_final_oof_prediction_analysis"
FRACTIONS = [.01, .02, .03, .05, .075, .1, .125, .15, .2, .3, .4, .5]


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
        raise ValueError("An explicit breast cancer eligibility rule is required")
    predictions = pd.read_csv(PACKAGE / "trainval_5cv_scores_long.csv")
    validate_predictions(predictions)
    phenotype = pd.read_csv(PHENOTYPE)
    splits = pd.read_csv(SPLITS)
    assert phenotype.sampleid.is_unique
    assert not splits.duplicated(["sampleid", "disease"]).any()
    assert splits.groupby("sampleid").fold.nunique().eq(1).all()
    run = json.loads((PACKAGE / "run_record.json").read_text())
    fits = pd.read_csv(PACKAGE / "tabpfn_fit_summary.csv").set_index("disease")
    fold_audit = pd.read_csv(PACKAGE / "audit_by_fold.csv").set_index(["disease", "fold"])
    baseline = pd.read_csv(BASELINE)
    holdout = pd.read_csv(PACKAGE / "holdout_scores_long.csv")
    assert not set(predictions.sampleid) & set(holdout.sampleid)
    assert holdout.label.isna().all()
    archive_path = PACKAGE / "Go_Gene_final_phase_submission_with_trainval_5cv_standardized_logits.zip"
    sources = [PACKAGE / x for x in ["trainval_5cv_scores_long.csv", "holdout_scores_long.csv",
                                    "run_record.json", "tabpfn_fit_summary.csv", "audit_by_fold.csv"]]
    sources += [archive_path, BASELINE, PHENOTYPE, SPLITS, Path(args.eligibility_config),
                Path(__file__), Path("src/decipehr/prediction_diagnostics.py"),
                Path("src/decipehr/metrics.py"), Path(run["script"])]
    pooled, folds, audits, clipping, fold_distribution = [], [], [], [], []
    bc = None
    with zipfile.ZipFile(archive_path) as archive:
        for disease, g in predictions.groupby("disease", sort=False):
            g = g.copy().reset_index(drop=True)
            fit = fits.loc[disease]
            assert fit.train_prediction_mode == "shared_5fold_oof"
            assert sorted(g.fold.unique()) == list(range(5))
            eligible = filter_eligible(phenotype, disease, rules)
            assert set(g.sampleid) == set(eligible.sampleid)
            if disease == "bc":
                assert eligible.sex.eq("Female").all()
            joined = g.merge(eligible[["sampleid", disease, "age_sd"]], on="sampleid", validate="one_to_one")
            assert joined.label.eq(joined[disease]).all()
            joined = joined.merge(splits.loc[splits.disease.eq(disease), ["sampleid", "fold"]],
                                  on="sampleid", suffixes=("", "_expected"), validate="one_to_one")
            assert joined.fold.eq(joined.fold_expected).all()
            assert len(joined) == len(g) == fit.n_context
            g = joined.merge(baseline.loc[baseline.disease.eq(disease),
                                         ["sampleid", "fold", "y_true", "y_prob"]],
                             on=["sampleid", "fold"], how="left", validate="one_to_one")
            assert len(g) == len(joined) and g.y_true.eq(g.label).all()
            assert g.y_prob.between(0, 1, inclusive="neither").all()
            submission_path = PACKAGE / f"Go Gene!_{disease}_trainval_5cv.txt"
            submitted = pd.read_csv(submission_path, sep="\t")
            zipped = pd.read_csv(io.BytesIO(archive.read(f"Go Gene!_{disease}.txt")), sep="\t")
            score_col = f"{disease}_score"
            assert submitted.sampleid.is_unique and zipped.sampleid.is_unique
            for source_frame in [submitted, zipped.loc[zipped.sampleid.isin(g.sampleid)]]:
                matched = g.merge(source_frame, on="sampleid", validate="one_to_one")
                assert len(matched) == len(g)
                assert np.allclose(matched.score, matched[score_col], rtol=0, atol=1e-12)
            sources.append(submission_path)
            assert np.allclose(g.score * fit.logit_train_sd + fit.logit_train_mean,
                               g.direct_logit, rtol=0, atol=1e-12)
            assert np.allclose(expit(g.direct_logit), np.clip(g.direct_probability, fit.clip_eps, 1-fit.clip_eps),
                               rtol=0, atol=1e-12)
            assert np.array_equal(risk_deciles(g.score.to_numpy()) == 10,
                                  risk_deciles(g.direct_probability.to_numpy()) == 10)
            y = g.label.to_numpy(dtype=int)
            for model, p, score in [("submitted", g.direct_probability.to_numpy(), g.score.to_numpy()),
                                    ("covariates", g.y_prob.to_numpy(), g.y_prob.to_numpy())]:
                result = metric_row(y, p, score)
                assert abs(result["auc"] - roc_auc_score(y, score)) < 1e-12
                pooled.append({"disease": disease, "model": model, **result})
                for fold, group in g.groupby("fold"):
                    yi = group.label.to_numpy(dtype=int)
                    pi = group.direct_probability.to_numpy() if model == "submitted" else group.y_prob.to_numpy()
                    si = group.score.to_numpy() if model == "submitted" else pi
                    folds.append({"disease": disease, "model": model, "fold": fold,
                                  **metric_row(yi, pi, si)})
            pooled_top = risk_deciles(g.score.to_numpy()) == 10
            g["pooled_top10"] = pooled_top
            for fold, group in g.groupby("fold"):
                expected = fold_audit.loc[(disease, fold)]
                assert len(group) == expected.n_validation
                assert group.label.sum() == expected.validation_cases
                assert len(g) - len(group) == expected.n_context
                assert y.sum() - group.label.sum() == expected.context_cases
                fold_distribution.append({"disease": disease, "fold": fold, "n": len(group),
                                          "cases": int(group.label.sum()), "score_mean": group.score.mean(),
                                          "score_sd": group.score.std(ddof=0),
                                          "probability_mean": group.direct_probability.mean(),
                                          "pooled_top10_n": int(group.pooled_top10.sum()),
                                          "pooled_top10_cases": int(group.loc[group.pooled_top10, "label"].sum())})
            clipping.append({"disease": disease, "n_clipped": int(((g.direct_probability < fit.clip_eps) |
                              (g.direct_probability > 1-fit.clip_eps)).sum()),
                             "r2_native_probability": nagelkerke_r2(y, g.direct_probability),
                             "r2_probability_recovered_from_submitted_score": nagelkerke_r2(y, expit(g.direct_logit)),
                             "native_and_submitted_top10_identical": True})
            audits.append({"disease": disease, "n": len(g), "cases": int(y.sum()), "folds": 5,
                           "eligible_ids_exact": True, "labels_exact": True, "fold_assignments_exact": True,
                           "submission_txt_and_zip_scores_exact": True, "baseline_same_patients_and_folds": True,
                           "n_prs_features": int(fit.n_prs_features)})
            if disease == "bc":
                bc = g
    assert bc is not None
    pd.DataFrame(pooled).to_csv(out / "pooled_metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "cohort_audit.csv", index=False)
    pd.DataFrame(clipping).to_csv(out / "score_transform_audit.csv", index=False)
    pd.DataFrame(fold_distribution).to_csv(out / "fold_prediction_distributions.csv", index=False)
    bc[["sampleid", "fold", "label", "score", "direct_logit", "direct_probability", "y_prob"]].to_csv(
        out / "bc_oof_predictions_private.csv", index=False)
    y = bc.label.to_numpy(dtype=int)
    probabilities = {"submitted": bc.direct_probability.to_numpy(), "covariates": bc.y_prob.to_numpy()}
    scores = {"submitted": bc.score.to_numpy(), "covariates": bc.y_prob.to_numpy()}
    deciles, thresholds, histograms, curves = [], [], [], []
    for model, p in probabilities.items():
        deciles.append(decile_table(y, p, scores[model], probabilities["covariates"]).assign(model=model))
        thresholds.append(threshold_table(y, scores[model], FRACTIONS).assign(model=model))
        fpr, tpr, _ = roc_curve(y, scores[model])
        precision, recall, _ = precision_recall_curve(y, scores[model])
        curves += [pd.DataFrame({"model": model, "curve": "roc", "x": fpr, "y": tpr}),
                   pd.DataFrame({"model": model, "curve": "pr", "x": recall, "y": precision})]
        for scale, values, bins in [("probability", p, np.linspace(0, 1, 101)),
                                    ("score", scores[model], np.linspace(scores[model].min(), scores[model].max(), 101))]:
            for label in [0, 1]:
                a = values[y == label]
                counts, edges = np.histogram(a, bins=bins)
                histograms.append(pd.DataFrame({"model": model, "scale": scale, "label": label,
                                                "left": edges[:-1], "right": edges[1:], "count": counts,
                                                "density": counts / len(a) / np.diff(edges)}))
    pd.concat(deciles).to_csv(out / "bc_deciles.csv", index=False)
    pd.concat(thresholds).to_csv(out / "bc_thresholds.csv", index=False)
    pd.concat(histograms).to_csv(out / "bc_histograms.csv", index=False)
    pd.concat(curves).to_csv(out / "bc_roc_pr_curves.csv", index=False)
    order = np.argsort(-scores["submitted"], kind="stable")
    bands = []
    boundaries = [0, .01, .02, .05, .1, .2, .5, 1]
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        idx = order[int(np.ceil(lower * len(y))):int(np.ceil(upper * len(y)))]
        low, high = proportion_confint(y[idx].sum(), len(idx), method="wilson")
        bands.append({"rank_from": lower, "rank_to": upper, "n": len(idx), "cases": int(y[idx].sum()),
                      "observed": y[idx].mean(), "ci_low": low, "ci_high": high,
                      "predicted": probabilities["submitted"][idx].mean()})
    pd.DataFrame(bands).to_csv(out / "bc_tail_bands.csv", index=False)
    print("Exact OOF-to-submission matching, labels, eligibility, and all five folds passed.", flush=True)
    draws, intervals = paired_bootstrap(y, probabilities, args.n_bootstrap, args.seed)
    # Verify the bootstrap ORs against the actual submitted (clipped, standardized) scores.
    rng = np.random.default_rng(args.seed)
    submitted_draws = draws.loc[draws.model.eq("submitted")].set_index("repeat")
    for repeat in range(args.n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        value = top_decile_odds_ratio(y[idx], scores["submitted"][idx])["top10_vs_lower90_or"]
        assert abs(value - submitted_draws.loc[repeat, "top10_vs_lower90_or"]) < 1e-12
    draws.to_csv(out / "bc_bootstrap_draws.csv", index=False)
    intervals.to_csv(out / "bc_intervals.csv", index=False)
    model_points = pd.DataFrame(pooled).query("disease == 'bc'").set_index("model")
    before = draws.loc[draws.model.eq("covariates")].set_index("repeat")
    after = draws.loc[draws.model.eq("submitted")].set_index("repeat")
    delta_rows = []
    for metric in ["auc", "nagelkerke_r2", "top10_vs_lower90_or", "brier", "log_loss"]:
        diff = after[metric] - before[metric]
        low, high = diff.quantile([.025, .975])
        delta_rows.append({"metric": metric, "delta": model_points.loc["submitted", metric] -
                           model_points.loc["covariates", metric], "low": low, "high": high})
    pd.DataFrame(delta_rows).to_csv(out / "bc_paired_deltas.csv", index=False)
    selection = Path("outputs/tabpfn3_final_trainval_selected_drop3_drop2_drop3_inputs/bc.selected_final_models.csv")
    sources.append(selection)
    selection_summary = pd.read_csv(selection)[["family", "selection_metric"]].to_dict("records")
    record = {
        "scope": "Pooled five-fold OOF from the exact confirmed May final submission package",
        "package": str(PACKAGE), "training_performed_in_this_analysis": False,
        "score_definition": "Submitted standardized clipped logits for OR90; stored native TabPFN probabilities for R2/Brier/log loss",
        "metadata": "Submitted TabPFN model already includes covariates; no additional logistic wrapper is applied",
        "baseline": "Previously saved covariates-only OOF probabilities with exactly matched patient IDs, labels and folds",
        "primary_aggregation": "Pooled patient-level OOF; fold-specific metrics are diagnostics, not averaged primary estimates",
        "bootstrap": {"replicates": args.n_bootstrap, "seed": args.seed,
                      "method": "Paired non-stratified patient resampling, percentile 95% CI; rerank each resample; fixed predictions; no refit"},
        "limitations": ["OOF protects TabPFN fitting from each evaluation patient's own label",
                        "PRS parameters and panel were selected using validation information; this is not fully nested pipeline evaluation",
                        "OOF evaluates the submitted configuration using fold models, not the single all-trainval fitted holdout model",
                        "Bootstrap intervals condition on predictions and do not include model selection or refitting uncertainty",
                        "Final holdout labels are unavailable and are not used here"],
        "bc_prs_selection": selection_summary,
        "checks": {"all_disease_patient_fold_and_label_matches": True, "female_eligibility": True,
                   "oof_scores_equal_submission_txt_and_zip": True, "holdout_disjoint": True,
                   "all_bootstrap_or_values_match_actual_submitted_score_ranking": True},
        "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels"]},
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(sources)},
    }
    (out / "analysis_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(model_points[["n", "cases", "auc", "top10_vs_lower90_or", "nagelkerke_r2"]].to_string(), flush=True)
    print(pd.DataFrame(delta_rows).to_string(index=False), flush=True)
    print(f"OOF analysis saved to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
