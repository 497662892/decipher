#!/usr/bin/env python3
"""Extend the frozen May-submission OOF diagnostics to CAD and T2D."""

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
from analyze_may_final_oof import BASELINE, FRACTIONS, PACKAGE, PHENOTYPE, SPLITS
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
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

DEFAULT_ROOT = "outputs/may_final_cad_t2d_oof_prediction_analysis"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=["cad", "T2D"], required=True)
    parser.add_argument("--eligibility-config", required=True)
    parser.add_argument("--out-root", default=DEFAULT_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    os.umask(0o077)
    disease = args.disease
    out = Path(args.out_root) / disease.lower()
    out.mkdir(parents=True, exist_ok=True)
    rules = load_eligibility_rules(args.eligibility_config)
    predictions = pd.read_csv(PACKAGE / "trainval_5cv_scores_long.csv")
    validate_predictions(predictions)
    phenotype = pd.read_csv(PHENOTYPE)
    splits = pd.read_csv(SPLITS)
    baseline = pd.read_csv(BASELINE)
    holdout = pd.read_csv(PACKAGE / "holdout_scores_long.csv")
    fit = pd.read_csv(PACKAGE / "tabpfn_fit_summary.csv").set_index("disease").loc[disease]
    fold_audit = pd.read_csv(PACKAGE / "audit_by_fold.csv").set_index(["disease", "fold"])
    run = json.loads((PACKAGE / "run_record.json").read_text())
    assert phenotype.sampleid.is_unique
    assert not splits.duplicated(["sampleid", "disease"]).any()
    assert splits.groupby("sampleid").fold.nunique().eq(1).all()
    assert not set(predictions.sampleid) & set(holdout.sampleid)
    assert holdout.label.isna().all()
    assert fit.train_prediction_mode == "shared_5fold_oof"
    eligible = filter_eligible(phenotype, disease, rules)
    g = predictions.loc[predictions.disease.eq(disease)].copy().reset_index(drop=True)
    assert set(g.sampleid) == set(eligible.sampleid)
    assert sorted(g.fold.unique()) == list(range(5))
    g = g.merge(eligible[["sampleid", disease]], on="sampleid", validate="one_to_one")
    assert g.label.eq(g[disease]).all()
    g = g.merge(splits.loc[splits.disease.eq(disease), ["sampleid", "fold"]],
                on="sampleid", suffixes=("", "_expected"), validate="one_to_one")
    assert g.fold.eq(g.fold_expected).all() and len(g) == fit.n_context
    g = g.merge(baseline.loc[baseline.disease.eq(disease), ["sampleid", "fold", "y_true", "y_prob"]],
                on=["sampleid", "fold"], how="left", validate="one_to_one")
    assert g.y_true.eq(g.label).all() and g.y_prob.between(0, 1, inclusive="neither").all()
    submission_path = PACKAGE / f"Go Gene!_{disease}_trainval_5cv.txt"
    archive_path = PACKAGE / "Go_Gene_final_phase_submission_with_trainval_5cv_standardized_logits.zip"
    with zipfile.ZipFile(archive_path) as archive:
        zipped = pd.read_csv(io.BytesIO(archive.read(f"Go Gene!_{disease}.txt")), sep="\t")
    for submitted in [pd.read_csv(submission_path, sep="\t"), zipped.loc[zipped.sampleid.isin(g.sampleid)]]:
        matched = g.merge(submitted, on="sampleid", validate="one_to_one")
        assert len(matched) == len(g)
        assert np.allclose(matched.score, matched[f"{disease}_score"], rtol=0, atol=1e-12)
    assert np.allclose(g.score * fit.logit_train_sd + fit.logit_train_mean,
                       g.direct_logit, rtol=0, atol=1e-12)
    assert np.allclose(expit(g.direct_logit), np.clip(g.direct_probability, fit.clip_eps, 1-fit.clip_eps),
                       rtol=0, atol=1e-12)
    y = g.label.to_numpy(dtype=int)
    probabilities = {"submitted": g.direct_probability.to_numpy(), "covariates": g.y_prob.to_numpy()}
    scores = {"submitted": g.score.to_numpy(), "covariates": g.y_prob.to_numpy()}
    assert np.array_equal(risk_deciles(scores["submitted"]), risk_deciles(probabilities["submitted"]))
    pooled, folds, deciles, thresholds, histograms, curves = [], [], [], [], [], []
    for model, p in probabilities.items():
        score = scores[model]
        point = metric_row(y, p, score)
        assert abs(point["auc"]-roc_auc_score(y, score)) < 1e-12
        assert abs(point["auprc"]-average_precision_score(y, score)) < 1e-12
        pooled.append({"disease": disease, "model": model, **point})
        for fold, group in g.groupby("fold"):
            idx = group.index.to_numpy()
            folds.append({"disease": disease, "model": model, "fold": fold,
                          **metric_row(y[idx], p[idx], score[idx])})
        deciles.append(decile_table(y, p, score, probabilities["covariates"]).assign(model=model))
        thresholds.append(threshold_table(y, score, FRACTIONS).assign(model=model))
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        curves += [pd.DataFrame({"model": model, "curve": "roc", "x": fpr, "y": tpr}),
                   pd.DataFrame({"model": model, "curve": "pr", "x": recall, "y": precision})]
        for scale, values, bins in [("probability", p, np.linspace(0, 1, 81)),
                                    ("score", score, np.linspace(score.min(), score.max(), 81))]:
            for label in [0, 1]:
                a = values[y == label]
                counts, edges = np.histogram(a, bins=bins)
                histograms.append(pd.DataFrame({"model": model, "scale": scale, "label": label,
                                                "left": edges[:-1], "right": edges[1:], "count": counts,
                                                "density": counts / len(a) / np.diff(edges)}))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(out / "pooled_metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "fold_metrics.csv", index=False)
    for name, frames in [("deciles", deciles), ("thresholds", thresholds),
                          ("histograms", histograms), ("roc_pr_curves", curves)]:
        pd.concat(frames).to_csv(out / f"{name}.csv", index=False)
    g[["sampleid", "fold", "label", "score", "direct_logit", "direct_probability", "y_prob"]].to_csv(
        out / "oof_predictions_private.csv", index=False)
    top = risk_deciles(scores["submitted"]) == 10
    base_top = risk_deciles(scores["covariates"]) == 10
    distribution = []
    for fold, group in g.groupby("fold"):
        expected = fold_audit.loc[(disease, fold)]
        assert len(group) == expected.n_validation and group.label.sum() == expected.validation_cases
        assert len(g)-len(group) == expected.n_context
        assert y.sum()-group.label.sum() == expected.context_cases
        selected = top[group.index]
        distribution.append({"fold": fold, "n": len(group), "cases": int(group.label.sum()),
                             "score_mean": group.score.mean(), "score_sd": group.score.std(ddof=0),
                             "probability_mean": group.direct_probability.mean(),
                             "pooled_top10_n": int(selected.sum()),
                             "pooled_top10_cases": int(group.loc[selected, "label"].sum())})
    pd.DataFrame(distribution).to_csv(out / "fold_prediction_distributions.csv", index=False)
    membership = []
    for name, mask in [("both", top & base_top), ("model_only", top & ~base_top),
                        ("baseline_only", ~top & base_top), ("neither", ~top & ~base_top)]:
        n, cases = int(mask.sum()), int(y[mask].sum())
        low, high = proportion_confint(cases, n, method="wilson")
        membership.append({"group": name, "n": n, "cases": cases, "observed": cases/n,
                           "ci_low": low, "ci_high": high})
    pd.DataFrame(membership).to_csv(out / "top10_membership.csv", index=False)
    order = np.argsort(-scores["submitted"], kind="stable")
    bands = []
    boundaries = [0, .01, .02, .05, .1, .2, .5, 1]
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        idx = order[int(np.ceil(lower*len(y))):int(np.ceil(upper*len(y)))]
        low, high = proportion_confint(y[idx].sum(), len(idx), method="wilson")
        bands.append({"rank_from": lower, "rank_to": upper, "n": len(idx), "cases": int(y[idx].sum()),
                      "observed": y[idx].mean(), "ci_low": low, "ci_high": high,
                      "predicted": probabilities["submitted"][idx].mean()})
    pd.DataFrame(bands).to_csv(out / "tail_bands.csv", index=False)
    print(f"{disease}: submission ZIP/TXT, patients, labels, eligibility and five folds match.", flush=True)
    draws, intervals = paired_bootstrap(y, probabilities, args.n_bootstrap, args.seed)
    rng = np.random.default_rng(args.seed)
    submitted_draws = draws.loc[draws.model.eq("submitted")].set_index("repeat")
    for repeat in range(args.n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        odds = top_decile_odds_ratio(y[idx], scores["submitted"][idx])["top10_vs_lower90_or"]
        assert abs(odds-submitted_draws.loc[repeat, "top10_vs_lower90_or"]) < 1e-12
    draws.to_csv(out / "bootstrap_draws.csv", index=False)
    intervals.to_csv(out / "intervals.csv", index=False)
    points = pooled.set_index("model")
    before = draws.loc[draws.model.eq("covariates")].set_index("repeat")
    after = draws.loc[draws.model.eq("submitted")].set_index("repeat")
    deltas = []
    for metric in ["auc", "nagelkerke_r2", "top10_vs_lower90_or", "brier", "log_loss"]:
        low, high = (after[metric]-before[metric]).quantile([.025, .975])
        deltas.append({"metric": metric, "delta": points.loc["submitted", metric]-points.loc["covariates", metric],
                       "low": low, "high": high})
    pd.DataFrame(deltas).to_csv(out / "paired_deltas.csv", index=False)
    pd.DataFrame([{"disease": disease,
                   "n_clipped": int(((g.direct_probability < fit.clip_eps) |
                                      (g.direct_probability > 1-fit.clip_eps)).sum()),
                   "r2_native_probability": nagelkerke_r2(y, probabilities["submitted"]),
                   "r2_probability_recovered_from_submitted_score": nagelkerke_r2(y, expit(g.direct_logit)),
                   "native_and_submitted_deciles_identical": True}]).to_csv(out / "score_transform_audit.csv", index=False)
    selection_path = Path("outputs/tabpfn3_final_trainval_selected_drop3_drop2_drop3_inputs") / f"{disease}.selected_final_models.csv"
    selection = pd.read_csv(selection_path)[["family", "selection_metric"]]
    assert len(selection) == int(fit.n_prs_features)
    checks = {"eligible_patients_exact": True, "labels_exact": True, "shared_patient_folds_exact": True,
              "submission_txt_and_zip_scores_exact": True, "baseline_same_patients_and_folds": True,
              "holdout_disjoint_and_unlabelled": True, "bootstrap_or_uses_exact_submitted_ranking": True}
    pd.DataFrame([{"disease": disease, "n": len(y), "cases": int(y.sum()), "folds": 5, **checks}]).to_csv(
        out / "cohort_audit.csv", index=False)
    sources = [PACKAGE / f for f in ["trainval_5cv_scores_long.csv", "holdout_scores_long.csv", "run_record.json",
                                     "tabpfn_fit_summary.csv", "audit_by_fold.csv"]]
    sources += [BASELINE, PHENOTYPE, SPLITS, submission_path, archive_path, selection_path,
                Path(args.eligibility_config), Path(__file__), Path("scripts/analyze_may_final_oof.py"),
                Path("src/decipehr/prediction_diagnostics.py"), Path("src/decipehr/metrics.py"), Path(run["script"])]
    record = {"disease": disease, "scope": "Pooled five-fold OOF from the confirmed final May submission",
              "package": str(PACKAGE), "training_performed": False, "additional_evaluation_model_fitted": False,
              "score_definition": "Submitted standardized clipped logits for ranking; saved native probabilities for R2, Brier and log loss",
              "baseline": "Saved covariates-only OOF, with exactly matched patients, labels and folds",
              "bootstrap": {"replicates": args.n_bootstrap, "seed": args.seed,
                            "method": "Paired patient resampling; percentile 95% CI; rerank each resample; fixed predictions; no refit"},
              "prs_selection": selection.to_dict("records"),
              "limitations": ["The model includes PRS and metadata; gains are not exclusively genetic effects",
                              "Each patient is excluded from its TabPFN fitting context, but PRS/panel selection is not fully nested",
                              "OOF assesses the submitted configuration, not the single full-trainval fitted holdout instance",
                              "Bootstrap conditions on frozen predictions and excludes refitting and selection uncertainty",
                              "Final holdout labels are unavailable; no final holdout evaluation is made"],
              "checks": checks, "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels"]},
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (out / "analysis_record.json").write_text(json.dumps(record, indent=2)+"\n")
    print(points[["n", "cases", "auc", "top10_vs_lower90_or", "nagelkerke_r2"]].to_string(), flush=True)
    print(pd.DataFrame(deltas).to_string(index=False), flush=True)
    print(f"Saved {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
