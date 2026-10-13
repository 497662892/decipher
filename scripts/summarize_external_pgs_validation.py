"""Verify completed official-validation runs and report the external PGS ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.ancestry_performance import paired_bootstrap_comparisons
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import load_weight_lock, sha256_file, write_json
from decipehr.external_pgs_benchmark import DEFAULT_METHODS, METHODS, summarize_benchmark
from decipehr.tabpfn_ensemble import stratified_oof_splits


def ids_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(sorted(frame["sampleid"].astype(str))).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--scoring-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-phenotype", default="data/processed/phenotype.csv")
    parser.add_argument("--validation-phenotype", default="data/processed/official_validation_phenotype.csv")
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new report directory")
    scoring = Path(args.scoring_dir)
    lock_path = scoring / "weights.scored.lock.json"
    lock = load_weight_lock(lock_path)
    score_run = json.loads((scoring / "scoring_run.json").read_text())
    assert score_run["weight_lock_sha256"] == sha256_file(lock_path)
    assert score_run["scores_sha256"] == sha256_file(scoring / "scores.long.csv")
    assert score_run["qc_sha256"] == sha256_file(scoring / "scoring_qc.csv")
    qc = pd.read_csv(scoring / "scoring_qc.csv")
    assert set(qc.loc[qc.included, "model"]) == {r["id"] for r in lock["resources"]}
    train = pd.read_csv(args.train_phenotype, dtype={"sampleid": str})
    val = pd.read_csv(args.validation_phenotype, dtype={"sampleid": str})
    assert train.sampleid.is_unique and val.sampleid.is_unique
    assert not set(train.sampleid) & set(val.sampleid)
    rules = load_eligibility_rules(args.eligibility_config)
    frames, audits, run_hashes = [], [], {}
    checkpoint_hashes = set()
    methods, primary_method = None, None
    for disease in ("T2D", "cad", "bc"):
        directory = Path(args.runs_root) / disease
        run = json.loads((directory / "run.json").read_text())
        assert run["status"] == "complete" and run["evaluation_protocol"] == "official_train_val"
        run_methods = tuple(run["arguments"]["methods"].split(","))
        assert len(run_methods) == len(set(run_methods))
        assert set(run_methods) in (set(DEFAULT_METHODS), set(METHODS))
        run_primary = run.get("primary_method", "tabpfn_prs_pc_soft")
        assert run_primary in run_methods
        if methods is None:
            methods, primary_method = run_methods, run_primary
        assert run_methods == methods and run_primary == primary_method
        assert run["arguments"]["normalization"] == "none" and run["arguments"]["score_set"] == "all"
        assert run["tabpfn_model_version"] == "v3" and run["n_estimators"] == 16
        assert run["input_sha256"]["weight_lock"] == sha256_file(lock_path)
        for key, digest in run["input_sha256"].items():
            assert sha256_file(run["arguments"][key]) == digest, f"Changed input: {disease}/{key}"
        assert run["input_sha256"]["train_phenotype"] == sha256_file(args.train_phenotype)
        assert run["input_sha256"]["validation_phenotype"] == sha256_file(args.validation_phenotype)
        checkpoint_hashes.add(run["tabpfn_checkpoint_sha256"])
        assert sha256_file(run["arguments"]["model_path"]) == run["tabpfn_checkpoint_sha256"]
        context = filter_eligible(train, disease, rules).sort_values("sampleid").reset_index(drop=True)
        query = filter_eligible(val, disease, rules).sort_values("sampleid").reset_index(drop=True)
        pred = pd.read_csv(directory / "validation_predictions.csv", dtype={"sampleid": str})
        assert pred.evaluation_protocol.eq("official_train_val").all() and pred.disease.eq(disease).all()
        assert set(pred.method) == set(methods) and not pred.duplicated(["sampleid", "method"]).any()
        assert len(pred) == len(query) * len(methods)
        assert np.isfinite(pred[["y_prob", "wrapper_prob"]]).all().all()
        assert pred[["y_prob", "wrapper_prob"]].ge(0).all().all()
        assert pred[["y_prob", "wrapper_prob"]].le(1).all().all()
        truth = query.set_index("sampleid")[disease]
        for _, rows in pred.groupby("method"):
            assert set(rows.sampleid) == set(query.sampleid)
            np.testing.assert_array_equal(rows.y_true, rows.sampleid.map(truth))
        fits = pd.read_csv(directory / disease / "fit_audit.csv")
        assert len(fits) == len(methods) * 4
        final = fits.loc[fits.fit_scope == "official_train_context"]
        assert len(final) == len(methods) and set(final.method) == set(methods)
        assert final.context_ids_sha256.eq(ids_hash(context)).all()
        assert final.query_ids_sha256.eq(ids_hash(query)).all()
        assert final.n_context.eq(len(context)).all() and final.n_query.eq(len(query)).all()
        inner = stratified_oof_splits(context[disease], requested_splits=3, seed=run["arguments"]["seed"])
        for index, (context_idx, query_idx) in enumerate(inner):
            rows = fits.loc[fits.fit_scope == f"wrapper_inner_{index}"]
            assert len(rows) == len(methods) and set(rows.method) == set(methods)
            assert rows.context_ids_sha256.eq(ids_hash(context.iloc[context_idx])).all()
            assert rows.query_ids_sha256.eq(ids_hash(context.iloc[query_idx])).all()
            assert rows.n_context.eq(len(context_idx)).all() and rows.n_query.eq(len(query_idx)).all()
        ids = {x["id"] for x in lock["resources"] if x["disease"] == disease}
        assert ids and set(run["score_banks"][disease]) == ids
        for selection in fits.selected_models:
            assert set(json.loads(selection)).issubset(ids)
        best = final.loc[final.method == "best_single_lr"]
        audits.append({"disease": disease, "n_context": len(context), "n_validation": len(query),
                       "validation_cases": int(query[disease].sum()), "n_external_scores": len(ids),
                       "train_selected_single": json.loads(best.iloc[0].selected_models)[0] if len(best) else None,
                       "selected_C": best.iloc[0].C if len(best) else None})
        frames.append(pred)
        run_hashes[disease] = {name: sha256_file(directory / name) for name in (
            "run.json", "validation_predictions.csv", f"{disease}/fit_audit.csv", f"{disease}/preprocessing.csv",
        )}
    assert len(checkpoint_hashes) == 1, "Disease runs must use the same TabPFN checkpoint"
    predictions = pd.concat(frames, ignore_index=True)
    metrics, comparisons = summarize_benchmark(predictions, n_bootstraps=args.n_bootstraps)
    for disease in run_hashes:
        saved = pd.read_csv(Path(args.runs_root) / disease / "summary_metrics.csv")
        recomputed = metrics.loc[metrics.disease.eq(disease)].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, recomputed, check_dtype=False, rtol=1e-9, atol=1e-12)
    wide = predictions.pivot(index=["sampleid", "disease", "y_true", "ancestry_group"],
                             columns="method", values="y_prob").reset_index()
    secondary = pd.concat([
        paired_bootstrap_comparisons(
            frame, comparisons=(("full_vs_train_selected_single", "best_single_lr", "tabpfn_prs_pc_soft"),),
            metric_names=["auc"], n_bootstraps=args.n_bootstraps, seed=42,
        )
        for _, frame in wide.groupby("disease", sort=True)
    ], ignore_index=True) if {"best_single_lr", "tabpfn_prs_pc_soft"}.issubset(methods) else pd.DataFrame()
    if not secondary.empty:
        secondary["exploratory"] = True
        secondary["inference_scope"] = "conditional_on_frozen_validation_predictions"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("validation_predictions.csv", predictions), ("summary_metrics.csv", metrics),
                        ("paired_auc_comparisons.csv", comparisons), ("secondary_single_comparisons.csv", secondary),
                        ("cohort_audit.csv", pd.DataFrame(audits))):
        if not frame.empty:
            frame.to_csv(out / name, index=False)
    rows = metrics.loc[metrics.ancestry_group.eq("ALL") & metrics.prediction_scale.eq("raw")]
    report = ["# External PGS ensemble: official train to validation", "",
              "All SNP weights are external and frozen. The official train cohort supplies the full context;",
              "selection, preprocessing and wrapper calibration use train only. Official val supplies evaluation labels.",
              f"Reported primary model: {primary_method}.",
              "", "| Trait | Context | Val | Cases | Scores | Train-selected single |",
              "|---|---:|---:|---:|---:|---|"]
    for a in audits:
        report.append(f"| {a['disease']} | {a['n_context']} | {a['n_validation']} | {a['validation_cases']} | "
                      f"{a['n_external_scores']} | {a['train_selected_single'] or 'not fitted'} |")
    if primary_method == "tabpfn_prs_pc":
        report.extend(["", "The PC-only default was adopted after reviewing the initial validation results.",
                       "This is a subsequent development choice; the original eight-arm results remain archived."])
    report.extend(["", f"{int(qc.included.sum())} of {len(qc)} frozen candidates passed the predeclared "
                   f"{score_run['min_overlap']:.0%} variant-coverage threshold. Coverage filtering used no outcome data."])
    amendment_hash = None
    if lock.get("amendment"):
        amendment = json.loads(Path(lock["amendment"]).read_text())
        assert amendment["validation_evaluation_started"] is False
        assert amendment["min_overlap_unchanged"] == score_run["min_overlap"]
        assert amendment["amended_lock_sha256"] == score_run["source_weight_lock_sha256"]
        amendment_hash = sha256_file(lock["amendment"])
        report.extend(["", "Before outcome evaluation, two original BC candidates failed fixed coverage QC.",
                       "European MegaPRS (PGS004869) and East Asian PRS-CS (PGS018538) were added to preserve a multi-score BC bank.",
                       "The 75% threshold was unchanged. protocol_amendment.json records this decision.",
                       "The TPMI source model used a broader ICD definition and included male controls; target BC evaluation remains female-only."])
    excluded = qc.loc[~qc.included]
    if not excluded.empty:
        report.extend(["", "| Excluded score | Trait | Matched / published variants | Coverage |",
                       "|---|---|---:|---:|"])
        for r in excluded.to_dict("records"):
            report.append(f"| {r['model']} | {r['disease']} | {r['n_matched_variants']} / "
                          f"{r['n_published_variants']} | {r['coverage']:.3%} |")
    report.extend(["", "## Raw validation performance", "",
                   "| Trait | Method | AUC | OR90 | Nagelkerke R² | Brier |", "|---|---|---:|---:|---:|---:|"])
    for r in rows.to_dict("records"):
        report.append(f"| {r['disease']} | {r['method']} | {r['auc']:.4f} | {r['top10_vs_lower90_or']:.4f} | "
                      f"{r['nagelkerke_r2']:.4f} | {r['brier']:.4f} |")
    if not secondary.empty:
        report.extend(["", "## Ensemble versus train-selected single score", "",
                       "These secondary paired comparisons are exploratory. Positive differences favor the ensemble.",
                       "The reference is one train-selected PGS plus age, sex, PCs and genetic similarities in ridge LR.",
                       "The full model uses all retained PGS plus the same covariates in TabPFN.",
                       "This compares whole prediction pipelines; without a single-PGS TabPFN arm, the gain cannot be attributed solely to multi-PGS integration.",
                       "", "| Trait | ΔAUC | Conditional 95% CI |", "|---|---:|---|"])
        for r in secondary.loc[secondary.ancestry_group.eq("ALL")].to_dict("records"):
            report.append(f"| {r['disease']} | {r['candidate_minus_reference']:.4f} | "
                          f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    if not comparisons.empty:
        report.extend(["", "## Added value of ancestry-aware integration", "",
                       "The original two soft-similarity contrasts per trait retain a six-comparison family. Positive differences favor the full model.",
                       "full_vs_pc tests adding five genetic-similarity weights beyond age, sex and PCs in TabPFN.",
                       "full_vs_linear_interactions compares full TabPFN against ridge LR with all PGS and PGS-by-PC/similarity interactions.",
                       "", "| Trait | Contrast | ΔAUC | Conditional 95% CI | Bonferroni 99.1667% CI |",
                       "|---|---|---:|---|---|"])
        for r in comparisons.loc[comparisons.ancestry_group.eq("ALL")].to_dict("records"):
            report.append(f"| {r['disease']} | {r['comparison']} | {r['candidate_minus_reference']:.4f} | "
                          f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
                          f"[{r['familywise_ci_low']:.4f}, {r['familywise_ci_high']:.4f}] |")
    report.extend(["", "Genetic-similarity weights are not ancestry proportions. Breast cancer is restricted to eligible females.",
                   "summary_metrics.csv also includes wrapper and ancestry subgroup results. AMBIGUOUS is retained.",
                   "Bootstrap intervals condition on the fitted predictors and exclude refitting uncertainty.",
                   "The official validation cohort was used in prior project comparisons; it is not a newly untouched external test.",
                   "", f"Variant coverage and exclusions: {scoring / 'scoring_qc.csv'}.", ""])
    (out / "REPORT.md").write_text("\n".join(report))
    write_json(out / "VERIFIED.json", {
        "status": "complete", "evaluation_protocol": "official_train_val", "diseases": list(run_hashes),
        "methods": list(methods), "primary_method": primary_method, "source_runs_sha256": run_hashes,
        "weight_lock_sha256": sha256_file(lock_path), "scoring_run_sha256": sha256_file(scoring / "scoring_run.json"),
        "protocol_amendment_sha256": amendment_hash,
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
        "checks": ["fixed_official_split", "female_bc_eligibility", "every_validation_patient_scored",
                   "configured_method_roster", "train_only_context_hashes", "train_only_inner_fold_hashes",
                   "frozen_weights_and_input_hashes", "matched_phenotypes", "same_checkpoint", "recomputed_metrics"],
    })
    print(rows[["disease", "method", "auc", "nagelkerke_r2", "top10_vs_lower90_or"]].to_string(index=False))
    print(f"Verified report: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
