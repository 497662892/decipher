"""Audit and report matched-input DiscoDivas comparisons on official validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.ancestry_performance import paired_bootstrap_comparisons
from decipehr.discodivas import COMPARISON_METHODS, REFERENCE_PCS, comparison_bank
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_pgs_benchmark import _ids_hash, summarize_benchmark
from decipehr.tabpfn_ensemble import stratified_oof_splits


def verify_harmonization(directory, run, train, val):
    """Reconstruct the final transformation independently from frozen source scores."""
    paths = run["input_paths"]
    scores = pd.read_csv(paths["scores"], dtype={"sampleid": str})
    raw = scores.loc[scores.disease == run["arguments"]["disease"]].pivot(
        index="sampleid", columns="model", values="prs",
    )
    ids = [*train.sampleid, *val.sampleid]
    raw = raw.loc[ids, run["source_ids"]].to_numpy(dtype=float)
    n = len(train)
    components = (raw - raw[:n].mean(axis=0)) / raw[:n].std(axis=0, ddof=1)
    metas = components @ np.asarray(run["published_combination"])
    pcs = pd.read_csv(paths["pcs"], sep="\t", dtype={"sampleid": str})
    x = pcs.set_index("sampleid").loc[ids, REFERENCE_PCS].to_numpy(dtype=float)
    # Rescale predictors for numerical stability; the fitted OLS subspace is unchanged.
    x = (x - x[:n].mean(axis=0)) / x[:n].std(axis=0)
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(x[:n], metas[:n], rcond=None)[0]
    residual = metas - x @ beta
    z = (residual - residual[:n].mean(axis=0)) / residual[:n].std(axis=0, ddof=1)
    actual = pd.read_csv(directory / "final_harmonized_scores.csv", dtype={"sampleid": str})
    actual = actual.set_index("sampleid").loc[ids]
    np.testing.assert_allclose(actual[run["model_ids"]], z, rtol=1e-8, atol=1e-8)
    weights = pd.read_csv(directory / "interpolation_coefficients.csv", dtype={"sampleid": str})
    weights = weights.set_index("sampleid").loc[ids]
    a = weights[[f"disco_a_{g}" for g in run["reference_groups"]]].to_numpy()
    np.testing.assert_allclose(a.sum(axis=1), 1, atol=1e-10)
    np.testing.assert_allclose(actual.discodivas_prs, (z * a).sum(axis=1), atol=1e-8)
    official = directory / "official_final_scores"
    mapping = pd.read_csv(official / "patient_mapping.tsv", sep="\t", dtype=str)
    prediction = pd.read_csv(official / "combined.tsv.gz", sep="\t", dtype={"IID": str})
    prediction = mapping.merge(prediction, on="IID", validate="one_to_one").set_index("sampleid")
    np.testing.assert_allclose(prediction.loc[ids, "PRS"], actual.discodivas_prs, atol=1e-10)
    provenance = json.loads((official / "upstream_provenance.json").read_text())
    assert sha256_file(provenance["upstream_path"]) == provenance["upstream_sha256"]
    assert sha256_file(provenance["launcher_path"]) == provenance["launcher_sha256"]
    assert provenance["compatibility_option"] == "datatable.fread.datatable=FALSE"
    assert provenance["regress_PCA"] is False
    assert provenance["quality_parameters"] == [1] * len(run["reference_groups"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--previous-runs-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError("Use a new report directory")
    previous_verification_path = Path(args.previous_runs_root).parent / "report/VERIFIED.json"
    previous_verification = json.loads(previous_verification_path.read_text())
    assert previous_verification["status"] == "complete"
    predictions, counts, hashes, matched_pairs, full_pairs = [], [], {}, [], []
    common_hashes = None
    for disease in ("T2D", "cad", "bc"):
        directory = Path(args.runs_root) / disease
        run_path = directory / "run.json"
        run = json.loads(run_path.read_text())
        assert run["status"] == "complete" and run["official_final_scores_verified"]
        assert run["evaluation_protocol"] == "official_train_val"
        assert run["methods"] == list(COMPARISON_METHODS)
        assert run["quality_parameters"] == [1] * len(run["reference_groups"])
        assert run["tabpfn_version"] == "v3" and run["n_estimators"] == 16
        assert run["inner_folds"] == 3 and run["ridge_C_grid"] == [0.1, 1, 10]
        for key, digest in run["input_sha256"].items():
            assert sha256_file(run["input_paths"][key]) == digest, f"Changed input: {key}"
        for name, digest in run["source_sha256"].items():
            assert sha256_file(name) == digest, f"Changed evaluation source: {name}"
        for name, digest in run["output_sha256"].items():
            assert sha256_file(directory / name) == digest, f"Changed result: {disease}/{name}"
        if common_hashes is None:
            common_hashes = run["input_sha256"]
        assert run["input_sha256"] == common_hashes
        paths = run["input_paths"]
        bank = comparison_bank(disease, pd.read_csv(paths["coefficient_mapping"]))
        assert run["source_ids"] == list(bank.source_ids)
        assert run["model_ids"] == bank.model_ids and run["reference_groups"] == list(bank.groups)
        np.testing.assert_array_equal(run["published_combination"], bank.combination)
        train_all = pd.read_csv(paths["train_phenotype"], dtype={"sampleid": str})
        val_all = pd.read_csv(paths["validation_phenotype"], dtype={"sampleid": str})
        assert train_all.sampleid.is_unique and val_all.sampleid.is_unique
        assert not set(train_all.sampleid) & set(val_all.sampleid)
        rules = load_eligibility_rules(paths["eligibility_config"])
        train = filter_eligible(train_all, disease, rules).sort_values("sampleid").reset_index(drop=True)
        val = filter_eligible(val_all, disease, rules).sort_values("sampleid").reset_index(drop=True)
        assert len(train) == run["n_context"] and len(val) == run["n_query"]
        pred = pd.read_csv(directory / "validation_predictions.csv", dtype={"sampleid": str})
        assert set(pred.method) == set(COMPARISON_METHODS)
        assert not pred.duplicated(["sampleid", "method"]).any()
        for _, rows in pred.groupby("method"):
            assert set(rows.sampleid) == set(val.sampleid)
            np.testing.assert_array_equal(rows.y_true, rows.sampleid.map(val.set_index("sampleid")[disease]))
        assert np.isfinite(pred[["y_prob", "wrapper_prob"]]).all().all()
        assert pred[["y_prob", "wrapper_prob"]].ge(0).all().all()
        assert pred[["y_prob", "wrapper_prob"]].le(1).all().all()
        assert pred.disease.eq(disease).all() and pred.fold.eq(0).all()
        fits = pd.read_csv(directory / "fit_audit.csv")
        assert len(fits) == 12
        for method, rows in fits.groupby("method"):
            expected = ["discodivas_prs"] if method == "discodivas_lr" else run["model_ids"]
            assert all(json.loads(value) == expected for value in rows.selected_models)
            if method.endswith("_lr"):
                assert rows.C.isin(run["ridge_C_grid"]).all()
        final = fits.loc[fits.fit_scope == "official_train_context"]
        assert set(final.method) == set(COMPARISON_METHODS)
        assert final.context_ids_sha256.eq(_ids_hash(train)).all()
        assert final.query_ids_sha256.eq(_ids_hash(val)).all()
        assert final.n_context.eq(len(train)).all() and final.n_query.eq(len(val)).all()
        allowed = {(_ids_hash(train), _ids_hash(val), len(train), len(val))}
        seed = run["arguments"]["seed"]
        folds = stratified_oof_splits(train[disease], requested_splits=3, seed=seed)
        for i, (fit_indices, query_indices) in enumerate(folds):
            fit, held = train.iloc[fit_indices], train.iloc[query_indices]
            rows = fits.loc[fits.fit_scope == f"wrapper_inner_{i}"]
            assert len(rows) == 3 and set(rows.method) == set(COMPARISON_METHODS)
            assert rows.context_ids_sha256.eq(_ids_hash(fit)).all()
            assert rows.query_ids_sha256.eq(_ids_hash(held)).all()
            assert rows.n_context.eq(len(fit)).all() and rows.n_query.eq(len(held)).all()
            allowed.add((_ids_hash(fit), _ids_hash(held), len(fit), len(held)))
            for a, b in stratified_oof_splits(fit[disease], requested_splits=3, seed=seed + i):
                allowed.add((_ids_hash(fit.iloc[a]), _ids_hash(fit.iloc[b]), len(a), len(b)))
        fit_log = pd.read_csv(directory / "harmonization_fit_audit.csv")
        assert len(fit_log) == 85
        observed = set(fit_log[["context_ids_sha256", "query_ids_sha256",
                                "n_context", "n_query"]].itertuples(index=False, name=None))
        assert observed == allowed
        verify_harmonization(directory, run, train, val)
        regenerated, _ = summarize_benchmark(pred, n_bootstraps=args.n_bootstraps)
        stored = pd.read_csv(directory / "summary_metrics.csv")
        pd.testing.assert_frame_equal(regenerated, stored, check_dtype=False, atol=1e-10)
        previous_dir = Path(args.previous_runs_root) / disease
        for name, digest in previous_verification["source_runs_sha256"][disease].items():
            assert sha256_file(previous_dir / name) == digest, f"Changed historical output: {name}"
        previous = json.loads((previous_dir / "run.json").read_text())
        assert previous["status"] == "complete" and previous["evaluation_protocol"] == "official_train_val"
        assert previous["n_estimators"] == 16 and previous["tabpfn_model_version"] == "v3"
        assert previous["arguments"]["seed"] == seed
        old_fits = pd.read_csv(previous_dir / disease / "fit_audit.csv")
        old_final = old_fits.loc[old_fits.method.eq("tabpfn_prs_pc") &
                                  old_fits.fit_scope.eq("official_train_context")]
        assert len(old_final) == 1
        assert old_final.context_ids_sha256.iloc[0] == _ids_hash(train)
        assert old_final.query_ids_sha256.iloc[0] == _ids_hash(val)
        assert json.loads(old_final.selected_models.iloc[0]) == previous["score_banks"][disease]
        for key in ("train_phenotype", "validation_phenotype", "scores", "weight_lock"):
            assert previous["input_sha256"][key] == run["input_sha256"][key]
        assert previous["tabpfn_checkpoint_sha256"] == run["input_sha256"]["model_path"]
        old = pd.read_csv(previous_dir / "validation_predictions.csv", dtype={"sampleid": str})
        old = old.loc[old.method == "tabpfn_prs_pc"].copy()
        assert old.sampleid.is_unique and set(old.sampleid) == set(val.sampleid)
        np.testing.assert_array_equal(old.y_true, old.sampleid.map(val.set_index("sampleid")[disease]))
        old_metrics, _ = summarize_benchmark(old, n_bootstraps=args.n_bootstraps)
        stored_old = pd.read_csv(previous_dir / "summary_metrics.csv")
        stored_old = stored_old.loc[stored_old.method.eq("tabpfn_prs_pc")].reset_index(drop=True)
        pd.testing.assert_frame_equal(old_metrics, stored_old, check_dtype=False, atol=1e-10)
        old["method"] = "full_bank_tabpfn_pc"
        both = pd.concat([pred, old], ignore_index=True)
        for scale, column in (("raw", "y_prob"), ("official_wrapper", "wrapper_prob")):
            wide = both.pivot(index=["sampleid", "disease", "y_true", "ancestry_group"],
                              columns="method", values=column).reset_index()
            assert not wide.isna().any().any()
            paired = paired_bootstrap_comparisons(
                wide, n_bootstraps=args.n_bootstraps, comparisons=(
                    ("matched_tabpfn_vs_discodivas", "discodivas_lr", "matched_tabpfn_pc"),
                    ("matched_tabpfn_vs_linear", "matched_multi_prs_lr", "matched_tabpfn_pc"),
                ), metric_names=["auc"], family_size=6,
            ).assign(prediction_scale=scale)
            matched_pairs.append(paired)
            full_pairs.append(paired_bootstrap_comparisons(
                wide, n_bootstraps=args.n_bootstraps,
                comparisons=(("full_bank_tabpfn_vs_discodivas", "discodivas_lr", "full_bank_tabpfn_pc"),),
                metric_names=["auc"], family_size=3,
            ).assign(prediction_scale=scale, exploratory=True))
        predictions.append(both)
        counts.append({"disease": disease, "n_train": len(train), "n_validation": len(val),
                       "validation_cases": int(val[disease].sum()),
                       "matched_models": len(run["model_ids"]),
                       "original_full_bank_scores": len(previous["score_banks"][disease]),
                       "source_ids": ",".join(run["source_ids"]),
                       "reference_groups": ",".join(run["reference_groups"])})
        hashes[disease] = {"run": sha256_file(run_path),
                           "previous_run": sha256_file(previous_dir / "run.json"),
                           "previous_predictions": sha256_file(previous_dir / "validation_predictions.csv")}
    combined = pd.concat(predictions, ignore_index=True)
    metrics, _ = summarize_benchmark(combined, n_bootstraps=args.n_bootstraps)
    primary = pd.concat(matched_pairs, ignore_index=True)
    primary["exploratory"] = (
        primary.ancestry_group.ne("ALL") | primary.prediction_scale.ne("raw")
        | primary.cases.lt(20) | (primary.n - primary.cases).lt(20)
    )
    secondary = pd.concat(full_pairs, ignore_index=True)
    out.mkdir(parents=True)
    artifacts = {"validation_predictions.csv": combined, "summary_metrics.csv": metrics,
                 "matched_paired_auc.csv": primary, "full_bank_paired_auc.csv": secondary,
                 "cohort_and_inputs.csv": pd.DataFrame(counts)}
    for name, frame in artifacts.items():
        frame.to_csv(out / name, index=False)
    headline = primary.loc[
        primary.ancestry_group.eq("ALL") & primary.prediction_scale.eq("raw")
        & primary.comparison.eq("matched_tabpfn_vs_discodivas")
    ]
    n_positive = int(headline.candidate_minus_reference.gt(0).sum())
    n_adjusted = int(headline.familywise_ci_low.gt(0).sum())
    report = [
        "# DiscoDivas versus ancestry-aware TabPFN", "",
        "Verified official train-context / validation-query comparison for all three traits.",
        f"Matched-input PC-only TabPFN has higher overall AUROC than DiscoDivas in {n_positive}/3 traits;",
        f"the family-adjusted paired intervals exclude zero in its favor in {n_adjusted}/3 traits.",
        "This comparison supports stronger prediction in this configuration; it does not isolate ancestry",
        "awareness as the cause or establish improvement in every ancestry subgroup.",
        "External SNP weights and D-PRISM combination coefficients remain fixed. Preprocessing,",
        "linear integration, hyperparameter selection, and calibration fit only training patients.",
        "The official validation cohort has previously informed development; this is a development comparison.",
        "", "## Cohorts and frozen inputs", "",
        "| Trait | Train | Val | Val cases | Matched models | Previous full bank | Reference proxies |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in counts:
        report.append(f"| {row['disease']} | {row['n_train']} | {row['n_validation']} | "
                      f"{row['validation_cases']} | {row['matched_models']} | "
                      f"{row['original_full_bank_scores']} | {row['reference_groups']} |")
    report += [
        "", "T2D uses five ancestry-tuned D-PRISM metascores built from 25 existing components",
        "using published coefficients and train-fitted component scaling. BC uses three heterogeneous",
        "external models and eligible females only. CAD uses EUR-tuned GPS_Mult and EAS-targeted MIXPRS+;",
        "the mixed-tuning PGS004696 has no identified PC center and remains only in the original full bank.",
        "", "## Performance", "",
        "All matched arms receive the same PC-residualized, standardized score inputs and age/sex/PC1-10 covariates.",
        "DiscoDivas combines scores with official individual-specific coefficients, then fits a logistic predictor.",
        "Linear integration uses the component scores directly. Matched TabPFN uses the same components and covariates.",
        "The existing full-bank TabPFN has different score inputs and preprocessing; compare it as a whole pipeline.",
    ]
    for scale in ("raw", "official_wrapper"):
        report += ["", f"### {scale}", "",
                   "| Trait | Method | AUC | OR90 | Nagelkerke R² | Brier |",
                   "|---|---|---:|---:|---:|---:|"]
        rows = metrics.loc[metrics.ancestry_group.eq("ALL") & metrics.prediction_scale.eq(scale)]
        for row in rows.to_dict("records"):
            report.append(f"| {row['disease']} | {row['method']} | {row['auc']:.6f} | "
                          f"{row['top10_vs_lower90_or']:.4f} | {row['nagelkerke_r2']:.6f} | "
                          f"{row['brier']:.6f} |")
    report += ["", "## Primary matched-input AUROC differences", "",
               "Positive values favor matched TabPFN. Family size six: two contrasts across three traits.",
               "Bootstrap intervals condition on the stored predictions and exclude refitting uncertainty.",
               "", "| Trait | Comparison | Difference | 95% CI | Family-adjusted CI |",
               "|---|---|---:|---|---|"]
    rows = primary.loc[primary.ancestry_group.eq("ALL") & primary.prediction_scale.eq("raw")]
    for row in rows.to_dict("records"):
        report.append(f"| {row['disease']} | {row['comparison']} | {row['candidate_minus_reference']:+.6f} | "
                      f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
                      f"[{row['familywise_ci_low']:+.6f}, {row['familywise_ci_high']:+.6f}] |")
    report += ["", "## Previous full-bank TabPFN comparison", "",
               "Exploratory whole-pipeline contrasts, with different score banks/preprocessing.",
               "Positive values favor the existing full-bank PC-only TabPFN.", "",
               "| Trait | AUC difference versus DiscoDivas | 95% CI |",
               "|---|---:|---|"]
    rows = secondary.loc[secondary.ancestry_group.eq("ALL") & secondary.prediction_scale.eq("raw")]
    for row in rows.to_dict("records"):
        report.append(f"| {row['disease']} | {row['candidate_minus_reference']:+.6f} | "
                      f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] |")
    report += ["", "## Ancestry subgroup comparison", "",
               "Exploratory raw AUROC differences: matched TabPFN minus DiscoDivas. Intervals below",
               "are unadjusted 95% paired intervals. A positive overall result does not imply all groups benefit.",
               "Groups with fewer than 20 cases or controls have particularly limited precision.", "",
               "| Trait | Group | N | Cases | AUC difference | 95% CI |",
               "|---|---|---:|---:|---:|---|"]
    rows = primary.loc[primary.ancestry_group.ne("ALL") & primary.prediction_scale.eq("raw")
                       & primary.comparison.eq("matched_tabpfn_vs_discodivas")]
    for row in rows.to_dict("records"):
        report.append(f"| {row['disease']} | {row['ancestry_group']} | {row['n']} | {row['cases']} | "
                      f"{row['candidate_minus_reference']:+.6f} | "
                      f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] |")
    report += [
        "", "## Method scope and verification", "",
        "This applies the official DiscoDivas interpolation to frozen external models, with 1000G population",
        "median proxies in a common PC space. It is not a reproduction of the paper's discovery/tuning cohorts.",
        "The original public PC-loading projection failed the 90% marker-coverage gate (53,571/59,529).",
        "Before disease evaluation, PLINK PCA was therefore fitted on 1000G alone using those 53,571 common",
        "markers. Both reference and target were projected with the same loadings and fixed external frequencies.",
        "Reference medians were recomputed in that space. No target patient entered PCA fitting.",
        "", "The upstream R file remains unchanged. The launcher sets datatable.fread.datatable=FALSE",
        "because upstream's regress.PCA=FALSE branch assumes data.frame positional indexing. Without",
        "that option, the real-script test detected a constant output. Sorted surrogate IDs preserve merge order.",
        "The adapter fits PRS-PC residualization and scaling only on each context, including nested inner folds.",
        "The quality parameters are fixed to one. Full-context combined scores were independently checked",
        "against the official R script; final preprocessing was reconstructed from raw frozen scores.",
        "", "All matched arms use the same evaluation protocol; TabPFN uses v3 with 16 estimators,",
        "the same official split, and the same covariates; TabPFN receives no soft-similarity features.",
        "Wrapper metrics use train-only nested OOF calibration. Subgroup and full-bank contrasts are exploratory.",
        "OR90 compares disease odds in the highest-scoring 10% with the lower 90% of validation patients.",
        "Nagelkerke R² uses held-out predicted-probability log likelihood relative to an intercept-only model.",
        "Calibration intercept/slope are validation diagnostics only and are included in summary_metrics.csv.",
        "Small groups with fewer than 20 cases or controls require caution. AMBIGUOUS does not prove admixture.",
        "", "Sources: [DiscoDivas](https://github.com/YunfengRuan/DiscoDivas),",
        "[D-PRISM](https://github.com/ahuertach/D-PRISM-T2D-PRS),",
        "[MIXPRS+](https://doi.org/10.1038/s41588-026-02637-4).", "",
    ]
    (out / "REPORT.md").write_text("\n".join(report))
    write_json(out / "VERIFIED.json", {
        "status": "verified", "evaluation_protocol": "official_train_val",
        "diseases": ["T2D", "cad", "bc"], "run_sha256": hashes,
        "n_bootstraps": args.n_bootstraps, "primary_family_size": 6,
        "previous_verification_sha256": sha256_file(previous_verification_path),
        "report_generator_sha256": sha256_file(__file__),
        "verified": ["patient sets and labels", "female BC eligibility", "frozen inputs and code",
                     "full train context and all nested fit membership hashes", "train-only transformations",
                     "official R interpolation", "metrics recomputed", "matched historical predictions"],
        "output_sha256": {name: sha256_file(out / name) for name in [*artifacts, "REPORT.md"]},
    })
    print(f"Verified all three traits: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
