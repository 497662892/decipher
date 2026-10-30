"""Plain multivariable logistic regression on the frozen PRS development cohorts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from report_prs_distillation_bio import paired_bootstrap
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import AGE, PCS, assert_boundary, id_hash
from decipehr.prs_distillation.reporting import basic_metrics, calibration
from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer

BIO = Path("outputs/prs_distillation_bio/exploration_20261026_3569648f99a8")
SOURCE = Path("outputs/prs_distillation_v1/phase_a_fulltrain_20261026_3d4334d2_0161ad36d950")
LR = Path("outputs/prs_distillation_v1/joint_logistic_comparison_fulltrain_20261026")
FILTERS = Path("outputs/prs_distillation_bio/validation_kinship_filters_20261027")
TRAITS = ["T2D", "cad", "bc"]
COHORTS = ["original_val", "exclude_shared_group", "exclude_direct_0177"]
ARMS = ["LR_UNPENALIZED", "LR_L2", "M0_SUP", "M0_KD", "TabPFN_v3"]


def read(path):
    return json.loads(Path(path).read_text())


def transformer(frame, prs):
    return TabPFNFeatureTransformer.fit(
        frame, prs_cols=prs, covariates=[AGE, "sex", *PCS], include_covariates=True)


def fit_plain(context, query, prs, directory, *, allow_family_overlap=False):
    """Fit preprocessing and unpenalized MLE using context labels only."""
    if "y" in query:
        raise ValueError("Query labels must not be supplied to model fitting")
    assert_boundary(context, query, require_group_disjoint=not allow_family_overlap)
    prep = transformer(context, prs)
    # Remove one dummy per categorical feature; the intercept represents its reference.
    references = [f"{c.output_prefix}__{c.categories[0]}" for c in prep.categorical_features]
    columns = [c for c in prep.output_columns if c not in references]
    x = prep.transform(context)[columns].to_numpy()
    xq = prep.transform(query)[columns].to_numpy()
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=20000, tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        # sklearn 1.8 maps C=inf to penalty=None and emits this redundant message.
        warnings.filterwarnings("ignore", message="Setting penalty=None will ignore the C and l1_ratio parameters", category=UserWarning)
        model.fit(x, context.y.to_numpy())
    np.testing.assert_array_equal(model.classes_, [0, 1])
    p = model.predict_proba(xq)[:, 1]
    np.testing.assert_allclose(p, expit(xq @ model.coef_[0] + model.intercept_[0]), atol=1e-15)
    design = np.column_stack([np.ones(len(context)), x])
    residual = model.predict_proba(x)[:, 1] - context.y.to_numpy()
    gradient = float(np.max(np.abs(design.T @ residual / len(context))))
    if gradient > 1e-5:
        raise ValueError(f"Unpenalized likelihood gradient too large: {gradient}")
    directory.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"feature": ["intercept", *columns],
                  "coefficient": [model.intercept_[0], *model.coef_[0]]}).to_csv(
                      directory / "coefficients.csv", index=False)
    write_json(directory / "preprocessing.json", asdict(prep))
    record = {"n_context": len(context), "n_query": len(query),
              "fit_ids_hash": id_hash(context), "query_ids_hash": id_hash(query),
              "features": columns, "reference_dummies": references,
              "penalty": "none", "C": "infinity", "solver": "lbfgs", "tol": 1e-9,
              "max_iter": 20000, "iterations": int(model.n_iter_[0]),
              "gradient_max_abs": gradient, "design_rank": int(np.linalg.matrix_rank(design)),
              "design_columns": design.shape[1], "query_labels_used": False,
              "family_overlap_allowed": allow_family_overlap,
              "training_log_loss": basic_metrics(context.y, model.predict_proba(x)[:, 1])["log_loss"]}
    write_json(directory / "FIT.json", record)
    return p, record


def metrics(y, p):
    return {**basic_metrics(y, p), "nagelkerke_r2": nagelkerke_r2(y, p),
            "average_precision": average_precision_score(y, p), **calibration(y, p)}


def aligned(frame, ids, column="probability"):
    assert frame.sampleid.is_unique and set(frame.sampleid) == set(ids)
    return frame.set_index("sampleid").loc[ids, column].to_numpy()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eligibility-config", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    private, report = out / "private", out / "report"
    private.mkdir()
    report.mkdir()
    audit = read(SOURCE / "audit_report.json")
    assert sha256_file(args.eligibility_config) == audit["source_sha256"][str(args.eligibility_config)]
    rules = load_eligibility_rules(args.eligibility_config)
    assert "bc" in rules
    lr_receipt, filter_receipt, bio_receipt = [read(p / n) for p, n in
        [(LR, "VERIFIED.json"), (FILTERS, "VERIFIED.json"), (BIO, "COMPLETE.json")]]
    verify_files(LR, lr_receipt["output_sha256"])
    verify_files(Path.cwd(), lr_receipt["input_sha256"])
    verify_files(FILTERS, {n: filter_receipt["output_sha256"][n] for n in
                         ["private/cohort_membership.parquet", "report/metrics.csv", "report/cohort_counts.csv"]})
    assert filter_receipt["train_unchanged"] and not filter_receipt["refitted"]
    split_receipt = read(SOURCE / "split_receipt.json")
    assert sha256_file(SOURCE / "split_manifest.parquet") == split_receipt["sha256"]
    split = pd.read_parquet(SOURCE / "split_manifest.parquet").set_index("sampleid")
    membership = pd.read_parquet(FILTERS / "private/cohort_membership.parquet")
    used = {str(p): sha256_file(p) for p in [
        SOURCE / "audit_report.json", SOURCE / "split_manifest.parquet", LR / "VERIFIED.json",
        LR / "PROTOCOL.json", FILTERS / "VERIFIED.json", FILTERS / "private/cohort_membership.parquet",
        BIO / "COMPLETE.json", args.eligibility_config]}
    protocol = {
        "primary_baseline": "LR_UNPENALIZED", "secondary_baseline": "frozen train-selected L2 LR, C=0.1",
        "features": "same frozen PRS panels + linear age_sd + sex dummies + PC1-PC10",
        "prs_counts": {"T2D": 26, "cad": 7, "bc": 5},
        "training": "true labels only, no teacher targets, no interactions, unrestricted signs",
        "unpenalized_tuning": "none; convergence settings fixed before reading evaluation labels",
        "preprocessing": "fit within each training context; one reference dummy removed for unpenalized MLE",
        "main_assessment": "all eligible official train -> official val, direct probabilities",
        "training_oof": "same frozen final_teacher_fold, five patient/available-kinship-group folds",
        "validation_filters": "reuse exact prior original/shared-group/direct-0.177 memberships; unresolved retained",
        "teacher_student_refit": False, "new_fits": "three full-train LR models and fifteen training OOF LR models",
        "development_status": "post-hoc development comparison; official val has previously been inspected",
        "bootstrap": "1000 paired available-kinship-group resamples, conditional on fixed predictions; no multiplicity correction",
        "age_units": "age_sd, not years", "eligibility_config": str(args.eligibility_config),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]},
        "script_sha256": sha256_file(__file__), "input_sha256": used}
    write_json(out / "PROTOCOL.json", protocol)
    prepared, fits, oof_metrics = {}, [], []
    for trait in TRAITS:
        d, e, manifest = load_inputs(SOURCE, trait)
        for frame in [d, e]:
            assert set(filter_eligible(frame, trait, rules).sampleid) == set(frame.sampleid)
        ids, prs = e.sampleid.tolist(), manifest.column.tolist()
        assert len(prs) == protocol["prs_counts"][trait]
        used[str(SOURCE / trait / "input_receipt.json")] = sha256_file(SOURCE / trait / "input_receipt.json")
        arrays = {}
        p, record = fit_plain(d, e, prs, private / trait / "final_model", allow_family_overlap=True)
        arrays["LR_UNPENALIZED"] = p
        fits.append({"trait": trait, "scope": "final_train", **record})
        folds = split.loc[d.sampleid, "final_teacher_fold"].to_numpy()
        q, coverage = np.full(len(d), np.nan), np.zeros(len(d), int)
        assert set(folds) == set(range(5))
        for fold in range(5):
            mask = folds == fold
            q[mask], record = fit_plain(d.loc[~mask], d.loc[mask].drop(columns="y"), prs,
                                       private / trait / "oof" / str(fold))
            coverage[mask] += 1
            fits.append({"trait": trait, "scope": f"outer_fold_{fold}", **record})
        assert (coverage == 1).all() and np.isfinite(q).all()
        d[["sampleid", "kinship_group", "y"]].assign(fold=folds, probability=q).to_parquet(
            private / trait / "training_oof.parquet", index=False)
        oof_metrics.append({"trait": trait, "arm": "LR_UNPENALIZED", **metrics(d.y.to_numpy(), q)})
        old_oof = pd.read_parquet(LR / trait / "training_oof.parquet")
        np.testing.assert_array_equal(aligned(old_oof, d.sampleid.tolist(), "fold"), folds)
        np.testing.assert_array_equal(aligned(old_oof, d.sampleid.tolist(), "y"), d.y.to_numpy())
        oof_metrics.append({"trait": trait, "arm": "LR_L2", **metrics(d.y.to_numpy(), aligned(old_oof, d.sampleid.tolist()))})
        old = pd.read_parquet(LR / trait / "validation_predictions.parquet")
        arrays["LR_L2"] = aligned(old, ids)
        # Verify existing predictions from plain coefficients, without deserializing model files.
        coefficients = pd.read_csv(LR / trait / "final_model/coefficients.csv").set_index("feature").coefficient
        design = transformer(d, prs).transform(e)
        np.testing.assert_allclose(expit(design.to_numpy() @ coefficients.loc[design.columns].to_numpy()
                                        + coefficients.loc["intercept"]), arrays["LR_L2"], atol=1e-12, rtol=0)
        core = BIO / "core" / trait
        seal = core / "PREDICTIONS_COMPLETE.json"
        assert sha256_file(seal) == bio_receipt["upstream_receipt_sha256"][str(seal)]
        receipt = read(seal)
        assert not receipt["evaluation_labels_read"]
        verify_files(core, {"direct_predictions.parquet": receipt["output_sha256"]["direct_predictions.parquet"]})
        frame = pd.read_parquet(core / "direct_predictions.parquet", filters=[("seed", "=", 17)])
        for arm in ["M0_SUP", "M0_KD"]:
            arrays[arm] = aligned(frame.loc[frame.arm.eq(arm)], ids)
        arrays["TabPFN_v3"] = aligned(frame.loc[frame.arm.eq("M0_KD")], ids, "p_teacher")
        for path in [seal, core / "direct_predictions.parquet", LR / trait / "validation_predictions.parquet",
                     LR / trait / "final_model/coefficients.csv", LR / trait / "training_oof.parquet"]:
            used[str(path)] = sha256_file(path)
        member = membership.loc[membership.trait.eq(trait)].set_index("sampleid")
        assert member.index.is_unique and set(member.index) == set(ids)
        member = member.loc[ids]
        np.testing.assert_array_equal(member.kinship_group, e.kinship_group)
        masks = {"original_val": np.ones(len(e), bool), "exclude_shared_group": ~member.remove_rule1.to_numpy(),
                 "exclude_direct_0177": ~member.remove_rule2.to_numpy()}
        predictions = e[["sampleid", "kinship_group"]].copy()
        for arm, probability in arrays.items():
            predictions[arm] = probability
        predictions.to_parquet(private / trait / "validation_predictions.parquet", index=False)
        prepared[trait] = (e, arrays, masks)
        print(f"Locked {trait}: train={len(d)}, val={len(e)}, 6 converged unpenalized fits; frozen L2 replay exact", flush=True)
    pd.DataFrame(fits).to_csv(report / "training_details.csv", index=False)
    pd.DataFrame(oof_metrics).to_csv(report / "training_oof_metrics.csv", index=False)
    write_json(out / "PREDICTIONS_COMPLETE.json", {
        "validation_labels_read": False, "new_fits": len(fits), "input_sha256": used,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*") if p.is_file()}})
    label_path = Path(audit["paths"]["evaluation"])
    assert sha256_file(label_path) == audit["source_sha256"][str(label_path)]
    labels = pd.read_csv(label_path, dtype={"sampleid": str}, usecols=["sampleid", *TRAITS]).set_index("sampleid")
    rows, gaps = [], []
    # Reuse the tested bootstrap's comparison slots; keep only M0_KD contrasts.
    slots = {"H_SELECTED": "M0_KD", "M0_KD": "LR_UNPENALIZED", "M0_SUP": "LR_L2",
             "G_A6_KD": "M0_SUP", "TabPFN_v3": "TabPFN_v3"}
    for trait, (e, arrays, masks) in prepared.items():
        y = labels.loc[e.sampleid, trait].to_numpy(int)
        for cohort, mask in masks.items():
            for arm, p in arrays.items():
                rows.append({"trait": trait, "cohort": cohort, "arm": arm, **metrics(y[mask], p[mask])})
            delta = paired_bootstrap(y[mask], {k: arrays[v][mask] for k, v in slots.items()},
                                     e.loc[mask, "kinship_group"].to_numpy(), "nested", replicates=1000)
            delta = delta.loc[delta.arm.eq("H_SELECTED")].copy()
            delta["arm"], delta["reference"] = delta.arm.map(slots), delta.reference.map(slots)
            gaps.append(delta.assign(trait=trait, cohort=cohort))
            print(f"Evaluated {trait}/{cohort}: n={mask.sum()}", flush=True)
    measured, differences = pd.DataFrame(rows), pd.concat(gaps, ignore_index=True)
    old = pd.read_csv(BIO / "report/val/metrics.csv").set_index(["trait", "arm"])
    for row in measured.loc[measured.cohort.eq("original_val") & measured.arm.ne("LR_UNPENALIZED")].itertuples():
        keys = ["auroc", "log_loss", "brier", "nagelkerke_r2"]
        np.testing.assert_allclose([getattr(row, k) for k in keys],
            old.loc[(row.trait, "LR" if row.arm == "LR_L2" else row.arm), keys].to_numpy(float), atol=1e-12, rtol=0)
    old = pd.read_csv(FILTERS / "report/metrics.csv").set_index(["trait", "cohort", "arm"])
    for row in measured.loc[measured.arm.isin(["M0_KD", "TabPFN_v3"])].itertuples():
        keys = ["auroc", "log_loss", "brier", "nagelkerke_r2"]
        np.testing.assert_allclose([getattr(row, k) for k in keys], old.loc[(row.trait, row.cohort, row.arm), keys].to_numpy(float), atol=1e-12, rtol=0)
    for row in differences.itertuples():
        part = measured.loc[measured.trait.eq(row.trait) & measured.cohort.eq(row.cohort)].set_index("arm")
        np.testing.assert_allclose(row.delta, part.loc[row.arm, row.metric] - part.loc[row.reference, row.metric], atol=1e-14)
    measured.to_csv(report / "metrics.csv", index=False)
    differences.to_csv(report / "paired_differences.csv", index=False)
    (report / "cohort_counts.csv").write_bytes((FILTERS / "report/cohort_counts.csv").read_bytes())
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    verify_files(Path.cwd(), used)
    write_json(out / "VERIFIED.json", {"status": "completed_and_verified", "new_fits": len(fits),
        "all_fits_converged": True, "training_only_preprocessing": True, "shared_five_fold_oof": True,
        "eligible_bc_only": True, "original_metrics_reproduced": True, "filtered_metrics_reproduced": True,
        "frozen_l2_coefficient_replay": True, "no_teacher_student_refit": True,
        "label_source_sha256": sha256_file(label_path), "input_sha256": used,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*") if p.is_file()}})
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
