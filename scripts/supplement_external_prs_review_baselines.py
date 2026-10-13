"""Audit plain single-PRS baselines and ancestry comparisons for the PRS review.

Reuses frozen predictions except for train-only CAD single-score selection.
Patient-level artifacts stay in the ignored analysis directory.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.ancestry_performance import paired_bootstrap_comparisons, prediction_metrics
from decipehr.discodivas import comparison_bank
from decipehr.eligibility import load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_pgs_benchmark import prepare_official_validation_data
from decipehr.internal_prs_benchmark import evaluate_partition, ids_hash, inner_splits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/external_pgs/progress_review_20261012"
OUT = BASE / "academic_revision_20261012"
CAD = ROOT / "outputs/external_pgs/cad_expansion_20261012"
CORRECTION = ROOT / "outputs/external_pgs/correction_20261011"
GROUPS = ("AFR-like", "AMR-like", "EAS-like", "EUR-like", "SAS-like", "AMBIGUOUS")
SOURCES = {}
CHECKS = []


def track(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = sha256_file(path)
    return path


def read(path, **kwargs):
    return pd.read_csv(track(path), dtype={"sampleid": str, "IID": str},
                       float_precision="round_trip", **kwargs)


def verify(path, key, parent):
    receipt = json.loads(track(path).read_text())
    for name, digest in receipt[key].items():
        assert sha256_file(track(parent / name)) == digest, name


def cad_single_baseline():
    out = OUT / "cad_single_best"
    if (out / "VERIFIED.json").exists():
        verify(out / "VERIFIED.json", "output_sha256", out)
        return read(out / "predictions.csv"), read(out / "selection.csv")
    out.mkdir(exist_ok=True)
    protocol = json.loads(track(CAD / "PROTOCOL.json").read_text())
    paths = protocol["source_paths"]
    bank = CAD / "banks/expanded"
    lock = json.loads(track(bank / "weights.scored.lock.json").read_text())
    ids = [r["id"] for r in lock["resources"]]
    assert len(ids) == 7
    context, query = prepare_official_validation_data(
        train_phenotype=read(ROOT / paths["train"]),
        validation_phenotype=read(ROOT / paths["development"]),
        scores=read(bank / "scores.long.csv"), soft_weights=None,
        hard_labels=read(ROOT / paths["ancestry"], sep="\t"), disease="cad", model_ids=ids,
        eligibility_rules=load_eligibility_rules(track(ROOT / paths["eligibility"])),
    )
    for frame in (context, query):
        frame["kinship_group"] = "patient_only_" + frame.sampleid
    folds = read(CAD / "shared_folds.csv")
    data = context.merge(folds[["sampleid", "fold"]], on="sampleid", validate="one_to_one")
    assert len(data) == len(context) == 5811 and len(query) == 2852
    partitions = [("development", -1, context, query)]
    partitions += [("training_oof", f, data.loc[data.fold.ne(f)], data.loc[data.fold.eq(f)])
                   for f in sorted(data.fold.unique())]
    predictions, selections, fits, parameters = [], [], [], []
    for scope, fold, current, target in partitions:
        current, target = current.reset_index(drop=True), target.reset_index(drop=True)
        seed = 42 if fold == -1 else 42 + int(fold)
        p, s, audit, config = evaluate_partition(
            current, target, disease="cad", full_ids=ids,
            bank=comparison_bank("cad", pd.DataFrame()), methods=("best_single_lr",), seed=seed,
        )
        final = audit.loc[audit.stage.eq("final")].iloc[0]
        assert final.context_ids_sha256 == ids_hash(current)
        assert final.query_ids_sha256 == ids_hash(target)
        for inner_fold, (a, b) in enumerate(inner_splits(current, "cad", seed)):
            rows = audit.loc[audit.stage.eq("inner") & audit.inner_fold.eq(inner_fold)]
            assert len(rows) == 28
            assert rows.context_ids_sha256.eq(ids_hash(current.iloc[a])).all()
            assert rows.query_ids_sha256.eq(ids_hash(current.iloc[b])).all()
        chosen = s.loc[s.selected.eq(True)]
        assert len(chosen) == 1 and len(s) == 28
        assert chosen.iloc[0].inner_auc == s.inner_auc.max()
        predictions.append(p.assign(scope=scope, fold=fold))
        selections.append(s.assign(scope=scope, fold=fold))
        fits.append(audit.assign(scope=scope, fold=fold))
        parameters.append({"scope": scope, "fold": int(fold), "parameters": config})
        print(f"CAD single-best {scope}/{fold}: {chosen.iloc[0].configuration}", flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    selection = pd.concat(selections, ignore_index=True)
    assert not pred.duplicated(["scope", "sampleid"]).any()
    pred.to_csv(out / "predictions.csv", index=False)
    selection.to_csv(out / "selection.csv", index=False)
    pd.concat(fits, ignore_index=True).to_csv(out / "fit_audit.csv", index=False)
    write_json(out / "parameters.json", parameters)
    write_json(out / "VERIFIED.json", {
        "status": "verified", "training_only_selection": True, "candidate_scores": ids,
        "C_grid": [0.01, 0.1, 1, 10], "partitions": 6, "fits_per_partition": 85,
        "shared_outer_folds_verified": True, "inner_context_hashes_verified": True,
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
    })
    return pred, selection


def normalized(frame):
    return frame.rename(columns={"y_prob": "probability", "y_true": "label",
                                 "ancestry_group": "hard_label"})


def main():
    track(OUT / "ANALYSIS_PLAN.json")
    verify(CAD / "report/VERIFIED.json", "output_sha256", CAD / "report")
    verify(BASE / "REPORT_VERIFICATION.json", "files_sha256", ROOT)
    for name in ["src/decipehr/internal_prs_benchmark.py", "src/decipehr/ancestry_performance.py",
                 "src/decipehr/external_pgs_benchmark.py", "src/decipehr/metrics.py"]:
        track(ROOT / name)
    cad_predictions, cad_selection = cad_single_baseline()
    points, paired, choices, all_predictions = [], [], [], []
    expected = read(BASE / "current_development_metrics.csv")
    for disease in ("T2D", "cad", "bc"):
        original = read(CORRECTION / "development" / disease / "predictions.csv")
        for method, frame in original.groupby("method"):
            values = prediction_metrics(frame.label, frame.probability)
            old = expected.loc[expected.disease.eq(disease) & expected.method.eq(method)].iloc[0]
            for key in ("auc", "nagelkerke_r2", "brier", "log_loss"):
                assert abs(values[key] - old[key]) < 1e-12
        selection = read(CORRECTION / "development" / disease / "selection.csv")
        if disease == "cad":
            selection = cad_selection.loc[cad_selection.scope.eq("development")]
        chosen = selection.loc[selection.method.eq("best_single_lr") & selection.selected.eq(True)].iloc[0]
        candidate = json.loads(chosen.configuration)
        choices.append({"disease": disease, "selected_score": candidate["prs"][0], "C": candidate["c"],
                        "training_inner_auc": chosen.inner_auc,
                        "candidate_scores": {"T2D": 31, "cad": 7, "bc": 3}[disease],
                        "selection": "training-only three-fold pooled AUC; log loss tie-break"})
        keep = ["covariates_lr", "best_single_lr", "full_lr", "tabpfn_full", "tabpfn_single",
                "tabpfn_matched", "selected_disco"]
        primary = original.loc[original.method.isin(keep)].copy()
        if disease == "cad":
            primary = primary.loc[~primary.method.isin(["best_single_lr", "full_lr", "tabpfn_full"])]
            latest = normalized(read(CAD / "evaluation/development/expanded/validation_predictions.csv"))
            latest["method"] = latest.method.map({"multi_prs_pc_lr": "full_lr", "tabpfn_prs_pc": "tabpfn_full"})
            single = cad_predictions.loc[cad_predictions.scope.eq("development")]
            primary = pd.concat([primary, latest, single], ignore_index=True)
        if disease == "T2D":
            p25 = pd.read_parquet(track(ROOT / "outputs/external_pgs/dprism25_comparison_20261011/report/predictions.parquet"))
            p25 = p25.loc[p25.evaluation.eq("development") & p25.method.eq("tabpfn_dprism25")].copy()
            p25["method"] = "tabpfn_disco_comparator"
            primary = pd.concat([primary, p25], ignore_index=True)
        else:
            matched = primary.loc[primary.method.eq("tabpfn_matched")].copy()
            matched["method"] = "tabpfn_disco_comparator"
            primary = pd.concat([primary, matched], ignore_index=True)
        primary = primary[["sampleid", "disease", "label", "hard_label", "method", "probability"]]
        primary["disease"] = disease
        assert not primary.duplicated(["sampleid", "method"]).any()
        truth = original.loc[original.method.eq("tabpfn_full")].set_index("sampleid").sort_index()
        for method, frame in primary.groupby("method"):
            frame = frame.set_index("sampleid").sort_index()
            assert frame.index.equals(truth.index)
            np.testing.assert_array_equal(frame.label, truth.label)
            np.testing.assert_array_equal(frame.hard_label, truth.hard_label)
            groups = [("ALL", frame), *[(g, frame.loc[frame.hard_label.eq(g)]) for g in GROUPS]]
            for group, part in groups:
                values = prediction_metrics(part.label, part.probability)
                points.append({"disease": disease, "method": method, "ancestry_group": group,
                               "small_group": min(values["cases"], values["controls"]) < 20, **values})
        all_predictions.append(primary)
        wide = primary.pivot(index=["sampleid", "label", "hard_label"], columns="method", values="probability").reset_index()
        assert not wide.isna().any().any()
        wide = wide.rename(columns={"label": "y_true", "hard_label": "ancestry_group"})
        wide["disease"] = disease
        comparisons = (("full_vs_single_lr", "best_single_lr", "tabpfn_full"),
                       ("matched_vs_disco", "selected_disco", "tabpfn_disco_comparator"))
        # Separate metric calls use the existing fast AUC implementation.
        for metric in ("auc", "nagelkerke_r2"):
            print(f"Paired intervals: {disease}, {metric}, ALL and six subgroups", flush=True)
            result = paired_bootstrap_comparisons(wide, n_bootstraps=2000, seed=20261012,
                comparisons=comparisons, metric_names=(metric,), family_size=12)
            subgroup = result.ancestry_group.ne("ALL")
            result.loc[subgroup, ["familywise_ci_low", "familywise_ci_high", "family_size"]] = np.nan
            result["inference"] = np.where(subgroup, "exploratory nominal 95% CI", "post-hoc 12-comparison Bonferroni")
            paired.append(result)
        CHECKS.append({"disease": disease, "n": len(truth), "cases": int(truth.label.sum()),
                       "all_method_patient_sets_labels_and_ancestry_equal": True})
    pd.DataFrame(points).to_csv(OUT / "development_metrics.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(OUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT / "single_best_selection.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(OUT / "predictions.private.parquet", index=False)
    nested = []
    for scope, frame in cad_predictions.groupby("scope"):
        nested.append({"scope": scope, "method": "best_single_lr_nested", **prediction_metrics(frame.label, frame.probability)})
    pd.DataFrame(nested).to_csv(OUT / "cad_nested_single_metrics.csv", index=False)
    cad_selection.loc[cad_selection.selected.eq(True), ["scope", "fold", "configuration", "inner_auc", "inner_log_loss"]].to_csv(OUT / "cad_nested_single_choices.csv", index=False)
    for path, digest in SOURCES.items():
        assert sha256_file(ROOT / path) == digest
    write_json(OUT / "VERIFIED.json", {
        "status": "verified", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_checks": CHECKS, "source_sha256": SOURCES,
        "point_metrics_recomputed": True, "single_best_selection_uses_training_only": True,
        "new_model_fits": "CAD only: 6 contexts x (7 PRS x 4 C x 3 inner folds + 1 final fit) = 510 logistic fits",
        "new_tabpfn_or_disco_fits": 0, "overall_family_size": 12,
        "subgroup_intervals": "nominal 95%, exploratory; no simultaneous inference",
        "script_sha256": sha256_file(Path(__file__)),
        "output_sha256": {p.name: sha256_file(p) for p in OUT.glob("*.csv")},
    })
    print(pd.DataFrame(choices).to_string(index=False), flush=True)
    print("Verified baseline and subgroup supplements.", flush=True)


if __name__ == "__main__":
    main()
