"""Export frozen external-PRS OOF/holdout predictions in the historical GoGene format."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_pgs_benchmark import _ids_hash
from decipehr.io import read_table
from decipehr.submission import standardized_logit_submission

DISEASES = ("T2D", "cad", "bc")
HISTORICAL = Path("submissions/final_phase/Go Gene!_audited_pipeline_with_trainval_5cv")
GITHUB_REFERENCE = "https://github.com/nyu-shenlab/dreams-team-gogene/tree/main/output"
# Checked against the user's GitHub output files on 2026-10-13.
GITHUB_SHA256 = {
    "T2D": "672af0b8322e496ccf6c457372bf6f143c65c891021225b722f13e0a79501ddb",
    "cad": "eb4f8daffb9e667e5c1b55f743649c4b2632f88792aade844604114368d1fc5a",
    "bc": "2cb818d2ea3522b781f378b42fd8d843cfca1909190bd5db795c3e40e6bee98f",
}


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return read_table(path, dtype={"sampleid": str}, float_precision="round_trip", **kwargs)


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", default="outputs/external_pgs/frozen_internal_external_trainval_20261013")
    parser.add_argument("--holdout-dir", default="outputs/external_pgs/holdout_wrapper_20261012")
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--out-dir", default="submissions/final_phase/gogene_external_pgs_20261013")
    args = parser.parse_args()
    comparison, holdout_root, out = map(Path, [args.comparison_dir, args.holdout_dir, args.out_dir])
    if out.exists():
        raise FileExistsError("Use a new output directory; existing exports are never overwritten")
    sources: dict[str, str] = {}

    def track(path, expected=None):
        path = str(Path(path).resolve())
        if path not in sources:
            sources[path] = sha256_file(path)
        if expected is not None and sources[path] != expected:
            raise ValueError(f"Frozen source hash mismatch: {path}")
        return sources[path]

    def record(path, expected=None):
        track(path, expected)
        return json.loads(Path(path).read_text())

    protocol = record(comparison / "PROTOCOL.json")
    report = record(comparison / "report/VERIFIED.json")
    final = record(holdout_root / "FINAL_VERIFIED.json")
    if report["status"] != "complete" or final["status"] != "complete":
        raise ValueError("Both source runs must have completed verification")
    track(comparison / "PROTOCOL.json", report["protocol_sha256"])
    for path, digest in protocol["input_sha256"].items():
        track(path, digest)
    for base, receipt in [(comparison / "report", report), (holdout_root, final)]:
        for name, digest in receipt["output_sha256"].items():
            track(base / name, digest)
        for path, digest in receipt["source_receipts_sha256"].items():
            track(path, digest)
    selection = record(holdout_root / "SELECTION.json", final["selection_sha256"])
    track(args.eligibility_config, track(protocol["paths"]["eligibility"]))
    rules = load_eligibility_rules(args.eligibility_config)
    phenotype = read_csv(protocol["paths"]["phenotype"])
    holdout = read_csv("data/processed/final/holdout_test_covariates.csv")
    splits = read_csv(protocol["paths"]["splits"])
    if set(holdout.columns) & set(DISEASES) or set(phenotype.sampleid) & set(holdout.sampleid):
        raise ValueError("Holdout labels or train/holdout patient overlap")
    oof = read_csv(comparison / "report/oof_predictions.csv")
    oof = oof.loc[oof.arm.eq("external")].copy()
    query = read_csv(holdout_root / "holdout_predictions.long.csv")
    # Original holdout CSVs serialized native float32 probabilities compactly.
    query["y_prob"] = query.y_prob.astype(np.float32).astype(float)
    if (set(oof.disease) != set(DISEASES) or set(query.disease) != set(DISEASES)
            or not oof.method.eq("tabpfn_prs_pc").all() or not query.method.eq("tabpfn_prs_pc").all()):
        raise ValueError("Unexpected diseases or predictive methods")
    if oof.groupby("sampleid").fold.nunique().ne(1).any():
        raise ValueError("A patient belongs to different folds across diseases")

    parts, summaries, submissions = [], [], {}
    for disease in DISEASES:
        receipt_path = holdout_root / "predictions" / disease / "VERIFIED.json"
        receipt = record(receipt_path)
        for name, digest in receipt["input_sha256"].items():
            track(receipt["input_paths"][name], digest)
        for name, digest in receipt["output_sha256"].items():
            track(receipt_path.parent / name, digest)
        for key, protocol_key in [("checkpoint", "checkpoint"), ("phenotype", "phenotype"),
                                  ("context_scores", f"{disease}_external_scores")]:
            if receipt["input_sha256"][key] != track(protocol["paths"][protocol_key]):
                raise ValueError("OOF and holdout inputs differ")
        if (receipt["model_ids"] != protocol["models"][disease]["external"]
                or receipt["model_ids"] != selection["diseases"][disease]["model_ids"]
                or receipt["arguments"]["seed"] != protocol["seed"]
                or receipt["n_estimators"] != 16
                or receipt["prediction_method"] != "tabpfn_prs_pc"
                or receipt["tabpfn_version"] != report["versions"]["tabpfn"]):
            raise ValueError("OOF and holdout model settings differ")
        track("data/processed/final/holdout_test_covariates.csv", receipt["input_sha256"]["holdout"])
        train_cohort = filter_eligible(phenotype, disease, rules)
        query_cohort = filter_eligible(holdout, disease, rules)
        train_pred = oof.loc[oof.disease.eq(disease)].copy()
        query_pred = query.loc[query.disease.eq(disease)].copy()
        data = pd.read_parquet(comparison / "inputs" / disease / "external.parquet")
        for fold in range(5):
            directory = comparison / "folds" / disease / "external" / str(fold)
            fold_record = record(directory / "COMPLETE.json")
            if (fold_record["status"] != "complete"
                    or fold_record["protocol_sha256"] != track(comparison / "PROTOCOL.json")
                    or fold_record["versions"] != report["versions"]):
                raise ValueError("Incomplete or incompatible outer fold")
            for name, digest in fold_record["output_sha256"].items():
                track(directory / name, digest)
            columns = ["sampleid", "disease", "fold", "y_true", "y_prob"]
            saved = read_csv(directory / "predictions.csv")
            current = train_pred.loc[train_pred.fold.eq(fold)]
            pd.testing.assert_frame_equal(
                saved[columns].sort_values("sampleid").reset_index(drop=True),
                current[columns].sort_values("sampleid").reset_index(drop=True), check_exact=True,
            )
            fit = read_csv(directory / "fit_audit.csv").query("fit_scope == 'outer_context'")
            if (len(fit) != 1 or fit.iloc[0].context_ids_sha256 != _ids_hash(data.loc[data.fold.ne(fold)])
                    or fit.iloc[0].query_ids_sha256 != _ids_hash(data.loc[data.fold.eq(fold)])):
                raise ValueError("OOF context/query patient hash mismatch")
        full = read_csv(receipt_path.parent / "fit_audit.csv").query("fit_scope == 'full_trainval_context'")
        if (len(full) != 1 or full.iloc[0].context_ids_sha256 != _ids_hash(train_cohort)
                or full.iloc[0].query_ids_sha256 != _ids_hash(query_cohort)):
            raise ValueError("Holdout was not predicted from the complete eligible train+development context")
        scores, parameters = standardized_logit_submission(
            train_pred, query_pred, train_ids=train_cohort.sampleid, holdout_ids=query_cohort.sampleid,
            folds=splits.loc[splits.disease.eq(disease), ["sampleid", "fold"]],
        )
        labels = train_cohort.set_index("sampleid")[disease].loc[train_pred.sampleid]
        if not np.array_equal(labels, train_pred.y_true):
            raise ValueError("OOF labels differ from the eligible phenotype")
        historical_file = HISTORICAL / f"Go Gene!_{disease}.txt"
        track(historical_file, GITHUB_SHA256[disease])
        historical = read_csv(historical_file)
        if list(historical.columns) != ["sampleid", f"{disease}_score"]:
            raise ValueError("Unexpected historical schema")
        if historical.sampleid.duplicated().any() or set(historical.sampleid) != set(scores.sampleid):
            raise ValueError("New submission cohort differs from the historical GitHub cohort")
        submissions[disease] = scores.set_index("sampleid").loc[historical.sampleid, ["score"]].reset_index().rename(
            columns={"score": f"{disease}_score"},
        )
        parts.append(scores.assign(disease=disease))
        tr = scores.loc[scores.cohort.eq("trainval_oof")]
        ho = scores.loc[scores.cohort.eq("holdout")]
        y = train_cohort.set_index("sampleid")[disease].loc[tr.sampleid].to_numpy()
        summaries.append({
            "disease": disease, "n_trainval": len(tr), "n_holdout": len(ho), "n_combined": len(scores),
            **parameters, "oof_score_mean": float(tr.score.mean()), "oof_score_sd_ddof0": float(tr.score.std(ddof=0)),
            "holdout_score_min": float(ho.score.min()), "holdout_score_max": float(ho.score.max()),
            "oof_clipped": int(((tr.direct_probability < 0.001) | (tr.direct_probability > 0.999)).sum()),
            "holdout_clipped": int(((ho.direct_probability < 0.001) | (ho.direct_probability > 0.999)).sum()),
            "oof_auc_direct_probability": roc_auc_score(y, tr.direct_probability),
            "oof_auc_submission_score": roc_auc_score(y, tr.score),
        })
        print(f"Verified {disease}: five OOF folds, full train+development holdout, historical cohort", flush=True)

    track(__file__)
    track("src/decipehr/submission.py")
    out.mkdir(parents=True, exist_ok=False)
    (out / "output").mkdir()
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(out / "scores.long.csv", index=False)
    wide = combined.loc[combined.cohort.eq("holdout")].pivot(index="sampleid", columns="disease", values="score")
    wide = wide.reindex(index=holdout.sampleid, columns=list(DISEASES)).rename(columns={d: f"{d}_score" for d in DISEASES})
    wide.reset_index().to_csv(out / "holdout_standardized_logits.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "standardization.csv", index=False)
    files = []
    for disease, frame in submissions.items():
        path = out / "output" / f"gogene_{disease}.txt"
        frame.to_csv(path, sep="\t", index=False, lineterminator="\n")
        pd.testing.assert_frame_equal(frame, read_csv(path), check_exact=True)
        files.append(path)
    archive_path = out / "gogene_external_pgs_standardized_logits.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or any(archive.read(p.name) != p.read_bytes() for p in files):
            raise ValueError("Submission ZIP readback failed")
    (out / "README.md").write_text(
        "# External PRS submission on the historical GoGene scale\n\n"
        "output/gogene_{T2D,cad,bc}.txt are tab-separated, two-column submission files.\n"
        "They contain train+development shared-five-fold OOF predictions followed by holdout,\n"
        "with precisely the historical GitHub patient set/order and disease score headers.\n"
        "The ZIP contains these three files only. holdout_standardized_logits.csv is holdout-only.\n\n"
        "Scores = (logit(clip(direct TabPFN probability, 0.001, 0.999)) - OOF mean) / OOF SD (ddof=0).\n"
        "The mean/SD use only train+development OOF logits, separately for each disease.\n"
        "Holdout predictions use the full eligible train+development context. BC is female-only;\n"
        "ineligible BC entries are blank only in the wide holdout CSV, and omitted from the BC TXT.\n\n"
        "These scores are standardized logits, not probabilities or Logistic-wrapper outputs.\n"
        "Existing probability artifacts remain unchanged. No model was retrained for this export.\n"
        "This restores submission format/scale; it does not establish an improvement in prediction.\n"
        "The PRS panels have prior development selection; OOF is conditional on frozen panels,\n"
        "not end-to-end nested PRS selection. Holdout labels are unavailable; no holdout metrics.\n\n"
        "standardization.csv includes clipping counts and OOF AUC before/after the export transform.\n"
        "scores.long.csv preserves the direct probabilities, logits, cohort, and OOF fold.\n"
        "run_record.json and VERIFIED.json preserve source hashes and export verification.\n"
        "These local patient-level outputs have not been published or uploaded.\n",
    )
    write_json(out / "run_record.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "prediction_source": "external arm y_prob (direct TabPFN); wrapper_prob is not used",
        "holdout_read_dtype": "float32 restored then promoted to float64", "score_scale": "trainval_5cv_oof_logit_mean_sd",
        "train_prediction_mode": "shared_5fold_oof", "holdout_prediction_mode": "full_trainval_context",
        "historical_github_reference": GITHUB_REFERENCE, "historical_file_sha256": GITHUB_SHA256,
        "models": {d: protocol["models"][d]["external"] for d in DISEASES},
        "source_versions": report["versions"], "standardization": summaries, "input_sha256": sources,
        "model_retrained": False, "end_to_end_nested_prs_cv": False,
        "holdout_labels_available": False, "holdout_metrics_available": False, "published": False,
    })
    # Independent file readback: use log(p) - log1p(-p), without the export helper.
    restored = read_csv(out / "scores.long.csv")
    maximum_error = 0.0
    for row in summary.itertuples():
        part = restored.loc[restored.disease.eq(row.disease)]
        p = np.clip(part.direct_probability.to_numpy(), row.clip_eps, 1 - row.clip_eps)
        logit = np.log(p) - np.log1p(-p)
        train_logit = logit[part.cohort.eq("trainval_oof")]
        expected = (logit - train_logit.mean()) / train_logit.std(ddof=0)
        error = float(np.max(np.abs(expected - part.score.to_numpy())))
        inverse_error = float(np.max(np.abs(expit(part.score * row.logit_trainval_oof_sd + row.logit_trainval_oof_mean) - p)))
        if error > 1e-12 or inverse_error > 1e-12:
            raise ValueError("Independent score/inverse-probability replay failed")
        maximum_error = max(maximum_error, error, inverse_error)
    for path, digest in sources.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Source changed during export: {path}")
    write_json(out / "VERIFIED.json", {
        "status": "complete", "source_sha256_verified": len(sources), "external_oof_folds_verified": 15,
        "historical_schema_patient_set_and_order_matched": True, "shared_patient_folds_verified": True,
        "female_bc_eligibility_verified": True, "holdout_used_for_standardization": False,
        "full_trainval_holdout_context_verified": True, "wrapper_prob_used": False,
        "independent_replay_max_abs_error": maximum_error, "zip_readback_verified": True,
        "original_prediction_artifacts_unchanged": True, "holdout_metrics_available": False,
        "n_combined_patient_disease_rows": len(combined), "n_holdout_rows_wide": len(wide),
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()},
    })
    print(summary[["disease", "n_trainval", "n_holdout", "n_combined", "oof_auc_direct_probability", "oof_auc_submission_score"]].to_string(index=False))
    print(f"Verified submission package: {out}")


if __name__ == "__main__":
    main()
