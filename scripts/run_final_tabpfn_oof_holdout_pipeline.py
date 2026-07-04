#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from build_tabpfn_ensemble_prs import (
    _csv_from_config,
    _fit_predict,
    _labels,
    _load_tabpfn_classifier,
    _normalise_tabpfn_model_version,
    _require_mapping,
    _required_value,
)

from decipehr.config import load_config
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.ensemble import pivot_long_prs_scores
from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns, validate_unique
from decipehr.tabpfn_ensemble import logit_from_probability

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_ELIGIBILITY_CONFIG = "configs/disease_eligibility.yaml"
DEFAULT_TRAIN_PHENOTYPE = "data/processed/final/train_val_phenotype.csv"
DEFAULT_SPLITS = "data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv"
DEFAULT_HOLDOUT_COVARIATES = "data/raw/holdout_phase/phenotype_files/Holdout_Cov_file.txt"


def _read_table_auto(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep="\t")
    return read_table(path)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _format_template(path_template: str, disease: str) -> str:
    return path_template.format(disease=disease)


def _score_col(disease: str) -> str:
    return f"{disease}_score"


def _check_holdout_covariates(
    *,
    official: pd.DataFrame,
    processed: pd.DataFrame | None,
    patient_col: str,
    covariates: list[str],
) -> dict[str, Any]:
    validate_unique(official, patient_col, "official holdout covariates")
    report: dict[str, Any] = {
        "n_official_holdout": int(len(official)),
        "official_duplicate_sampleids": 0,
        "processed_holdout_checked": processed is not None,
    }
    if processed is None:
        return report

    validate_unique(processed, patient_col, "processed holdout covariates")
    report["n_processed_holdout"] = int(len(processed))
    report["processed_same_order_as_official"] = (
        processed[patient_col].astype(str).tolist() == official[patient_col].astype(str).tolist()
    )
    if not report["processed_same_order_as_official"]:
        raise ValueError("Processed holdout covariates are not in official Holdout_Cov_file order")

    require_columns(official, [patient_col, *covariates], "official holdout covariates")
    require_columns(processed, [patient_col, *covariates], "processed holdout covariates")
    merged = official[[patient_col, *covariates]].merge(
        processed[[patient_col, *covariates]],
        on=patient_col,
        suffixes=("_official", "_processed"),
        validate="one_to_one",
    )
    mismatches: dict[str, int] = {}
    max_abs_diff: dict[str, float | None] = {}
    for covariate in covariates:
        left = merged[f"{covariate}_official"]
        right = merged[f"{covariate}_processed"]
        if pd.api.types.is_numeric_dtype(left) or pd.to_numeric(left, errors="coerce").notna().mean() >= 0.95:
            a = pd.to_numeric(left, errors="coerce")
            b = pd.to_numeric(right, errors="coerce")
            diff = (a - b).abs()
            mismatches[covariate] = int((~np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)).sum())
            max_abs_diff[covariate] = float(diff.max()) if len(diff) else None
        else:
            mismatch = left.astype(str).to_numpy() != right.astype(str).to_numpy()
            mismatches[covariate] = int(np.sum(mismatch))
            max_abs_diff[covariate] = None
    report["covariate_mismatch_counts"] = mismatches
    report["covariate_max_abs_diff"] = max_abs_diff
    if any(value != 0 for value in mismatches.values()):
        raise ValueError(f"Official and processed holdout covariates differ: {mismatches}")
    return report


def _eligible_mask(frame: pd.DataFrame, disease: str, eligibility_rules: dict[str, dict[str, Any]]) -> pd.Series:
    eligible = filter_eligible(frame, disease, eligibility_rules)
    return frame.index.isin(eligible.index)


def _disease_data(
    *,
    disease: str,
    train_phenotype: pd.DataFrame,
    holdout_covariates: pd.DataFrame,
    splits: pd.DataFrame,
    train_prs: pd.DataFrame,
    holdout_prs: pd.DataFrame,
    eligibility_rules: dict[str, dict[str, Any]],
    patient_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    disease_splits = splits.loc[splits["disease"].astype(str) == disease, [patient_col, "fold"]].copy()
    if disease_splits.empty:
        raise ValueError(f"No split rows found for disease {disease}")
    if disease_splits.duplicated(patient_col).any():
        examples = disease_splits.loc[disease_splits.duplicated(patient_col), patient_col].head(5)
        raise ValueError(f"Duplicate split rows for {disease}: {examples.tolist()}")

    prs_cols = [col for col in train_prs.columns if col != patient_col]
    if not prs_cols:
        raise ValueError(f"No selected PRS columns found for {disease}")
    missing_holdout_cols = [col for col in prs_cols if col not in holdout_prs.columns]
    if missing_holdout_cols:
        raise ValueError(f"Holdout PRS scores are missing columns for {disease}: {missing_holdout_cols}")

    train = train_phenotype.merge(disease_splits, on=patient_col, how="inner", validate="one_to_one")
    train = train.loc[_eligible_mask(train, disease, eligibility_rules)].copy().reset_index(drop=True)
    train = train.merge(train_prs, on=patient_col, how="left", validate="one_to_one")
    holdout = holdout_covariates.copy().reset_index(drop=True)
    holdout = holdout.loc[_eligible_mask(holdout, disease, eligibility_rules)].copy().reset_index(drop=True)
    holdout = holdout.merge(holdout_prs[[patient_col, *prs_cols]], on=patient_col, how="left", validate="one_to_one")

    for context, frame in [("train-val", train), ("holdout", holdout)]:
        missing = frame.loc[frame[prs_cols].isna().any(axis=1), patient_col]
        if not missing.empty:
            raise ValueError(
                f"{context} selected PRS scores are missing for {len(missing)} eligible {disease} patients. "
                f"Examples: {missing.head(5).tolist()}"
            )

    folds = sorted(train["fold"].dropna().astype(int).unique().tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected folds [0,1,2,3,4] for {disease}; got {folds}")

    audit = {
        "disease": disease,
        "n_trainval_eligible": int(len(train)),
        "n_holdout_eligible": int(len(holdout)),
        "fold_counts": {str(k): int(v) for k, v in train["fold"].astype(int).value_counts().sort_index().items()},
        "fold_case_counts": {
            str(fold): int(train.loc[train["fold"].astype(int) == fold, disease].astype(int).sum())
            for fold in folds
        },
        "n_prs_features": int(len(prs_cols)),
        "prs_features": prs_cols,
    }
    return train, holdout, prs_cols, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final TabPFN pipeline in one pass: shared 5-fold OOF train-val scores, "
            "full train-val context holdout scores, and train-val-standardized final outputs."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--covariates", default=None)
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--train-phenotype", default=DEFAULT_TRAIN_PHENOTYPE)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--holdout-covariates", default=DEFAULT_HOLDOUT_COVARIATES)
    parser.add_argument("--processed-holdout-covariates", default="data/processed/final/holdout_test_covariates.csv")
    parser.add_argument("--eligibility-config", default=DEFAULT_ELIGIBILITY_CONFIG)
    parser.add_argument("--train-prs-long-scores", required=True)
    parser.add_argument("--holdout-prs-long-scores", required=True)
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--score-col", default="prs")
    parser.add_argument("--clip-eps", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tabpfn-model-version", default="v3")
    parser.add_argument("--tabpfn-kwargs", default="{}")
    parser.add_argument("--tabpfn-model-cache-dir", default=None)
    parser.add_argument("--allow-browser", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--team-name", default="TeamName")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.clip_eps <= 0 or args.clip_eps >= 0.5:
        raise ValueError("--clip-eps must be between 0 and 0.5")
    if args.tabpfn_model_cache_dir:
        os.environ["TABPFN_MODEL_CACHE_DIR"] = args.tabpfn_model_cache_dir
    if not args.allow_browser:
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")

    classifier_kwargs = json.loads(args.tabpfn_kwargs)
    if not isinstance(classifier_kwargs, dict):
        raise ValueError("--tabpfn-kwargs must decode to a JSON object")

    config = load_config(args.config, args.config_override)
    columns = _require_mapping(config, "columns")
    labels = _labels(args, columns)
    covariates = parse_csv_list(args.covariates or _csv_from_config(columns.get("covariates"), "columns.covariates"))
    patient_col = args.patient_col or _required_value(columns, "patient_id", "columns")

    train_phenotype = read_table(args.train_phenotype)
    splits = read_table(args.splits)
    holdout_covariates = _read_table_auto(args.holdout_covariates)
    processed_holdout = (
        _read_table_auto(args.processed_holdout_covariates)
        if args.processed_holdout_covariates and Path(args.processed_holdout_covariates).exists()
        else None
    )
    require_columns(train_phenotype, [patient_col, *labels, *covariates], "train phenotype")
    require_columns(splits, [patient_col, "disease", "fold"], "shared CV splits")
    require_columns(holdout_covariates, [patient_col, *covariates], "holdout covariates")
    validate_unique(train_phenotype, patient_col, "train phenotype")
    validate_unique(holdout_covariates, patient_col, "holdout covariates")

    if set(train_phenotype[patient_col].astype(str)) & set(holdout_covariates[patient_col].astype(str)):
        raise ValueError("Train-val and holdout sample IDs overlap")

    holdout_audit = _check_holdout_covariates(
        official=holdout_covariates,
        processed=processed_holdout,
        patient_col=patient_col,
        covariates=covariates,
    )

    eligibility_rules = load_eligibility_rules(args.eligibility_config)
    classifier_cls = _load_tabpfn_classifier()
    model_version, model_version_label = _normalise_tabpfn_model_version(args.tabpfn_model_version)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_cache: dict[str, pd.DataFrame] = {}
    disease_audits: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    feature_summaries: list[pd.DataFrame] = []
    trainval_long_parts: list[pd.DataFrame] = []
    holdout_long_parts: list[pd.DataFrame] = []
    combined_long_parts: list[pd.DataFrame] = []
    submission_records: list[dict[str, Any]] = []
    submission_files: list[Path] = []

    for disease in labels:
        train_long = long_cache.setdefault(
            _format_template(args.train_prs_long_scores, disease),
            read_table(_format_template(args.train_prs_long_scores, disease)),
        )
        holdout_long = long_cache.setdefault(
            _format_template(args.holdout_prs_long_scores, disease),
            read_table(_format_template(args.holdout_prs_long_scores, disease)),
        )
        train_prs = pivot_long_prs_scores(
            train_long,
            disease,
            patient_col=patient_col,
            disease_col=args.disease_col,
            model_col=args.model_col,
            score_col=args.score_col,
            context="final train-val long PRS scores",
        )
        holdout_prs = pivot_long_prs_scores(
            holdout_long,
            disease,
            patient_col=patient_col,
            disease_col=args.disease_col,
            model_col=args.model_col,
            score_col=args.score_col,
            context="final holdout long PRS scores",
        )
        train_data, holdout_data, prs_cols, data_audit = _disease_data(
            disease=disease,
            train_phenotype=train_phenotype,
            holdout_covariates=holdout_covariates,
            splits=splits,
            train_prs=train_prs,
            holdout_prs=holdout_prs,
            eligibility_rules=eligibility_rules,
            patient_col=patient_col,
        )

        oof_probs = np.full(len(train_data), np.nan, dtype=float)
        for fold in [0, 1, 2, 3, 4]:
            context = train_data.loc[train_data["fold"].astype(int) != fold].reset_index(drop=True)
            validation = train_data.loc[train_data["fold"].astype(int) == fold].reset_index(drop=True)
            y_context = context[disease].astype(int)
            if y_context.nunique() < 2:
                raise ValueError(f"Fold {fold} context for {disease} has fewer than two classes")

            fold_probs, fold_features = _fit_predict(
                classifier_cls=classifier_cls,
                context=context,
                y_context=y_context,
                eval_frame=validation,
                prs_cols=prs_cols,
                covariates=covariates,
                include_covariates=True,
                device=args.device,
                classifier_kwargs=classifier_kwargs,
                model_version=model_version,
            )
            validation_indices = train_data.index[train_data["fold"].astype(int) == fold].to_numpy()
            oof_probs[validation_indices] = fold_probs
            fold_summaries.append(
                {
                    "disease": disease,
                    "fold": int(fold),
                    "n_context": int(len(context)),
                    "context_cases": int(y_context.sum()),
                    "n_validation": int(len(validation)),
                    "validation_cases": int(validation[disease].astype(int).sum()),
                    "probability_min": float(np.nanmin(fold_probs)),
                    "probability_max": float(np.nanmax(fold_probs)),
                }
            )
            feature_summaries.append(
                fold_features.assign(
                    disease=disease,
                    fold=int(fold),
                    feature_fit_scope="trainval_5cv_context",
                )
            )

        if np.isnan(oof_probs).any():
            missing = train_data.loc[np.isnan(oof_probs), patient_col].head(5).tolist()
            raise ValueError(f"OOF prediction missing for {disease}; examples: {missing}")

        holdout_probs, holdout_features = _fit_predict(
            classifier_cls=classifier_cls,
            context=train_data,
            y_context=train_data[disease].astype(int),
            eval_frame=holdout_data,
            prs_cols=prs_cols,
            covariates=covariates,
            include_covariates=True,
            device=args.device,
            classifier_kwargs=classifier_kwargs,
            model_version=model_version,
        )
        feature_summaries.append(
            holdout_features.assign(
                disease=disease,
                fold=-1,
                feature_fit_scope="holdout_full_trainval_context",
            )
        )

        train_logits = logit_from_probability(oof_probs, eps=args.clip_eps)
        holdout_logits = logit_from_probability(holdout_probs, eps=args.clip_eps)
        logit_mean = float(np.mean(train_logits))
        logit_sd = float(np.std(train_logits, ddof=0))
        if not np.isfinite(logit_mean):
            logit_mean = 0.0
        if not np.isfinite(logit_sd) or logit_sd <= 0:
            logit_sd = 1.0

        train_scores = (train_logits - logit_mean) / logit_sd
        holdout_scores = (holdout_logits - logit_mean) / logit_sd
        if not np.isclose(float(np.mean(train_scores)), 0.0, atol=1e-12):
            raise ValueError(f"Train-val standardized mean is not zero for {disease}")
        if not np.isclose(float(np.std(train_scores, ddof=0)), 1.0, atol=1e-12):
            raise ValueError(f"Train-val standardized sd is not one for {disease}")

        train_score_frame = pd.DataFrame(
            {
                patient_col: train_data[patient_col].to_numpy(),
                "disease": disease,
                "cohort": "train-val",
                "fold": train_data["fold"].astype(int).to_numpy(),
                "label": train_data[disease].astype(int).to_numpy(),
                "score": train_scores,
                "direct_logit": train_logits,
                "direct_probability": oof_probs,
            }
        )
        holdout_score_frame = pd.DataFrame(
            {
                patient_col: holdout_data[patient_col].to_numpy(),
                "disease": disease,
                "cohort": "holdout",
                "fold": -1,
                "label": np.nan,
                "score": holdout_scores,
                "direct_logit": holdout_logits,
                "direct_probability": holdout_probs,
            }
        )
        combined_score_frame = pd.concat([train_score_frame, holdout_score_frame], ignore_index=True)
        trainval_long_parts.append(train_score_frame)
        holdout_long_parts.append(holdout_score_frame)
        combined_long_parts.append(combined_score_frame)

        score_col = _score_col(disease)
        submission = combined_score_frame[[patient_col, "score"]].rename(columns={"score": score_col})
        if submission[patient_col].duplicated().any():
            raise ValueError(f"Combined submission file for {disease} has duplicate sample IDs")
        if submission[score_col].isna().any():
            raise ValueError(f"Combined submission file for {disease} has missing scores")
        submission_file = out_dir / f"{args.team_name}_{disease}.txt"
        submission.to_csv(submission_file, sep="\t", index=False, lineterminator="\n")
        submission_files.append(submission_file)

        train_file = out_dir / f"{args.team_name}_{disease}_trainval_5cv.txt"
        holdout_file = out_dir / f"{args.team_name}_{disease}_holdout.txt"
        train_score_frame[[patient_col, "score"]].rename(columns={"score": score_col}).to_csv(
            train_file,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        holdout_score_frame[[patient_col, "score"]].rename(columns={"score": score_col}).to_csv(
            holdout_file,
            sep="\t",
            index=False,
            lineterminator="\n",
        )

        data_audit.update(
            {
                "n_combined": int(len(combined_score_frame)),
                "trainval_standardized_mean_ddof0": float(np.mean(train_scores)),
                "trainval_standardized_sd_ddof0": float(np.std(train_scores, ddof=0)),
                "trainval_standardized_sd_ddof1": float(np.std(train_scores, ddof=1)),
                "holdout_standardized_mean": float(np.mean(holdout_scores)),
                "holdout_standardized_sd_ddof0": float(np.std(holdout_scores, ddof=0)),
                "holdout_standardized_min": float(np.min(holdout_scores)),
                "holdout_standardized_max": float(np.max(holdout_scores)),
                "logit_trainval_oof_mean": logit_mean,
                "logit_trainval_oof_sd": logit_sd,
            }
        )
        disease_audits.append(data_audit)
        fit_summaries.append(
            {
                "disease": disease,
                "strategy": "covariate_adjusted",
                "train_prediction_mode": "shared_5fold_oof",
                "holdout_prediction_mode": "full_trainval_context",
                "score_scale": "trainval_5cv_oof_logit_mean_sd",
                "clip_eps": float(args.clip_eps),
                "tabpfn_model_version": model_version_label,
                "tabpfn_model_path": str(classifier_kwargs.get("model_path", "auto")),
                "n_context": int(len(train_data)),
                "context_cases": int(train_data[disease].astype(int).sum()),
                "n_holdout_scored": int(len(holdout_data)),
                "n_prs_features": int(len(prs_cols)),
                "n_covariates_requested": int(len(covariates)),
                "logit_train_mean": logit_mean,
                "logit_train_sd": logit_sd,
            }
        )
        submission_records.append(
            {
                "team_name": args.team_name,
                "phenotype": disease,
                "submission_file": str(submission_file),
                "trainval_5cv_file": str(train_file),
                "holdout_file": str(holdout_file),
                "columns": f"{patient_col}; {score_col}",
                "n_combined_rows": int(len(submission)),
                "n_trainval_rows": int(len(train_score_frame)),
                "n_holdout_rows": int(len(holdout_score_frame)),
                "score_type": "standardized TabPFN-3 direct logit; train-val is shared 5-CV OOF",
                "score_min": float(submission[score_col].min()),
                "score_max": float(submission[score_col].max()),
                "score_mean": float(submission[score_col].mean()),
            }
        )

    combined_long = pd.concat(combined_long_parts, ignore_index=True)
    trainval_long = pd.concat(trainval_long_parts, ignore_index=True)
    holdout_long = pd.concat(holdout_long_parts, ignore_index=True)
    write_table(combined_long, out_dir / "combined_trainval_5cv_and_holdout_scores_long.csv")
    write_table(trainval_long, out_dir / "trainval_5cv_scores_long.csv")
    write_table(holdout_long, out_dir / "holdout_scores_long.csv")
    write_table(pd.DataFrame(disease_audits), out_dir / "audit_by_disease.csv")
    write_table(pd.DataFrame(fold_summaries), out_dir / "audit_by_fold.csv")
    write_table(pd.DataFrame(fit_summaries), out_dir / "tabpfn_fit_summary.csv")
    write_table(pd.DataFrame(submission_records), out_dir / "submission_manifest.csv")
    if feature_summaries:
        write_table(pd.concat(feature_summaries, ignore_index=True), out_dir / "tabpfn_feature_summary.csv")

    team_slug = args.team_name.replace("!", "").replace(" ", "_")
    zip_path = out_dir / f"{team_slug}_final_phase_submission_with_trainval_5cv_standardized_logits.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for submission_file in submission_files:
            zf.write(submission_file, arcname=submission_file.name)

    readme_lines = [
        "# Final Phase Submission - Audited TabPFN 5-CV Pipeline",
        "",
        f"Team name: {args.team_name}",
        "",
        "This directory was generated by `scripts/run_final_tabpfn_oof_holdout_pipeline.py`.",
        "",
        "Pipeline definition:",
        "- train-val scores: shared project 5-fold out-of-fold TabPFN-3 predictions.",
        "- holdout scores: TabPFN-3 predictions using all train-val eligible patients as context.",
        "- features: selected final PRS inputs plus covariates (`covariate_adjusted`).",
        "- standardization: mean/sd of the entire train-val 5-CV OOF logits for each disease.",
        "- holdout data are not used to estimate standardization parameters.",
        "",
        "| Disease | Combined rows | Train-val rows | Holdout rows | Train mean | Train SD ddof0 | Holdout mean | Holdout SD ddof0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in disease_audits:
        readme_lines.append(
            "| {disease} | {n_combined} | {n_trainval_eligible} | {n_holdout_eligible} | "
            "{trainval_standardized_mean_ddof0:.6g} | {trainval_standardized_sd_ddof0:.6g} | "
            "{holdout_standardized_mean:.6g} | {holdout_standardized_sd_ddof0:.6g} |".format(**audit)
        )
    (out_dir / "README_final_submission_with_trainval_5cv.md").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )

    run_record = {
        "team_name": args.team_name,
        "zip_file": str(zip_path),
        "script": "scripts/run_final_tabpfn_oof_holdout_pipeline.py",
        "train_phenotype": args.train_phenotype,
        "splits": args.splits,
        "holdout_covariates": args.holdout_covariates,
        "train_prs_long_scores": args.train_prs_long_scores,
        "holdout_prs_long_scores": args.holdout_prs_long_scores,
        "labels": labels,
        "covariates": covariates,
        "holdout_covariate_audit": holdout_audit,
        "disease_audits": disease_audits,
        "fit_summaries": fit_summaries,
    }
    (out_dir / "run_record.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(pd.DataFrame(submission_records).to_string(index=False))
    print("\nAudit by disease:")
    print(
        pd.DataFrame(disease_audits)[
            [
                "disease",
                "n_trainval_eligible",
                "n_holdout_eligible",
                "n_combined",
                "trainval_standardized_mean_ddof0",
                "trainval_standardized_sd_ddof0",
                "holdout_standardized_mean",
                "holdout_standardized_sd_ddof0",
                "holdout_standardized_min",
                "holdout_standardized_max",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote final zip to {zip_path}")


if __name__ == "__main__":
    main()
