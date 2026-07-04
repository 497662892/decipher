from __future__ import annotations

import argparse
import json
import os
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
    _oof_probabilities,
    _require_mapping,
    _required_value,
    _strategies,
)

from decipehr.config import load_config
from decipehr.eligibility import eligible_counts, filter_eligible, load_eligibility_rules
from decipehr.ensemble import pivot_long_prs_scores
from decipehr.io import read_table, write_table
from decipehr.metrics import (
    binary_metrics,
    challenge_metrics,
    odds_ratio_per_sd,
    risk_stratification_or,
)
from decipehr.models import fit_predict_logistic
from decipehr.schema import parse_csv_list, require_columns, validate_unique
from decipehr.tabpfn_ensemble import TabPFNStrategy, logit_from_probability, scale_validation_logits

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_ELIGIBILITY_CONFIG = "configs/disease_eligibility.yaml"


def _class_weight(value: str) -> str | None:
    if value == "balanced":
        return "balanced"
    if value == "none":
        return None
    raise ValueError(f"Unsupported class weight: {value}")


def _calibration_methods(value: str) -> list[str]:
    methods = parse_csv_list(value)
    allowed = {"prs_only", "prs_covariate"}
    unknown = [method for method in methods if method not in allowed]
    if unknown:
        raise ValueError(f"Unsupported calibration methods: {unknown}; expected one of {sorted(allowed)}")
    if not methods:
        raise ValueError("At least one calibration method is required")
    return methods


def _metrics_for_predictions(pred: pd.DataFrame, score_col: str = "y_prob") -> dict[str, float | int]:
    base = binary_metrics(pred["y_true"], pred[score_col])
    ors = risk_stratification_or(pred["y_true"], pred[score_col])
    challenge = challenge_metrics(pred["y_true"], pred[score_col])
    return {
        "auc": base.auc,
        "auprc": base.auprc,
        "brier": base.brier,
        "prevalence": base.prevalence,
        "n": base.n,
        "or_per_sd": odds_ratio_per_sd(pred["y_true"], pred[score_col]),
        **challenge,
        **ors,
    }


def _read_long_scores(path_template: str, disease: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    path = path_template.format(disease=disease)
    if path not in cache:
        cache[path] = read_table(path)
    return cache[path]


def _build_outer_fold_scores(
    *,
    classifier_cls,
    disease: str,
    strategy: TabPFNStrategy,
    context: pd.DataFrame,
    validation: pd.DataFrame,
    prs_cols: list[str],
    covariates: list[str],
    patient_col: str,
    inner_folds: int,
    seed: int,
    fold: int,
    device: str | None,
    classifier_kwargs: dict[str, Any],
    model_version,
    model_version_label: str,
    train_prediction_mode: str,
    clip_eps: float,
    score_scale: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    include_covariates = strategy == "covariate_adjusted"
    if context.empty or validation.empty:
        raise ValueError(f"Empty outer CV split for {disease}, fold={fold}")
    y_context = context[disease].astype(int)
    if y_context.nunique() < 2:
        raise ValueError(f"TabPFN context labels for {disease}, fold={fold} have fewer than two classes")

    if train_prediction_mode == "oof":
        train_probs, train_features, actual_inner_folds = _oof_probabilities(
            classifier_cls=classifier_cls,
            context=context,
            y_context=y_context,
            prs_cols=prs_cols,
            covariates=covariates,
            include_covariates=include_covariates,
            n_folds=inner_folds,
            seed=seed + int(fold),
            device=device,
            classifier_kwargs=classifier_kwargs,
            model_version=model_version,
        )
    elif train_prediction_mode == "in_context":
        train_probs, feature_summary = _fit_predict(
            classifier_cls=classifier_cls,
            context=context,
            y_context=y_context,
            eval_frame=context,
            prs_cols=prs_cols,
            covariates=covariates,
            include_covariates=include_covariates,
            device=device,
            classifier_kwargs=classifier_kwargs,
            model_version=model_version,
        )
        train_features = feature_summary.assign(oof_fold=-1)
        actual_inner_folds = 1
    else:
        raise ValueError(f"Unsupported train prediction mode: {train_prediction_mode}")

    validation_probs, validation_features = _fit_predict(
        classifier_cls=classifier_cls,
        context=context,
        y_context=y_context,
        eval_frame=validation,
        prs_cols=prs_cols,
        covariates=covariates,
        include_covariates=include_covariates,
        device=device,
        classifier_kwargs=classifier_kwargs,
        model_version=model_version,
    )

    train_logits = logit_from_probability(train_probs, eps=clip_eps)
    validation_logits = logit_from_probability(validation_probs, eps=clip_eps)
    train_scaled, validation_scaled, logit_mean, logit_sd = scale_validation_logits(
        train_logits=train_logits,
        validation_logits=validation_logits,
        mode=score_scale,
    )

    train_scored = context.copy()
    validation_scored = validation.copy()
    train_scored["prs"] = train_scaled
    validation_scored["prs"] = validation_scaled

    feature_summary = pd.concat(
        [
            train_features.assign(feature_fit_scope="outer_train_oof"),
            validation_features.assign(oof_fold=-1, feature_fit_scope="outer_validation_full_context"),
        ],
        ignore_index=True,
    )
    feature_summary.insert(0, "outer_fold", fold)
    feature_summary.insert(0, "strategy", strategy)
    feature_summary.insert(0, "disease", disease)

    summary = {
        "disease": disease,
        "strategy": strategy,
        "outer_fold": int(fold),
        "n_context": int(len(context)),
        "context_cases": int(y_context.sum()),
        "n_validation": int(len(validation)),
        "validation_cases": int(validation[disease].astype(int).sum()),
        "n_prs_features": int(len(prs_cols)),
        "n_covariates_requested": int(len(covariates) if include_covariates else 0),
        "train_prediction_mode": train_prediction_mode,
        "n_inner_oof_folds": int(actual_inner_folds),
        "probability_min": float(np.nanmin(train_probs)),
        "probability_max": float(np.nanmax(train_probs)),
        "validation_probability_min": float(np.nanmin(validation_probs)),
        "validation_probability_max": float(np.nanmax(validation_probs)),
        "clip_eps": float(clip_eps),
        "score_scale": score_scale,
        "tabpfn_model_version": model_version_label,
        "tabpfn_model_path": str(classifier_kwargs.get("model_path", "auto")),
        "logit_train_mean": float(logit_mean),
        "logit_train_sd": float(logit_sd),
    }
    return train_scored, validation_scored, feature_summary, summary


def _feature_cols_for_calibration(method: str, covariates: list[str]) -> list[str]:
    if method == "prs_only":
        return ["prs"]
    if method == "prs_covariate":
        return [*covariates, "prs"]
    raise ValueError(f"Unsupported calibration method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a TabPFN PRS ensemble with the shared final train-val outer 5-CV split. "
            "Base PRS inputs must be OOF scores selected from each family within the same CV."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--model-name", default="tabpfn_ensemble")
    parser.add_argument("--disease", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--covariates", default=None)
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--phenotype", default="data/processed/final/train_val_phenotype.csv")
    parser.add_argument(
        "--splits",
        default="data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv",
    )
    parser.add_argument("--eligibility-config", default=DEFAULT_ELIGIBILITY_CONFIG)
    parser.add_argument("--train-prs-long-scores", required=True)
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--score-col", default="prs")
    parser.add_argument("--strategies", default="prs_only,covariate_adjusted")
    parser.add_argument("--inner-folds", type=int, default=None)
    parser.add_argument("--train-prediction-mode", choices=["oof", "in_context"], default="oof")
    parser.add_argument("--calibration-methods", default="prs_covariate")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--clip-eps", type=float, default=1e-5)
    parser.add_argument("--score-scale", choices=["train_sd", "none"], default="train_sd")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tabpfn-model-version", default="default")
    parser.add_argument("--tabpfn-kwargs", default="{}")
    parser.add_argument("--tabpfn-model-cache-dir", default=None)
    parser.add_argument("--allow-browser", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
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
    seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
    inner_folds = args.inner_folds if args.inner_folds is not None else int(config.get("project", {}).get("n_folds", 5))
    strategies = _strategies(args.strategies)
    calibration_methods = _calibration_methods(args.calibration_methods)
    class_weight = _class_weight(args.class_weight)

    phenotype = read_table(args.phenotype)
    splits = read_table(args.splits)
    require_columns(phenotype, [patient_col, *labels, *covariates], "phenotype")
    require_columns(splits, [patient_col, "disease", "fold"], "CV splits")
    validate_unique(phenotype, patient_col, "phenotype")

    eligibility_rules = load_eligibility_rules(args.eligibility_config)
    classifier_cls = _load_tabpfn_classifier()
    model_version, model_version_label = _normalise_tabpfn_model_version(args.tabpfn_model_version)
    long_cache: dict[str, pd.DataFrame] = {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    tabpfn_score_rows: list[pd.DataFrame] = []
    feature_summaries: list[pd.DataFrame] = []
    fit_summaries: list[dict[str, Any]] = []

    for disease in labels:
        train_long = _read_long_scores(args.train_prs_long_scores, disease, long_cache)
        train_prs = pivot_long_prs_scores(
            train_long,
            disease,
            patient_col=patient_col,
            disease_col=args.disease_col,
            model_col=args.model_col,
            score_col=args.score_col,
            context="TabPFN CV long PRS scores",
        )
        prs_cols = [col for col in train_prs.columns if col != patient_col]
        if not prs_cols:
            raise ValueError(f"No selected PRS columns were found for {disease}")

        disease_splits = splits.loc[splits["disease"].astype(str) == disease, [patient_col, "fold"]].copy()
        if disease_splits.empty:
            raise ValueError(f"No CV split rows found for disease {disease}")
        if disease_splits.duplicated(patient_col).any():
            examples = disease_splits.loc[disease_splits.duplicated(patient_col), patient_col].head(5)
            raise ValueError(f"Duplicate split rows for {disease}: {examples.tolist()}")

        data = phenotype.merge(disease_splits, on=patient_col, how="inner", validate="one_to_one")
        data = filter_eligible(data, disease, eligibility_rules).reset_index(drop=True)
        if data.empty:
            raise ValueError(f"No eligible patients remain for disease {disease}")
        data = data.merge(train_prs, on=patient_col, how="left", validate="one_to_one")
        missing_prs = data.loc[data[prs_cols].isna().any(axis=1), patient_col]
        if not missing_prs.empty:
            raise ValueError(
                f"TabPFN CV input is missing selected PRS scores for {len(missing_prs)} eligible "
                f"{disease} patients. Examples: {missing_prs.head(5).tolist()}"
            )

        for fold in sorted(data["fold"].dropna().astype(int).unique()):
            context = data.loc[data["fold"].astype(int) != fold].copy().reset_index(drop=True)
            validation = data.loc[data["fold"].astype(int) == fold].copy().reset_index(drop=True)
            for strategy in strategies:
                train_scored, validation_scored, feature_summary, fit_summary = _build_outer_fold_scores(
                    classifier_cls=classifier_cls,
                    disease=disease,
                    strategy=strategy,
                    context=context,
                    validation=validation,
                    prs_cols=prs_cols,
                    covariates=covariates,
                    patient_col=patient_col,
                    inner_folds=inner_folds,
                    seed=seed,
                    fold=int(fold),
                    device=args.device,
                    classifier_kwargs=classifier_kwargs,
                    model_version=model_version,
                    model_version_label=model_version_label,
                    train_prediction_mode=args.train_prediction_mode,
                    clip_eps=args.clip_eps,
                    score_scale=args.score_scale,
                )
                feature_summaries.append(feature_summary)
                fit_summaries.append(fit_summary)

                score_frame = validation_scored[[patient_col, "prs"]].copy()
                score_frame["disease"] = disease
                score_frame["fold"] = int(fold)
                score_frame["strategy"] = strategy
                tabpfn_score_rows.append(score_frame[[patient_col, "disease", "fold", "strategy", "prs"]])

                for calibration_method in calibration_methods:
                    feature_cols = _feature_cols_for_calibration(calibration_method, covariates)
                    method = f"{args.model_name}_{strategy}_{calibration_method}"
                    pred = fit_predict_logistic(
                        train=train_scored,
                        val=validation_scored,
                        label_col=disease,
                        feature_cols=feature_cols,
                        patient_col=patient_col,
                        method=method,
                        fold=int(fold),
                        class_weight=class_weight,
                    ).predictions
                    pred["strategy"] = strategy
                    pred["calibration_method"] = calibration_method
                    prediction_frames.append(pred)

                    row: dict[str, Any] = {
                        "disease": disease,
                        "method": method,
                        "strategy": strategy,
                        "calibration_method": calibration_method,
                        "fold": int(fold),
                    }
                    row.update(_metrics_for_predictions(pred))
                    fold_metric_rows.append(row)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    fold_metrics = pd.DataFrame(fold_metric_rows)
    write_table(predictions, out_dir / "oof_predictions.csv")
    write_table(fold_metrics, out_dir / "fold_metrics.csv")
    if tabpfn_score_rows:
        write_table(pd.concat(tabpfn_score_rows, ignore_index=True), out_dir / "tabpfn_oof_prs_scores.csv")
    if feature_summaries:
        write_table(pd.concat(feature_summaries, ignore_index=True), out_dir / "tabpfn_cv_feature_summary.csv")
    if fit_summaries:
        write_table(pd.DataFrame(fit_summaries), out_dir / "tabpfn_cv_fit_summary.csv")

    summary_rows = []
    if not predictions.empty:
        group_cols = ["disease", "method", "strategy", "calibration_method"]
        for keys, frame in predictions.groupby(group_cols, sort=True, dropna=False):
            row = dict(zip(group_cols, keys, strict=True))
            row.update(_metrics_for_predictions(frame))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_table(summary, out_dir / "summary_metrics.csv")
    if not summary.empty and {"auc", "nagelkerke_r2"}.issubset(summary.columns):
        best = (
            summary.sort_values(
                ["disease", "calibration_method", "auc", "nagelkerke_r2"],
                ascending=[True, True, False, False],
                kind="mergesort",
            )
            .groupby(["disease", "calibration_method"], as_index=False)
            .head(1)
        )
        write_table(best, out_dir / "best_by_disease.csv")

    write_table(eligible_counts(phenotype, labels, eligibility_rules), out_dir / "eligibility_counts.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary_metrics": summary_rows,
                "labels": labels,
                "strategies": list(strategies),
                "calibration_methods": calibration_methods,
                "train_prediction_mode": args.train_prediction_mode,
                "inner_folds": inner_folds,
                "class_weight": args.class_weight,
                "tabpfn_model_version": model_version_label,
                "tabpfn_model_path": str(classifier_kwargs.get("model_path", "auto")),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote TabPFN ensemble 5-CV evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
