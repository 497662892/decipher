#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.eligibility import eligible_counts, filter_eligible, load_eligibility_rules
from decipehr.io import read_table, write_table
from decipehr.metrics import (
    binary_metrics,
    nagelkerke_r2,
    odds_ratio_per_sd,
    risk_stratification_or,
    top_decile_odds_ratio,
)
from decipehr.schema import parse_csv_list, require_columns
from decipehr.validation import (
    iter_bootstrap_indices,
    prepare_official_validation_phenotype,
    summarize_repeated_metrics,
)

DEFAULT_LABELLED_PHENOTYPE = (
    "data/raw/holdout_phase/phenotype_files/"
    "Training_Phase_Cov_file_with_Phenotype.txt"
)
DEFAULT_VALIDATION_COVARIATES = (
    "data/raw/training_phase/phenotype_files/"
    "Testing_Cov_file.txt"
)


def _sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _load_validation_phenotype(
    phenotype_path: str | None,
    labelled_phenotype_path: str,
    validation_covariates_path: str,
    patient_col: str,
    labels: list[str],
    covariates: list[str],
) -> pd.DataFrame:
    if phenotype_path:
        phenotype = read_table(phenotype_path)
        require_columns(phenotype, [patient_col, *labels, *covariates], "validation phenotype")
        return phenotype
    labelled = read_table(labelled_phenotype_path)
    validation_covariates = read_table(validation_covariates_path)
    return prepare_official_validation_phenotype(
        labelled_phenotype=labelled,
        validation_covariates=validation_covariates,
        patient_col=patient_col,
        label_cols=labels,
        covariate_cols=covariates,
    )


def _fit_summary_lookup(
    fit_summary_path: str | None,
    strategy: str | None,
) -> dict[str, tuple[float, float, str]]:
    if not fit_summary_path:
        return {}
    summary = read_table(fit_summary_path)
    require_columns(
        summary,
        ["disease", "logit_train_mean", "logit_train_sd", "score_scale"],
        "TabPFN fit summary",
    )
    if strategy and "strategy" in summary.columns:
        summary = summary.loc[summary["strategy"].astype(str) == strategy].copy()
    if summary.empty:
        raise ValueError(f"No TabPFN fit summary rows remain after filtering strategy={strategy!r}")

    lookup: dict[str, tuple[float, float, str]] = {}
    for disease, frame in summary.groupby("disease", dropna=False):
        if len(frame) != 1:
            raise ValueError(
                f"Expected one fit summary row for disease={disease!r}; got {len(frame)}. "
                "Pass --strategy to disambiguate multi-strategy fit summaries."
            )
        row = frame.iloc[0]
        mean = float(row["logit_train_mean"])
        sd = float(row["logit_train_sd"])
        score_scale = str(row["score_scale"])
        if not math.isfinite(mean):
            mean = 0.0
        if not math.isfinite(sd) or sd <= 0:
            sd = 1.0
        lookup[str(disease)] = (mean, sd, score_scale)
    return lookup


def _scores_to_probability(
    score: pd.Series,
    disease: str,
    transform: str,
    fit_lookup: dict[str, tuple[float, float, str]],
) -> tuple[np.ndarray, str]:
    raw = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    resolved = transform
    if transform == "auto":
        resolved = "tabpfn_logit" if disease in fit_lookup else "sigmoid"

    if resolved == "identity":
        if np.nanmin(raw) < 0 or np.nanmax(raw) > 1:
            raise ValueError(
                f"Direct score for {disease} is outside [0, 1], so --score-transform identity "
                "cannot be used for probability metrics."
            )
        prob = raw
    elif resolved == "sigmoid":
        prob = _sigmoid(raw)
    elif resolved == "tabpfn_logit":
        if disease not in fit_lookup:
            raise ValueError(
                f"Missing TabPFN fit summary row for {disease}; pass --fit-summary or use "
                "--score-transform sigmoid/identity."
            )
        mean, sd, score_scale = fit_lookup[disease]
        logits = raw * sd + mean if score_scale == "train_sd" else raw
        prob = _sigmoid(logits)
    else:
        raise ValueError(f"Unsupported score transform: {transform}")

    return np.clip(prob, 0.0, 1.0), resolved


def _metrics_for_direct_predictions(pred: pd.DataFrame) -> dict[str, float | int]:
    base = binary_metrics(pred["y_true"], pred["y_prob"])
    top = top_decile_odds_ratio(pred["y_true"], pred["score"])
    strat = risk_stratification_or(pred["y_true"], pred["score"])
    return {
        "auc": base.auc,
        "auprc": base.auprc,
        "brier": base.brier,
        "prevalence": base.prevalence,
        "n": base.n,
        "or_per_sd": odds_ratio_per_sd(pred["y_true"], pred["score"]),
        **top,
        "nagelkerke_r2": nagelkerke_r2(pred["y_true"], pred["y_prob"]),
        **strat,
    }


def _bootstrap_metrics(
    pred: pd.DataFrame,
    patient_col: str,
    n_bootstraps: int,
    seed: int,
    sample_size: int | None,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    metric_rows = []
    sample_rows = []
    for repeat, indices in iter_bootstrap_indices(
        len(pred),
        n_resamples=n_bootstraps,
        seed=seed,
        sample_size=sample_size,
    ):
        boot = pred.iloc[indices]
        metric_row = {"repeat": int(repeat)}
        metric_row.update(_metrics_for_direct_predictions(boot))
        metric_rows.append(metric_row)
        sample_rows.append(
            {
                "repeat": int(repeat),
                "bootstrap_n": int(len(boot)),
                "bootstrap_unique_patients": int(boot[patient_col].nunique()),
                "bootstrap_cases": int(boot["y_true"].astype(int).sum()),
            }
        )
    return metric_rows, sample_rows


def _summarize_point_metrics(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    require_columns(metrics, group_cols, "point metrics")
    exclude = set(group_cols)
    metric_cols = [
        col
        for col in metrics.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(metrics[col])
    ]
    if not metric_cols:
        return metrics[group_cols].drop_duplicates().reset_index(drop=True)
    summary = metrics.groupby(group_cols, dropna=False, sort=True)[metric_cols].mean().reset_index()
    summary = summary.rename(columns={col: f"{col}_mean" for col in metric_cols})
    summary.insert(len(group_cols), "n_repeats", 0)
    return summary


def _resolve_n_bootstraps(n_bootstraps: int | None, n_repeats: int | None) -> int:
    if n_bootstraps is not None:
        return n_bootstraps
    if n_repeats is not None:
        return n_repeats
    return 1000


def _model_fields(model_name: str | None) -> dict[str, str]:
    return {"model_name": model_name} if model_name else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score official validation directly from supplied model scores. This does not "
            "fit or refit a logistic calibration model."
        )
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--phenotype", default=None)
    parser.add_argument("--labelled-phenotype", default=DEFAULT_LABELLED_PHENOTYPE)
    parser.add_argument("--validation-covariates", default=DEFAULT_VALIDATION_COVARIATES)
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--covariates", default="")
    parser.add_argument("--prs-scores", required=True, help="Wide validation score table.")
    parser.add_argument("--prs-prefix", default="prs_")
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument(
        "--score-transform",
        choices=["auto", "tabpfn_logit", "sigmoid", "identity"],
        default="auto",
        help=(
            "How to convert direct scores to probabilities for Brier/Nagelkerke. "
            "auto uses tabpfn_logit when --fit-summary has a disease row, otherwise sigmoid."
        ),
    )
    parser.add_argument(
        "--fit-summary",
        default=None,
        help="Optional TabPFN fit summary with logit_train_mean/logit_train_sd for inverse scaling.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Optional strategy value used to filter multi-strategy TabPFN fit summaries.",
    )
    parser.add_argument("--n-bootstraps", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--bootstrap-size", type=int, default=None)
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--validation-phenotype-out", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.point_only:
        args.n_bootstraps = 0

    patient_col = args.patient_col
    labels = parse_csv_list(args.labels)
    covariates = parse_csv_list(args.covariates)
    n_bootstraps = _resolve_n_bootstraps(args.n_bootstraps, args.n_repeats)

    validation_phenotype = _load_validation_phenotype(
        phenotype_path=args.phenotype,
        labelled_phenotype_path=args.labelled_phenotype,
        validation_covariates_path=args.validation_covariates,
        patient_col=patient_col,
        labels=labels,
        covariates=covariates,
    )
    scores = read_table(args.prs_scores)
    require_columns(scores, [patient_col], "validation scores")
    score_cols = [col for col in scores.columns if col != patient_col]
    validation_phenotype = validation_phenotype.drop(
        columns=[col for col in score_cols if col in validation_phenotype.columns]
    )
    validation_phenotype = validation_phenotype.merge(scores, on=patient_col, how="left", validate="one_to_one")

    eligibility_rules = load_eligibility_rules(args.eligibility_config)
    fit_lookup = _fit_summary_lookup(args.fit_summary, args.strategy)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.validation_phenotype_out:
        write_table(validation_phenotype, args.validation_phenotype_out)

    prediction_frames = []
    point_rows = []
    bootstrap_metric_rows = []
    bootstrap_sample_rows = []
    fit_rows = []
    model_fields = _model_fields(args.model_name)

    for disease_idx, disease in enumerate(labels):
        score_col = f"{args.prs_prefix}{disease}"
        require_columns(validation_phenotype, [score_col], "validation scores")
        validation_data = filter_eligible(validation_phenotype, disease, eligibility_rules).reset_index(
            drop=True
        )
        if validation_data.empty:
            raise ValueError(f"No eligible validation patients remain for disease {disease}")
        missing = validation_data.loc[validation_data[score_col].isna(), patient_col]
        if not missing.empty:
            raise ValueError(
                f"Direct validation scores are missing for {len(missing)} eligible {disease} patients. "
                f"Examples: {missing.head(5).tolist()}"
            )

        y_prob, resolved_transform = _scores_to_probability(
            validation_data[score_col],
            disease=disease,
            transform=args.score_transform,
            fit_lookup=fit_lookup,
        )
        method = f"direct_score_{disease}"
        pred = pd.DataFrame(
            {
                patient_col: validation_data[patient_col].to_numpy(),
                "fold": 0,
                "disease": disease,
                "method": method,
                "y_true": validation_data[disease].astype(int).to_numpy(),
                "score": pd.to_numeric(validation_data[score_col], errors="coerce").to_numpy(dtype=float),
                "y_prob": y_prob,
                "score_col": score_col,
                "score_transform": resolved_transform,
            }
        )

        row = {
            **model_fields,
            "disease": disease,
            "method": method,
            "score_col": score_col,
            "score_transform": resolved_transform,
        }
        row.update(_metrics_for_direct_predictions(pred))
        point_rows.append(row)
        fit_rows.append(
            {
                **model_fields,
                "disease": disease,
                "method": method,
                "score_col": score_col,
                "score_transform": resolved_transform,
                "n_fit": 0,
                "n_prediction": int(len(validation_data)),
                "fit_cases": 0,
                "prediction_cases": int(validation_data[disease].astype(int).sum()),
            }
        )
        rows, sample_rows = _bootstrap_metrics(
            pred=pred,
            patient_col=patient_col,
            n_bootstraps=n_bootstraps,
            seed=args.seed + disease_idx,
            sample_size=args.bootstrap_size,
        )
        for metric_row in rows:
            metric_row.update(
                {
                    **model_fields,
                    "disease": disease,
                    "method": method,
                    "score_col": score_col,
                    "score_transform": resolved_transform,
                }
            )
            bootstrap_metric_rows.append(metric_row)
        for sample_row in sample_rows:
            sample_row.update(
                {
                    **model_fields,
                    "disease": disease,
                    "method": method,
                    "score_col": score_col,
                    "score_transform": resolved_transform,
                }
            )
            bootstrap_sample_rows.append(sample_row)
        if args.write_predictions:
            prediction_frames.append(pred)

    point_metrics = pd.DataFrame(point_rows)
    bootstrap_metrics = pd.DataFrame(bootstrap_metric_rows)
    bootstrap_samples = pd.DataFrame(bootstrap_sample_rows)
    fit_summary = pd.DataFrame(fit_rows)

    write_table(point_metrics, out_dir / "point_metrics.csv")
    write_table(bootstrap_metrics, out_dir / "bootstrap_metrics.csv")
    write_table(bootstrap_metrics, out_dir / "repeat_metrics.csv")
    write_table(bootstrap_samples, out_dir / "bootstrap_sample_summary.csv")
    write_table(fit_summary, out_dir / "fit_prediction_summary.csv")
    write_table(fit_summary, out_dir / "repeat_split_summary.csv")
    validation_counts = eligible_counts(validation_phenotype, labels, eligibility_rules)
    write_table(validation_counts, out_dir / "validation_eligibility_counts.csv")
    write_table(validation_counts, out_dir / "eligibility_counts.csv")

    summary_group_cols = ["model_name", "disease", "method", "score_col", "score_transform"]
    summary_source = bootstrap_metrics if n_bootstraps > 0 else point_metrics
    present_group_cols = [col for col in summary_group_cols if col in summary_source.columns]
    if n_bootstraps > 0:
        summary = summarize_repeated_metrics(bootstrap_metrics, present_group_cols)
    else:
        summary = _summarize_point_metrics(point_metrics, present_group_cols)
    write_table(summary, out_dir / "summary_metrics.csv")
    if args.write_predictions:
        write_table(pd.concat(prediction_frames, ignore_index=True), out_dir / "predictions.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_mode = "point-only" if n_bootstraps == 0 else "bootstrap"
    print(f"Wrote direct-score official {validation_mode} validation outputs to {out_dir}")


if __name__ == "__main__":
    main()
