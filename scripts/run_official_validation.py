from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from decipehr.eligibility import eligible_counts, filter_eligible, load_eligibility_rules
from decipehr.io import read_table, write_table
from decipehr.metrics import (
    binary_metrics,
    challenge_metrics,
    odds_ratio_per_sd,
    risk_stratification_or,
)
from decipehr.models import fit_predict_logistic
from decipehr.schema import parse_csv_list, require_columns
from decipehr.validation import (
    iter_bootstrap_indices,
    prepare_official_validation_phenotype,
    summarize_repeated_metrics,
)

DEFAULT_TRAIN_PHENOTYPE = (
    "data/raw/training_phase/phenotype_files/"
    "Training_Cov_file.txt"
)
DEFAULT_LABELLED_PHENOTYPE = (
    "data/raw/holdout_phase/phenotype_files/"
    "Training_Phase_Cov_file_with_Phenotype.txt"
)
DEFAULT_VALIDATION_COVARIATES = (
    "data/raw/training_phase/phenotype_files/"
    "Testing_Cov_file.txt"
)


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


def _load_phenotype(
    phenotype_path: str,
    patient_col: str,
    labels: list[str],
    covariates: list[str],
    context: str,
) -> pd.DataFrame:
    phenotype = read_table(phenotype_path)
    require_columns(phenotype, [patient_col, *labels, *covariates], context)
    return phenotype


def _load_validation_phenotype(
    phenotype_path: str | None,
    labelled_phenotype_path: str,
    validation_covariates_path: str,
    patient_col: str,
    labels: list[str],
    covariates: list[str],
) -> pd.DataFrame:
    if phenotype_path:
        return _load_phenotype(
            phenotype_path=phenotype_path,
            patient_col=patient_col,
            labels=labels,
            covariates=covariates,
            context="validation phenotype",
        )
    labelled = read_table(labelled_phenotype_path)
    validation_covariates = read_table(validation_covariates_path)
    return prepare_official_validation_phenotype(
        labelled_phenotype=labelled,
        validation_covariates=validation_covariates,
        patient_col=patient_col,
        label_cols=labels,
        covariate_cols=covariates,
    )


def _merge_prs_scores(
    phenotype: pd.DataFrame,
    scores_path: str | None,
    patient_col: str,
    context: str,
) -> pd.DataFrame:
    if not scores_path:
        return phenotype
    scores = read_table(scores_path)
    require_columns(scores, [patient_col], context)
    return phenotype.merge(scores, on=patient_col, how="left", validate="one_to_one")


def _method_specs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    labels: list[str],
    covariates: list[str],
    prs_prefix: str,
) -> list[tuple[str, list[str], str | None]]:
    methods: list[tuple[str, list[str], str | None]] = [("covariate_only", covariates, None)]
    for label in labels:
        prs_col = f"{prs_prefix}{label}"
        train_has_prs = prs_col in train.columns
        validation_has_prs = prs_col in validation.columns
        if train_has_prs != validation_has_prs:
            raise ValueError(
                f"PRS column {prs_col!r} must be present in both model-training and "
                "validation data. Pass both --train-prs-scores and --prs-scores."
            )
        if train_has_prs:
            methods.append((f"prs_covariate_{label}", [*covariates, prs_col], label))
    return methods


def _long_prs_group_map(
    prs_long: pd.DataFrame | None,
    disease: str,
    patient_col: str,
    param_cols: list[str],
    context: str,
) -> dict[tuple[object, ...], tuple[dict[str, object], pd.DataFrame]]:
    if prs_long is None:
        return {}
    disease_scores = prs_long.loc[prs_long["disease"] == disease].copy()
    if disease_scores.empty:
        return {}
    require_columns(disease_scores, [patient_col, "prs", *param_cols], context)
    if not param_cols:
        return {(): ({}, disease_scores[[patient_col, "prs"]].copy())}

    groups = {}
    for values, frame in disease_scores.groupby(param_cols, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        params = dict(zip(param_cols, values, strict=True))
        groups[values] = (params, frame[[patient_col, "prs"]].copy())
    return groups


def _class_weight(value: str) -> str | None:
    if value == "balanced":
        return "balanced"
    if value == "none":
        return None
    raise ValueError(f"Unsupported class weight: {value}")


def _predict_once(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    disease: str,
    method: str,
    feature_cols: list[str],
    patient_col: str,
    class_weight: str | None,
) -> pd.DataFrame:
    fit_data = train.copy()
    eval_data = validation.copy()
    features = list(feature_cols)
    if not features:
        fit_data["_intercept_only"] = 1.0
        eval_data["_intercept_only"] = 1.0
        features = ["_intercept_only"]
    return fit_predict_logistic(
        train=fit_data,
        val=eval_data,
        label_col=disease,
        feature_cols=features,
        patient_col=patient_col,
        method=method,
        fold=0,
        class_weight=class_weight,
    ).predictions.rename(columns={"fold": "model_fit"})


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
        metric_row.update(_metrics_for_predictions(boot))
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
    """Expose point estimates with the same *_mean columns used by bootstrap summaries."""
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
            "Fit logistic models once on the model-training phenotype, predict the "
            "official validation set, then estimate metrics with bootstrap resamples."
        )
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model label copied into metric outputs for cross-model comparisons.",
    )
    parser.add_argument(
        "--train-phenotype",
        default=DEFAULT_TRAIN_PHENOTYPE,
        help=(
            "Phenotype used to fit the logistic model. For validation this should be "
            "the original training phenotype; for final holdout scoring this should be "
            "the combined training+validation labelled phenotype."
        ),
    )
    parser.add_argument(
        "--phenotype",
        default=None,
        help="Optional prepared official validation phenotype. If omitted, labels are joined from official files.",
    )
    parser.add_argument("--labelled-phenotype", default=DEFAULT_LABELLED_PHENOTYPE)
    parser.add_argument("--validation-covariates", default=DEFAULT_VALIDATION_COVARIATES)
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--covariates", default="")
    parser.add_argument(
        "--train-prs-scores",
        default=None,
        help="Optional wide PRS table for model-training patients.",
    )
    parser.add_argument("--prs-scores", default=None, help="Optional wide PRS table for validation patients.")
    parser.add_argument(
        "--train-prs-long-scores",
        default=None,
        help="Optional long PRS table for model-training patients.",
    )
    parser.add_argument(
        "--prs-long-scores",
        default=None,
        help="Optional long PRS table for validation patients.",
    )
    parser.add_argument("--prs-param-cols", default="clump_r2,p_threshold")
    parser.add_argument("--prs-prefix", default="prs_")
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--n-bootstraps", type=int, default=None)
    parser.add_argument(
        "--point-only",
        action="store_true",
        help="Skip bootstrap resampling and write summary_metrics.csv from point estimates.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Deprecated alias for --n-bootstraps.",
    )
    parser.add_argument(
        "--bootstrap-size",
        type=int,
        default=None,
        help="Rows per bootstrap resample. Defaults to the full eligible validation set size.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=None,
        help="Deprecated and ignored. Bootstrap scoring uses --bootstrap-size instead.",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Deprecated and ignored. Bootstrap resamples are sampled with replacement.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Class weight for logistic regression. Official-style scoring uses unweighted logistic regression.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write one row per validation prediction.",
    )
    parser.add_argument("--validation-phenotype-out", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.point_only:
        args.n_bootstraps = 0

    patient_col = args.patient_col
    labels = parse_csv_list(args.labels)
    covariates = parse_csv_list(args.covariates)
    prs_param_cols = parse_csv_list(args.prs_param_cols)
    n_bootstraps = _resolve_n_bootstraps(args.n_bootstraps, args.n_repeats)
    class_weight = _class_weight(args.class_weight)

    if args.eval_size is not None:
        print("--eval-size is ignored for bootstrap scoring; use --bootstrap-size to override resample size.")
    if args.no_stratify:
        print("--no-stratify is ignored for bootstrap scoring.")

    train_phenotype = _load_phenotype(
        phenotype_path=args.train_phenotype,
        patient_col=patient_col,
        labels=labels,
        covariates=covariates,
        context="model-training phenotype",
    )
    validation_phenotype = _load_validation_phenotype(
        phenotype_path=args.phenotype,
        labelled_phenotype_path=args.labelled_phenotype,
        validation_covariates_path=args.validation_covariates,
        patient_col=patient_col,
        labels=labels,
        covariates=covariates,
    )
    train_phenotype = _merge_prs_scores(
        train_phenotype,
        args.train_prs_scores,
        patient_col=patient_col,
        context="training PRS scores",
    )
    validation_phenotype = _merge_prs_scores(
        validation_phenotype,
        args.prs_scores,
        patient_col=patient_col,
        context="validation PRS scores",
    )

    train_prs_long = None
    if args.train_prs_long_scores:
        train_prs_long = read_table(args.train_prs_long_scores)
        require_columns(
            train_prs_long,
            [patient_col, "disease", "prs", *prs_param_cols],
            "training long PRS scores",
        )
    validation_prs_long = None
    if args.prs_long_scores:
        if not args.train_prs_long_scores:
            raise ValueError(
                "PRS logistic fitting requires --train-prs-long-scores together with --prs-long-scores."
            )
        validation_prs_long = read_table(args.prs_long_scores)
        require_columns(
            validation_prs_long,
            [patient_col, "disease", "prs", *prs_param_cols],
            "validation long PRS scores",
        )

    eligibility_rules = load_eligibility_rules(args.eligibility_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.validation_phenotype_out:
        write_table(validation_phenotype, args.validation_phenotype_out)

    methods = _method_specs(train_phenotype, validation_phenotype, labels, covariates, args.prs_prefix)
    prediction_frames = []
    point_rows = []
    bootstrap_metric_rows = []
    bootstrap_sample_rows = []
    fit_rows = []
    model_fields = _model_fields(args.model_name)

    for disease_idx, disease in enumerate(labels):
        train_data = filter_eligible(train_phenotype, disease, eligibility_rules).reset_index(drop=True)
        validation_data = filter_eligible(validation_phenotype, disease, eligibility_rules).reset_index(
            drop=True
        )
        if train_data.empty:
            raise ValueError(f"No eligible model-training patients remain for disease {disease}")
        if validation_data.empty:
            raise ValueError(f"No eligible validation patients remain for disease {disease}")

        for method, feature_cols, applies_to in methods:
            if applies_to is not None and applies_to != disease:
                continue
            pred = _predict_once(
                train=train_data,
                validation=validation_data,
                disease=disease,
                method=method,
                feature_cols=feature_cols,
                patient_col=patient_col,
                class_weight=class_weight,
            )
            row = {**model_fields, "disease": disease, "method": method}
            row.update(_metrics_for_predictions(pred))
            point_rows.append(row)
            fit_rows.append(
                {
                    **model_fields,
                    "disease": disease,
                    "method": method,
                    "n_fit": int(len(train_data)),
                    "n_prediction": int(len(validation_data)),
                    "fit_cases": int(train_data[disease].astype(int).sum()),
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
                metric_row.update({**model_fields, "disease": disease, "method": method})
                bootstrap_metric_rows.append(metric_row)
            for sample_row in sample_rows:
                sample_row.update({**model_fields, "disease": disease, "method": method})
                bootstrap_sample_rows.append(sample_row)
            if args.write_predictions:
                prediction_frames.append(pred)

        train_groups = _long_prs_group_map(
            train_prs_long,
            disease,
            patient_col,
            prs_param_cols,
            "training long PRS scores",
        )
        validation_groups = _long_prs_group_map(
            validation_prs_long,
            disease,
            patient_col,
            prs_param_cols,
            "validation long PRS scores",
        )
        for key, (params, validation_score_frame) in validation_groups.items():
            if key not in train_groups:
                raise ValueError(f"Missing training PRS scores for disease={disease}, params={params}")
            _, train_score_frame = train_groups[key]
            fit_data = train_data.merge(train_score_frame, on=patient_col, how="inner", validate="one_to_one")
            eval_data = validation_data.merge(
                validation_score_frame,
                on=patient_col,
                how="inner",
                validate="one_to_one",
            )
            if fit_data.empty or eval_data.empty:
                raise ValueError(f"No PRS-scored rows for disease={disease}, params={params}")
            method = f"prs_covariate_{disease}"
            pred = _predict_once(
                train=fit_data,
                validation=eval_data,
                disease=disease,
                method=method,
                feature_cols=[*covariates, "prs"],
                patient_col=patient_col,
                class_weight=class_weight,
            )
            for column, value in params.items():
                pred[column] = value
            row = {**model_fields, "disease": disease, "method": method, **params}
            row.update(_metrics_for_predictions(pred))
            point_rows.append(row)
            fit_rows.append(
                {
                    **model_fields,
                    "disease": disease,
                    "method": method,
                    **params,
                    "n_fit": int(len(fit_data)),
                    "n_prediction": int(len(eval_data)),
                    "fit_cases": int(fit_data[disease].astype(int).sum()),
                    "prediction_cases": int(eval_data[disease].astype(int).sum()),
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
                metric_row.update({**model_fields, "disease": disease, "method": method, **params})
                bootstrap_metric_rows.append(metric_row)
            for sample_row in sample_rows:
                sample_row.update({**model_fields, "disease": disease, "method": method, **params})
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
    write_table(
        eligible_counts(train_phenotype, labels, eligibility_rules),
        out_dir / "train_eligibility_counts.csv",
    )
    validation_counts = eligible_counts(validation_phenotype, labels, eligibility_rules)
    write_table(validation_counts, out_dir / "validation_eligibility_counts.csv")
    write_table(validation_counts, out_dir / "eligibility_counts.csv")
    summary_group_cols = ["model_name", "disease", "method", *prs_param_cols]
    summary_source = bootstrap_metrics if n_bootstraps > 0 else point_metrics
    present_group_cols = [col for col in summary_group_cols if col in summary_source.columns]
    if n_bootstraps > 0:
        summary = summarize_repeated_metrics(bootstrap_metrics, present_group_cols)
    else:
        summary = _summarize_point_metrics(point_metrics, present_group_cols)
    write_table(summary, out_dir / "summary_metrics.csv")
    if args.write_predictions:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        write_table(predictions, out_dir / "predictions.csv")
    summary_rows = summary.to_dict("records")
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_mode = "point-only" if n_bootstraps == 0 else "bootstrap"
    print(f"Wrote official {validation_mode} validation outputs to {out_dir}")


if __name__ == "__main__":
    main()
