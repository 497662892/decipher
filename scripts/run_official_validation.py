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
    iter_repeated_holdout_indices,
    prepare_official_validation_phenotype,
    summarize_repeated_metrics,
)

DEFAULT_LABELLED_PHENOTYPE = (
    "/gpfs/data/deciphEHRlab/deciphEHRdream/holdout_phase/phenotype_files/"
    "Training_Phase_Cov_file_with_Phenotype.txt"
)
DEFAULT_VALIDATION_COVARIATES = (
    "/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/phenotype_files/"
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


def _method_specs(
    phenotype: pd.DataFrame,
    labels: list[str],
    covariates: list[str],
    prs_prefix: str,
) -> list[tuple[str, list[str], str | None]]:
    methods: list[tuple[str, list[str], str | None]] = [("covariate_only", covariates, None)]
    for label in labels:
        prs_col = f"{prs_prefix}{label}"
        if prs_col in phenotype.columns:
            methods.append((f"prs_covariate_{label}", [*covariates, prs_col], label))
    return methods


def _long_prs_groups(
    prs_long: pd.DataFrame | None,
    disease: str,
    patient_col: str,
    param_cols: list[str],
) -> list[tuple[dict[str, object], pd.DataFrame]]:
    if prs_long is None:
        return []
    disease_scores = prs_long.loc[prs_long["disease"] == disease].copy()
    if disease_scores.empty:
        return []
    require_columns(disease_scores, [patient_col, "prs", *param_cols], "long PRS scores")
    if not param_cols:
        return [({}, disease_scores[[patient_col, "prs"]].copy())]

    groups = []
    for values, frame in disease_scores.groupby(param_cols, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        params = dict(zip(param_cols, values, strict=True))
        groups.append((params, frame[[patient_col, "prs"]].copy()))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated holdout validation using the official validation labels. "
            "Each repeat fits on validation samples not selected for prediction."
        )
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
    parser.add_argument("--prs-scores", default=None, help="Optional wide table with patient ID and prs_<label> columns.")
    parser.add_argument(
        "--prs-long-scores",
        default=None,
        help="Optional long table with patient ID, disease, prs, and PRS parameter columns.",
    )
    parser.add_argument("--prs-param-cols", default="clump_r2,p_threshold")
    parser.add_argument("--prs-prefix", default="prs_")
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable label-stratified sampling of the 1000 prediction samples.",
    )
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write one row per predicted sample per repeat. This can be large for 1000 repeats.",
    )
    parser.add_argument("--validation-phenotype-out", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    patient_col = args.patient_col
    labels = parse_csv_list(args.labels)
    covariates = parse_csv_list(args.covariates)
    prs_param_cols = parse_csv_list(args.prs_param_cols)
    phenotype = _load_validation_phenotype(
        phenotype_path=args.phenotype,
        labelled_phenotype_path=args.labelled_phenotype,
        validation_covariates_path=args.validation_covariates,
        patient_col=patient_col,
        labels=labels,
        covariates=covariates,
    )
    if args.prs_scores:
        prs = read_table(args.prs_scores)
        require_columns(prs, [patient_col], "prs_scores")
        phenotype = phenotype.merge(prs, on=patient_col, how="left", validate="one_to_one")
    prs_long = None
    if args.prs_long_scores:
        prs_long = read_table(args.prs_long_scores)
        require_columns(prs_long, [patient_col, "disease", "prs", *prs_param_cols], "long PRS scores")

    eligibility_rules = load_eligibility_rules(args.eligibility_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.validation_phenotype_out:
        write_table(phenotype, args.validation_phenotype_out)

    methods = _method_specs(phenotype, labels, covariates, args.prs_prefix)
    prediction_frames = []
    metric_rows = []
    split_rows = []

    for disease in labels:
        disease_data = filter_eligible(phenotype, disease, eligibility_rules)
        if disease_data.empty:
            raise ValueError(f"No eligible validation patients remain for disease {disease}")
        disease_data = disease_data.reset_index(drop=True)
        for repeat, fit_idx, eval_idx in iter_repeated_holdout_indices(
            disease_data,
            label_col=disease,
            eval_size=args.eval_size,
            n_repeats=args.n_repeats,
            seed=args.seed,
            stratify=not args.no_stratify,
        ):
            split_rows.append(
                {
                    "disease": disease,
                    "repeat": int(repeat),
                    "n_fit": int(len(fit_idx)),
                    "n_prediction": int(len(eval_idx)),
                    "fit_cases": int(disease_data.iloc[fit_idx][disease].astype(int).sum()),
                    "prediction_cases": int(disease_data.iloc[eval_idx][disease].astype(int).sum()),
                }
            )
            base_fit_data = disease_data.iloc[fit_idx].copy()
            base_eval_data = disease_data.iloc[eval_idx].copy()
            for method, feature_cols, applies_to in methods:
                if applies_to is not None and applies_to != disease:
                    continue
                fit_data = base_fit_data.copy()
                eval_data = base_eval_data.copy()
                features = list(feature_cols)
                if not features:
                    fit_data["_intercept_only"] = 1.0
                    eval_data["_intercept_only"] = 1.0
                    features = ["_intercept_only"]
                pred = fit_predict_logistic(
                    train=fit_data,
                    val=eval_data,
                    label_col=disease,
                    feature_cols=features,
                    patient_col=patient_col,
                    method=method,
                    fold=int(repeat),
                ).predictions.rename(columns={"fold": "repeat"})
                row = {"disease": disease, "method": method, "repeat": int(repeat)}
                row.update(_metrics_for_predictions(pred))
                metric_rows.append(row)
                if args.write_predictions:
                    prediction_frames.append(pred)
            for params, score_frame in _long_prs_groups(prs_long, disease, patient_col, prs_param_cols):
                fit_data = base_fit_data.merge(score_frame, on=patient_col, how="inner")
                eval_data = base_eval_data.merge(score_frame, on=patient_col, how="inner")
                if fit_data.empty or eval_data.empty:
                    raise ValueError(
                        f"No PRS-scored rows for disease={disease}, repeat={repeat}, params={params}"
                    )
                method = f"prs_covariate_{disease}"
                pred = fit_predict_logistic(
                    train=fit_data,
                    val=eval_data,
                    label_col=disease,
                    feature_cols=[*covariates, "prs"],
                    patient_col=patient_col,
                    method=method,
                    fold=int(repeat),
                ).predictions.rename(columns={"fold": "repeat"})
                for column, value in params.items():
                    pred[column] = value
                row = {"disease": disease, "method": method, "repeat": int(repeat), **params}
                row.update(_metrics_for_predictions(pred))
                metric_rows.append(row)
                if args.write_predictions:
                    prediction_frames.append(pred)

    repeat_metrics = pd.DataFrame(metric_rows)
    split_summary = pd.DataFrame(split_rows)
    write_table(repeat_metrics, out_dir / "repeat_metrics.csv")
    write_table(split_summary, out_dir / "repeat_split_summary.csv")
    write_table(eligible_counts(phenotype, labels, eligibility_rules), out_dir / "eligibility_counts.csv")
    summary_group_cols = ["disease", "method", *prs_param_cols]
    present_group_cols = [col for col in summary_group_cols if col in repeat_metrics.columns]
    summary = summarize_repeated_metrics(repeat_metrics, present_group_cols)
    write_table(summary, out_dir / "summary_metrics.csv")
    if args.write_predictions:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        write_table(predictions, out_dir / "repeated_holdout_predictions.csv")
    summary_rows = summary.to_dict("records")
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote official repeated-holdout validation outputs to {out_dir}")


if __name__ == "__main__":
    main()
