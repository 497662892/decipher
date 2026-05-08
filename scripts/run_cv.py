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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared-fold CV for baseline model comparison.")
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--covariates", default="")
    parser.add_argument("--prs-scores", default=None, help="Optional table with patient_id and PRS columns.")
    parser.add_argument("--prs-prefix", default="prs_", help="Expected PRS column prefix, e.g. prs_disease1.")
    parser.add_argument(
        "--eligibility-config",
        default=None,
        help="Optional YAML with disease-specific eligible patient rules.",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    patient_col = args.patient_col
    labels = parse_csv_list(args.labels)
    covariates = parse_csv_list(args.covariates)
    phenotype = read_table(args.phenotype)
    splits = read_table(args.splits)
    eligibility_rules = load_eligibility_rules(args.eligibility_config)
    require_columns(phenotype, [patient_col, *labels, *covariates], "phenotype")
    require_columns(splits, [patient_col, "fold"], "splits")
    disease_specific_splits = "disease" in splits.columns
    if disease_specific_splits:
        require_columns(splits, [patient_col, "disease", "fold"], "splits")
    else:
        data = phenotype.merge(splits[[patient_col, "fold"]], on=patient_col, how="inner")

    methods: list[tuple[str, list[str], str | None]] = [("covariate_only", covariates, None)]
    if args.prs_scores:
        prs = read_table(args.prs_scores)
        require_columns(prs, [patient_col], "prs_scores")
        if disease_specific_splits:
            phenotype = phenotype.merge(prs, on=patient_col, how="left")
        else:
            data = data.merge(prs, on=patient_col, how="left")
        for label in labels:
            prs_col = f"{args.prs_prefix}{label}"
            data_columns = phenotype.columns if disease_specific_splits else data.columns
            if prs_col in data_columns:
                methods.append((f"prs_covariate_{label}", [*covariates, prs_col], label))

    predictions = []
    metric_rows = []
    for disease in labels:
        if disease_specific_splits:
            disease_splits = splits.loc[splits["disease"] == disease, [patient_col, "fold"]]
            if disease_splits.empty:
                raise ValueError(f"Disease-specific split file has no rows for {disease}")
            if disease_splits[patient_col].duplicated().any():
                examples = disease_splits.loc[disease_splits[patient_col].duplicated(), patient_col].head(5)
                raise ValueError(f"Split file has duplicate patients for {disease}: {examples.tolist()}")
            disease_data = phenotype.merge(disease_splits, on=patient_col, how="inner")
        else:
            disease_data = data
        disease_data = filter_eligible(disease_data, disease, eligibility_rules)
        if disease_data.empty:
            raise ValueError(f"No eligible patients remain for disease {disease}")
        for method, feature_cols, applies_to in methods:
            if applies_to is not None and applies_to != disease:
                continue
            features = list(feature_cols)
            if not features:
                disease_data["_intercept_only"] = 1.0
                features = ["_intercept_only"]
            for fold in sorted(disease_data["fold"].unique()):
                train = disease_data[disease_data["fold"] != fold].copy()
                val = disease_data[disease_data["fold"] == fold].copy()
                if val.empty:
                    continue
                pred = fit_predict_logistic(
                    train=train,
                    val=val,
                    label_col=disease,
                    feature_cols=features,
                    patient_col=patient_col,
                    method=method,
                    fold=int(fold),
                ).predictions
                predictions.append(pred)
                row = {"disease": disease, "method": method, "fold": int(fold)}
                row.update(_metrics_for_predictions(pred))
                metric_rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    write_table(oof, out_dir / "oof_predictions.csv")
    write_table(fold_metrics, out_dir / "fold_metrics.csv")
    eligibility_base = phenotype if disease_specific_splits else data
    write_table(eligible_counts(eligibility_base, labels, eligibility_rules), out_dir / "eligibility_counts.csv")

    summary_rows = []
    for (disease, method), frame in oof.groupby(["disease", "method"]):
        row = {"disease": disease, "method": method}
        row.update(_metrics_for_predictions(frame))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_table(summary, out_dir / "summary_metrics.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote CV outputs to {out_dir}")


if __name__ == "__main__":
    main()
