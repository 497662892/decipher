from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from decipehr.eligibility import filter_eligible, load_eligibility_rules
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
    parser = argparse.ArgumentParser(description="Evaluate fold-aware LDpred2 PRS grids.")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--phenotype", default="data/processed/phenotype.csv")
    parser.add_argument("--splits", default="data/processed/splits/disease_specific_split_seed42.csv")
    parser.add_argument("--scores", required=True, help="Long table from collect_ldpred2_cv_scores.py.")
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--covariates", default="")
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    patient_col = args.patient_col
    covariates = parse_csv_list(args.covariates)
    phenotype = read_table(args.phenotype)
    splits = read_table(args.splits)
    scores = read_table(args.scores)
    eligibility_rules = load_eligibility_rules(args.eligibility_config)

    require_columns(phenotype, [patient_col, args.disease, *covariates], "phenotype")
    require_columns(splits, [patient_col, "disease", "fold"], "splits")
    require_columns(
        scores,
        [patient_col, "disease", "fold", "h2", "p", "sparse", "prs"],
        "LDpred2 scores",
    )

    disease_splits = splits.loc[splits["disease"] == args.disease, [patient_col, "fold"]]
    if disease_splits.empty:
        raise ValueError(f"No split rows found for disease {args.disease}")
    data = phenotype.merge(disease_splits, on=patient_col, how="inner")
    data = filter_eligible(data, args.disease, eligibility_rules)
    if data.empty:
        raise ValueError(f"No eligible patients remain for disease {args.disease}")

    disease_scores = scores.loc[scores["disease"] == args.disease].copy()
    if disease_scores.empty:
        raise ValueError(f"No LDpred2 scores found for disease {args.disease}")
    duplicate_keys = [patient_col, "fold", "h2", "p", "sparse"]
    if disease_scores.duplicated(duplicate_keys).any():
        examples = disease_scores.loc[disease_scores.duplicated(duplicate_keys), duplicate_keys].head(5)
        raise ValueError(f"Duplicate LDpred2 score rows detected: {examples.to_dict('records')}")

    predictions = []
    fold_metric_rows = []
    methods = [("ldpred2_prs_only", ["prs"])]
    if covariates:
        methods.append(("ldpred2_prs_covariate", [*covariates, "prs"]))

    grouped = disease_scores.groupby(["h2", "p", "sparse"], sort=True)
    for (h2, p_value, sparse), param_scores in grouped:
        for fold in sorted(data["fold"].dropna().astype(int).unique()):
            fold_scores = param_scores.loc[
                param_scores["fold"].astype(int) == fold,
                [patient_col, "prs"],
            ]
            if fold_scores.empty:
                raise ValueError(
                    f"Missing LDpred2 scores for disease={args.disease}, fold={fold}, "
                    f"h2={h2}, p={p_value}, sparse={sparse}"
                )
            missing_scores = data.loc[~data[patient_col].isin(fold_scores[patient_col]), patient_col]
            if not missing_scores.empty:
                raise ValueError(
                    f"LDpred2 scores are missing {len(missing_scores)} eligible patients for "
                    f"disease={args.disease}, fold={fold}, h2={h2}, p={p_value}, sparse={sparse}. "
                    f"Examples: {missing_scores.head(5).tolist()}"
                )
            scored = data.merge(fold_scores, on=patient_col, how="left", validate="one_to_one")
            if scored["prs"].isna().any():
                examples = scored.loc[scored["prs"].isna(), patient_col].head(5).tolist()
                raise ValueError(
                    f"LDpred2 scores contain missing PRS values after merge for "
                    f"disease={args.disease}, fold={fold}, h2={h2}, p={p_value}, sparse={sparse}: "
                    f"{examples}"
                )
            train = scored.loc[scored["fold"].astype(int) != fold].copy()
            val = scored.loc[scored["fold"].astype(int) == fold].copy()
            if train.empty or val.empty:
                raise ValueError(
                    f"Empty train/validation split for disease={args.disease}, fold={fold}, "
                    f"h2={h2}, p={p_value}, sparse={sparse}"
                )
            for method, feature_cols in methods:
                pred = fit_predict_logistic(
                    train=train,
                    val=val,
                    label_col=args.disease,
                    feature_cols=feature_cols,
                    patient_col=patient_col,
                    method=method,
                    fold=int(fold),
                ).predictions
                pred["h2"] = float(h2)
                pred["p"] = float(p_value)
                pred["sparse"] = int(sparse)
                predictions.append(pred)
                row = {
                    "disease": args.disease,
                    "method": method,
                    "fold": int(fold),
                    "h2": float(h2),
                    "p": float(p_value),
                    "sparse": int(sparse),
                }
                row.update(_metrics_for_predictions(pred))
                fold_metric_rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    write_table(oof, out_dir / "oof_predictions.csv")
    write_table(fold_metrics, out_dir / "fold_metrics.csv")

    summary_rows = []
    for (disease, method, h2, p_value, sparse), frame in oof.groupby(
        ["disease", "method", "h2", "p", "sparse"],
        sort=True,
    ):
        row = {
            "disease": disease,
            "method": method,
            "h2": float(h2),
            "p": float(p_value),
            "sparse": int(sparse),
        }
        row.update(_metrics_for_predictions(frame))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_table(summary, out_dir / "summary_metrics.csv")
    if {"auc", "nagelkerke_r2"}.issubset(summary.columns):
        best = summary.sort_values(
            ["method", "auc", "nagelkerke_r2"],
            ascending=[True, False, False],
            kind="mergesort",
        ).groupby("method", as_index=False).head(1)
        write_table(best, out_dir / "best_by_method.csv")
        primary_method = "ldpred2_prs_covariate" if covariates else "ldpred2_prs_only"
        primary = summary.loc[summary["method"] == primary_method].copy()
        if not primary.empty:
            write_table(primary, out_dir / "primary_metrics.csv")
            best_primary = primary.sort_values(
                ["auc", "nagelkerke_r2"],
                ascending=[False, False],
                kind="mergesort",
            ).head(1)
            write_table(best_primary, out_dir / "best_primary_model.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote LDpred2 CV evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
