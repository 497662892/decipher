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
from decipehr.pt import plink_sscore_columns, read_plink_sscore
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


def _score_column(columns: list[str]) -> str:
    for score_name in ["BETA", "prs"]:
        try:
            return plink_sscore_columns(columns, 1, [score_name])[0]
        except ValueError:
            pass
    sum_cols = [col for col in columns if col.upper().endswith("_SUM")]
    avg_cols = [col for col in columns if col.upper().endswith("_AVG")]
    for col in sum_cols + avg_cols:
        if col.upper() not in {"NAMED_ALLELE_DOSAGE_SUM", "ALLELE_CT"}:
            return col
    raise ValueError("PLINK .sscore output is missing a PRS score column")


def _manifest_score_path(task: dict[str, object], score_path_column: str) -> Path:
    value = task.get(score_path_column)
    if value in (None, ""):
        for fallback in ["score_file", "sscore"]:
            value = task.get(fallback)
            if value not in (None, ""):
                break
    if value in (None, ""):
        raise ValueError(f"Manifest row is missing score path column {score_path_column!r}")
    return Path(str(value))


def _read_score_frame(
    task: dict[str, object],
    *,
    score_kind: str,
    score_path_column: str,
    patient_col: str,
    method_name: str,
    param_cols: list[str],
) -> pd.DataFrame:
    score_path = _manifest_score_path(task, score_path_column)
    if not score_path.exists() or score_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing PRS score output: {score_path}")

    if score_kind == "long":
        frame = read_table(score_path)
        require_columns(frame, [patient_col, "prs"], f"long PRS scores {score_path}")
        out_cols = [patient_col, "prs", *[col for col in param_cols if col in frame.columns]]
        out = frame[out_cols].copy()
    elif score_kind == "sscore":
        frame = read_plink_sscore(score_path, patient_col=patient_col)
        score_col = _score_column(list(frame.columns))
        out = frame[[patient_col, score_col]].rename(columns={score_col: "prs"}).copy()
    else:
        raise ValueError(f"Unsupported score kind: {score_kind}")

    out["disease"] = str(task["disease"])
    out["fold"] = int(task["fold"])
    out["source_score_file"] = str(score_path)
    out["method_family"] = method_name
    for col in param_cols:
        if col not in out.columns:
            if col not in task or task[col] in (None, ""):
                raise ValueError(f"Parameter column {col!r} is absent from scores and manifest")
            out[col] = task[col]
    out["prs"] = pd.to_numeric(out["prs"], errors="coerce")
    out = out.dropna(subset=[patient_col, "prs"])
    return out[[patient_col, "disease", "fold", "method_family", *param_cols, "prs", "source_score_file"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fold-aware PRS grids from per-fold score files.")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--score-kind", choices=["long", "sscore"], required=True)
    parser.add_argument("--score-path-column", default="score_file")
    parser.add_argument("--param-cols", default="")
    parser.add_argument("--phenotype", default="data/processed/final/train_val_phenotype.csv")
    parser.add_argument(
        "--splits",
        default="data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv",
    )
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--covariates", default="sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd")
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--scores-out",
        default=None,
        help=(
            "Optional raw OOF PRS long-table output. Defaults to a sibling scores directory "
            "next to --out-dir, named {disease}.{method_name}_cv_scores_long.csv."
        ),
    )
    args = parser.parse_args()

    patient_col = args.patient_col
    covariates = parse_csv_list(args.covariates)
    param_cols = parse_csv_list(args.param_cols)

    phenotype = read_table(args.phenotype)
    splits = read_table(args.splits)
    manifest = read_table(args.score_manifest)
    eligibility_rules = load_eligibility_rules(args.eligibility_config)

    require_columns(phenotype, [patient_col, args.disease, *covariates], "phenotype")
    require_columns(splits, [patient_col, "disease", "fold"], "splits")
    require_columns(manifest, ["disease", "fold"], "score manifest")
    if args.score_path_column not in manifest.columns and not {"score_file", "sscore"} & set(manifest.columns):
        raise ValueError(
            f"Score manifest needs {args.score_path_column!r}, or one of score_file/sscore columns"
        )

    disease_manifest = manifest.loc[manifest["disease"].astype(str) == args.disease].copy()
    if disease_manifest.empty:
        raise ValueError(f"No score manifest rows found for disease {args.disease}")
    disease_splits = splits.loc[splits["disease"] == args.disease, [patient_col, "fold"]].copy()
    if disease_splits.empty:
        raise ValueError(f"No split rows found for disease {args.disease}")

    data = phenotype.merge(disease_splits, on=patient_col, how="inner")
    data = filter_eligible(data, args.disease, eligibility_rules)
    if data.empty:
        raise ValueError(f"No eligible patients remain for disease {args.disease}")

    score_frames = [
        _read_score_frame(
            task,
            score_kind=args.score_kind,
            score_path_column=args.score_path_column,
            patient_col=patient_col,
            method_name=args.method_name,
            param_cols=param_cols,
        )
        for task in disease_manifest.to_dict("records")
    ]
    scores = pd.concat(score_frames, ignore_index=True)
    duplicate_keys = [patient_col, "fold", *param_cols]
    if scores.duplicated(duplicate_keys).any():
        examples = scores.loc[scores.duplicated(duplicate_keys), duplicate_keys].head(5)
        raise ValueError(f"Duplicate score rows detected: {examples.to_dict('records')}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_scores_out = (
        Path(args.scores_out)
        if args.scores_out
        else out_dir.parent / "scores" / f"{args.disease}.{args.method_name}_cv_scores_long.csv"
    )
    raw_scores_out.parent.mkdir(parents=True, exist_ok=True)
    raw_score_cols = [patient_col, "disease", "fold", *param_cols, "prs", "source_score_file"]
    write_table(scores[raw_score_cols].copy(), raw_scores_out)

    predictions = []
    fold_metric_rows = []
    methods = [(f"{args.method_name}_prs_only", ["prs"])]
    if covariates:
        methods.append((f"{args.method_name}_prs_covariate", [*covariates, "prs"]))

    group_cols = param_cols or ["method_family"]
    grouped = scores.groupby(group_cols, sort=True, dropna=False)
    for param_values, param_scores in grouped:
        if not isinstance(param_values, tuple):
            param_values = (param_values,)
        param_payload = dict(zip(group_cols, param_values, strict=True))
        for fold in sorted(data["fold"].dropna().astype(int).unique()):
            fold_scores = param_scores.loc[param_scores["fold"].astype(int) == fold, [patient_col, "prs"]]
            if fold_scores.empty:
                raise ValueError(f"Missing scores for disease={args.disease}, fold={fold}, params={param_payload}")
            missing_scores = data.loc[~data[patient_col].isin(fold_scores[patient_col]), patient_col]
            if not missing_scores.empty:
                raise ValueError(
                    f"Scores are missing {len(missing_scores)} eligible patients for disease={args.disease}, "
                    f"fold={fold}, params={param_payload}. Examples: {missing_scores.head(5).tolist()}"
                )
            scored = data.merge(fold_scores, on=patient_col, how="left", validate="one_to_one")
            train = scored.loc[scored["fold"].astype(int) != fold].copy()
            val = scored.loc[scored["fold"].astype(int) == fold].copy()
            if train.empty or val.empty:
                raise ValueError(f"Empty train/validation split for disease={args.disease}, fold={fold}")
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
                for key, value in param_payload.items():
                    pred[key] = value
                predictions.append(pred)
                row = {"disease": args.disease, "method": method, "fold": int(fold), **param_payload}
                row.update(_metrics_for_predictions(pred))
                fold_metric_rows.append(row)

    oof = pd.concat(predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    write_table(oof, out_dir / "oof_predictions.csv")
    write_table(fold_metrics, out_dir / "fold_metrics.csv")

    summary_rows = []
    summary_group_cols = ["disease", "method", *group_cols]
    for keys, frame in oof.groupby(summary_group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(summary_group_cols, keys, strict=True))
        row.update(_metrics_for_predictions(frame))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_table(summary, out_dir / "summary_metrics.csv")
    if {"auc", "nagelkerke_r2"}.issubset(summary.columns):
        best = (
            summary.sort_values(["method", "auc", "nagelkerke_r2"], ascending=[True, False, False], kind="mergesort")
            .groupby("method", as_index=False)
            .head(1)
        )
        write_table(best, out_dir / "best_by_method.csv")
        primary_method = f"{args.method_name}_prs_covariate" if covariates else f"{args.method_name}_prs_only"
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
        json.dumps(summary_rows, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote raw {args.method_name} CV PRS scores to {raw_scores_out}")
    print(f"Wrote {args.method_name} CV evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
