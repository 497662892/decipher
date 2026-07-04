from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns


@dataclass(frozen=True)
class CVFamilySpec:
    name: str
    cv_root: str
    score_template: str
    metrics_template: str
    param_cols: tuple[str, ...]


DEFAULT_FAMILIES: dict[str, CVFamilySpec] = {
    "pt": CVFamilySpec(
        name="pt",
        cv_root="outputs/pt_cv_auc_r2_final_trainval",
        score_template="{disease}/scores/{disease}.pt_cv_scores_long.csv",
        metrics_template="{disease}/evaluation/summary_metrics.csv",
        param_cols=("clump_r2", "p_threshold"),
    ),
    "ldpred2": CVFamilySpec(
        name="ldpred2",
        cv_root="outputs/ldpred2_cv_auc_r2_final_trainval",
        score_template="{disease}/scores/{disease}.ldpred2_cv_scores_long.csv",
        metrics_template="{disease}/evaluation/summary_metrics.csv",
        param_cols=("h2", "p", "sparse"),
    ),
    "lassosum2": CVFamilySpec(
        name="lassosum2",
        cv_root="outputs/lassosum2_cv_auc_r2_final_trainval",
        score_template="{disease}/scores/{disease}.lassosum2_cv_scores_long.csv",
        metrics_template="{disease}/evaluation/summary_metrics.csv",
        param_cols=("delta", "lambda"),
    ),
    "prscs": CVFamilySpec(
        name="prscs",
        cv_root="outputs/prscs_cv_auc_r2_final_trainval",
        score_template="{disease}/scores/{disease}.prscs_cv_scores_long.csv",
        metrics_template="{disease}/evaluation/summary_metrics.csv",
        param_cols=("phi",),
    ),
    "sdpr": CVFamilySpec(
        name="sdpr",
        cv_root="outputs/sdpr_cv_auc_r2_final_trainval",
        score_template="{disease}/scores/{disease}.sdpr_cv_scores_long.csv",
        metrics_template="{disease}/evaluation/summary_metrics.csv",
        param_cols=("model", "a"),
    ),
}


def _path(root: str | Path, template: str, disease: str) -> Path:
    return Path(root) / template.format(disease=disease)


def _resolve_metric_column(metrics: pd.DataFrame, requested: str) -> str:
    if requested in metrics.columns:
        return requested
    if requested.endswith("_mean"):
        base = requested.removesuffix("_mean")
        if base in metrics.columns:
            return base
    mean_name = f"{requested}_mean"
    if mean_name in metrics.columns:
        return mean_name
    raise ValueError(
        f"Selection metric {requested!r} was not found. Available columns include: "
        f"{metrics.columns[:20].tolist()}"
    )


def _metric_direction(metric_col: str, requested: str) -> str:
    metric = metric_col.removesuffix("_mean").lower()
    requested_metric = requested.removesuffix("_mean").lower()
    if metric == "brier" or requested_metric == "brier":
        return "min"
    return "max"


def _filter_metrics(
    metrics: pd.DataFrame,
    *,
    disease: str,
    family: str,
    method_template: str,
) -> pd.DataFrame:
    require_columns(metrics, ["disease", "method"], "CV metrics")
    expected_method = method_template.format(disease=disease, family=family)
    filtered = metrics.loc[
        (metrics["disease"].astype(str) == disease) & (metrics["method"].astype(str) == expected_method)
    ].copy()
    if filtered.empty:
        raise ValueError(f"No CV metric rows for disease={disease!r}, method={expected_method!r}")
    return filtered


def _select_best_row(
    metrics: pd.DataFrame,
    *,
    disease: str,
    family: str,
    method_template: str,
    metric: str,
    direction: str,
    param_cols: tuple[str, ...],
) -> tuple[pd.Series, str, str]:
    filtered = _filter_metrics(metrics, disease=disease, family=family, method_template=method_template)
    metric_col = _resolve_metric_column(filtered, metric)
    selected_direction = _metric_direction(metric_col, metric) if direction == "auto" else direction
    filtered[metric_col] = pd.to_numeric(filtered[metric_col], errors="coerce")
    filtered = filtered.loc[filtered[metric_col].notna()].copy()
    if filtered.empty:
        raise ValueError(f"No finite values for selection metric {metric_col!r}")
    sort_param_cols = [col for col in param_cols if col in filtered.columns]
    filtered = filtered.sort_values(
        by=[metric_col, *sort_param_cols],
        ascending=[selected_direction == "min", *([True] * len(sort_param_cols))],
        kind="mergesort",
    )
    return filtered.iloc[0], metric_col, selected_direction


def _numeric_scalar(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _normal_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _match_param(series: pd.Series, selected_value: Any) -> pd.Series:
    selected_numeric = _numeric_scalar(selected_value)
    if selected_numeric is not None:
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_mask = np.isclose(
            numeric_series.to_numpy(dtype=float),
            selected_numeric,
            rtol=1e-8,
            atol=1e-12,
            equal_nan=False,
        )
        if bool(np.any(numeric_mask)):
            return pd.Series(numeric_mask, index=series.index)
    return series.map(_normal_string) == _normal_string(selected_value)


def _select_score_rows(
    scores: pd.DataFrame,
    *,
    selected: pd.Series,
    splits: pd.DataFrame | None,
    family: str,
    disease: str,
    patient_col: str,
    disease_col: str,
    fold_col: str,
    score_col: str,
    param_cols: tuple[str, ...],
    score_path: Path,
) -> pd.DataFrame:
    require_columns(scores, [patient_col, disease_col, fold_col, score_col, *param_cols], f"{family} CV scores")
    mask = scores[disease_col].astype(str) == disease
    for column in param_cols:
        if column not in selected.index or pd.isna(selected[column]):
            raise ValueError(f"Selected {family} CV row has no value for parameter {column!r}")
        mask &= _match_param(scores[column], selected[column])
    matched = scores.loc[mask].copy()
    if matched.empty:
        selected_params = {col: selected[col] for col in param_cols}
        raise ValueError(f"No {family} CV score rows matched selected parameters: {selected_params}")

    if splits is not None:
        require_columns(splits, [patient_col, disease_col, fold_col], "CV splits")
        split_fold_col = "__split_fold"
        disease_splits = splits.loc[splits[disease_col].astype(str) == disease, [patient_col, fold_col]].copy()
        if disease_splits.empty:
            raise ValueError(f"No CV split rows found for disease {disease}")
        if disease_splits.duplicated(patient_col).any():
            examples = disease_splits.loc[disease_splits.duplicated(patient_col), patient_col].head(5)
            raise ValueError(f"Duplicate CV split rows for {disease}: {examples.tolist()}")
        disease_splits = disease_splits.rename(columns={fold_col: split_fold_col})
        matched = matched.merge(disease_splits, on=patient_col, how="inner", validate="many_to_one")
        matched_fold = pd.to_numeric(matched[fold_col], errors="coerce")
        split_fold = pd.to_numeric(matched[split_fold_col], errors="coerce")
        matched = matched.loc[matched_fold == split_fold].copy()
        missing_patients = disease_splits.loc[~disease_splits[patient_col].isin(matched[patient_col]), patient_col]
        if not missing_patients.empty:
            raise ValueError(
                f"Selected {family} CV scores are missing OOF rows for {len(missing_patients)} "
                f"{disease} patients. Examples: {missing_patients.head(5).tolist()}"
            )
        matched[fold_col] = matched[split_fold_col].to_numpy()

    duplicate_keys = [patient_col, disease_col]
    if matched.duplicated(duplicate_keys).any():
        examples = matched.loc[matched.duplicated(duplicate_keys), [*duplicate_keys, fold_col]].head(5)
        raise ValueError(f"Duplicate selected {family} OOF score rows: {examples.to_dict('records')}")
    matched[score_col] = pd.to_numeric(matched[score_col], errors="coerce")
    if matched[score_col].isna().any():
        examples = matched.loc[matched[score_col].isna(), [patient_col, score_col]].head(5)
        raise ValueError(f"Selected {family} CV scores contain nonnumeric PRS values: {examples.to_dict('records')}")

    output = matched[[patient_col, disease_col, fold_col, score_col]].copy()
    output = output.rename(columns={disease_col: "disease", fold_col: "fold", score_col: "prs"})
    output["model"] = family
    output["family"] = family
    output["source_model"] = matched["model"].astype(str).to_numpy() if "model" in matched.columns else family
    for column in param_cols:
        if column == "model":
            output["selected_model"] = matched[column].to_numpy()
        else:
            output[column] = matched[column].to_numpy()
    output["source_score_file"] = str(score_path)
    return output


def _selection_record(
    *,
    family: str,
    disease: str,
    selected: pd.Series,
    metric_col: str,
    direction: str,
    metrics_path: Path,
    score_path: Path,
    param_cols: tuple[str, ...],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "disease": disease,
        "family": family,
        "selection_metric": metric_col,
        "selection_direction": direction,
        "selection_metric_value": selected[metric_col],
        "metrics_file": str(metrics_path),
        "train_score_file": str(score_path),
    }
    for column in param_cols:
        record[column] = selected[column] if column in selected.index else np.nan
    for column in [
        "auc",
        "auprc",
        "brier",
        "or_per_sd",
        "top10_vs_lower90_or",
        "nagelkerke_r2",
        "prevalence",
        "n",
    ]:
        if column in selected.index:
            record[column] = selected[column]
    return record


def _process_family(
    *,
    spec: CVFamilySpec,
    splits: pd.DataFrame | None,
    disease: str,
    metric: str,
    direction: str,
    method_template: str,
    patient_col: str,
    disease_col: str,
    fold_col: str,
    score_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics_path = _path(spec.cv_root, spec.metrics_template, disease)
    score_path = _path(spec.cv_root, spec.score_template, disease)
    missing = [path for path in [metrics_path, score_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CV files for {spec.name} {disease}: {missing}")

    metrics = read_table(metrics_path)
    selected, metric_col, selected_direction = _select_best_row(
        metrics,
        disease=disease,
        family=spec.name,
        method_template=method_template,
        metric=metric,
        direction=direction,
        param_cols=spec.param_cols,
    )
    scores = read_table(score_path)
    selected_scores = _select_score_rows(
        scores,
        selected=selected,
        splits=splits,
        family=spec.name,
        disease=disease,
        patient_col=patient_col,
        disease_col=disease_col,
        fold_col=fold_col,
        score_col=score_col,
        param_cols=spec.param_cols,
        score_path=score_path,
    )
    selected_scores["selection_metric"] = metric_col
    selected_scores["selection_metric_value"] = selected[metric_col]
    record = _selection_record(
        family=spec.name,
        disease=disease,
        selected=selected,
        metric_col=metric_col,
        direction=selected_direction,
        metrics_path=metrics_path,
        score_path=score_path,
        param_cols=spec.param_cols,
    )
    return selected_scores, record


def _apply_overrides(args: argparse.Namespace, specs: dict[str, CVFamilySpec]) -> dict[str, CVFamilySpec]:
    overrides = {
        "pt": args.pt_cv_root,
        "ldpred2": args.ldpred2_cv_root,
        "lassosum2": args.lassosum2_cv_root,
        "prscs": args.prscs_cv_root,
        "sdpr": args.sdpr_cv_root,
    }
    updated = {}
    for name, spec in specs.items():
        updated[name] = CVFamilySpec(
            name=spec.name,
            cv_root=overrides[name] or spec.cv_root,
            score_template=spec.score_template,
            metrics_template=spec.metrics_template,
            param_cols=spec.param_cols,
        )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select each PRS family's best final train-val 5-CV model and write fold-safe "
            "OOF long inputs for a TabPFN ensemble."
        )
    )
    parser.add_argument("--disease", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--families", default="pt,ldpred2,lassosum2,prscs,sdpr")
    parser.add_argument("--selection-metric", default="auc")
    parser.add_argument("--selection-direction", choices=["auto", "max", "min"], default="auto")
    parser.add_argument("--selection-method", default="{family}_prs_covariate")
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--score-col", default="prs")
    parser.add_argument(
        "--splits",
        default=None,
        help=(
            "Optional patient-level fold assignment. When provided, raw fold-model scores are "
            "filtered to each patient's held-out fold to produce true OOF ensemble inputs."
        ),
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pt-cv-root", default=None)
    parser.add_argument("--ldpred2-cv-root", default=None)
    parser.add_argument("--lassosum2-cv-root", default=None)
    parser.add_argument("--prscs-cv-root", default=None)
    parser.add_argument("--sdpr-cv-root", default=None)
    args = parser.parse_args()

    if bool(args.disease) == bool(args.labels):
        raise ValueError("Use exactly one of --disease or --labels")
    diseases = [args.disease] if args.disease else parse_csv_list(args.labels)
    families = parse_csv_list(args.families)
    unknown = [family for family in families if family not in DEFAULT_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown PRS families: {unknown}; expected one of {sorted(DEFAULT_FAMILIES)}")

    specs = _apply_overrides(args, DEFAULT_FAMILIES)
    splits = read_table(args.splits) if args.splits else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for disease in diseases:
        parts = []
        records = []
        missing_messages = []
        for family in families:
            try:
                selected_scores, record = _process_family(
                    spec=specs[family],
                    splits=splits,
                    disease=disease,
                    metric=args.selection_metric,
                    direction=args.selection_direction,
                    method_template=args.selection_method,
                    patient_col=args.patient_col,
                    disease_col=args.disease_col,
                    fold_col=args.fold_col,
                    score_col=args.score_col,
                )
            except (FileNotFoundError, ValueError) as exc:
                if args.allow_missing:
                    missing_messages.append(f"{family}: {exc}")
                    continue
                raise
            parts.append(selected_scores)
            records.append(record)
        if not parts:
            raise ValueError(f"No TabPFN CV ensemble inputs were built for {disease}; missing={missing_messages}")
        train_long = pd.concat(parts, ignore_index=True)
        if train_long.duplicated([args.patient_col, "disease", "model"]).any():
            examples = train_long.loc[
                train_long.duplicated([args.patient_col, "disease", "model"]),
                [args.patient_col, "disease", "model"],
            ].head(5)
            raise ValueError(f"Duplicate selected ensemble input rows: {examples.to_dict('records')}")
        selected_models = pd.DataFrame(records)
        write_table(train_long, out_dir / f"{disease}.train_long.csv")
        write_table(selected_models, out_dir / f"{disease}.selected_single_models.csv")
        all_records.extend(records)
        print(f"Wrote {len(records)} selected CV PRS families for {disease} to {out_dir / f'{disease}.train_long.csv'}")
        for message in missing_messages:
            print(f"Skipped missing family for {disease}: {message}")

    if all_records:
        write_table(pd.DataFrame(all_records), out_dir / "selected_single_models.csv")


if __name__ == "__main__":
    main()
