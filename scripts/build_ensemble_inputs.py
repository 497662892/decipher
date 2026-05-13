from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns, validate_unique


@dataclass(frozen=True)
class FamilySpec:
    name: str
    validation_root: str
    score_root: str
    train_scores_template: str
    validation_scores_template: str
    param_cols: tuple[str, ...]


DEFAULT_FAMILIES: dict[str, FamilySpec] = {
    "pt": FamilySpec(
        name="pt",
        validation_root="outputs/official_validation_pt",
        score_root="outputs/pt_full_train",
        train_scores_template="{disease}/scores/{disease}.pt_full_train_training_scores_long.csv",
        validation_scores_template="{disease}/scores/{disease}.pt_full_train_scores_long.csv",
        param_cols=("clump_r2", "p_threshold"),
    ),
    "ldpred2": FamilySpec(
        name="ldpred2",
        validation_root="outputs/official_validation_ldpred2_ukbb",
        score_root="outputs/ldpred2_ukbb_full_train",
        train_scores_template="{disease}/scores/{disease}.ldpred2_full_train_training_scores_long.csv",
        validation_scores_template="{disease}/scores/{disease}.ldpred2_full_train_scores_long.csv",
        param_cols=("h2", "p", "sparse"),
    ),
    "lassosum2": FamilySpec(
        name="lassosum2",
        validation_root="outputs/official_validation_lassosum2_ukbb",
        score_root="outputs/lassosum2_ukbb_full_train",
        train_scores_template="{disease}/scores/{disease}.lassosum2_full_train_training_scores_long.csv",
        validation_scores_template="{disease}/scores/{disease}.lassosum2_full_train_scores_long.csv",
        param_cols=("delta", "lambda"),
    ),
    "prscs": FamilySpec(
        name="prscs",
        validation_root="outputs/official_validation_prscs_ukbb_eur",
        score_root="outputs/prscs_full_train_ukbb_eur",
        train_scores_template="{disease}/scores/{disease}.prscs_full_train_training_scores_long.csv",
        validation_scores_template="{disease}/scores/{disease}.prscs_full_train_scores_long.csv",
        param_cols=("phi",),
    ),
    "sdpr": FamilySpec(
        name="sdpr",
        validation_root="outputs/official_validation_sdpr",
        score_root="outputs/sdpr_full_train",
        train_scores_template="{disease}/scores/{disease}.sdpr_full_train_training_scores_long.csv",
        validation_scores_template="{disease}/scores/{disease}.sdpr_full_train_scores_long.csv",
        param_cols=("model",),
    ),
    "sbayesrc": FamilySpec(
        name="sbayesrc",
        validation_root="outputs/official_validation_sbayesrc",
        score_root="outputs/sbayesrc_full_train",
        train_scores_template="{disease}/scores/{disease}.sbayesrc_full_train_training_scores_long.csv",
        validation_scores_template="{disease}/scores/{disease}.sbayesrc_full_train_scores_long.csv",
        param_cols=("model",),
    ),
    "megaprs": FamilySpec(
        name="megaprs",
        validation_root="outputs/official_validation_megaprs",
        score_root="outputs/megaprs_full_train",
        train_scores_template="{disease}/scores/{disease}.megaprs_full_train_training_scores_long.csv",
        validation_scores_template="{disease}/scores/{disease}.megaprs_full_train_scores_long.csv",
        param_cols=("model",),
    ),
}


def _format_template(template: str, disease: str) -> str:
    return template.format(disease=disease)


def _path(root: str | Path, template: str, disease: str) -> Path:
    return Path(root) / _format_template(template, disease)


def _metrics_path(spec: FamilySpec, disease: str, metrics_kind: str) -> Path | None:
    root = Path(spec.validation_root) / disease
    summary = root / "summary_metrics.csv"
    point = root / "point_metrics.csv"
    if metrics_kind == "summary":
        return summary if summary.exists() else None
    if metrics_kind == "point":
        return point if point.exists() else None
    if summary.exists():
        return summary
    if point.exists():
        return point
    return None


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
    method_template: str,
) -> pd.DataFrame:
    require_columns(metrics, ["disease", "method"], "validation metrics")
    expected_method = method_template.format(disease=disease)
    filtered = metrics.loc[
        (metrics["disease"].astype(str) == disease) & (metrics["method"].astype(str) == expected_method)
    ].copy()
    if filtered.empty:
        raise ValueError(f"No validation metrics rows for disease={disease!r}, method={expected_method!r}")
    return filtered


def _sort_key_frame(frame: pd.DataFrame, metric_col: str, param_cols: tuple[str, ...]) -> pd.DataFrame:
    sort_cols = [metric_col, *[col for col in param_cols if col in frame.columns]]
    return frame[sort_cols].copy()


def _select_best_row(
    metrics: pd.DataFrame,
    *,
    disease: str,
    method_template: str,
    metric: str,
    direction: str,
    param_cols: tuple[str, ...],
) -> tuple[pd.Series, str, str]:
    filtered = _filter_metrics(metrics, disease=disease, method_template=method_template)
    metric_col = _resolve_metric_column(filtered, metric)
    selected_direction = _metric_direction(metric_col, metric) if direction == "auto" else direction
    filtered[metric_col] = pd.to_numeric(filtered[metric_col], errors="coerce")
    filtered = filtered.loc[filtered[metric_col].notna()].copy()
    if filtered.empty:
        raise ValueError(f"No finite values for selection metric {metric_col!r}")
    ascending = selected_direction == "min"
    sort_keys = _sort_key_frame(filtered, metric_col, param_cols)
    filtered = filtered.assign(__sort_metric=sort_keys[metric_col])
    filtered = filtered.sort_values(
        by=["__sort_metric", *[col for col in param_cols if col in filtered.columns]],
        ascending=[ascending, *([True] * len([col for col in param_cols if col in filtered.columns]))],
        kind="mergesort",
    )
    return filtered.iloc[0].drop(labels=["__sort_metric"]), metric_col, selected_direction


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
    selected_string = _normal_string(selected_value)
    return series.map(_normal_string) == selected_string


def _select_score_rows(
    scores: pd.DataFrame,
    *,
    selected: pd.Series,
    family: str,
    disease: str,
    patient_col: str,
    disease_col: str,
    score_col: str,
    param_cols: tuple[str, ...],
    context: str,
) -> pd.DataFrame:
    require_columns(scores, [patient_col, disease_col, score_col], context)
    missing_params = [col for col in param_cols if col not in scores.columns]
    if missing_params:
        raise ValueError(f"{context} is missing selected parameter columns: {missing_params}")
    mask = scores[disease_col].astype(str) == disease
    for column in param_cols:
        if column not in selected.index or pd.isna(selected[column]):
            raise ValueError(f"Selected {family} metrics row has no value for parameter {column!r}")
        mask &= _match_param(scores[column], selected[column])
    matched = scores.loc[mask].copy()
    if matched.empty:
        selected_params = {col: selected[col] for col in param_cols}
        raise ValueError(f"No {context} rows matched {family} selected parameters: {selected_params}")
    validate_unique(matched, patient_col, f"{context} matched rows for {family}")
    matched[score_col] = pd.to_numeric(matched[score_col], errors="coerce")
    if matched[score_col].isna().any():
        examples = matched.loc[matched[score_col].isna(), [patient_col, score_col]].head(5).to_dict("records")
        raise ValueError(f"{context} has nonnumeric PRS values after selection: {examples}")

    output = matched[[patient_col, disease_col, score_col]].copy()
    output = output.rename(columns={disease_col: "disease", score_col: "prs"})
    if "model" in matched.columns:
        output["source_model"] = matched["model"].astype(str).to_numpy()
    else:
        output["source_model"] = family
    output["model"] = family
    output["family"] = family
    for column in param_cols:
        output[column] = matched[column].to_numpy()
    if "source_score_file" in matched.columns:
        output["source_score_file"] = matched["source_score_file"].to_numpy()
    if "sscore" in matched.columns:
        output["source_score_file"] = matched["sscore"].to_numpy()
    return output


def _selection_record(
    *,
    family: str,
    disease: str,
    selected: pd.Series,
    metric_col: str,
    direction: str,
    metrics_path: Path,
    train_path: Path,
    validation_path: Path,
    param_cols: tuple[str, ...],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "disease": disease,
        "family": family,
        "selection_metric": metric_col,
        "selection_direction": direction,
        "selection_metric_value": selected[metric_col],
        "metrics_file": str(metrics_path),
        "train_score_file": str(train_path),
        "validation_score_file": str(validation_path),
    }
    for column in param_cols:
        record[column] = selected[column] if column in selected.index else np.nan
    for column in [
        "auc_mean",
        "auprc_mean",
        "brier_mean",
        "or_per_sd_mean",
        "top10_vs_lower90_or_mean",
        "nagelkerke_r2_mean",
        "auc",
        "auprc",
        "brier",
        "or_per_sd",
        "top10_vs_lower90_or",
        "nagelkerke_r2",
    ]:
        if column in selected.index:
            record[column] = selected[column]
    return record


def _process_family(
    *,
    spec: FamilySpec,
    disease: str,
    metric: str,
    direction: str,
    metrics_kind: str,
    method_template: str,
    patient_col: str,
    disease_col: str,
    score_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics_path = _metrics_path(spec, disease, metrics_kind)
    if metrics_path is None:
        raise FileNotFoundError(f"No metrics file found for {spec.name} {disease}")
    train_path = _path(spec.score_root, spec.train_scores_template, disease)
    validation_path = _path(spec.score_root, spec.validation_scores_template, disease)
    missing_paths = [path for path in [train_path, validation_path] if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing score files for {spec.name} {disease}: {missing_paths}")

    metrics = read_table(metrics_path)
    selected, metric_col, selected_direction = _select_best_row(
        metrics,
        disease=disease,
        method_template=method_template,
        metric=metric,
        direction=direction,
        param_cols=spec.param_cols,
    )
    train_scores = read_table(train_path)
    validation_scores = read_table(validation_path)
    train_selected = _select_score_rows(
        train_scores,
        selected=selected,
        family=spec.name,
        disease=disease,
        patient_col=patient_col,
        disease_col=disease_col,
        score_col=score_col,
        param_cols=spec.param_cols,
        context=f"{spec.name} training scores",
    )
    validation_selected = _select_score_rows(
        validation_scores,
        selected=selected,
        family=spec.name,
        disease=disease,
        patient_col=patient_col,
        disease_col=disease_col,
        score_col=score_col,
        param_cols=spec.param_cols,
        context=f"{spec.name} validation scores",
    )
    record = _selection_record(
        family=spec.name,
        disease=disease,
        selected=selected,
        metric_col=metric_col,
        direction=selected_direction,
        metrics_path=metrics_path,
        train_path=train_path,
        validation_path=validation_path,
        param_cols=spec.param_cols,
    )
    train_selected["selection_metric"] = metric_col
    train_selected["selection_metric_value"] = selected[metric_col]
    validation_selected["selection_metric"] = metric_col
    validation_selected["selection_metric_value"] = selected[metric_col]
    return train_selected, validation_selected, record


def _apply_overrides(args: argparse.Namespace, specs: dict[str, FamilySpec]) -> dict[str, FamilySpec]:
    overrides = {
        "pt": (args.pt_validation_root, args.pt_score_root),
        "ldpred2": (args.ldpred2_validation_root, args.ldpred2_score_root),
        "lassosum2": (args.lassosum2_validation_root, args.lassosum2_score_root),
        "prscs": (args.prscs_validation_root, args.prscs_score_root),
        "sdpr": (args.sdpr_validation_root, args.sdpr_score_root),
        "sbayesrc": (args.sbayesrc_validation_root, args.sbayesrc_score_root),
        "megaprs": (args.megaprs_validation_root, args.megaprs_score_root),
    }
    updated = {}
    for name, spec in specs.items():
        validation_root, score_root = overrides[name]
        updated[name] = FamilySpec(
            name=spec.name,
            validation_root=validation_root or spec.validation_root,
            score_root=score_root or spec.score_root,
            train_scores_template=spec.train_scores_template,
            validation_scores_template=spec.validation_scores_template,
            param_cols=spec.param_cols,
        )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best validation-tuned single PRS from each non-annotated family "
            "and combine the selected predictions into long ensemble input files."
        )
    )
    parser.add_argument("--disease", default=None, help="Single disease to build.")
    parser.add_argument("--labels", default=None, help="Comma-separated diseases to build.")
    parser.add_argument("--families", default="pt,ldpred2,lassosum2,prscs,sdpr")
    parser.add_argument(
        "--selection-metric",
        default="top10_vs_lower90_or_mean",
        help="Metric used to choose the winner within each family.",
    )
    parser.add_argument("--selection-direction", choices=["auto", "max", "min"], default="auto")
    parser.add_argument("--metrics-kind", choices=["auto", "summary", "point"], default="auto")
    parser.add_argument(
        "--selection-method",
        default="prs_covariate_{disease}",
        help="Validation method row to select from; {disease} is expanded.",
    )
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--score-col", default="prs")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pt-validation-root", default=None)
    parser.add_argument("--pt-score-root", default=None)
    parser.add_argument("--ldpred2-validation-root", default=None)
    parser.add_argument("--ldpred2-score-root", default=None)
    parser.add_argument("--lassosum2-validation-root", default=None)
    parser.add_argument("--lassosum2-score-root", default=None)
    parser.add_argument("--prscs-validation-root", default=None)
    parser.add_argument("--prscs-score-root", default=None)
    parser.add_argument("--sdpr-validation-root", default=None)
    parser.add_argument("--sdpr-score-root", default=None)
    parser.add_argument("--sbayesrc-validation-root", default=None)
    parser.add_argument("--sbayesrc-score-root", default=None)
    parser.add_argument("--megaprs-validation-root", default=None)
    parser.add_argument("--megaprs-score-root", default=None)
    args = parser.parse_args()

    if bool(args.disease) == bool(args.labels):
        raise ValueError("Use exactly one of --disease or --labels")
    diseases = [args.disease] if args.disease else parse_csv_list(args.labels)
    families = parse_csv_list(args.families)
    unknown = [family for family in families if family not in DEFAULT_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown PRS families: {unknown}; expected one of {sorted(DEFAULT_FAMILIES)}")
    specs = _apply_overrides(args, DEFAULT_FAMILIES)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for disease in diseases:
        train_parts = []
        validation_parts = []
        disease_records = []
        missing_messages = []
        for family in families:
            try:
                train_selected, validation_selected, record = _process_family(
                    spec=specs[family],
                    disease=disease,
                    metric=args.selection_metric,
                    direction=args.selection_direction,
                    metrics_kind=args.metrics_kind,
                    method_template=args.selection_method,
                    patient_col=args.patient_col,
                    disease_col=args.disease_col,
                    score_col=args.score_col,
                )
            except (FileNotFoundError, ValueError) as exc:
                if args.allow_missing:
                    missing_messages.append(f"{family}: {exc}")
                    continue
                raise
            train_parts.append(train_selected)
            validation_parts.append(validation_selected)
            disease_records.append(record)
        if not train_parts or not validation_parts:
            raise ValueError(f"No ensemble inputs were built for {disease}; missing={missing_messages}")
        train_long = pd.concat(train_parts, ignore_index=True)
        validation_long = pd.concat(validation_parts, ignore_index=True)
        selected_models = pd.DataFrame(disease_records)
        write_table(train_long, out_dir / f"{disease}.train_long.csv")
        write_table(validation_long, out_dir / f"{disease}.validation_long.csv")
        write_table(selected_models, out_dir / f"{disease}.selected_single_models.csv")
        all_records.extend(disease_records)
        print(
            f"Wrote {len(disease_records)} selected PRS families for {disease} to "
            f"{out_dir / f'{disease}.train_long.csv'} and {out_dir / f'{disease}.validation_long.csv'}"
        )
        for message in missing_messages:
            print(f"Skipped missing family for {disease}: {message}")
    if all_records:
        write_table(pd.DataFrame(all_records), out_dir / "selected_single_models.csv")


if __name__ == "__main__":
    main()
