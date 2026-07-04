from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import require_columns

DEFAULT_METRICS = (
    "auc",
    "auprc",
    "brier",
    "or_per_sd",
    "top10_vs_lower90_or",
    "nagelkerke_r2",
    "top_vs_bottom",
    "top_vs_middle",
    "bottom_vs_middle",
)
LOWER_IS_BETTER = {"brier"}
METRIC_FILE_NAMES = ("summary_metrics.csv", "point_metrics.csv")


@dataclass(frozen=True)
class PerformanceSpec:
    model_id: str
    path: Path
    model_group: str = ""
    setting: str = ""
    strategy: str = ""
    disease: str = ""
    method: str = ""
    method_template: str = ""
    metrics_kind: str = "auto"
    extra: dict[str, Any] | None = None


def metric_direction(metric: str) -> str:
    metric_name = metric.removesuffix("_mean").removesuffix("_std").removesuffix("_sd")
    return "min" if metric_name.lower() in LOWER_IS_BETTER else "max"


def resolve_metric_column(metrics: pd.DataFrame, requested: str) -> str:
    if requested in metrics.columns:
        return requested
    if requested.endswith("_mean"):
        base = requested.removesuffix("_mean")
        if base in metrics.columns:
            return base
    mean_name = f"{requested}_mean"
    if mean_name in metrics.columns:
        return mean_name
    raise ValueError(f"Metric {requested!r} was not found in columns: {metrics.columns.tolist()}")


def _std_column(metrics: pd.DataFrame, metric: str) -> str | None:
    for suffix in ("_std", "_sd"):
        col = f"{metric}{suffix}"
        if col in metrics.columns:
            return col
    return None


def _clean_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _metrics_path(validation_dir: Path, metrics_kind: str) -> Path | None:
    summary = validation_dir / "summary_metrics.csv"
    point = validation_dir / "point_metrics.csv"
    if metrics_kind == "summary":
        return summary if summary.exists() else None
    if metrics_kind == "point":
        return point if point.exists() else None
    if summary.exists():
        return summary
    if point.exists():
        return point
    return None


def find_metric_files(path: Path, metrics_kind: str = "auto") -> list[Path]:
    path = Path(path)
    if path.is_file():
        if path.name not in METRIC_FILE_NAMES:
            raise ValueError(f"Metrics file must be named one of {METRIC_FILE_NAMES}: {path}")
        return [path]
    direct = _metrics_path(path, metrics_kind)
    if direct is not None:
        return [direct]

    validation_dirs = {
        candidate.parent
        for name in METRIC_FILE_NAMES
        for candidate in path.rglob(name)
        if candidate.is_file()
    }
    files = []
    for validation_dir in sorted(validation_dirs):
        metrics_path = _metrics_path(validation_dir, metrics_kind)
        if metrics_path is not None:
            files.append(metrics_path)
    return files


def _infer_strategy(validation_dir: Path) -> str:
    if validation_dir.name.startswith("official_validation_"):
        return validation_dir.name.removeprefix("official_validation_")
    return ""


def infer_spec_from_model_arg(raw: str, *, metrics_kind: str = "auto") -> PerformanceSpec:
    if "=" not in raw:
        raise ValueError("--model entries must use MODEL_ID=PATH")
    model_id, raw_path = raw.split("=", maxsplit=1)
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("--model MODEL_ID cannot be empty")
    return PerformanceSpec(model_id=model_id, path=Path(raw_path), metrics_kind=metrics_kind)


def read_manifest(path: Path, *, default_metrics_kind: str = "auto") -> list[PerformanceSpec]:
    manifest = read_table(path)
    require_columns(manifest, ["model_id", "path"], str(path))
    reserved = {
        "model_id",
        "path",
        "model_group",
        "setting",
        "strategy",
        "disease",
        "method",
        "method_template",
        "metrics_kind",
    }
    specs = []
    for row in manifest.to_dict(orient="records"):
        extra = {key: value for key, value in row.items() if key not in reserved}
        specs.append(
            PerformanceSpec(
                model_id=_clean_optional(row["model_id"]),
                path=Path(_clean_optional(row["path"])),
                model_group=_clean_optional(row.get("model_group")),
                setting=_clean_optional(row.get("setting")),
                strategy=_clean_optional(row.get("strategy")),
                disease=_clean_optional(row.get("disease")),
                method=_clean_optional(row.get("method")),
                method_template=_clean_optional(row.get("method_template")),
                metrics_kind=_clean_optional(row.get("metrics_kind")) or default_metrics_kind,
                extra=extra,
            )
        )
    return specs


def _expected_method(disease: str, spec: PerformanceSpec, default_method_template: str) -> str:
    if spec.method:
        return spec.method
    template = spec.method_template or default_method_template
    if not template:
        return ""
    return template.format(disease=disease)


def _collect_metrics_file(
    *,
    metrics_path: Path,
    spec: PerformanceSpec,
    default_method_template: str,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    frame = read_table(metrics_path)
    require_columns(frame, ["disease", "method"], str(metrics_path))
    if spec.disease:
        frame = frame.loc[frame["disease"].astype(str) == spec.disease].copy()
    if frame.empty:
        return frame

    if spec.method or spec.method_template or default_method_template:
        masks = []
        for disease in frame["disease"].astype(str).drop_duplicates():
            expected = _expected_method(disease, spec, default_method_template)
            if expected:
                masks.append(
                    (frame["disease"].astype(str) == disease)
                    & (frame["method"].astype(str) == expected)
                )
        if masks:
            mask = masks[0]
            for next_mask in masks[1:]:
                mask = mask | next_mask
            frame = frame.loc[mask].copy()
        else:
            frame = frame.iloc[0:0].copy()
    if frame.empty:
        return frame

    validation_dir = metrics_path.parent
    metadata = {
        "model_id": spec.model_id,
        "model_group": spec.model_group,
        "setting": spec.setting,
        "strategy": spec.strategy or _infer_strategy(validation_dir),
        "metrics_kind": "summary" if metrics_path.name == "summary_metrics.csv" else "point",
        "metrics_file": str(metrics_path),
        "validation_dir": str(validation_dir),
    }
    for key, value in metadata.items():
        frame.insert(0, key, value)
    if spec.extra:
        for key, value in spec.extra.items():
            if key not in frame.columns:
                frame[key] = value

    for metric in metrics:
        try:
            metric_col = resolve_metric_column(frame, metric)
        except ValueError:
            continue
        frame[metric] = pd.to_numeric(frame[metric_col], errors="coerce")
        std_col = _std_column(frame, metric)
        if std_col is not None:
            frame[f"{metric}_std"] = pd.to_numeric(frame[std_col], errors="coerce")
    return frame


def collect_performance(
    specs: list[PerformanceSpec],
    *,
    default_method_template: str = "prs_covariate_{disease}",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    frames = []
    for spec in specs:
        for metrics_path in find_metric_files(spec.path, spec.metrics_kind):
            frame = _collect_metrics_file(
                metrics_path=metrics_path,
                spec=spec,
                default_method_template=default_method_template,
                metrics=metrics,
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise ValueError("No matching performance rows were found")
    return pd.concat(frames, ignore_index=True)


def add_metric_ranks(
    performance: pd.DataFrame,
    *,
    rank_metric: str,
    direction: str = "auto",
) -> pd.DataFrame:
    ranked = performance.copy()
    metric_col = resolve_metric_column(ranked, rank_metric)
    selected_direction = metric_direction(metric_col) if direction == "auto" else direction
    ranked[metric_col] = pd.to_numeric(ranked[metric_col], errors="coerce")
    ranked = ranked.sort_values(
        by=["disease", metric_col, "model_id", "setting"],
        ascending=[True, selected_direction == "min", True, True],
        kind="mergesort",
    )
    ranked["rank_metric"] = metric_col
    ranked["rank_direction"] = selected_direction
    ranked["rank"] = ranked.groupby("disease", sort=False)[metric_col].rank(
        method="first",
        ascending=(selected_direction == "min"),
    )
    return ranked.sort_values(["disease", "rank"], kind="mergesort")


def add_baseline_deltas(
    performance: pd.DataFrame,
    *,
    baseline_model_id: str,
    baseline_setting: str = "",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    baseline = performance.loc[performance["model_id"].astype(str) == baseline_model_id].copy()
    if baseline_setting:
        baseline = baseline.loc[baseline["setting"].astype(str) == baseline_setting].copy()
    if baseline.empty:
        raise ValueError(f"Baseline model_id={baseline_model_id!r} was not found")
    duplicated = baseline["disease"].duplicated()
    if duplicated.any():
        examples = baseline.loc[duplicated, ["disease", "method", "setting"]].to_dict(orient="records")
        raise ValueError(f"Baseline must have one row per disease; duplicated rows include {examples}")

    baseline_cols = ["disease", *[metric for metric in metrics if metric in baseline.columns]]
    baseline = baseline[baseline_cols].rename(
        columns={metric: f"baseline_{metric}" for metric in baseline_cols if metric != "disease"}
    )
    compared = performance.merge(baseline, on="disease", how="left")
    for metric in metrics:
        baseline_col = f"baseline_{metric}"
        if metric in compared.columns and baseline_col in compared.columns:
            compared[f"delta_{metric}"] = compared[metric] - compared[baseline_col]
    return compared


def wide_performance_table(
    performance: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    index_cols = [
        col
        for col in ("model_id", "model_group", "setting", "strategy", "method", "metrics_kind")
        if col in performance.columns
    ]
    value_cols = [metric for metric in metrics if metric in performance.columns]
    if not value_cols:
        return pd.DataFrame(columns=index_cols)
    melted = performance[index_cols + ["disease"] + value_cols].melt(
        id_vars=index_cols + ["disease"],
        value_vars=value_cols,
        var_name="metric",
        value_name="value",
    )
    melted = melted.loc[melted["value"].notna()].copy()
    melted["disease_metric"] = melted["disease"].astype(str) + "_" + melted["metric"].astype(str)
    wide = melted.pivot_table(
        index=index_cols,
        columns="disease_metric",
        values="value",
        aggfunc="first",
        sort=True,
    )
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def write_performance_outputs(
    performance: pd.DataFrame,
    *,
    out: Path,
    best_out: Path | None,
    wide_out: Path | None,
    rank_metric: str,
    rank_direction: str,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> tuple[Path, Path, Path | None]:
    ranked = add_metric_ranks(performance, rank_metric=rank_metric, direction=rank_direction)
    write_table(ranked, out)
    selected_best_out = best_out or out.with_name(f"{out.stem}.best.csv")
    best = ranked.groupby("disease", as_index=False, sort=False).head(1)
    write_table(best, selected_best_out)
    selected_wide_out = wide_out
    if selected_wide_out is not None:
        write_table(wide_performance_table(ranked, metrics=metrics), selected_wide_out)
    return out, selected_best_out, selected_wide_out


def parse_metric_list(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_METRICS
    parsed = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not parsed:
        raise ValueError("At least one metric is required")
    return parsed


def finite_metric_count(performance: pd.DataFrame, metric: str) -> int:
    if metric not in performance.columns:
        return 0
    values = pd.to_numeric(performance[metric], errors="coerce")
    return int(np.isfinite(values).sum())
