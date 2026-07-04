from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns


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
    raise ValueError(f"Metric {requested!r} was not found in columns: {metrics.columns.tolist()}")


def _metric_direction(metric_col: str, requested: str) -> str:
    metric = metric_col.removesuffix("_mean").lower()
    requested_metric = requested.removesuffix("_mean").lower()
    if metric == "brier" or requested_metric == "brier":
        return "min"
    return "max"


def _read_manifest(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    manifest = read_table(path)
    require_columns(manifest, ["disease", "setting_id"], "ensemble grid manifest")
    return manifest


def _metrics_file(validation_dir: Path, metrics_kind: str) -> Path | None:
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


def _collect_one(
    *,
    metrics_path: Path,
    disease: str,
    method_template: str,
    setting_id: str,
    strategy: str,
) -> pd.DataFrame:
    metrics = read_table(metrics_path)
    require_columns(metrics, ["disease", "method"], str(metrics_path))
    expected_method = method_template.format(disease=disease)
    rows = metrics.loc[
        (metrics["disease"].astype(str) == disease) & (metrics["method"].astype(str) == expected_method)
    ].copy()
    if rows.empty:
        return rows
    rows.insert(0, "strategy", strategy)
    rows.insert(0, "setting_id", setting_id)
    rows.insert(0, "grid_disease", disease)
    rows["metrics_file"] = str(metrics_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and rank official validation results from ensemble grid runs."
    )
    parser.add_argument("--grid-root", default="outputs/ensemble_grid")
    parser.add_argument("--disease", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--metrics-kind", choices=["auto", "summary", "point"], default="auto")
    parser.add_argument("--selection-metric", default="top10_vs_lower90_or_mean")
    parser.add_argument("--selection-direction", choices=["auto", "max", "min"], default="auto")
    parser.add_argument("--selection-method", default="prs_covariate_{disease}")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if bool(args.disease) and bool(args.labels):
        raise ValueError("Use --disease or --labels, not both")
    diseases = [args.disease] if args.disease else parse_csv_list(args.labels)
    grid_root = Path(args.grid_root)
    if not diseases:
        diseases = sorted(path.name for path in grid_root.iterdir() if path.is_dir())

    frames = []
    for disease in diseases:
        disease_dir = grid_root / disease
        if not disease_dir.exists():
            continue
        for validation_dir in sorted(disease_dir.glob("*/official_validation_*")):
            if not validation_dir.is_dir():
                continue
            metrics_path = _metrics_file(validation_dir, args.metrics_kind)
            if metrics_path is None:
                continue
            setting_id = validation_dir.parent.name
            strategy = validation_dir.name.removeprefix("official_validation_")
            frame = _collect_one(
                metrics_path=metrics_path,
                disease=disease,
                method_template=args.selection_method,
                setting_id=setting_id,
                strategy=strategy,
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise ValueError("No ensemble grid validation metrics were found")
    summary = pd.concat(frames, ignore_index=True)
    manifest = _read_manifest(args.manifest)
    if manifest is not None:
        summary = summary.merge(
            manifest.drop(columns=["command"], errors="ignore"),
            on=["disease", "setting_id"],
            how="left",
            suffixes=("", "_manifest"),
        )
    metric_col = _resolve_metric_column(summary, args.selection_metric)
    direction = _metric_direction(metric_col, args.selection_metric) if args.selection_direction == "auto" else args.selection_direction
    summary[metric_col] = pd.to_numeric(summary[metric_col], errors="coerce")
    ranked = summary.loc[summary[metric_col].notna()].sort_values(
        by=metric_col,
        ascending=(direction == "min"),
        kind="mergesort",
    )
    out_path = Path(args.out) if args.out else grid_root / "ensemble_grid_summary.csv"
    best_path = out_path.with_name(f"{out_path.stem}.best.csv")
    write_table(ranked, out_path)
    best = ranked.groupby("grid_disease", as_index=False, sort=False).head(1)
    write_table(best, best_path)
    print(f"Wrote ranked ensemble grid results to {out_path}")
    print(f"Wrote best ensemble grid rows to {best_path}")


if __name__ == "__main__":
    main()
