from __future__ import annotations

import argparse
from pathlib import Path

from decipehr.performance import (
    add_baseline_deltas,
    collect_performance,
    finite_metric_count,
    infer_spec_from_model_arg,
    parse_metric_list,
    read_manifest,
    write_performance_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect official validation metrics across PRS, ensemble, and TabPFN settings. "
            "Inputs can be explicit MODEL_ID=PATH entries or a manifest CSV/TSV."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help=(
            "Model label and metrics path/root. PATH may be a metrics CSV, an official_validation "
            "directory, or a directory scanned recursively for metrics files. Repeatable."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional CSV/TSV with columns model_id,path and optional model_group,setting,"
            "strategy,disease,method,method_template,metrics_kind plus arbitrary tags."
        ),
    )
    parser.add_argument(
        "--method-template",
        default="prs_covariate_{disease}",
        help=(
            "Default method to keep from each metrics file. Use an empty string to keep all methods. "
            "Manifest method or method_template overrides this per row."
        ),
    )
    parser.add_argument(
        "--metrics-kind",
        choices=["auto", "summary", "point"],
        default="auto",
        help="Choose summary_metrics.csv, point_metrics.csv, or summary if available.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metrics to normalize and compare. Defaults to common validation metrics.",
    )
    parser.add_argument("--rank-metric", default="auc", help="Metric used for ranks and best rows.")
    parser.add_argument("--rank-direction", choices=["auto", "max", "min"], default="auto")
    parser.add_argument(
        "--baseline-model-id",
        default=None,
        help="Optional model_id used to add delta_<metric> columns within each disease.",
    )
    parser.add_argument(
        "--baseline-setting",
        default="",
        help="Optional setting filter if the baseline model has multiple rows per disease.",
    )
    parser.add_argument("--out", default="outputs/performance/model_performance.csv")
    parser.add_argument("--best-out", default=None)
    parser.add_argument(
        "--wide-out",
        default=None,
        help="Optional wide output with one row per model/setting and disease_metric columns.",
    )
    args = parser.parse_args()

    metrics = parse_metric_list(args.metrics)
    specs = [infer_spec_from_model_arg(raw, metrics_kind=args.metrics_kind) for raw in args.model]
    if args.manifest:
        specs.extend(read_manifest(Path(args.manifest), default_metrics_kind=args.metrics_kind))
    if not specs:
        raise ValueError("Provide at least one --model MODEL_ID=PATH or --manifest")

    performance = collect_performance(
        specs,
        default_method_template=args.method_template,
        metrics=metrics,
    )
    if args.baseline_model_id:
        performance = add_baseline_deltas(
            performance,
            baseline_model_id=args.baseline_model_id,
            baseline_setting=args.baseline_setting,
            metrics=metrics,
        )

    if finite_metric_count(performance, args.rank_metric) == 0:
        raise ValueError(f"Rank metric {args.rank_metric!r} has no finite values")

    out_path, best_path, wide_path = write_performance_outputs(
        performance,
        out=Path(args.out),
        best_out=Path(args.best_out) if args.best_out else None,
        wide_out=Path(args.wide_out) if args.wide_out else None,
        rank_metric=args.rank_metric,
        rank_direction=args.rank_direction,
        metrics=metrics,
    )
    print(f"Wrote model performance rows to {out_path}")
    print(f"Wrote best rows to {best_path}")
    if wide_path is not None:
        print(f"Wrote wide comparison table to {wide_path}")


if __name__ == "__main__":
    main()
