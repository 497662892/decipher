from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.ancestry_normalization import (
    infer_score_columns,
    normalize_by_hard_label,
    normalize_by_soft_weights,
)
from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns, validate_unique


@dataclass(frozen=True)
class ModelSpec:
    family: str
    disease: str
    strategy: str
    train_prs: Path
    validation_prs: Path
    raw_metrics: Path
    raw_metric_row: dict[str, object] | None = None
    selected_long_scores: bool = False


POINT_METRIC_COLUMNS = {
    "auc",
    "auprc",
    "brier",
    "prevalence",
    "n",
    "or_per_sd",
    "top10_vs_lower90_or",
    "top10_n",
    "lower90_n",
    "top10_cases",
    "lower90_cases",
    "nagelkerke_r2",
    "top_vs_bottom",
    "top_vs_middle",
    "bottom_vs_middle",
}

SELECTION_PARAMETER_COLUMNS = (
    "clump_r2",
    "p_threshold",
    "h2",
    "p",
    "sparse",
    "delta",
    "lambda",
    "phi",
    "model",
)


def _read_existing(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _metrics_path(path: str | Path, metrics_kind: str) -> Path:
    path = Path(path)
    if metrics_kind == "point" and path.name == "summary_metrics.csv":
        return path.with_name("point_metrics.csv")
    return path


def _point_row_as_summary_style(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    for metric in POINT_METRIC_COLUMNS:
        if metric in out and f"{metric}_mean" not in out:
            out[f"{metric}_mean"] = out[metric]
    return out


def _ensemble_specs(path: str | Path, family: str, metrics_kind: str) -> list[ModelSpec]:
    df = _read_existing(path)
    specs = []
    for row in df.to_dict("records"):
        out_dir = Path(str(row["out_dir"]))
        strategy = str(row["strategy"])
        specs.append(
            ModelSpec(
                family=family,
                disease=str(row["disease"]),
                strategy=strategy,
                train_prs=out_dir / strategy / "train_prs_scores.csv",
                validation_prs=out_dir / strategy / "validation_prs_scores.csv",
                raw_metrics=_metrics_path(str(row["metrics_file"]), metrics_kind),
            )
        )
    return specs


def _selected_point_metric_row(row: dict[str, object], metrics_path: Path) -> dict[str, object] | None:
    metrics = _read_existing(metrics_path)
    if metrics.empty:
        return None
    disease = str(row["disease"])
    if "disease" in metrics.columns:
        metrics = metrics.loc[metrics["disease"].astype(str) == disease].copy()
    if "method" in metrics.columns:
        expected = f"prs_covariate_{disease}"
        metrics = metrics.loc[metrics["method"].astype(str) == expected].copy()
    if "model" in metrics.columns and not _is_missing(row.get("source_model")):
        metrics = metrics.loc[_series_matches(metrics["model"], row["source_model"])].copy()

    for column in SELECTION_PARAMETER_COLUMNS:
        if column == "model" and not _is_missing(row.get("source_model")):
            continue
        if column not in metrics.columns or column not in row or _is_missing(row.get(column)):
            continue
        metrics = metrics.loc[_series_matches(metrics[column], row[column])].copy()

    if metrics.empty:
        return None
    return metrics.iloc[0].to_dict()


def _raw_metric_row(row: dict[str, object], raw_metrics: Path, metrics_kind: str) -> dict[str, object]:
    if metrics_kind == "point":
        point_row = _selected_point_metric_row(row, raw_metrics)
        if point_row is not None:
            return point_row
        print(f"Falling back to selected summary row; no matching point metric row in {raw_metrics}")
    return dict(row)


def _single_best_specs(path: str | Path, metrics_kind: str) -> list[ModelSpec]:
    df = _read_existing(path)
    require_columns(
        df,
        ["disease", "family", "metrics_file", "train_score_file", "validation_score_file"],
        "selected single-model manifest",
    )
    specs = []
    for row in df.to_dict("records"):
        family = str(row["family"])
        disease = str(row["disease"])
        raw_metrics = _metrics_path(str(row["metrics_file"]), metrics_kind)
        raw_metric_row = _raw_metric_row(row, raw_metrics, metrics_kind)
        raw_metric_row.update(
            {
                "family": family,
                "disease": disease,
                "strategy": "single_best",
                "normalization": "raw",
                "metrics_file": str(raw_metrics),
            }
        )
        specs.append(
            ModelSpec(
                family=family,
                disease=disease,
                strategy="single_best",
                train_prs=Path(str(row["train_score_file"])),
                validation_prs=Path(str(row["validation_score_file"])),
                raw_metrics=raw_metrics,
                raw_metric_row=raw_metric_row,
                selected_long_scores=True,
            )
        )
    return specs


def _tabpfn_specs(path: str | Path, root: str | Path, family: str, metrics_kind: str) -> list[ModelSpec]:
    df = _read_existing(path)
    specs = []
    root = Path(root)
    for row in df.to_dict("records"):
        disease = str(row["disease"])
        setting = str(row["setting"])
        strategy = str(row["strategy"])
        setting_dir = root / setting
        strategy_dir = setting_dir / strategy
        specs.append(
            ModelSpec(
                family=family,
                disease=disease,
                strategy=strategy,
                train_prs=strategy_dir / "train_prs_scores.csv",
                validation_prs=strategy_dir / "validation_prs_scores.csv",
                raw_metrics=_metrics_path(
                    setting_dir / f"official_validation_{strategy}" / "summary_metrics.csv",
                    metrics_kind,
                ),
            )
        )
    return specs


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _series_matches(series: pd.Series, value: object) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value) and numeric_series.notna().any():
        return np.isclose(numeric_series.to_numpy(dtype=float), float(numeric_value), rtol=1e-8, atol=1e-12)
    return series.astype("string").str.strip() == str(value).strip()


def _filter_selected_long_scores(
    path: Path,
    *,
    spec: ModelSpec,
    patient_col: str,
    context: str,
) -> pd.DataFrame:
    if spec.raw_metric_row is None:
        raise ValueError("Selected long-score filtering requires raw_metric_row")
    frame = read_table(path)
    require_columns(frame, [patient_col, "prs"], context)
    if "disease" in frame.columns:
        frame = frame.loc[frame["disease"].astype(str) == spec.disease].copy()

    row = spec.raw_metric_row
    if "model" in frame.columns and not _is_missing(row.get("source_model")):
        frame = frame.loc[_series_matches(frame["model"], row["source_model"])].copy()

    for column in ["clump_r2", "p_threshold", "h2", "p", "sparse", "delta", "lambda", "phi", "model"]:
        if column not in frame.columns or column not in row or _is_missing(row.get(column)):
            continue
        if column == "model" and not _is_missing(row.get("source_model")):
            continue
        frame = frame.loc[_series_matches(frame[column], row[column])].copy()

    if frame.empty:
        raise ValueError(f"No rows remain after selected-model filtering for {context}: {path}")
    validate_unique(frame, patient_col, context)
    return frame[[patient_col, "prs"]].rename(columns={"prs": f"prs_{spec.disease}"})


def _load_raw_score_frames(
    spec: ModelSpec,
    *,
    spec_dir: Path,
    patient_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    if not spec.selected_long_scores:
        return read_table(spec.train_prs), read_table(spec.validation_prs), spec.train_prs, spec.validation_prs

    train = _filter_selected_long_scores(
        spec.train_prs,
        spec=spec,
        patient_col=patient_col,
        context=f"{spec.family}/{spec.disease} selected training scores",
    )
    validation = _filter_selected_long_scores(
        spec.validation_prs,
        spec=spec,
        patient_col=patient_col,
        context=f"{spec.family}/{spec.disease} selected validation scores",
    )
    train_path = spec_dir / "raw.train_prs_scores.csv"
    validation_path = spec_dir / "raw.validation_prs_scores.csv"
    write_table(train, train_path)
    write_table(validation, validation_path)
    return train, validation, train_path, validation_path


def _drop_scores_without_ancestry(
    scores: pd.DataFrame,
    ancestry: pd.DataFrame,
    *,
    patient_col: str,
    context: str,
) -> pd.DataFrame:
    require_columns(scores, [patient_col], f"{context} scores")
    require_columns(ancestry, [patient_col], f"{context} ancestry")
    keep_ids = set(ancestry[patient_col].astype(str))
    observed = scores[patient_col].astype(str)
    keep = observed.isin(keep_ids)
    if keep.all():
        return scores
    examples = observed.loc[~keep].head(5).tolist()
    print(f"Dropping {int((~keep).sum())} {context} score rows without 1000G ancestry: {examples}")
    return scores.loc[keep].reset_index(drop=True)


def _load_patient_frame(path: str | Path, patient_col: str) -> pd.DataFrame:
    frame = read_table(path)
    if patient_col not in frame.columns and "IID" in frame.columns:
        frame = frame.rename(columns={"IID": patient_col})
    return frame


def _metric_row(
    path: Path,
    spec: ModelSpec,
    normalization: str,
    metrics_kind: str,
) -> dict[str, object] | None:
    if normalization == "raw" and spec.raw_metric_row is not None and path == spec.raw_metrics:
        row = dict(spec.raw_metric_row)
        return _point_row_as_summary_style(row) if metrics_kind == "point" else row
    if not path.exists():
        return None
    metrics = pd.read_csv(path)
    if "disease" in metrics.columns:
        metrics = metrics.loc[metrics["disease"].astype(str) == spec.disease]
    if "method" in metrics.columns:
        expected = f"prs_covariate_{spec.disease}"
        preferred = metrics.loc[metrics["method"].astype(str) == expected]
        if not preferred.empty:
            metrics = preferred
    if metrics.empty:
        return None
    row = metrics.iloc[0].to_dict()
    row.update(
        {
            "family": spec.family,
            "disease": spec.disease,
            "strategy": spec.strategy,
            "normalization": normalization,
            "metrics_file": str(path),
        }
    )
    if metrics_kind == "point":
        row = _point_row_as_summary_style(row)
    return row


def _validate(
    *,
    spec: ModelSpec,
    normalization: str,
    train_prs: Path,
    validation_prs: Path,
    out_dir: Path,
    n_bootstraps: int,
    force: bool,
) -> None:
    if not force and (out_dir / "summary_metrics.csv").exists():
        print(f"Reusing existing validation outputs in {out_dir}")
        return
    command = [
        sys.executable,
        "scripts/validate_official_model.py",
        "--model-name",
        f"{spec.family}_{spec.disease}_{normalization}",
        "--disease",
        spec.disease,
        "--train-prs-scores",
        str(train_prs),
        "--validation-prs-scores",
        str(validation_prs),
        "--n-bootstraps",
        str(n_bootstraps),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hard/soft 1000G ancestry-normalized PRS scores for existing best models."
    )
    parser.add_argument(
        "--families",
        default="l2_ensemble,elasticnet_ensemble,tabpfn_grid",
        help="Comma-separated subset of single_best, l2_ensemble, elasticnet_ensemble, tabpfn_grid.",
    )
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument(
        "--hard-labels",
        default=(
            "outputs/final_ancestry_1000g/combined/"
            "target_1000G_hard_labels_pc10_margin0.5.train_val.tsv"
        ),
    )
    parser.add_argument(
        "--soft-weights",
        default=(
            "outputs/final_ancestry_1000g/combined/"
            "target_1000G_soft_weights_pc10_tau1.train_val.tsv"
        ),
    )
    parser.add_argument("--l2-best", default="outputs/ensemble_grid_nonannotated_a0p20/ensemble_grid_summary.best.csv")
    parser.add_argument(
        "--single-best",
        default="outputs/ensemble_inputs_nonannotated_a0p20_plus_megaprs_sbayesrc/selected_single_models.csv",
    )
    parser.add_argument(
        "--elasticnet-best",
        default="outputs/ensemble_grid_nonannotated_a0p20_elasticnet/ensemble_grid_summary.best.csv",
    )
    parser.add_argument(
        "--tabpfn-best",
        default="outputs/tabpfn_grid_nonannotated_a0p20/tabpfn_grid_best_by_disease.csv",
    )
    parser.add_argument("--tabpfn-root", default="outputs/tabpfn_grid_nonannotated_a0p20")
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument(
        "--point-only",
        action="store_true",
        help="Evaluate hard/soft normalized models without bootstrap and compare point estimates.",
    )
    parser.add_argument(
        "--metrics-kind",
        choices=["summary", "point"],
        default="summary",
        help="Use bootstrap summary metrics or point metrics for raw/hard/soft comparison.",
    )
    parser.add_argument(
        "--validate-raw",
        action="store_true",
        help="Rerun raw PRS validation in the output directory instead of reusing existing raw metrics.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun validation even if summary_metrics.csv exists.")
    parser.add_argument("--out-root", default="outputs/ancestry_norm_best_models")
    args = parser.parse_args()
    if args.point_only:
        args.n_bootstraps = 0
        args.metrics_kind = "point"
        args.validate_raw = True

    selected = set(parse_csv_list(args.families))
    specs: list[ModelSpec] = []
    if "single_best" in selected:
        specs.extend(_single_best_specs(args.single_best, args.metrics_kind))
    if "l2_ensemble" in selected:
        specs.extend(_ensemble_specs(args.l2_best, "l2_ensemble", args.metrics_kind))
    if "elasticnet_ensemble" in selected:
        specs.extend(_ensemble_specs(args.elasticnet_best, "elasticnet_ensemble", args.metrics_kind))
    if "tabpfn_grid" in selected:
        specs.extend(_tabpfn_specs(args.tabpfn_best, args.tabpfn_root, "tabpfn_grid", args.metrics_kind))
    if not specs:
        raise ValueError("No model specs were found for the requested families")

    hard_labels = _load_patient_frame(args.hard_labels, args.patient_col)
    soft_weights = _load_patient_frame(args.soft_weights, args.patient_col)
    out_root = Path(args.out_root)
    manifest_rows = []
    metric_rows = []
    for spec in specs:
        if not spec.train_prs.exists() or not spec.validation_prs.exists():
            print(f"Skipping missing PRS scores for {spec.family}/{spec.disease}: {spec.train_prs}")
            continue
        spec_dir = out_root / spec.family / spec.disease
        spec_dir.mkdir(parents=True, exist_ok=True)
        train_scores, validation_scores, raw_train_prs, raw_validation_prs = _load_raw_score_frames(
            spec,
            spec_dir=spec_dir,
            patient_col=args.patient_col,
        )
        if spec.selected_long_scores:
            train_scores = _drop_scores_without_ancestry(
                train_scores,
                hard_labels,
                patient_col=args.patient_col,
                context=f"{spec.family}/{spec.disease} training",
            )
            validation_scores = _drop_scores_without_ancestry(
                validation_scores,
                hard_labels,
                patient_col=args.patient_col,
                context=f"{spec.family}/{spec.disease} validation",
            )
            write_table(train_scores, raw_train_prs)
            write_table(validation_scores, raw_validation_prs)
        score_cols = infer_score_columns(train_scores, patient_col=args.patient_col)
        hard = normalize_by_hard_label(
            train_scores=train_scores,
            validation_scores=validation_scores,
            hard_labels=hard_labels,
            patient_col=args.patient_col,
            score_cols=score_cols,
        )
        soft = normalize_by_soft_weights(
            train_scores=train_scores,
            validation_scores=validation_scores,
            soft_weights=soft_weights,
            patient_col=args.patient_col,
            score_cols=score_cols,
        )
        hard_train = spec_dir / "hard_z.train_prs_scores.csv"
        hard_validation = spec_dir / "hard_z.validation_prs_scores.csv"
        soft_train = spec_dir / "soft_z.train_prs_scores.csv"
        soft_validation = spec_dir / "soft_z.validation_prs_scores.csv"
        write_table(hard.train, hard_train)
        write_table(hard.validation, hard_validation)
        write_table(soft.train, soft_train)
        write_table(soft.validation, soft_validation)
        write_table(
            pd.concat([hard.parameters, soft.parameters], ignore_index=True),
            spec_dir / "normalization_parameters.csv",
        )

        raw_out = spec_dir / "official_validation_raw"
        hard_out = spec_dir / "official_validation_hard_z"
        soft_out = spec_dir / "official_validation_soft_z"
        if args.validate_raw:
            _validate(
                spec=spec,
                normalization="raw",
                train_prs=raw_train_prs,
                validation_prs=raw_validation_prs,
                out_dir=raw_out,
                n_bootstraps=args.n_bootstraps,
                force=args.force,
            )
        _validate(
            spec=spec,
            normalization="hard_z",
            train_prs=hard_train,
            validation_prs=hard_validation,
            out_dir=hard_out,
            n_bootstraps=args.n_bootstraps,
            force=args.force,
        )
        _validate(
            spec=spec,
            normalization="soft_z",
            train_prs=soft_train,
            validation_prs=soft_validation,
            out_dir=soft_out,
            n_bootstraps=args.n_bootstraps,
            force=args.force,
        )

        manifest_rows.append(
            {
                "family": spec.family,
                "disease": spec.disease,
                "strategy": spec.strategy,
                "raw_train_prs": str(raw_train_prs),
                "raw_validation_prs": str(raw_validation_prs),
                "raw_metrics": str(spec.raw_metrics),
                "raw_validation_metrics": str(raw_out / f"{args.metrics_kind}_metrics.csv")
                if args.validate_raw
                else str(spec.raw_metrics),
                "hard_train_prs": str(hard_train),
                "hard_validation_prs": str(hard_validation),
                "soft_train_prs": str(soft_train),
                "soft_validation_prs": str(soft_validation),
            }
        )
        raw_metric_path = raw_out / f"{args.metrics_kind}_metrics.csv" if args.validate_raw else spec.raw_metrics
        for normalization, path in [
            ("raw", raw_metric_path),
            ("hard_z", hard_out / f"{args.metrics_kind}_metrics.csv"),
            ("soft_z", soft_out / f"{args.metrics_kind}_metrics.csv"),
        ]:
            row = _metric_row(path, spec, normalization, args.metrics_kind)
            if row is not None:
                metric_rows.append(row)

    write_table(pd.DataFrame(manifest_rows), out_root / "model_manifest.csv")
    metrics = pd.DataFrame(metric_rows)
    write_table(metrics, out_root / "summary_metrics.long.csv")
    preferred = [
        "family",
        "disease",
        "strategy",
        "normalization",
        "auc_mean",
        "auprc_mean",
        "brier_mean",
        "or_per_sd_mean",
        "top10_vs_lower90_or_mean",
        "nagelkerke_r2_mean",
        "metrics_file",
    ]
    present = [column for column in preferred if column in metrics.columns]
    if present:
        write_table(metrics[present], out_root / "summary_metrics.compact.csv")
    if not metrics.empty:
        delta_rows = []
        group_cols = ["family", "disease", "strategy"]
        for _, frame in metrics.groupby(group_cols, sort=True):
            raw = frame.loc[frame["normalization"].astype(str) == "raw"]
            if raw.empty:
                continue
            raw_row = raw.iloc[0]
            for row in frame.to_dict("records"):
                out = dict(row)
                for metric in [
                    "auc_mean",
                    "auprc_mean",
                    "brier_mean",
                    "or_per_sd_mean",
                    "top10_vs_lower90_or_mean",
                    "nagelkerke_r2_mean",
                ]:
                    if metric in frame.columns:
                        out[f"{metric}_delta_vs_raw"] = row.get(metric) - raw_row.get(metric)
                delta_rows.append(out)
        deltas = pd.DataFrame(delta_rows)
        if not deltas.empty:
            delta_cols = [
                "family",
                "disease",
                "strategy",
                "normalization",
                "auc_mean",
                "auc_mean_delta_vs_raw",
                "top10_vs_lower90_or_mean",
                "top10_vs_lower90_or_mean_delta_vs_raw",
                "nagelkerke_r2_mean",
                "nagelkerke_r2_mean_delta_vs_raw",
                "brier_mean",
                "brier_mean_delta_vs_raw",
                "metrics_file",
            ]
            write_table(deltas[[column for column in delta_cols if column in deltas.columns]], out_root / "summary_metrics.deltas.csv")
            if "top10_vs_lower90_or_mean" in deltas.columns:
                best_top10 = deltas.loc[
                    deltas.groupby(["family", "disease", "strategy"])["top10_vs_lower90_or_mean"].idxmax()
                ].sort_values(["family", "disease"])
                write_table(best_top10, out_root / "summary_metrics.best_by_top10.csv")
            if "auc_mean" in deltas.columns:
                best_auc = deltas.loc[
                    deltas.groupby(["family", "disease", "strategy"])["auc_mean"].idxmax()
                ].sort_values(["family", "disease"])
                write_table(best_auc, out_root / "summary_metrics.best_by_auc.csv")
    print(f"Wrote ancestry-normalization model comparison to {out_root}")


if __name__ == "__main__":
    main()
