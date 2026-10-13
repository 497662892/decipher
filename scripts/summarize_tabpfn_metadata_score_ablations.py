#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decipehr.config import load_yaml
from decipehr.io import read_table, write_table
from decipehr.mechanism_ablations import (
    align_paired_predictions,
    clipped_logit,
    paired_stratified_bootstrap,
    prediction_metrics,
    validate_prediction_frame,
)
from decipehr.metadata_score_ablations import (
    MetadataScoreAblation,
    build_metadata_score_ablations,
)
from decipehr.schema import require_columns


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key {key!r} must be a mapping")
    return value


def _list(mapping: dict[str, Any], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config value {key!r} must be a non-empty list")
    return [str(item) for item in value]


def _load_completed_outputs(
    *,
    config: dict[str, Any],
    input_root: Path,
    diseases: list[str],
    ablations: tuple[MetadataScoreAblation, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    project = _mapping(config, "project")
    metadata = _mapping(config, "metadata")
    expected_final = _list(metadata, "final")
    expected_folds = set(range(int(project["n_outer_folds"])))
    prediction_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    contract_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for disease in diseases:
        for ablation in ablations:
            task_dir = input_root / ablation.name / disease
            prediction_path = task_dir / "oof_predictions.csv"
            fit_path = task_dir / "tabpfn_cv_fit_summary.csv"
            record_path = task_dir / "ablation_run_record.json"
            for path in [prediction_path, fit_path, record_path]:
                if not path.is_file():
                    raise FileNotFoundError(f"Missing completed ablation artifact: {path}")

            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["disease"] != disease or record["ablation"] != ablation.name:
                raise ValueError(f"Task identity mismatch in {record_path}")
            if record["score_covariates"] != list(ablation.score_covariates):
                raise ValueError(f"Score-covariate contract mismatch in {record_path}")
            if record["final_covariates"] != expected_final:
                raise ValueError(f"Final-covariate contract mismatch in {record_path}")
            records.append(record)

            predictions = read_table(prediction_path)
            require_columns(
                predictions,
                ["sampleid", "fold", "disease", "method", "y_true", "y_prob"],
                f"{disease} {ablation.name} OOF predictions",
            )
            if set(predictions["disease"].astype(str)) != {disease}:
                raise ValueError(f"Disease mismatch in {prediction_path}")
            if set(predictions["fold"].astype(int)) != expected_folds:
                raise ValueError(f"Outer-fold mismatch in {prediction_path}")
            if predictions["sampleid"].duplicated().any():
                raise ValueError(f"Duplicate patient predictions in {prediction_path}")
            predictions = predictions.assign(
                variant=ablation.name,
                model_family="tabpfn_score_full_wrapper",
                score=clipped_logit(predictions["y_prob"], eps=1e-12),
            )
            prediction_frames.append(
                predictions[
                    [
                        "sampleid",
                        "disease",
                        "fold",
                        "variant",
                        "model_family",
                        "y_true",
                        "y_prob",
                        "score",
                    ]
                ]
            )

            fit = read_table(fit_path)
            require_columns(
                fit,
                [
                    "outer_fold",
                    "n_covariates_requested",
                    "n_final_covariates_requested",
                    "train_prediction_mode",
                    "n_inner_oof_folds",
                    "score_covariates",
                    "final_covariates",
                ],
                f"{disease} {ablation.name} fit audit",
            )
            if len(fit) != len(expected_folds):
                raise ValueError(f"Expected one fit audit per outer fold in {fit_path}")
            if set(fit["outer_fold"].astype(int)) != expected_folds:
                raise ValueError(f"Fit-audit outer-fold mismatch in {fit_path}")
            if set(fit["train_prediction_mode"].astype(str)) != {"oof"}:
                raise ValueError(f"Non-OOF inner scoring detected in {fit_path}")
            expected_inner = int(project["n_inner_folds"])
            if set(fit["n_inner_oof_folds"].astype(int)) != {expected_inner}:
                raise ValueError(f"Inner-fold mismatch in {fit_path}")
            if set(fit["n_covariates_requested"].astype(int)) != {
                len(ablation.score_covariates)
            }:
                raise ValueError(f"Score-covariate count mismatch in {fit_path}")
            if set(fit["n_final_covariates_requested"].astype(int)) != {
                len(expected_final)
            }:
                raise ValueError(f"Final-covariate count mismatch in {fit_path}")
            fit_frames.append(fit.assign(ablation=ablation.name))
            contract_rows.append(
                {
                    "disease": disease,
                    "ablation": ablation.name,
                    "omitted_from_score": ";".join(ablation.omitted_from_score),
                    "score_covariates": ";".join(ablation.score_covariates),
                    "n_score_covariates": len(ablation.score_covariates),
                    "final_covariates": ";".join(expected_final),
                    "n_final_covariates": len(expected_final),
                    "outer_folds": len(expected_folds),
                    "inner_oof_folds": expected_inner,
                    "n_oof_predictions": len(predictions),
                    "prediction_file": str(prediction_path),
                    "fit_audit_file": str(fit_path),
                }
            )

    combined = pd.concat(prediction_frames, ignore_index=True)
    validate_prediction_frame(
        combined,
        patient_col="sampleid",
        expected_variants=[ablation.name for ablation in ablations],
    )
    return (
        combined,
        pd.concat(fit_frames, ignore_index=True),
        pd.DataFrame(contract_rows),
        records,
    )


def _metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for (disease, variant), frame in predictions.groupby(
        ["disease", "variant"], sort=True
    ):
        pooled_rows.append(
            {"disease": disease, "variant": variant, **prediction_metrics(frame)}
        )
        for fold, fold_frame in frame.groupby("fold", sort=True):
            fold_rows.append(
                {
                    "disease": disease,
                    "variant": variant,
                    "fold": int(fold),
                    **prediction_metrics(fold_frame),
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)


def _paired_comparisons(
    *,
    predictions: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    diseases: list[str],
    ablations: tuple[MetadataScoreAblation, ...],
    n_resamples: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_lookup = pooled_metrics.set_index(["disease", "variant"])
    nonreference = [ablation for ablation in ablations if ablation.name != "full"]
    jobs = [
        (disease_index, disease, ablation_index, ablation)
        for disease_index, disease in enumerate(diseases)
        for ablation_index, ablation in enumerate(nonreference)
    ]

    def run_job(job):
        disease_index, disease, ablation_index, ablation = job
        paired = align_paired_predictions(
            predictions,
            disease=disease,
            reference="full",
            challenger=ablation.name,
            patient_col="sampleid",
        )
        bootstrap = paired_stratified_bootstrap(
            paired,
            reference="full",
            challenger=ablation.name,
            n_resamples=n_resamples,
            seed=seed + disease_index * 100 + ablation_index,
            metrics=("auc", "brier"),
        ).assign(
            disease=disease,
            reference="full",
            challenger=ablation.name,
            comparison=f"{ablation.name} score vs full score",
        )
        reference_metrics = pooled_lookup.loc[(disease, "full")]
        challenger_metrics = pooled_lookup.loc[(disease, ablation.name)]
        row: dict[str, Any] = {
            "disease": disease,
            "comparison": f"{ablation.name} score vs full score",
            "reference": "full",
            "challenger": ablation.name,
            "omitted_from_score": ";".join(ablation.omitted_from_score),
            "n": int(reference_metrics["n"]),
        }
        for metric in [
            "auc",
            "auprc",
            "brier",
            "nagelkerke_r2",
            "or_per_sd",
            "top10_vs_lower90_or",
        ]:
            reference_value = float(reference_metrics[metric])
            challenger_value = float(challenger_metrics[metric])
            row[f"reference_{metric}"] = reference_value
            row[f"challenger_{metric}"] = challenger_value
            row[f"delta_{metric}"] = challenger_value - reference_value
        return row, bootstrap

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(run_job, jobs))
    return (
        pd.DataFrame([row for row, _ in results]),
        pd.concat([frame for _, frame in results], ignore_index=True),
    )


def _fmt(value: float, digits: int = 4, *, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _build_report(
    *,
    config: dict[str, Any],
    pooled_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    bootstrap: pd.DataFrame,
    contract: pd.DataFrame,
) -> str:
    project = _mapping(config, "project")
    metadata = _mapping(config, "metadata")
    paths = _mapping(config, "paths")
    auc_intervals = bootstrap.loc[bootstrap["metric"] == "auc", [
        "disease",
        "challenger",
        "ci_lower",
        "ci_upper",
        "p_two_sided",
    ]]
    comparison_lookup = comparisons.merge(
        auc_intervals,
        on=["disease", "challenger"],
        how="left",
        validate="one_to_one",
    ).set_index(["disease", "challenger"])

    rows = []
    arm_order = ["full", "no_age", "no_sex", "no_pcs", "no_age_sex"]
    for disease in _list(project, "diseases"):
        for arm in arm_order:
            metric = pooled_metrics.set_index(["disease", "variant"]).loc[
                (disease, arm)
            ]
            if arm == "full":
                delta = ci = p_value = "—"
            else:
                comparison = comparison_lookup.loc[(disease, arm)]
                delta = _fmt(float(comparison["delta_auc"]), signed=True)
                ci = (
                    f"[{_fmt(float(comparison['ci_lower']), signed=True)}, "
                    f"{_fmt(float(comparison['ci_upper']), signed=True)}]"
                )
                p_value = _fmt(float(comparison["p_two_sided"]), digits=3)
            rows.append(
                "| "
                + " | ".join(
                    [
                        disease,
                        arm,
                        str(
                            int(
                                contract.set_index(["disease", "ablation"]).loc[
                                    (disease, arm), "n_score_covariates"
                                ]
                            )
                        ),
                        str(int(metric["n"])),
                        _fmt(float(metric["auc"])),
                        delta,
                        ci,
                        p_value,
                        _fmt(float(metric["auprc"])),
                        _fmt(float(metric["brier"])),
                        _fmt(float(metric["nagelkerke_r2"])),
                    ]
                )
                + " |"
            )

    final_covariates = ", ".join(_list(metadata, "final"))
    return "\n".join(
        [
            "# TabPFN-3 metadata-score ablations with a fixed full-covariate wrapper",
            "",
            "## Contract",
            "",
            "Only the covariates used to generate the intermediate TabPFN-3 score vary. "
            f"Every final logistic model is fit with that score plus `{final_covariates}`.",
            "",
            f"- Model definition: `{project['model_definition']}`.",
            f"- Selected-PRS manifest: `{paths['prs_selection_manifest']}`.",
            f"- Outer comparison: shared {int(project['n_outer_folds'])}-fold patient split.",
            f"- Inner score generation: {int(project['n_inner_folds'])}-fold OOF within each outer-training fold.",
            "- Outer validation is prediction-only; no final holdout labels are used.",
            "- Breast-cancer eligibility is female-only.",
            "",
            "## Pooled outer-OOF results",
            "",
            "`Delta AUC` is ablated-score arm minus full-score arm. All arms retain the "
            "same 12-covariate final wrapper.",
            "",
            "| Disease | Score arm | Score metadata n | N | AUC | Delta AUC | 95% CI | Paired bootstrap p | AUPRC | Brier | Nagelkerke R2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Interpretation guardrails",
            "",
            "- These are nested-CV estimates of information lost from the TabPFN score, "
            "conditional on restoring all metadata in the final model.",
            "- The PRS panel was selected earlier, so this is a post-selection mechanism "
            "audit rather than an independent model-selection benchmark.",
            "- Paired bootstrap intervals capture patient-resampling uncertainty, not "
            "variation across TabPFN random seeds.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize completed TabPFN metadata-score ablations."
    )
    parser.add_argument(
        "--config", default="configs/tabpfn_metadata_score_ablations.yaml"
    )
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    project = _mapping(config, "project")
    paths = _mapping(config, "paths")
    metadata = _mapping(config, "metadata")
    summary_config = _mapping(config, "summary")
    diseases = _list(project, "diseases")
    ablations = build_metadata_score_ablations(
        sex=str(metadata["sex"]),
        age=str(metadata["age"]),
        pcs=_list(metadata, "pcs"),
        full_covariates=_list(metadata, "final"),
    )
    input_root = (
        Path(args.input_root) if args.input_root else Path(str(paths["output_root"]))
    )
    out_dir = Path(args.out_dir) if args.out_dir else input_root / "summary"
    n_resamples = (
        args.bootstrap_resamples
        if args.bootstrap_resamples is not None
        else int(summary_config["paired_bootstrap_resamples"])
    )
    n_jobs = args.n_jobs if args.n_jobs is not None else int(summary_config["n_jobs"])
    if n_resamples < 1 or n_jobs < 1:
        raise ValueError("Bootstrap resamples and n_jobs must be positive")

    predictions, fit_audit, contract, records = _load_completed_outputs(
        config=config,
        input_root=input_root,
        diseases=diseases,
        ablations=ablations,
    )
    pooled_metrics, fold_metrics = _metric_tables(predictions)
    comparisons, bootstrap = _paired_comparisons(
        predictions=predictions,
        pooled_metrics=pooled_metrics,
        diseases=diseases,
        ablations=ablations,
        n_resamples=n_resamples,
        seed=int(project["seed"]),
        n_jobs=n_jobs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(predictions, out_dir / "oof_predictions.csv")
    write_table(pooled_metrics, out_dir / "pooled_metrics.csv")
    write_table(fold_metrics, out_dir / "fold_metrics.csv")
    write_table(comparisons, out_dir / "comparison_point_metrics.csv")
    write_table(bootstrap, out_dir / "paired_bootstrap_deltas.csv")
    write_table(contract, out_dir / "contract_audit.csv")
    write_table(fit_audit, out_dir / "fit_audit_by_outer_fold.csv")
    report = _build_report(
        config=config,
        pooled_metrics=pooled_metrics,
        comparisons=comparisons,
        bootstrap=bootstrap,
        contract=contract,
    )
    (out_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    (out_dir / "run_record.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "input_root": str(input_root),
                "diseases": diseases,
                "ablations": [ablation.name for ablation in ablations],
                "n_tasks": len(records),
                "n_prediction_rows": len(predictions),
                "paired_bootstrap_resamples": n_resamples,
                "n_jobs": n_jobs,
                "report": str(out_dir / "RESULTS.md"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote corrected metadata-score ablation summary to {out_dir}")


if __name__ == "__main__":
    main()
