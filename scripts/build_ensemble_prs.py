from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import pandas as pd

from decipehr.config import load_config
from decipehr.eligibility import load_eligibility_rules
from decipehr.ensemble import (
    EnsembleResult,
    EnsembleStrategy,
    align_wide_prs_scores,
    fit_ensemble_scores,
    pivot_long_prs_scores,
)
from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_ELIGIBILITY_CONFIG = "configs/disease_eligibility.yaml"
DEFAULT_TRAIN_PHENOTYPE = (
    "/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/phenotype_files/"
    "Training_Cov_file.txt"
)
DEFAULT_VALIDATION_COVARIATES = (
    "/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/phenotype_files/"
    "Testing_Cov_file.txt"
)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config key {key!r} must be a mapping")
    return value


def _required_value(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value {context}.{key}")
    return str(value)


def _csv_from_config(value: object, context: str) -> str:
    if isinstance(value, str):
        parsed = parse_csv_list(value)
    elif isinstance(value, list):
        parsed = [str(item) for item in value]
    else:
        raise ValueError(f"Config value {context} must be a string or list")
    if not parsed:
        raise ValueError(f"Config value {context} must not be empty")
    return ",".join(parsed)


def _labels(args: argparse.Namespace, columns: dict[str, Any]) -> list[str]:
    if args.labels and args.disease:
        raise ValueError("Use either --labels or --disease, not both")
    if args.disease:
        return [args.disease]
    if args.labels:
        return parse_csv_list(args.labels)
    return parse_csv_list(_csv_from_config(columns.get("labels"), "columns.labels"))


def _strategies(value: str) -> list[EnsembleStrategy]:
    allowed = {"prs_only", "covariate_adjusted"}
    aliases = {"layered": "covariate_adjusted", "covariate": "covariate_adjusted"}
    parsed = []
    for item in parse_csv_list(value):
        strategy = aliases.get(item, item)
        if strategy not in allowed:
            raise ValueError(f"Unsupported ensemble strategy {item!r}; expected one of {sorted(allowed)}")
        parsed.append(strategy)  # type: ignore[arg-type]
    if not parsed:
        raise ValueError("At least one ensemble strategy is required")
    return parsed


def _class_weight(value: str) -> str | None:
    if value == "none":
        return None
    if value == "balanced":
        return "balanced"
    raise ValueError(f"Unsupported class weight: {value}")


def _load_validation_patients(path: str | None, patient_col: str) -> pd.DataFrame | None:
    if not path:
        return None
    frame = read_table(path)
    require_columns(frame, [patient_col], "validation covariates")
    return frame[[patient_col]].copy()


def _score_inputs_are_long(args: argparse.Namespace) -> bool:
    has_long = bool(args.train_prs_long_scores or args.validation_prs_long_scores)
    has_wide = bool(args.train_prs_scores or args.validation_prs_scores)
    if has_long and has_wide:
        raise ValueError("Use either long PRS inputs or wide PRS inputs, not both")
    if has_long:
        if not args.train_prs_long_scores or not args.validation_prs_long_scores:
            raise ValueError("Long PRS input requires both --train-prs-long-scores and --prs-long-scores")
        return True
    if not args.train_prs_scores or not args.validation_prs_scores:
        raise ValueError("Wide PRS input requires both --train-prs-scores and --prs-scores")
    return False


def _merge_strategy_results(results: list[dict[EnsembleStrategy, EnsembleResult]]) -> dict[EnsembleStrategy, EnsembleResult]:
    merged: dict[EnsembleStrategy, EnsembleResult] = {}
    for result_by_strategy in results:
        for strategy, result in result_by_strategy.items():
            if strategy not in merged:
                merged[strategy] = result
                continue
            previous = merged[strategy]
            train_scores = previous.train_scores.merge(
                result.train_scores,
                on=previous.train_scores.columns[0],
                how="outer",
                validate="one_to_one",
            )
            validation_scores = previous.validation_scores.merge(
                result.validation_scores,
                on=previous.validation_scores.columns[0],
                how="outer",
                validate="one_to_one",
            )
            merged[strategy] = EnsembleResult(
                train_scores=train_scores,
                validation_scores=validation_scores,
                weights=pd.concat([previous.weights, result.weights], ignore_index=True),
                fit_summary=pd.concat([previous.fit_summary, result.fit_summary], ignore_index=True),
            )
    return merged


def _build_from_long_inputs(
    *,
    args: argparse.Namespace,
    train_phenotype: pd.DataFrame,
    train_long: pd.DataFrame,
    validation_long: pd.DataFrame,
    labels: list[str],
    covariates: list[str],
    strategies: list[EnsembleStrategy],
    patient_col: str,
    eligibility_rules: dict[str, dict],
    class_weight: str | None,
) -> dict[EnsembleStrategy, EnsembleResult]:
    disease_results = []
    for disease in labels:
        train_prs = pivot_long_prs_scores(
            train_long,
            disease,
            patient_col=patient_col,
            disease_col=args.disease_col,
            model_col=args.model_col,
            score_col=args.score_col,
            context="training long PRS scores",
        )
        validation_prs = pivot_long_prs_scores(
            validation_long,
            disease,
            patient_col=patient_col,
            disease_col=args.disease_col,
            model_col=args.model_col,
            score_col=args.score_col,
            context="validation long PRS scores",
        )
        disease_results.append(
            fit_ensemble_scores(
                train_phenotype=train_phenotype,
                train_prs=train_prs,
                validation_prs=validation_prs,
                labels=[disease],
                covariates=covariates,
                strategies=strategies,
                patient_col=patient_col,
                eligibility_rules=eligibility_rules,
                penalty=args.penalty,
                c_value=args.C,
                l1_ratio=args.l1_ratio,
                class_weight=class_weight,
                seed=args.seed,
            )
        )
    return _merge_strategy_results(disease_results)


def _build_from_wide_inputs(
    *,
    args: argparse.Namespace,
    train_phenotype: pd.DataFrame,
    train_wide: pd.DataFrame,
    validation_wide: pd.DataFrame,
    labels: list[str],
    covariates: list[str],
    strategies: list[EnsembleStrategy],
    patient_col: str,
    eligibility_rules: dict[str, dict],
    class_weight: str | None,
) -> dict[EnsembleStrategy, EnsembleResult]:
    prs_cols = parse_csv_list(args.prs_cols)
    if not prs_cols:
        excluded = {patient_col, *labels, *covariates}
        prs_cols = [col for col in train_wide.columns if col not in excluded]
    if not prs_cols:
        raise ValueError("No PRS columns were inferred; pass --prs-cols for wide input")
    train_prs = align_wide_prs_scores(
        train_wide,
        patient_col=patient_col,
        prs_cols=prs_cols,
        context="training wide PRS scores",
    )
    validation_prs = align_wide_prs_scores(
        validation_wide,
        patient_col=patient_col,
        prs_cols=prs_cols,
        context="validation wide PRS scores",
    )
    return fit_ensemble_scores(
        train_phenotype=train_phenotype,
        train_prs=train_prs,
        validation_prs=validation_prs,
        labels=labels,
        covariates=covariates,
        strategies=strategies,
        patient_col=patient_col,
        eligibility_rules=eligibility_rules,
        penalty=args.penalty,
        c_value=args.C,
        l1_ratio=args.l1_ratio,
        class_weight=class_weight,
        seed=args.seed,
    )


def _reindex_scores(
    scores: pd.DataFrame,
    patient_order: pd.DataFrame | None,
    patient_col: str,
) -> pd.DataFrame:
    if patient_order is None:
        return scores
    return patient_order.merge(scores, on=patient_col, how="left", validate="one_to_one")


def _write_validation_commands(
    *,
    out_dir: Path,
    strategies: list[EnsembleStrategy],
    args: argparse.Namespace,
    labels_arg: str,
    covariates_arg: str,
    patient_col: str,
) -> None:
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for strategy in strategies:
        strategy_dir = out_dir / strategy
        model_name = f"{args.model_name}_{strategy}" if args.model_name else f"ensemble_{strategy}"
        command = [
            "uv",
            "run",
            "python",
            "scripts/validate_official_model.py",
            "--config",
            args.config,
            "--model-name",
            model_name,
            "--patient-col",
            patient_col,
            "--labels",
            labels_arg,
            "--covariates",
            covariates_arg,
            "--train-prs-scores",
            str(strategy_dir / "train_prs_scores.csv"),
            "--validation-prs-scores",
            str(strategy_dir / "validation_prs_scores.csv"),
            "--eligibility-config",
            args.eligibility_config,
            "--out-dir",
            str(out_dir / f"official_validation_{strategy}"),
        ]
        if args.config_override:
            command.extend(["--config-override", args.config_override])
        commands.append(shlex.join(command))
    (out_dir / "validate_ensemble_commands.sh").write_text("\n\n".join(commands) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit PRS-only and covariate-adjusted score-level ensemble PRS models on "
            "the labelled training set, then write official-validation-compatible "
            "single-PRS score files for training and validation patients."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--model-name", default="ensemble")
    parser.add_argument("--disease", default=None, help="Build one disease-specific ensemble.")
    parser.add_argument("--labels", default=None, help="Comma-separated disease labels.")
    parser.add_argument("--covariates", default=None, help="Comma-separated covariates for adjusted fitting.")
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--train-phenotype", default=None)
    parser.add_argument("--validation-covariates", default=None)
    parser.add_argument("--eligibility-config", default=DEFAULT_ELIGIBILITY_CONFIG)
    parser.add_argument("--train-prs-scores", default=None, help="Wide training PRS table.")
    parser.add_argument(
        "--validation-prs-scores",
        "--prs-scores",
        dest="validation_prs_scores",
        default=None,
        help="Wide validation PRS table.",
    )
    parser.add_argument("--prs-cols", default="", help="Comma-separated PRS columns for wide input.")
    parser.add_argument("--train-prs-long-scores", default=None, help="Long training PRS table.")
    parser.add_argument(
        "--validation-prs-long-scores",
        "--prs-long-scores",
        dest="validation_prs_long_scores",
        default=None,
        help="Long validation PRS table.",
    )
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--score-col", default="prs")
    parser.add_argument(
        "--strategies",
        default="prs_only,covariate_adjusted",
        help="Comma-separated strategies: prs_only,covariate_adjusted.",
    )
    parser.add_argument("--penalty", choices=["l1", "l2", "elasticnet", "none"], default="l2")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--l1-ratio", type=float, default=None)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--no-validation-commands",
        action="store_true",
        help="Do not write validate_ensemble_commands.sh.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.config_override)
    paths = _require_mapping(config, "paths")
    columns = _require_mapping(config, "columns")
    labels = _labels(args, columns)
    labels_arg = ",".join(labels)
    covariates = parse_csv_list(args.covariates or _csv_from_config(columns.get("covariates"), "columns.covariates"))
    covariates_arg = ",".join(covariates)
    patient_col = args.patient_col or _required_value(columns, "patient_id", "columns")
    train_phenotype_path = args.train_phenotype or str(
        paths.get("official_train_phenotype", DEFAULT_TRAIN_PHENOTYPE)
    )
    validation_covariates_path = args.validation_covariates or str(
        paths.get("official_validation_covariates", DEFAULT_VALIDATION_COVARIATES)
    )
    strategies = _strategies(args.strategies)
    class_weight = _class_weight(args.class_weight)
    seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
    args.seed = seed

    is_long = _score_inputs_are_long(args)
    train_phenotype = read_table(train_phenotype_path)
    require_columns(train_phenotype, [patient_col, *labels, *covariates], "training phenotype")
    validation_patients = _load_validation_patients(validation_covariates_path, patient_col)
    eligibility_rules = load_eligibility_rules(args.eligibility_config)

    if is_long:
        train_long = read_table(args.train_prs_long_scores)
        validation_long = read_table(args.validation_prs_long_scores)
        results = _build_from_long_inputs(
            args=args,
            train_phenotype=train_phenotype,
            train_long=train_long,
            validation_long=validation_long,
            labels=labels,
            covariates=covariates,
            strategies=strategies,
            patient_col=patient_col,
            eligibility_rules=eligibility_rules,
            class_weight=class_weight,
        )
    else:
        train_wide = read_table(args.train_prs_scores)
        validation_wide = read_table(args.validation_prs_scores)
        results = _build_from_wide_inputs(
            args=args,
            train_phenotype=train_phenotype,
            train_wide=train_wide,
            validation_wide=validation_wide,
            labels=labels,
            covariates=covariates,
            strategies=strategies,
            patient_col=patient_col,
            eligibility_rules=eligibility_rules,
            class_weight=class_weight,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_weights = []
    all_summaries = []
    for strategy, result in results.items():
        strategy_dir = out_dir / strategy
        train_scores = _reindex_scores(result.train_scores, train_phenotype[[patient_col]], patient_col)
        validation_scores = _reindex_scores(result.validation_scores, validation_patients, patient_col)
        write_table(train_scores, strategy_dir / "train_prs_scores.csv")
        write_table(validation_scores, strategy_dir / "validation_prs_scores.csv")
        write_table(result.weights, strategy_dir / "weights.csv")
        write_table(result.fit_summary, strategy_dir / "fit_summary.csv")
        all_weights.append(result.weights)
        all_summaries.append(result.fit_summary)
    if all_weights:
        write_table(pd.concat(all_weights, ignore_index=True), out_dir / "ensemble_weights.csv")
    if all_summaries:
        write_table(pd.concat(all_summaries, ignore_index=True), out_dir / "ensemble_fit_summary.csv")
    if not args.no_validation_commands:
        _write_validation_commands(
            out_dir=out_dir,
            strategies=strategies,
            args=args,
            labels_arg=labels_arg,
            covariates_arg=covariates_arg,
            patient_col=patient_col,
        )

    print(f"Wrote ensemble PRS outputs to {out_dir}")


if __name__ == "__main__":
    main()
