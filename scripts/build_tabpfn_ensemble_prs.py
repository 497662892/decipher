#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decipehr.config import load_config
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.ensemble import pivot_long_prs_scores
from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns, validate_unique
from decipehr.tabpfn_ensemble import (
    TabPFNFeatureTransformer,
    TabPFNStrategy,
    logit_from_probability,
    scale_validation_logits,
    stratified_oof_splits,
)

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_ELIGIBILITY_CONFIG = "configs/disease_eligibility.yaml"
DEFAULT_TRAIN_PHENOTYPE = (
    "data/raw/training_phase/phenotype_files/"
    "Training_Cov_file.txt"
)
DEFAULT_VALIDATION_COVARIATES = (
    "data/raw/training_phase/phenotype_files/"
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


def _strategies(value: str) -> list[TabPFNStrategy]:
    allowed = {"prs_only", "covariate_adjusted"}
    parsed = []
    for item in parse_csv_list(value):
        if item not in allowed:
            raise ValueError(f"Unsupported TabPFN strategy {item!r}; expected one of {sorted(allowed)}")
        parsed.append(item)  # type: ignore[arg-type]
    if not parsed:
        raise ValueError("At least one TabPFN strategy is required")
    return parsed


def _load_tabpfn_classifier():
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise SystemExit(
            "The local 'tabpfn' package is not installed in this Python environment. "
            "Install it locally, or run with an environment that already has TabPFN and "
            "local model weights available. This script imports 'tabpfn', not 'tabpfn_client'."
        ) from exc
    return TabPFNClassifier


def _normalise_tabpfn_model_version(value: str | None):
    if value in (None, "", "default", "auto"):
        return None, "default"
    try:
        from tabpfn.constants import ModelVersion
    except ImportError as exc:
        raise SystemExit(
            "The installed 'tabpfn' package does not expose tabpfn.constants.ModelVersion, "
            "so explicit --tabpfn-model-version selection is unavailable."
        ) from exc

    aliases = {
        "2": "V2",
        "v2": "V2",
        "V2": "V2",
        "2.5": "V2_5",
        "v2.5": "V2_5",
        "V2_5": "V2_5",
        "2.6": "V2_6",
        "v2.6": "V2_6",
        "V2_6": "V2_6",
        "3": "V3",
        "v3": "V3",
        "V3": "V3",
    }
    member_name = aliases.get(str(value), str(value))
    try:
        return getattr(ModelVersion, member_name), member_name.lower().replace("_", ".")
    except AttributeError as exc:
        available = ", ".join(
            sorted(name for name in dir(ModelVersion) if name.isupper())
        )
        raise ValueError(
            f"Unsupported TabPFN model version {value!r}. "
            f"The installed tabpfn package exposes: {available or 'none'}."
        ) from exc


def _new_classifier(
    classifier_cls,
    *,
    device: str | None,
    kwargs: dict[str, Any],
    model_version,
):
    init_kwargs = dict(kwargs)
    if device:
        signature = inspect.signature(classifier_cls)
        if "device" in signature.parameters:
            init_kwargs.setdefault("device", device)
    if model_version is not None:
        factory = getattr(classifier_cls, "create_default_for_version", None)
        if factory is None:
            raise ValueError(
                "The installed TabPFNClassifier does not support create_default_for_version; "
                "upgrade tabpfn or omit --tabpfn-model-version."
            )
        return factory(model_version, **init_kwargs)
    return classifier_cls(**init_kwargs)


def _positive_probability(model, x_eval: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x_eval.to_numpy(dtype=float)))
    classes = np.asarray(model.classes_)
    positive = np.where(classes == 1)[0]
    if len(positive) != 1:
        raise ValueError(f"Could not find positive class 1 in TabPFN classes_: {classes.tolist()}")
    return proba[:, int(positive[0])]


def _fit_predict(
    *,
    classifier_cls,
    context: pd.DataFrame,
    y_context: pd.Series,
    eval_frame: pd.DataFrame,
    prs_cols: list[str],
    covariates: list[str],
    include_covariates: bool,
    device: str | None,
    classifier_kwargs: dict[str, Any],
    model_version,
) -> tuple[np.ndarray, pd.DataFrame]:
    transformer = TabPFNFeatureTransformer.fit(
        context,
        prs_cols=prs_cols,
        covariates=covariates,
        include_covariates=include_covariates,
    )
    x_context = transformer.transform(context)
    x_eval = transformer.transform(eval_frame)
    model = _new_classifier(
        classifier_cls,
        device=device,
        kwargs=classifier_kwargs,
        model_version=model_version,
    )
    model.fit(x_context.to_numpy(dtype=float), y_context.astype(int).to_numpy())
    return _positive_probability(model, x_eval), transformer.summary()


def _oof_probabilities(
    *,
    classifier_cls,
    context: pd.DataFrame,
    y_context: pd.Series,
    prs_cols: list[str],
    covariates: list[str],
    include_covariates: bool,
    n_folds: int,
    seed: int,
    device: str | None,
    classifier_kwargs: dict[str, Any],
    model_version,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    probs = np.full(len(context), np.nan, dtype=float)
    feature_summaries = []
    splits = stratified_oof_splits(y_context, requested_splits=n_folds, seed=seed)
    for fold, (fit_idx, holdout_idx) in enumerate(splits):
        fold_probs, feature_summary = _fit_predict(
            classifier_cls=classifier_cls,
            context=context.iloc[fit_idx],
            y_context=y_context.iloc[fit_idx],
            eval_frame=context.iloc[holdout_idx],
            prs_cols=prs_cols,
            covariates=covariates,
            include_covariates=include_covariates,
            device=device,
            classifier_kwargs=classifier_kwargs,
            model_version=model_version,
        )
        probs[holdout_idx] = fold_probs
        feature_summaries.append(feature_summary.assign(oof_fold=fold))
    if np.isnan(probs).any():
        raise ValueError("TabPFN OOF prediction did not produce scores for all context rows")
    return probs, pd.concat(feature_summaries, ignore_index=True), len(splits)


def _build_strategy_scores(
    *,
    classifier_cls,
    disease: str,
    strategy: TabPFNStrategy,
    train_base: pd.DataFrame,
    validation_base: pd.DataFrame,
    prs_cols: list[str],
    covariates: list[str],
    eligibility_rules: dict[str, dict[str, Any]],
    patient_col: str,
    n_folds: int,
    seed: int,
    device: str | None,
    classifier_kwargs: dict[str, Any],
    model_version,
    model_version_label: str,
    train_prediction_mode: str,
    clip_eps: float,
    score_scale: str,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, Any]]:
    include_covariates = strategy == "covariate_adjusted"
    context = filter_eligible(train_base, disease, eligibility_rules).copy()
    validation_eval = filter_eligible(validation_base, disease, eligibility_rules).copy()
    if context.empty:
        raise ValueError(f"No eligible TabPFN context rows remain for {disease}")
    if validation_eval.empty:
        raise ValueError(f"No eligible TabPFN validation rows remain for {disease}")

    y_context = context[disease].astype(int)
    if y_context.nunique() < 2:
        raise ValueError(f"TabPFN context labels for {disease} have fewer than two classes")

    if train_prediction_mode == "oof":
        train_probs, oof_features, actual_folds = _oof_probabilities(
            classifier_cls=classifier_cls,
            context=context,
            y_context=y_context,
            prs_cols=prs_cols,
            covariates=covariates,
            include_covariates=include_covariates,
            n_folds=n_folds,
            seed=seed,
            device=device,
            classifier_kwargs=classifier_kwargs,
            model_version=model_version,
        )
    elif train_prediction_mode == "in_context":
        train_probs, feature_summary = _fit_predict(
            classifier_cls=classifier_cls,
            context=context,
            y_context=y_context,
            eval_frame=context,
            prs_cols=prs_cols,
            covariates=covariates,
            include_covariates=include_covariates,
            device=device,
            classifier_kwargs=classifier_kwargs,
            model_version=model_version,
        )
        oof_features = feature_summary.assign(oof_fold=-1)
        actual_folds = 1
    else:
        raise ValueError(f"Unsupported train prediction mode: {train_prediction_mode}")

    validation_probs, validation_features = _fit_predict(
        classifier_cls=classifier_cls,
        context=context,
        y_context=y_context,
        eval_frame=validation_eval,
        prs_cols=prs_cols,
        covariates=covariates,
        include_covariates=include_covariates,
        device=device,
        classifier_kwargs=classifier_kwargs,
        model_version=model_version,
    )

    train_logits = logit_from_probability(train_probs, eps=clip_eps)
    validation_logits = logit_from_probability(validation_probs, eps=clip_eps)
    train_scaled, validation_scaled, logit_mean, logit_sd = scale_validation_logits(
        train_logits=train_logits,
        validation_logits=validation_logits,
        mode=score_scale,
    )

    train_out = pd.Series(np.nan, index=train_base.index, dtype=float)
    validation_out = pd.Series(np.nan, index=validation_base.index, dtype=float)
    train_out.loc[context.index] = train_scaled
    validation_out.loc[validation_eval.index] = validation_scaled

    feature_summary = pd.concat(
        [
            oof_features.assign(feature_fit_scope="train_oof"),
            validation_features.assign(oof_fold=-1, feature_fit_scope="validation_full_context"),
        ],
        ignore_index=True,
    )
    feature_summary.insert(0, "strategy", strategy)
    feature_summary.insert(0, "disease", disease)

    summary = {
        "disease": disease,
        "strategy": strategy,
        "n_context": int(len(context)),
        "context_cases": int(y_context.sum()),
        "n_validation_scored": int(len(validation_eval)),
        "n_prs_features": int(len(prs_cols)),
        "n_covariates_requested": int(len(covariates) if include_covariates else 0),
        "train_prediction_mode": train_prediction_mode,
        "n_oof_folds": int(actual_folds),
        "probability_min": float(np.nanmin(train_probs)),
        "probability_max": float(np.nanmax(train_probs)),
        "validation_probability_min": float(np.nanmin(validation_probs)),
        "validation_probability_max": float(np.nanmax(validation_probs)),
        "clip_eps": float(clip_eps),
        "score_scale": score_scale,
        "tabpfn_model_version": model_version_label,
        "tabpfn_model_path": str(classifier_kwargs.get("model_path", "auto")),
        "logit_train_mean": float(logit_mean),
        "logit_train_sd": float(logit_sd),
    }
    return train_out, validation_out, feature_summary, summary


def _reindex_scores(scores: pd.DataFrame, patient_order: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    return patient_order.merge(scores, on=patient_col, how="left", validate="one_to_one")


def _read_long_scores(path_template: str, disease: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    path = path_template.format(disease=disease)
    if path not in cache:
        cache[path] = read_table(path)
    return cache[path]


def _write_validation_commands(
    *,
    out_dir: Path,
    strategies: list[TabPFNStrategy],
    args: argparse.Namespace,
    labels_arg: str,
    covariates_arg: str,
    patient_col: str,
) -> None:
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for strategy in strategies:
        strategy_dir = out_dir / strategy
        model_name = f"{args.model_name}_{strategy}" if args.model_name else f"tabpfn_{strategy}"
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
    (out_dir / "validate_tabpfn_ensemble_commands.sh").write_text(
        "\n\n".join(commands) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use local TabPFN as a nonlinear PRS ensemble/meta-model. Labelled training "
            "rows are used as TabPFN context; training scores are OOF by default, and "
            "official validation scores use all eligible training rows as context."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--model-name", default="tabpfn_ensemble")
    parser.add_argument("--disease", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--covariates", default=None)
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--train-phenotype", default=None)
    parser.add_argument("--validation-covariates", default=None)
    parser.add_argument("--eligibility-config", default=DEFAULT_ELIGIBILITY_CONFIG)
    parser.add_argument("--train-prs-long-scores", required=True)
    parser.add_argument("--validation-prs-long-scores", required=True)
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--score-col", default="prs")
    parser.add_argument("--strategies", default="prs_only,covariate_adjusted")
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--train-prediction-mode", choices=["oof", "in_context"], default="oof")
    parser.add_argument("--clip-eps", type=float, default=1e-5)
    parser.add_argument("--score-scale", choices=["train_sd", "none"], default="train_sd")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tabpfn-model-version",
        default="default",
        help="TabPFN model version to instantiate: default, v2, v2.5, v2.6, or v3 if supported by the installed package.",
    )
    parser.add_argument("--tabpfn-kwargs", default="{}", help="JSON kwargs passed to TabPFNClassifier.")
    parser.add_argument("--tabpfn-model-cache-dir", default=None)
    parser.add_argument("--allow-browser", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-validation-commands", action="store_true")
    args = parser.parse_args()

    if args.clip_eps <= 0 or args.clip_eps >= 0.5:
        raise ValueError("--clip-eps must be between 0 and 0.5")
    if args.tabpfn_model_cache_dir:
        os.environ["TABPFN_MODEL_CACHE_DIR"] = args.tabpfn_model_cache_dir
    if not args.allow_browser:
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    classifier_kwargs = json.loads(args.tabpfn_kwargs)
    if not isinstance(classifier_kwargs, dict):
        raise ValueError("--tabpfn-kwargs must decode to a JSON object")

    config = load_config(args.config, args.config_override)
    paths = _require_mapping(config, "paths")
    columns = _require_mapping(config, "columns")
    labels = _labels(args, columns)
    labels_arg = ",".join(labels)
    covariates = parse_csv_list(args.covariates or _csv_from_config(columns.get("covariates"), "columns.covariates"))
    covariates_arg = ",".join(covariates)
    patient_col = args.patient_col or _required_value(columns, "patient_id", "columns")
    seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
    n_folds = args.n_folds if args.n_folds is not None else int(config.get("project", {}).get("n_folds", 5))
    strategies = _strategies(args.strategies)

    train_phenotype_path = args.train_phenotype or str(
        paths.get("official_train_phenotype", DEFAULT_TRAIN_PHENOTYPE)
    )
    validation_covariates_path = args.validation_covariates or str(
        paths.get("official_validation_covariates", DEFAULT_VALIDATION_COVARIATES)
    )
    train_phenotype = read_table(train_phenotype_path)
    validation_covariates = read_table(validation_covariates_path)
    require_columns(train_phenotype, [patient_col, *labels, *covariates], "training phenotype")
    require_columns(validation_covariates, [patient_col, *covariates], "validation covariates")
    validate_unique(train_phenotype, patient_col, "training phenotype")
    validate_unique(validation_covariates, patient_col, "validation covariates")

    eligibility_rules = load_eligibility_rules(args.eligibility_config)
    classifier_cls = _load_tabpfn_classifier()
    model_version, model_version_label = _normalise_tabpfn_model_version(args.tabpfn_model_version)
    long_cache: dict[str, pd.DataFrame] = {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_feature_summaries = []
    all_fit_summaries = []

    strategy_train_scores = {
        strategy: train_phenotype[[patient_col]].copy()
        for strategy in strategies
    }
    strategy_validation_scores = {
        strategy: validation_covariates[[patient_col]].copy()
        for strategy in strategies
    }

    for disease in labels:
        train_long = _read_long_scores(args.train_prs_long_scores, disease, long_cache)
        validation_long = _read_long_scores(args.validation_prs_long_scores, disease, long_cache)
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
        prs_cols = [col for col in train_prs.columns if col != patient_col]
        missing_validation_cols = [col for col in prs_cols if col not in validation_prs.columns]
        if missing_validation_cols:
            raise ValueError(f"Validation PRS scores are missing columns: {missing_validation_cols}")

        train_base = train_phenotype.merge(train_prs, on=patient_col, how="inner", validate="one_to_one")
        validation_base = validation_covariates.merge(
            validation_prs,
            on=patient_col,
            how="inner",
            validate="one_to_one",
        )
        if train_base.empty or validation_base.empty:
            raise ValueError(f"No TabPFN ensemble rows available for {disease}")

        for strategy in strategies:
            train_score, validation_score, feature_summary, fit_summary = _build_strategy_scores(
                classifier_cls=classifier_cls,
                disease=disease,
                strategy=strategy,
                train_base=train_base,
                validation_base=validation_base,
                prs_cols=prs_cols,
                covariates=covariates,
                eligibility_rules=eligibility_rules,
                patient_col=patient_col,
                n_folds=n_folds,
                seed=seed,
                device=args.device,
                classifier_kwargs=classifier_kwargs,
                model_version=model_version,
                model_version_label=model_version_label,
                train_prediction_mode=args.train_prediction_mode,
                clip_eps=args.clip_eps,
                score_scale=args.score_scale,
            )
            train_score_by_patient = pd.DataFrame(
                {
                    patient_col: train_base[patient_col].to_numpy(),
                    f"prs_{disease}": train_score.to_numpy(),
                }
            )
            validation_score_by_patient = pd.DataFrame(
                {
                    patient_col: validation_base[patient_col].to_numpy(),
                    f"prs_{disease}": validation_score.to_numpy(),
                }
            )
            strategy_train_scores[strategy] = strategy_train_scores[strategy].merge(
                train_score_by_patient,
                on=patient_col,
                how="left",
                validate="one_to_one",
            )
            strategy_validation_scores[strategy] = strategy_validation_scores[strategy].merge(
                validation_score_by_patient,
                on=patient_col,
                how="left",
                validate="one_to_one",
            )
            all_feature_summaries.append(feature_summary)
            all_fit_summaries.append(fit_summary)

    for strategy in strategies:
        strategy_dir = out_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        write_table(
            _reindex_scores(strategy_train_scores[strategy], train_phenotype[[patient_col]], patient_col),
            strategy_dir / "train_prs_scores.csv",
        )
        write_table(
            _reindex_scores(
                strategy_validation_scores[strategy],
                validation_covariates[[patient_col]],
                patient_col,
            ),
            strategy_dir / "validation_prs_scores.csv",
        )

    if all_feature_summaries:
        write_table(pd.concat(all_feature_summaries, ignore_index=True), out_dir / "tabpfn_feature_summary.csv")
    if all_fit_summaries:
        write_table(pd.DataFrame(all_fit_summaries), out_dir / "tabpfn_fit_summary.csv")
    if not args.no_validation_commands:
        _write_validation_commands(
            out_dir=out_dir,
            strategies=strategies,
            args=args,
            labels_arg=labels_arg,
            covariates_arg=covariates_arg,
            patient_col=patient_col,
        )

    print(f"Wrote TabPFN ensemble PRS outputs to {out_dir}")


if __name__ == "__main__":
    main()
