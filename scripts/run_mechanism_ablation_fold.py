#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from build_tabpfn_ensemble_prs import (
    _fit_predict,
    _load_tabpfn_classifier,
    _normalise_tabpfn_model_version,
)

from decipehr.config import load_yaml
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.ensemble import pivot_long_prs_scores
from decipehr.io import read_table, write_table
from decipehr.mechanism_ablations import (
    build_tabpfn_variants,
    clipped_logit,
    fit_tuned_xgboost,
    fold_task,
    prior_corrected_additive_probability,
    validate_prediction_frame,
)
from decipehr.schema import require_columns, validate_unique


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


def _clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _resolve_task(
    *,
    args: argparse.Namespace,
    diseases: list[str],
    n_folds: int,
) -> tuple[str, int, int]:
    if args.task_id is not None:
        if args.disease is not None or args.fold is not None:
            raise ValueError("Use --task-id or --disease/--fold, not both")
        disease, fold = fold_task(args.task_id, diseases, n_folds=n_folds)
        return disease, fold, int(args.task_id)
    if args.disease is None or args.fold is None:
        raise ValueError("Provide either --task-id or both --disease and --fold")
    if args.disease not in diseases:
        raise ValueError(f"Unknown disease {args.disease!r}; expected one of {diseases}")
    if args.fold < 0 or args.fold >= n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}]")
    return args.disease, int(args.fold), diseases.index(args.disease) * n_folds + int(args.fold)


def _load_disease_data(
    *,
    config: dict[str, Any],
    disease: str,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    paths = _mapping(config, "paths")
    columns = _mapping(config, "columns")
    prs_config = _mapping(config, "prs")
    project = _mapping(config, "project")
    patient_col = str(columns["patient_id"])
    ancestry_patient_col = str(columns["ancestry_patient_id"])
    ancestry_source_col = str(columns["ancestry_group"])
    ancestry_feature = str(_mapping(config, "metadata")["ancestry_group_feature"])
    all_prs = _list(prs_config, "all")

    phenotype = read_table(str(paths["phenotype"]))
    splits = read_table(str(paths["splits"]))
    ancestry = read_table(str(paths["ancestry_labels"]))
    prs_long = read_table(
        str(paths["prs_long_template"]).format(disease=disease),
        usecols=[
            patient_col,
            str(columns["disease"]),
            str(columns["model"]),
            str(columns["score"]),
        ],
    )
    prs_wide = pivot_long_prs_scores(
        prs_long,
        disease,
        patient_col=patient_col,
        disease_col=str(columns["disease"]),
        model_col=str(columns["model"]),
        score_col=str(columns["score"]),
        context="mechanism-ablation all-PRS scores",
    )

    full_metadata = _list(_mapping(config, "metadata"), "full")
    require_columns(phenotype, [patient_col, disease, *full_metadata], "train-val phenotype")
    require_columns(splits, [patient_col, "disease", "fold"], "shared disease-specific folds")
    require_columns(ancestry, [ancestry_patient_col, ancestry_source_col], "1000G hard labels")
    require_columns(prs_wide, [patient_col, *all_prs], "all-PRS wide scores")
    validate_unique(phenotype, patient_col, "train-val phenotype")
    validate_unique(prs_wide, patient_col, "all-PRS wide scores")

    disease_splits = splits.loc[
        splits["disease"].astype(str) == disease,
        [patient_col, "fold"],
    ].copy()
    validate_unique(disease_splits, patient_col, f"{disease} folds")
    ancestry = ancestry[[ancestry_patient_col, ancestry_source_col]].rename(
        columns={ancestry_patient_col: patient_col, ancestry_source_col: ancestry_feature}
    )
    validate_unique(ancestry, patient_col, "1000G hard labels")

    data = phenotype.merge(disease_splits, on=patient_col, how="inner", validate="one_to_one")
    eligibility_rules = load_eligibility_rules(str(paths["eligibility_config"]))
    data = filter_eligible(data, disease, eligibility_rules).reset_index(drop=True)
    data = data.merge(ancestry, on=patient_col, how="left", validate="one_to_one")
    data = data.merge(prs_wide[[patient_col, *all_prs]], on=patient_col, how="left", validate="one_to_one")
    missing_columns = [ancestry_feature, *all_prs]
    missing = data.loc[data[missing_columns].isna().any(axis=1), patient_col]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} eligible {disease} patients are missing ancestry or PRS inputs"
        )
    observed_folds = sorted(data["fold"].astype(int).unique().tolist())
    expected_folds = list(range(int(project["n_folds"])))
    if observed_folds != expected_folds:
        raise ValueError(f"Expected folds {expected_folds} for {disease}; got {observed_folds}")

    context = data.loc[data["fold"].astype(int) != fold].copy().reset_index(drop=True)
    validation = data.loc[data["fold"].astype(int) == fold].copy().reset_index(drop=True)
    if context[disease].nunique() != 2 or validation[disease].nunique() != 2:
        raise ValueError(f"Both classes must be present in context and validation for {disease} fold {fold}")
    audit = {
        "disease": disease,
        "fold": int(fold),
        "n_all_eligible": int(len(data)),
        "n_context": int(len(context)),
        "context_cases": int(context[disease].astype(int).sum()),
        "n_validation": int(len(validation)),
        "validation_cases": int(validation[disease].astype(int).sum()),
        "all_prs": all_prs,
        "ancestry_group_counts_context": {
            str(key): int(value)
            for key, value in context[ancestry_feature].astype(str).value_counts().sort_index().items()
        },
        "ancestry_group_counts_validation": {
            str(key): int(value)
            for key, value in validation[ancestry_feature].astype(str).value_counts().sort_index().items()
        },
    }
    return context, validation, all_prs, audit


def _prediction_frame(
    *,
    validation: pd.DataFrame,
    patient_col: str,
    disease: str,
    fold: int,
    variant: str,
    model_family: str,
    probability: np.ndarray,
    score: np.ndarray | None = None,
    clip_eps: float,
) -> pd.DataFrame:
    probability = np.asarray(probability, dtype=float)
    if len(probability) != len(validation):
        raise ValueError(f"Prediction length mismatch for {variant}")
    resolved_score = clipped_logit(probability, eps=clip_eps) if score is None else np.asarray(score)
    return pd.DataFrame(
        {
            patient_col: validation[patient_col].to_numpy(),
            "disease": disease,
            "fold": int(fold),
            "variant": variant,
            "model_family": model_family,
            "y_true": validation[disease].astype(int).to_numpy(),
            "y_prob": probability,
            "score": resolved_score,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every prespecified TabPFN mechanism-ablation arm for one disease/fold."
    )
    parser.add_argument("--config", default="configs/mechanism_ablations.yaml")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    project = _mapping(config, "project")
    paths = _mapping(config, "paths")
    columns = _mapping(config, "columns")
    prs_config = _mapping(config, "prs")
    metadata = _mapping(config, "metadata")
    tabpfn_config = _mapping(config, "tabpfn")
    xgboost_config = _mapping(config, "xgboost")
    diseases = _list(project, "diseases")
    n_folds = int(project["n_folds"])
    disease, fold, task_id = _resolve_task(args=args, diseases=diseases, n_folds=n_folds)
    patient_col = str(columns["patient_id"])
    seed = int(project["seed"])
    clip_eps = float(tabpfn_config["clip_eps"])
    n_jobs = args.n_jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

    context, validation, all_prs, data_audit = _load_disease_data(
        config=config,
        disease=disease,
        fold=fold,
    )
    selected_by_disease = _mapping(prs_config, "selected")
    best_by_disease = _mapping(prs_config, "canonical_best_single")
    selected_prs = [str(item) for item in selected_by_disease[disease]]
    variants = build_tabpfn_variants(
        best_prs=str(best_by_disease[disease]),
        selected_prs=selected_prs,
        all_prs=all_prs,
        full_metadata=_list(metadata, "full"),
        age_sex=_list(metadata, "age_sex"),
        pcs=_list(metadata, "pcs"),
        ancestry_group_metadata=_list(metadata, "age_sex_ancestry_group"),
    )

    model_path = Path(str(tabpfn_config["model_path"]))
    if not model_path.is_file():
        raise FileNotFoundError(f"TabPFN checkpoint does not exist: {model_path}")
    classifier_cls = _load_tabpfn_classifier()
    model_version, model_version_label = _normalise_tabpfn_model_version(
        str(tabpfn_config["model_version"])
    )
    classifier_kwargs = {
        "n_estimators": int(tabpfn_config["n_estimators"]),
        "model_path": str(model_path),
        "random_state": seed + fold,
    }

    prediction_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    variant_rows: list[dict[str, Any]] = []
    probabilities: dict[str, np.ndarray] = {}
    for variant in variants:
        print(f"Running {disease} fold {fold}: {variant.name}", flush=True)
        probability, feature_summary = _fit_predict(
            classifier_cls=classifier_cls,
            context=context,
            y_context=context[disease].astype(int),
            eval_frame=validation,
            prs_cols=list(variant.prs_columns),
            covariates=list(variant.covariates),
            include_covariates=bool(variant.covariates),
            device=args.device,
            classifier_kwargs=classifier_kwargs,
            model_version=model_version,
        )
        probabilities[variant.name] = probability
        prediction_frames.append(
            _prediction_frame(
                validation=validation,
                patient_col=patient_col,
                disease=disease,
                fold=fold,
                variant=variant.name,
                model_family=variant.model_family,
                probability=probability,
                clip_eps=clip_eps,
            )
        )
        feature_frames.append(
            feature_summary.assign(
                disease=disease,
                fold=int(fold),
                variant=variant.name,
                model_family=variant.model_family,
            )
        )
        variant_rows.append(
            {
                "disease": disease,
                "variant": variant.name,
                "model_family": variant.model_family,
                "prs_columns": ";".join(variant.prs_columns),
                "covariates": ";".join(variant.covariates),
                "n_prs": len(variant.prs_columns),
                "n_covariates_requested": len(variant.covariates),
                "description": variant.description,
            }
        )
        _clear_cuda_cache()

    full_metadata = _list(metadata, "full")
    fixed_xgboost_parameters = {
        "n_estimators": int(xgboost_config["n_estimators"]),
        "subsample": float(xgboost_config["subsample"]),
        "colsample_bytree": float(xgboost_config["colsample_bytree"]),
        "reg_lambda": float(xgboost_config["reg_lambda"]),
    }
    print(f"Running {disease} fold {fold}: tuned XGBoost comparator", flush=True)
    xgboost_probability, xgboost_features, tuning, best_parameters = fit_tuned_xgboost(
        context=context,
        validation=validation,
        label_col=disease,
        prs_columns=selected_prs,
        covariates=full_metadata,
        parameter_grid={
            "max_depth": list(xgboost_config["max_depth"]),
            "learning_rate": list(xgboost_config["learning_rate"]),
            "min_child_weight": list(xgboost_config["min_child_weight"]),
        },
        fixed_parameters=fixed_xgboost_parameters,
        inner_folds=int(xgboost_config["inner_folds"]),
        seed=seed + fold,
        n_jobs=n_jobs,
    )
    xgboost_variant = "xgboost_selected_prs_full_metadata"
    prediction_frames.append(
        _prediction_frame(
            validation=validation,
            patient_col=patient_col,
            disease=disease,
            fold=fold,
            variant=xgboost_variant,
            model_family="xgboost",
            probability=xgboost_probability,
            clip_eps=clip_eps,
        )
    )
    feature_frames.append(
        xgboost_features.assign(
            disease=disease,
            fold=int(fold),
            variant=xgboost_variant,
            model_family="xgboost",
        )
    )
    variant_rows.append(
        {
            "disease": disease,
            "variant": xgboost_variant,
            "model_family": "xgboost",
            "prs_columns": ";".join(selected_prs),
            "covariates": ";".join(full_metadata),
            "n_prs": len(selected_prs),
            "n_covariates_requested": len(full_metadata),
            "description": "Inner-CV-tuned XGBoost on the exact joint-model transformed inputs",
        }
    )

    additive_variant = "tabpfn_separate_additive"
    print(f"Deriving {disease} fold {fold}: {additive_variant}", flush=True)
    additive_probability, additive_score = prior_corrected_additive_probability(
        probabilities["tabpfn_metadata_only"],
        probabilities["tabpfn_selected_prs_only"],
        training_prevalence=float(context[disease].astype(int).mean()),
        eps=clip_eps,
    )
    prediction_frames.append(
        _prediction_frame(
            validation=validation,
            patient_col=patient_col,
            disease=disease,
            fold=fold,
            variant=additive_variant,
            model_family="tabpfn_additive",
            probability=additive_probability,
            score=additive_score,
            clip_eps=clip_eps,
        )
    )
    variant_rows.append(
        {
            "disease": disease,
            "variant": additive_variant,
            "model_family": "tabpfn_additive",
            "prs_columns": ";".join(selected_prs),
            "covariates": ";".join(full_metadata),
            "n_prs": len(selected_prs),
            "n_covariates_requested": len(full_metadata),
            "description": (
                "Equal-weight sum of separate TabPFN metadata and PRS logits, with one "
                "training-prevalence logit subtracted to avoid double-counting the prior"
            ),
        }
    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_variants = [variant.name for variant in variants] + [xgboost_variant, additive_variant]
    validate_prediction_frame(
        predictions,
        patient_col=patient_col,
        expected_variants=expected_variants,
    )

    output_root = Path(str(paths["output_root"]))
    out_dir = Path(args.out_dir) if args.out_dir else output_root / "fold_predictions" / disease / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(predictions, out_dir / "predictions.csv")
    write_table(pd.concat(feature_frames, ignore_index=True), out_dir / "feature_audit.csv")
    write_table(pd.DataFrame(variant_rows), out_dir / "variant_manifest.csv")
    write_table(tuning.assign(disease=disease, fold=int(fold)), out_dir / "xgboost_tuning.csv")

    run_record = {
        "config": args.config,
        "task_id": task_id,
        "disease": disease,
        "fold": int(fold),
        "device": args.device,
        "seed": seed,
        "data_audit": data_audit,
        "selected_prs": selected_prs,
        "canonical_best_single_prs": str(best_by_disease[disease]),
        "tabpfn": {
            "model_version": model_version_label,
            "model_path": str(model_path),
            "n_estimators": int(tabpfn_config["n_estimators"]),
            "random_state": seed + fold,
            "clip_eps": clip_eps,
        },
        "xgboost": {
            "inner_folds": int(xgboost_config["inner_folds"]),
            "fixed_parameters": fixed_xgboost_parameters,
            "selected_parameters": best_parameters,
        },
        "variants": expected_variants,
    }
    (out_dir / "run_record.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(predictions):,} predictions for {disease} fold {fold} "
        f"across {len(expected_variants)} variants to {out_dir}"
    )


if __name__ == "__main__":
    main()
