#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from decipehr.mechanism_ablations import fit_tuned_xgboost, prediction_metrics
from decipehr.models import fit_predict_logistic
from decipehr.official_mechanism import (
    OfficialStage1Arm,
    build_official_stage1_arms,
    scaled_logit_scores,
)
from decipehr.schema import require_columns, validate_unique
from decipehr.tabpfn_ensemble import stratified_oof_splits


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key {key!r} must be a mapping")
    return value


def _list(mapping: dict[str, Any], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config value {key!r} must be a non-empty list")
    return [str(item) for item in value]


def _arms_for_disease(config: dict[str, Any], disease: str) -> tuple[OfficialStage1Arm, ...]:
    prs = _mapping(config, "prs")
    metadata = _mapping(config, "metadata")
    best = _mapping(prs, "canonical_best_single")
    columns = _mapping(config, "columns")
    return build_official_stage1_arms(
        best_prs=str(best[disease]),
        nonannotation_prs=_list(prs, "nonannotation"),
        all_prs=_list(prs, "all"),
        full_metadata=_list(metadata, "full"),
        sex=str(metadata["sex"]),
        age=str(metadata["age"]),
        pcs=_list(metadata, "pcs"),
        ancestry_group=str(columns["ancestry_group_feature"]),
    )


def _resolve_task(
    *,
    config: dict[str, Any],
    task_id: int | None,
    disease: str | None,
    arm_name: str | None,
) -> tuple[str, OfficialStage1Arm, int]:
    diseases = _list(_mapping(config, "project"), "diseases")
    reference_arms = _arms_for_disease(config, diseases[0])
    arm_names = [arm.name for arm in reference_arms]
    if task_id is not None:
        if disease is not None or arm_name is not None:
            raise ValueError("Use --task-id or --disease/--arm, not both")
        n_tasks = len(diseases) * len(arm_names)
        if task_id < 0 or task_id >= n_tasks:
            raise ValueError(f"task-id must be in [0, {n_tasks - 1}]")
        disease_index, arm_index = divmod(task_id, len(arm_names))
        resolved_disease = diseases[disease_index]
        return resolved_disease, _arms_for_disease(config, resolved_disease)[arm_index], task_id
    if disease is None or arm_name is None:
        raise ValueError("Provide --task-id or both --disease and --arm")
    if disease not in diseases:
        raise ValueError(f"Unknown disease {disease!r}; expected {diseases}")
    arms = {arm.name: arm for arm in _arms_for_disease(config, disease)}
    if arm_name not in arms:
        raise ValueError(f"Unknown arm {arm_name!r}; expected {sorted(arms)}")
    return disease, arms[arm_name], diseases.index(disease) * len(arms) + arm_names.index(arm_name)


def _load_disease_data(
    *,
    config: dict[str, Any],
    disease: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = _mapping(config, "paths")
    columns = _mapping(config, "columns")
    prs = _mapping(config, "prs")
    metadata = _mapping(config, "metadata")
    patient_col = str(columns["patient_id"])
    all_prs = _list(prs, "all")
    full_metadata = _list(metadata, "full")
    ancestry_patient_col = str(columns["ancestry_patient_id"])
    ancestry_source = str(columns["ancestry_group_source"])
    ancestry_feature = str(columns["ancestry_group_feature"])

    train_phenotype = read_table(str(paths["train_phenotype"]))
    validation_phenotype = read_table(str(paths["validation_phenotype"]))
    require_columns(
        train_phenotype,
        [patient_col, disease, *full_metadata],
        "official training phenotype",
    )
    require_columns(
        validation_phenotype,
        [patient_col, disease, *full_metadata],
        "official validation phenotype",
    )
    validate_unique(train_phenotype, patient_col, "official training phenotype")
    validate_unique(validation_phenotype, patient_col, "official validation phenotype")
    overlap = set(train_phenotype[patient_col].astype(str)) & set(
        validation_phenotype[patient_col].astype(str)
    )
    if overlap:
        raise ValueError(f"Official training and validation overlap for {len(overlap)} patients")

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
        context="official mechanism all-eight PRS scores",
    )
    require_columns(prs_wide, [patient_col, *all_prs], "official mechanism PRS scores")
    validate_unique(prs_wide, patient_col, "official mechanism PRS scores")

    ancestry = read_table(str(paths["ancestry_labels"]))
    require_columns(
        ancestry,
        [ancestry_patient_col, ancestry_source],
        "1000G hard ancestry labels",
    )
    ancestry = ancestry[[ancestry_patient_col, ancestry_source]].rename(
        columns={ancestry_patient_col: patient_col, ancestry_source: ancestry_feature}
    )
    validate_unique(ancestry, patient_col, "1000G hard ancestry labels")

    rules = load_eligibility_rules(str(paths["eligibility_config"]))
    train = filter_eligible(train_phenotype, disease, rules).reset_index(drop=True)
    validation = filter_eligible(validation_phenotype, disease, rules).reset_index(drop=True)
    train = train.merge(prs_wide, on=patient_col, how="left", validate="one_to_one")
    validation = validation.merge(prs_wide, on=patient_col, how="left", validate="one_to_one")
    train = train.merge(ancestry, on=patient_col, how="left", validate="one_to_one")
    validation = validation.merge(ancestry, on=patient_col, how="left", validate="one_to_one")

    required_inputs = [*all_prs, ancestry_feature]
    for label, frame in [("training", train), ("validation", validation)]:
        missing = frame.loc[frame[required_inputs].isna().any(axis=1), patient_col]
        if not missing.empty:
            raise ValueError(
                f"{len(missing)} official {label} patients lack PRS or ancestry inputs"
            )
        if frame[disease].astype(int).nunique() != 2:
            raise ValueError(f"Official {label} data for {disease} do not contain both classes")

    audit = {
        "disease": disease,
        "train_n": int(len(train)),
        "train_cases": int(train[disease].astype(int).sum()),
        "validation_n": int(len(validation)),
        "validation_cases": int(validation[disease].astype(int).sum()),
        "train_validation_overlap": 0,
        "all_prs": all_prs,
        "ancestry_train_counts": {
            str(key): int(value)
            for key, value in train[ancestry_feature]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "ancestry_validation_counts": {
            str(key): int(value)
            for key, value in validation[ancestry_feature]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return train, validation, audit


def _tabpfn_stage1(
    *,
    config: dict[str, Any],
    disease: str,
    arm: OfficialStage1Arm,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    device: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]
]:
    project = _mapping(config, "project")
    tabpfn = _mapping(config, "tabpfn")
    settings = _mapping(_mapping(tabpfn, "disease_settings"), disease)
    seed = int(project["seed"])
    model_path = Path(str(settings["model_path"]))
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing TabPFN checkpoint: {model_path}")

    classifier_cls = _load_tabpfn_classifier()
    model_version, model_version_label = _normalise_tabpfn_model_version(
        str(tabpfn["model_version"])
    )
    classifier_kwargs = {
        "n_estimators": int(settings["n_estimators"]),
        "model_path": str(model_path),
        "balance_probabilities": bool(settings["balance_probabilities"]),
        "random_state": seed,
    }
    y_train = train[disease].astype(int)
    splits = stratified_oof_splits(
        y_train,
        requested_splits=int(project["n_stage1_oof_folds"]),
        seed=seed,
    )
    train_probability = np.full(len(train), np.nan, dtype=float)
    oof_fold = np.full(len(train), -1, dtype=int)
    feature_frames: list[pd.DataFrame] = []
    include_covariates = bool(arm.covariates)
    for fold, (fit_indices, holdout_indices) in enumerate(splits):
        probability, features = _fit_predict(
            classifier_cls=classifier_cls,
            context=train.iloc[fit_indices],
            y_context=y_train.iloc[fit_indices],
            eval_frame=train.iloc[holdout_indices],
            prs_cols=list(arm.prs_columns),
            covariates=list(arm.covariates),
            include_covariates=include_covariates,
            device=device,
            classifier_kwargs=classifier_kwargs,
            model_version=model_version,
        )
        train_probability[holdout_indices] = probability
        oof_fold[holdout_indices] = fold
        feature_frames.append(features.assign(score_fit_scope="train_oof", oof_fold=fold))
    if np.isnan(train_probability).any() or (oof_fold < 0).any():
        raise ValueError("TabPFN Stage-1 OOF scoring did not cover every training patient")

    validation_probability, validation_features = _fit_predict(
        classifier_cls=classifier_cls,
        context=train,
        y_context=y_train,
        eval_frame=validation,
        prs_cols=list(arm.prs_columns),
        covariates=list(arm.covariates),
        include_covariates=include_covariates,
        device=device,
        classifier_kwargs=classifier_kwargs,
        model_version=model_version,
    )
    feature_frames.append(
        validation_features.assign(score_fit_scope="validation_full_context", oof_fold=-1)
    )
    train_score, validation_score, logit_mean, logit_sd = scaled_logit_scores(
        train_probability,
        validation_probability,
        clip_eps=float(tabpfn["clip_eps"]),
    )
    audit = {
        "model_family": "tabpfn",
        "model_version": model_version_label,
        "model_path": str(model_path),
        "n_estimators": int(settings["n_estimators"]),
        "balance_probabilities": bool(settings["balance_probabilities"]),
        "n_oof_folds": len(splits),
        "clip_eps": float(tabpfn["clip_eps"]),
        "score_scale": str(tabpfn["score_scale"]),
        "logit_train_mean": float(logit_mean),
        "logit_train_sd": float(logit_sd),
    }
    return (
        train_probability,
        validation_probability,
        train_score,
        validation_score,
        oof_fold,
        pd.concat(feature_frames, ignore_index=True),
        audit,
    )


def _xgboost_stage1(
    *,
    config: dict[str, Any],
    disease: str,
    arm: OfficialStage1Arm,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    n_jobs: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]
]:
    project = _mapping(config, "project")
    xgboost = _mapping(config, "xgboost")
    tabpfn = _mapping(config, "tabpfn")
    seed = int(project["seed"])
    splits = stratified_oof_splits(
        train[disease].astype(int),
        requested_splits=int(project["n_stage1_oof_folds"]),
        seed=seed,
    )
    parameter_grid = {
        "max_depth": list(xgboost["max_depth"]),
        "learning_rate": list(xgboost["learning_rate"]),
        "min_child_weight": list(xgboost["min_child_weight"]),
    }
    fixed_parameters = {
        "n_estimators": int(xgboost["n_estimators"]),
        "subsample": float(xgboost["subsample"]),
        "colsample_bytree": float(xgboost["colsample_bytree"]),
        "reg_lambda": float(xgboost["reg_lambda"]),
    }
    train_probability = np.full(len(train), np.nan, dtype=float)
    oof_fold = np.full(len(train), -1, dtype=int)
    tuning_frames: list[pd.DataFrame] = []
    best_by_scope: list[dict[str, Any]] = []
    for fold, (fit_indices, holdout_indices) in enumerate(splits):
        probability, _, tuning, best = fit_tuned_xgboost(
            context=train.iloc[fit_indices],
            validation=train.iloc[holdout_indices],
            label_col=disease,
            prs_columns=arm.prs_columns,
            covariates=arm.covariates,
            parameter_grid=parameter_grid,
            fixed_parameters=fixed_parameters,
            inner_folds=int(xgboost["inner_folds"]),
            seed=seed + fold,
            n_jobs=n_jobs,
        )
        train_probability[holdout_indices] = probability
        oof_fold[holdout_indices] = fold
        tuning_frames.append(tuning.assign(score_fit_scope="train_oof", oof_fold=fold))
        best_by_scope.append({"score_fit_scope": "train_oof", "oof_fold": fold, **best})
    if np.isnan(train_probability).any() or (oof_fold < 0).any():
        raise ValueError("XGBoost Stage-1 OOF scoring did not cover every training patient")

    validation_probability, _, tuning, best = fit_tuned_xgboost(
        context=train,
        validation=validation,
        label_col=disease,
        prs_columns=arm.prs_columns,
        covariates=arm.covariates,
        parameter_grid=parameter_grid,
        fixed_parameters=fixed_parameters,
        inner_folds=int(xgboost["inner_folds"]),
        seed=seed,
        n_jobs=n_jobs,
    )
    tuning_frames.append(tuning.assign(score_fit_scope="validation_full_context", oof_fold=-1))
    best_by_scope.append({"score_fit_scope": "validation_full_context", "oof_fold": -1, **best})
    train_score, validation_score, logit_mean, logit_sd = scaled_logit_scores(
        train_probability,
        validation_probability,
        clip_eps=float(tabpfn["clip_eps"]),
    )
    audit = {
        "model_family": "xgboost",
        "n_oof_folds": len(splits),
        "nested_tuning_folds": int(xgboost["inner_folds"]),
        "fixed_parameters": fixed_parameters,
        "best_parameters_by_scope": best_by_scope,
        "clip_eps": float(tabpfn["clip_eps"]),
        "score_scale": str(tabpfn["score_scale"]),
        "logit_train_mean": float(logit_mean),
        "logit_train_sd": float(logit_sd),
    }
    return (
        train_probability,
        validation_probability,
        train_score,
        validation_score,
        oof_fold,
        pd.concat(tuning_frames, ignore_index=True),
        audit,
    )


def _raw_stage1(
    *,
    arm: OfficialStage1Arm,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]
]:
    if len(arm.prs_columns) != 1:
        raise ValueError("Raw Stage-1 arm requires exactly one PRS column")
    column = arm.prs_columns[0]
    train_score = pd.to_numeric(train[column], errors="raise").to_numpy(dtype=float)
    validation_score = pd.to_numeric(validation[column], errors="raise").to_numpy(dtype=float)
    return (
        np.full(len(train), np.nan),
        np.full(len(validation), np.nan),
        train_score,
        validation_score,
        np.full(len(train), -1, dtype=int),
        pd.DataFrame(),
        {
            "model_family": "raw_prs",
            "raw_score_column": column,
            "n_oof_folds": 0,
            "score_scale": "standardized inside the Stage-2 training-only pipeline",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one leakage-safe official-validation Stage-1 arm and the fixed "
            "Stage-2 score-plus-full-metadata wrapper."
        )
    )
    parser.add_argument("--config", default="configs/official_mechanism_ablations.yaml")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    disease, arm, task_id = _resolve_task(
        config=config,
        task_id=args.task_id,
        disease=args.disease,
        arm_name=args.arm,
    )
    project = _mapping(config, "project")
    paths = _mapping(config, "paths")
    columns = _mapping(config, "columns")
    metadata = _mapping(config, "metadata")
    stage2 = _mapping(config, "stage2")
    patient_col = str(columns["patient_id"])
    final_metadata = _list(metadata, "full")
    expected_stage2_features = [*final_metadata, "stage1_score"]
    if _list(stage2, "features") != expected_stage2_features:
        raise ValueError("Stage-2 features must be exactly full metadata followed by stage1_score")
    if str(stage2["class_weight"]) != "none":
        raise ValueError("Official mechanism Stage 2 must use class_weight=none")

    print(f"Running task {task_id}: disease={disease}, arm={arm.name}", flush=True)
    train, validation, data_audit = _load_disease_data(config=config, disease=disease)
    n_jobs = args.n_jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    if arm.model_family == "tabpfn":
        outputs = _tabpfn_stage1(
            config=config,
            disease=disease,
            arm=arm,
            train=train,
            validation=validation,
            device=args.device,
        )
    elif arm.model_family == "xgboost":
        outputs = _xgboost_stage1(
            config=config,
            disease=disease,
            arm=arm,
            train=train,
            validation=validation,
            n_jobs=n_jobs,
        )
    elif arm.model_family == "raw_prs":
        outputs = _raw_stage1(arm=arm, train=train, validation=validation)
    else:
        raise ValueError(f"Unsupported model family: {arm.model_family}")

    (
        train_probability,
        validation_probability,
        train_score,
        validation_score,
        oof_fold,
        feature_or_tuning_audit,
        stage1_audit,
    ) = outputs
    train_stage2 = train.copy()
    validation_stage2 = validation.copy()
    train_stage2["stage1_score"] = train_score
    validation_stage2["stage1_score"] = validation_score
    prediction = fit_predict_logistic(
        train=train_stage2,
        val=validation_stage2,
        label_col=disease,
        feature_cols=expected_stage2_features,
        patient_col=patient_col,
        method=arm.name,
        fold=0,
        class_weight=None,
    ).predictions
    prediction["variant"] = arm.name
    prediction["model_family"] = arm.model_family
    prediction["stage1_score"] = validation_score
    prediction["score"] = prediction["y_prob"]
    metric_row = {
        "disease": disease,
        "variant": arm.name,
        "model_family": arm.model_family,
        **prediction_metrics(prediction),
    }

    train_scores = pd.DataFrame(
        {
            patient_col: train[patient_col].to_numpy(),
            "disease": disease,
            "y_true": train[disease].astype(int).to_numpy(),
            "stage1_probability": train_probability,
            "stage1_score": train_score,
            "oof_fold": oof_fold,
        }
    )
    validation_scores = pd.DataFrame(
        {
            patient_col: validation[patient_col].to_numpy(),
            "disease": disease,
            "y_true": validation[disease].astype(int).to_numpy(),
            "stage1_probability": validation_probability,
            "stage1_score": validation_score,
            "score_fit_scope": "official_train_full_context",
        }
    )
    out_dir = Path(str(paths["output_root"])) / arm.name / disease
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(train_scores, out_dir / "train_stage1_scores.csv")
    write_table(validation_scores, out_dir / "validation_stage1_scores.csv")
    write_table(prediction, out_dir / "official_validation_predictions.csv")
    write_table(pd.DataFrame([metric_row]), out_dir / "point_metrics.csv")
    if not feature_or_tuning_audit.empty:
        audit_name = "xgboost_tuning.csv" if arm.model_family == "xgboost" else "feature_audit.csv"
        write_table(feature_or_tuning_audit, out_dir / audit_name)

    run_record = {
        "config": args.config,
        "task_id": task_id,
        "disease": disease,
        "arm": arm.name,
        "description": arm.description,
        "model_family": arm.model_family,
        "prs_columns": list(arm.prs_columns),
        "stage1_covariates": list(arm.covariates),
        "stage1": stage1_audit,
        "stage2": {
            "model": str(stage2["model"]),
            "solver": str(stage2["solver"]),
            "max_iter": int(stage2["max_iter"]),
            "class_weight": None,
            "features": expected_stage2_features,
            "fit_cohort": "official training",
            "prediction_cohort": "official validation",
        },
        "data_audit": data_audit,
        "seed": int(project["seed"]),
    }
    (out_dir / "run_record.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Finished task {task_id}: disease={disease}, arm={arm.name}, "
        f"official_validation_auc={metric_row['auc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
