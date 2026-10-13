#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decipehr.ancestry_performance import (
    paired_bootstrap_comparisons,
    subgroup_metrics,
)
from decipehr.config import load_yaml
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.io import read_table, write_table
from decipehr.mechanism_ablations import (
    align_paired_predictions,
    paired_stratified_bootstrap,
    prediction_metrics,
    validate_prediction_frame,
)
from decipehr.models import fit_predict_logistic
from decipehr.official_mechanism import (
    Q3_ADDITIVE_ARM,
    Q3_METADATA_SCORE_COMPONENT,
    Q3_RAW_PRS_COMPONENT,
    OfficialStage1Arm,
    build_official_stage1_arms,
    official_comparison_specs,
    q3_additive_stage2_features,
)
from decipehr.schema import require_columns, validate_unique

ADDITIVE_ARM = Q3_ADDITIVE_ARM
ANCESTRY_MODELS = {
    "best_single_prs": "Raw best single PRS",
    "tabpfn_best_single": "TabPFN-3 single best + metadata",
    "tabpfn_nonannotation": "TabPFN-3 five non-annotation PRSs",
    "tabpfn_all8": "TabPFN-3 all eight PRSs",
}
ANCESTRY_COMPARISONS = (
    ("tabpfn_single_vs_raw_single", "best_single_prs", "tabpfn_best_single"),
    (
        "nonannotation_vs_tabpfn_single",
        "tabpfn_best_single",
        "tabpfn_nonannotation",
    ),
    ("all8_vs_nonannotation", "tabpfn_nonannotation", "tabpfn_all8"),
    ("all8_vs_raw_single", "best_single_prs", "tabpfn_all8"),
)


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
    columns = _mapping(config, "columns")
    return build_official_stage1_arms(
        best_prs=str(_mapping(prs, "canonical_best_single")[disease]),
        nonannotation_prs=_list(prs, "nonannotation"),
        all_prs=_list(prs, "all"),
        full_metadata=_list(metadata, "full"),
        sex=str(metadata["sex"]),
        age=str(metadata["age"]),
        pcs=_list(metadata, "pcs"),
        ancestry_group=str(columns["ancestry_group_feature"]),
    )


def _load_eligible_phenotypes(
    config: dict[str, Any], disease: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _mapping(config, "paths")
    columns = _mapping(config, "columns")
    metadata = _mapping(config, "metadata")
    patient_col = str(columns["patient_id"])
    full_metadata = _list(metadata, "full")
    train = read_table(str(paths["train_phenotype"]))
    validation = read_table(str(paths["validation_phenotype"]))
    require_columns(train, [patient_col, disease, *full_metadata], "training phenotype")
    require_columns(
        validation,
        [patient_col, disease, *full_metadata],
        "validation phenotype",
    )
    rules = load_eligibility_rules(str(paths["eligibility_config"]))
    return (
        filter_eligible(train, disease, rules).reset_index(drop=True),
        filter_eligible(validation, disease, rules).reset_index(drop=True),
    )


def _derive_additive_arm(config: dict[str, Any], disease: str, output_root: Path) -> None:
    project = _mapping(config, "project")
    columns = _mapping(config, "columns")
    metadata = _mapping(config, "metadata")
    prs = _mapping(config, "prs")
    patient_col = str(columns["patient_id"])
    final_metadata = _list(metadata, "full")
    raw_arm = "raw_best_prs"
    metadata_arm = "tabpfn_metadata_only"
    raw_dir = output_root / raw_arm / disease
    metadata_dir = output_root / metadata_arm / disease
    raw_train = read_table(raw_dir / "train_stage1_scores.csv")
    raw_validation = read_table(raw_dir / "validation_stage1_scores.csv")
    metadata_train = read_table(metadata_dir / "train_stage1_scores.csv")
    metadata_validation = read_table(metadata_dir / "validation_stage1_scores.csv")
    require_columns(
        raw_train,
        [patient_col, "y_true", "stage1_score"],
        f"{raw_arm} training component",
    )
    require_columns(
        raw_validation,
        [patient_col, "y_true", "stage1_score", "score_fit_scope"],
        f"{raw_arm} validation component",
    )
    require_columns(
        metadata_train,
        [patient_col, "y_true", "stage1_score", "oof_fold"],
        f"{metadata_arm} training component",
    )
    require_columns(
        metadata_validation,
        [patient_col, "y_true", "stage1_score", "score_fit_scope"],
        f"{metadata_arm} validation component",
    )
    expected_oof_folds = set(range(int(project["n_stage1_oof_folds"])))
    observed_oof_folds = set(
        pd.to_numeric(metadata_train["oof_fold"], errors="raise").astype(int).unique()
    )
    if observed_oof_folds != expected_oof_folds:
        raise ValueError(
            f"{disease}: metadata-only training component must cover OOF folds "
            f"{sorted(expected_oof_folds)}; got {sorted(observed_oof_folds)}"
        )
    for component_name, frame in [
        (raw_arm, raw_validation),
        (metadata_arm, metadata_validation),
    ]:
        observed_scopes = set(frame["score_fit_scope"].astype(str).unique())
        if observed_scopes != {"official_train_full_context"}:
            raise ValueError(
                f"{disease}: {component_name} validation score scope mismatch: "
                f"{sorted(observed_scopes)}"
            )

    train_components = (
        raw_train[[patient_col, "y_true", "stage1_score"]]
        .rename(columns={"stage1_score": Q3_RAW_PRS_COMPONENT})
        .merge(
            metadata_train[[patient_col, "y_true", "stage1_score", "oof_fold"]].rename(
                columns={"stage1_score": Q3_METADATA_SCORE_COMPONENT}
            ),
            on=[patient_col, "y_true"],
            how="inner",
            validate="one_to_one",
        )
    )
    validation_components = (
        raw_validation[[patient_col, "y_true", "stage1_score"]]
        .rename(columns={"stage1_score": Q3_RAW_PRS_COMPONENT})
        .merge(
            metadata_validation[[patient_col, "y_true", "stage1_score"]].rename(
                columns={"stage1_score": Q3_METADATA_SCORE_COMPONENT}
            ),
            on=[patient_col, "y_true"],
            how="inner",
            validate="one_to_one",
        )
    )
    if len(train_components) != len(raw_train) or len(train_components) != len(metadata_train):
        raise ValueError(f"{disease}: additive training component patient sets differ")
    if len(validation_components) != len(raw_validation) or len(validation_components) != len(
        metadata_validation
    ):
        raise ValueError(f"{disease}: additive validation component patient sets differ")

    train_scores = train_components[[patient_col, "y_true"]].copy()
    train_scores.insert(1, "disease", disease)
    train_scores[Q3_RAW_PRS_COMPONENT] = train_components[Q3_RAW_PRS_COMPONENT]
    train_scores[Q3_METADATA_SCORE_COMPONENT] = train_components[Q3_METADATA_SCORE_COMPONENT]
    train_scores["oof_fold"] = train_components["oof_fold"].astype(int)
    validation_scores = validation_components[[patient_col, "y_true"]].copy()
    validation_scores.insert(1, "disease", disease)
    validation_scores[Q3_RAW_PRS_COMPONENT] = validation_components[Q3_RAW_PRS_COMPONENT]
    validation_scores[Q3_METADATA_SCORE_COMPONENT] = validation_components[
        Q3_METADATA_SCORE_COMPONENT
    ]
    validation_scores["score_fit_scope"] = "official_train_full_context"

    train_phenotype, validation_phenotype = _load_eligible_phenotypes(config, disease)
    train_stage2 = train_phenotype.merge(
        train_scores[[patient_col, "y_true", Q3_METADATA_SCORE_COMPONENT, Q3_RAW_PRS_COMPONENT]],
        on=patient_col,
        how="inner",
        validate="one_to_one",
    )
    validation_stage2 = validation_phenotype.merge(
        validation_scores[
            [patient_col, "y_true", Q3_METADATA_SCORE_COMPONENT, Q3_RAW_PRS_COMPONENT]
        ],
        on=patient_col,
        how="inner",
        validate="one_to_one",
    )
    if len(train_stage2) != len(train_phenotype) or len(validation_stage2) != len(
        validation_phenotype
    ):
        raise ValueError(f"{disease}: additive scores do not cover eligible phenotypes")
    for cohort, frame in [("training", train_stage2), ("validation", validation_stage2)]:
        if not np.array_equal(
            frame[disease].astype(int).to_numpy(),
            frame["y_true"].astype(int).to_numpy(),
        ):
            raise ValueError(f"{disease}: {cohort} component labels do not match phenotype")
    stage2_features = list(q3_additive_stage2_features(final_metadata))
    prediction = fit_predict_logistic(
        train=train_stage2,
        val=validation_stage2,
        label_col=disease,
        feature_cols=stage2_features,
        patient_col=patient_col,
        method=ADDITIVE_ARM,
        fold=0,
        class_weight=None,
    ).predictions
    prediction["variant"] = ADDITIVE_ARM
    prediction["model_family"] = "logistic_additive_components"
    prediction[Q3_RAW_PRS_COMPONENT] = validation_stage2[Q3_RAW_PRS_COMPONENT].to_numpy()
    prediction[Q3_METADATA_SCORE_COMPONENT] = validation_stage2[
        Q3_METADATA_SCORE_COMPONENT
    ].to_numpy()
    prediction["score"] = prediction["y_prob"]
    metric_row = {
        "disease": disease,
        "variant": ADDITIVE_ARM,
        "model_family": "logistic_additive_components",
        **prediction_metrics(prediction),
    }
    out_dir = output_root / ADDITIVE_ARM / disease
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(train_scores, out_dir / "train_stage1_scores.csv")
    write_table(validation_scores, out_dir / "validation_stage1_scores.csv")
    write_table(prediction, out_dir / "official_validation_predictions.csv")
    write_table(pd.DataFrame([metric_row]), out_dir / "point_metrics.csv")
    (out_dir / "run_record.json").write_text(
        json.dumps(
            {
                "disease": disease,
                "arm": ADDITIVE_ARM,
                "description": (
                    "Final regression with full metadata plus two separate components: "
                    "the metadata-only TabPFN score and the raw best single PRS"
                ),
                "components": {
                    Q3_RAW_PRS_COMPONENT: {
                        "source_arm": raw_arm,
                        "prs_family": str(_mapping(prs, "canonical_best_single")[disease]),
                        "role": "independent raw PRS regression component",
                    },
                    Q3_METADATA_SCORE_COMPONENT: {
                        "source_arm": metadata_arm,
                        "training_score_scope": "five-fold OOF within official training",
                        "validation_score_scope": "complete official training context",
                        "role": "independent metadata-only TabPFN regression component",
                    },
                },
                "stage2": {
                    "model": "logistic_regression",
                    "class_weight": None,
                    "features": stage2_features,
                    "components_combined_before_regression": False,
                    "component_coefficients": "learned separately in the same regression",
                    "fit_cohort": "official training",
                    "prediction_cohort": "official validation",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _collect_outputs(
    config: dict[str, Any], output_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project = _mapping(config, "project")
    metadata = _mapping(config, "metadata")
    columns = _mapping(config, "columns")
    diseases = _list(project, "diseases")
    patient_col = str(columns["patient_id"])
    full_metadata = _list(metadata, "full")
    standard_stage2_features = [*full_metadata, "stage1_score"]
    additive_stage2_features = list(q3_additive_stage2_features(full_metadata))
    prediction_frames = []
    contract_rows = []
    expected_variants: list[str] | None = None
    for disease in diseases:
        arms = list(_arms_for_disease(config, disease))
        arm_names = [arm.name for arm in arms] + [ADDITIVE_ARM]
        if expected_variants is None:
            expected_variants = arm_names
        elif arm_names != expected_variants:
            raise ValueError("Expected arm names differ among diseases")
        arm_lookup = {arm.name: arm for arm in arms}
        for arm_name in arm_names:
            out_dir = output_root / arm_name / disease
            prediction = read_table(out_dir / "official_validation_predictions.csv")
            prediction_frames.append(prediction)
            record = json.loads((out_dir / "run_record.json").read_text(encoding="utf-8"))
            observed_features = list(record["stage2"]["features"])
            expected_features = (
                additive_stage2_features if arm_name == ADDITIVE_ARM else standard_stage2_features
            )
            if observed_features != expected_features:
                raise ValueError(f"{disease}/{arm_name}: Stage-2 feature contract mismatch")
            if record["stage2"]["class_weight"] is not None:
                raise ValueError(f"{disease}/{arm_name}: Stage 2 must use class_weight=None")
            if arm_name == ADDITIVE_ARM:
                best_prs = str(_mapping(_mapping(config, "prs"), "canonical_best_single")[disease])
                components = record.get("components", {})
                raw_component = components.get(Q3_RAW_PRS_COMPONENT, {})
                metadata_component = components.get(Q3_METADATA_SCORE_COMPONENT, {})
                if raw_component.get("source_arm") != "raw_best_prs":
                    raise ValueError(f"{disease}/{arm_name}: raw PRS component source mismatch")
                if raw_component.get("prs_family") != best_prs:
                    raise ValueError(f"{disease}/{arm_name}: best single PRS family mismatch")
                if metadata_component.get("source_arm") != "tabpfn_metadata_only":
                    raise ValueError(
                        f"{disease}/{arm_name}: metadata-only TabPFN component source mismatch"
                    )
                model_family = "logistic_additive_components"
                prs_columns = f"none in TabPFN; raw {best_prs} is a separate Stage-2 component"
                stage1_covariates = ";".join(full_metadata)
                description = str(record["description"])
            else:
                arm = arm_lookup[arm_name]
                model_family = arm.model_family
                prs_columns = ";".join(arm.prs_columns)
                stage1_covariates = ";".join(arm.covariates)
                description = arm.description
                if list(record["prs_columns"]) != list(arm.prs_columns):
                    raise ValueError(f"{disease}/{arm_name}: PRS input contract mismatch")
                if list(record["stage1_covariates"]) != list(arm.covariates):
                    raise ValueError(f"{disease}/{arm_name}: Stage-1 covariate mismatch")
            contract_rows.append(
                {
                    "disease": disease,
                    "variant": arm_name,
                    "model_family": model_family,
                    "stage1_prs": prs_columns,
                    "stage1_covariates": stage1_covariates,
                    "stage2_features": ";".join(observed_features),
                    "evaluation_cohort": "official validation",
                    "description": description,
                }
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    validate_prediction_frame(
        predictions,
        patient_col=patient_col,
        expected_variants=expected_variants,
    )
    metric_rows = []
    for (disease, variant), frame in predictions.groupby(["disease", "variant"], sort=True):
        metric_rows.append({"disease": disease, "variant": variant, **prediction_metrics(frame)})
    return predictions, pd.DataFrame(metric_rows), pd.DataFrame(contract_rows)


def _paired_comparisons(
    *,
    config: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project = _mapping(config, "project")
    summary = _mapping(config, "summary")
    columns = _mapping(config, "columns")
    diseases = _list(project, "diseases")
    patient_col = str(columns["patient_id"])
    n_resamples = int(summary["paired_bootstrap_resamples"])
    seed = int(project["seed"])
    lookup = metrics.set_index(["disease", "variant"])
    jobs = [
        (comparison_index, spec, disease_index, disease)
        for comparison_index, spec in enumerate(official_comparison_specs())
        for disease_index, disease in enumerate(diseases)
    ]

    def run_job(job):
        comparison_index, spec, disease_index, disease = job
        paired = align_paired_predictions(
            predictions,
            disease=disease,
            reference=spec.reference,
            challenger=spec.challenger,
            patient_col=patient_col,
        )
        bootstrap = paired_stratified_bootstrap(
            paired,
            reference=spec.reference,
            challenger=spec.challenger,
            n_resamples=n_resamples,
            seed=seed + comparison_index * 100 + disease_index,
            metrics=(
                "auc",
                "brier",
                "nagelkerke_r2",
                "top10_vs_lower90_or",
            ),
        ).assign(
            question=spec.question,
            comparison=spec.comparison,
            disease=disease,
            reference=spec.reference,
            challenger=spec.challenger,
            interpretation=spec.interpretation,
        )
        reference_metrics = lookup.loc[(disease, spec.reference)]
        challenger_metrics = lookup.loc[(disease, spec.challenger)]
        row: dict[str, Any] = {
            "question": spec.question,
            "comparison": spec.comparison,
            "disease": disease,
            "reference": spec.reference,
            "challenger": spec.challenger,
            "interpretation": spec.interpretation,
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

    with ThreadPoolExecutor(max_workers=int(summary["n_jobs"])) as executor:
        results = list(executor.map(run_job, jobs))
    return (
        pd.DataFrame([row for row, _ in results]),
        pd.concat([table for _, table in results], ignore_index=True),
    )


def _ancestry_analysis(
    *,
    config: dict[str, Any],
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project = _mapping(config, "project")
    paths = _mapping(config, "paths")
    columns = _mapping(config, "columns")
    summary = _mapping(config, "summary")
    patient_col = str(columns["patient_id"])
    ancestry_patient_col = str(columns["ancestry_patient_id"])
    ancestry_source = str(columns["ancestry_group_source"])
    ancestry = read_table(str(paths["ancestry_labels"]))[
        [ancestry_patient_col, ancestry_source]
    ].rename(columns={ancestry_patient_col: patient_col, ancestry_source: "ancestry_group"})
    validate_unique(ancestry, patient_col, "ancestry labels")
    variant_map = {
        "raw_best_prs": "best_single_prs",
        "tabpfn_best_prs_full_metadata": "tabpfn_best_single",
        "tabpfn_nonannotation_full_metadata": "tabpfn_nonannotation",
        "tabpfn_all8_full_metadata": "tabpfn_all8",
    }
    patient_frames = []
    metric_frames = []
    comparison_frames = []
    for disease_index, disease in enumerate(_list(project, "diseases")):
        subset = predictions.loc[
            predictions["disease"].astype(str).eq(disease)
            & predictions["variant"].isin(variant_map),
            [patient_col, "variant", "y_true", "y_prob"],
        ].copy()
        wide = subset.pivot(index=[patient_col, "y_true"], columns="variant", values="y_prob")
        wide = wide.rename(columns=variant_map).reset_index()
        if wide[list(variant_map.values())].isna().any().any():
            raise ValueError(f"{disease}: ancestry model predictions do not align")
        wide.insert(1, "disease", disease)
        wide = wide.merge(ancestry, on=patient_col, how="left", validate="one_to_one")
        if wide["ancestry_group"].isna().any():
            raise ValueError(f"{disease}: ancestry labels are incomplete")
        patient_frames.append(wide)
        metric_frames.append(
            subgroup_metrics(
                wide,
                group_col="ancestry_group",
                model_columns=ANCESTRY_MODELS,
            )
        )
        comparison_frames.append(
            paired_bootstrap_comparisons(
                wide,
                n_bootstraps=int(summary["paired_bootstrap_resamples"]),
                seed=int(project["seed"]) + disease_index * 100,
                group_col="ancestry_group",
                comparisons=ANCESTRY_COMPARISONS,
                metric_names=("auc",),
            )
        )
    return (
        pd.concat(patient_frames, ignore_index=True),
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(comparison_frames, ignore_index=True),
    )


def _fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _question_table(comparisons: pd.DataFrame, bootstrap: pd.DataFrame, question: int) -> str:
    table = comparisons.loc[comparisons["question"].astype(int).eq(question)].copy()
    official_metrics = (
        ("auc", "AUC"),
        ("top10_vs_lower90_or", "OR90"),
        ("nagelkerke_r2", "Nagelkerke R²"),
    )
    for metric, _ in official_metrics:
        inference = bootstrap.loc[
            bootstrap["question"].astype(int).eq(question) & bootstrap["metric"].eq(metric),
            ["comparison", "disease", "ci_lower", "ci_upper", "p_two_sided"],
        ].rename(
            columns={
                "ci_lower": f"{metric}_ci_lower",
                "ci_upper": f"{metric}_ci_upper",
                "p_two_sided": f"{metric}_p_two_sided",
            }
        )
        table = table.merge(inference, on=["comparison", "disease"], validate="one_to_one")

    def delta_with_ci(row: pd.Series, metric: str) -> str:
        return (
            f"{_fmt(row[f'delta_{metric}'], signed=True)} "
            f"[{_fmt(row[f'{metric}_ci_lower'], signed=True)}, "
            f"{_fmt(row[f'{metric}_ci_upper'], signed=True)}]"
        )

    display = pd.DataFrame(
        {
            "Comparison": table["comparison"],
            "Disease": table["disease"].str.upper(),
            "N": table["n"].astype(int),
        }
    )
    for metric, label in official_metrics:
        display[f"Reference {label}"] = table[f"reference_{metric}"].map(_fmt)
        display[f"Challenger {label}"] = table[f"challenger_{metric}"].map(_fmt)
        display[f"Delta {label} [95% CI]"] = table.apply(
            lambda row, current_metric=metric: delta_with_ci(row, current_metric),
            axis=1,
        )
        display[f"{label} paired p"] = table[f"{metric}_p_two_sided"].map(
            lambda value: _fmt(value, digits=3)
        )
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def _ancestry_table(metrics: pd.DataFrame) -> str:
    auc = metrics[["disease", "ancestry_group", "model", "n", "cases", "auc"]].copy()
    wide = auc.pivot(
        index=["disease", "ancestry_group", "n", "cases"],
        columns="model",
        values="auc",
    ).reset_index()
    display = pd.DataFrame(
        {
            "Disease": wide["disease"].str.upper(),
            "Ancestry group": wide["ancestry_group"],
            "N/cases": wide.apply(lambda row: f"{int(row['n'])}/{int(row['cases'])}", axis=1),
            "Raw single": wide["best_single_prs"].map(lambda value: _fmt(value, 3)),
            "TabPFN single": wide["tabpfn_best_single"].map(lambda value: _fmt(value, 3)),
            "Five non-annotation": wide["tabpfn_nonannotation"].map(lambda value: _fmt(value, 3)),
            "All 8": wide["tabpfn_all8"].map(lambda value: _fmt(value, 3)),
        }
    )
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def _ancestry_gain_table(comparisons: pd.DataFrame) -> str:
    primary = comparisons.loc[
        comparisons["comparison"].eq("tabpfn_single_vs_raw_single")
        & comparisons["metric"].eq("auc")
    ].copy()
    display = pd.DataFrame(
        {
            "Disease": primary["disease"].str.upper(),
            "Ancestry group": primary["ancestry_group"],
            "N/cases": primary.apply(lambda row: f"{int(row['n'])}/{int(row['cases'])}", axis=1),
            "Raw single AUC": primary["reference_value"].map(lambda value: _fmt(value, 3)),
            "TabPFN single AUC": primary["candidate_value"].map(lambda value: _fmt(value, 3)),
            "Delta AUC [95% CI]": primary.apply(
                lambda row: (
                    f"{_fmt(row['candidate_minus_reference'], 4, signed=True)} "
                    f"[{_fmt(row['ci_low'], 4, signed=True)}, "
                    f"{_fmt(row['ci_high'], 4, signed=True)}]"
                ),
                axis=1,
            ),
        }
    )
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def _render_html(
    *,
    config: dict[str, Any],
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    bootstrap: pd.DataFrame,
    contract: pd.DataFrame,
    ancestry_metrics: pd.DataFrame,
    ancestry_comparisons: pd.DataFrame,
) -> str:
    project = _mapping(config, "project")
    tabpfn = _mapping(config, "tabpfn")
    metric_display = metrics[
        ["disease", "variant", "n", "auc", "auprc", "brier", "nagelkerke_r2", "top10_vs_lower90_or"]
    ].copy()
    for column in ["auc", "auprc", "brier", "nagelkerke_r2", "top10_vs_lower90_or"]:
        metric_display[column] = metric_display[column].map(_fmt)
    metric_display["disease"] = metric_display["disease"].str.upper()
    metric_display = metric_display.rename(
        columns={
            "disease": "Disease",
            "variant": "Arm",
            "n": "N",
            "auc": "AUC",
            "auprc": "AUPRC",
            "brier": "Brier",
            "nagelkerke_r2": "Nagelkerke R²",
            "top10_vs_lower90_or": "OR90",
        }
    )
    contract_display = contract[
        [
            "disease",
            "variant",
            "stage1_prs",
            "stage1_covariates",
            "stage2_features",
            "evaluation_cohort",
        ]
    ].copy()
    contract_display = contract_display.rename(
        columns={
            "disease": "Disease",
            "variant": "Arm",
            "stage1_prs": "Stage 1 PRS",
            "stage1_covariates": "Stage 1 metadata",
            "stage2_features": "Stage 2 inputs",
            "evaluation_cohort": "Evaluation",
        }
    )
    question_titles = {
        1: "Single best, five non-annotation PRSs, and all eight PRSs",
        2: "Which metadata contribute inside the single-best-PRS Stage-1 score?",
        3: "Joint single-PRS–metadata modeling versus separate regression components",
        4: "TabPFN versus matched nested-tuned XGBoost",
        5: "Continuous PCs versus hard ancestry with the same single best PRS",
    }
    question_contracts = {
        1: (
            "Every row below is the final official-validation result from "
            "<code>M + Stage-1 score</code>, where "
            "<code>M = sex + age_sd + PC1_sd-PC10_sd</code>."
        ),
        2: (
            "Every Stage-1 arm uses the same disease-specific single-best PRS. Metadata are "
            "ablated only inside its TabPFN score; every final regression restores "
            "<code>M + Stage-1 score</code> before official-validation evaluation."
        ),
        3: (
            "<strong>Joint:</strong> <code>M + TabPFN(raw single-best PRS, M)</code>. "
            "<strong>Separate components:</strong> <code>M + TabPFN(M) + raw single-best "
            "PRS</code>. The metadata-only TabPFN score and raw PRS are two independent "
            "final-regression columns with separately learned coefficients; they are never "
            "added together beforehand. Joint is the reference, so deltas are "
            "<code>separate components − joint</code>."
        ),
        4: (
            "Both Stage-1 algorithms use the same five PRSs and metadata; every final "
            "regression uses <code>M + Stage-1 score</code>."
        ),
        5: (
            "Both Stage-1 arms use the same disease-specific single-best PRS. Only continuous "
            "PC1-PC10 versus one-hot hard ancestry differs inside Stage 1; every final "
            "regression uses <code>M + Stage-1 score</code>. Continuous PCs are the reference, "
            "so deltas are <code>hard ancestry − continuous PCs</code>."
        ),
    }
    question_sections = "".join(
        (
            f"<section id='q{question}'><h2>Question {question}. "
            f"{html.escape(title)}</h2>"
            f"<p>{question_contracts[question]}</p>"
            f"<div class='table-wrap'>{_question_table(comparisons, bootstrap, question)}</div>"
            "</section>"
        )
        for question, title in question_titles.items()
    )
    settings = _mapping(tabpfn, "disease_settings")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TabPFN-3 official-validation mechanism experiments</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #17202a; background: #f5f7fa; }}
main {{ max-width: 1500px; margin: auto; padding: 30px; }}
h1, h2 {{ color: #17365d; }} section {{ background: white; margin: 18px 0; padding: 22px; border-radius: 10px; box-shadow: 0 1px 5px #ccd3dc; }}
.alert {{ border-left: 6px solid #b42318; background: #fff1f0; padding: 14px; }}
.protocol {{ border-left: 6px solid #176b87; background: #eef8fb; padding: 14px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }} th, td {{ border: 1px solid #d8dee6; padding: 7px; text-align: left; }} th {{ background: #eaf0f7; position: sticky; top: 0; }}
.table-wrap {{ overflow-x: auto; }} code {{ background: #edf1f5; padding: 2px 4px; }}
</style></head><body><main>
<h1>TabPFN-3 mechanism experiments — corrected official-validation evaluation</h1>
<div class="alert"><strong>Superseded results:</strong> the earlier direct outer-OOF Question 1/3/4/5 metrics are not formal results and must not be used. This page contains only fixed-Stage-2 official-validation estimates.</div>
<section><h2>Canonical evaluation contract</h2>
<div class="protocol"><strong>Stage 1:</strong> TabPFN training scores are five-fold out-of-fold predictions within the official training cohort; official-validation scores are predictions from the complete official training context. <strong>Final regression:</strong> standard arms fit an unweighted logistic model on <code>M + Stage-1 score</code>, where <code>M = sex + age_sd + PC1_sd-PC10_sd</code>. The Question-3 separate-component reference instead fits <code>M + TabPFN(M) + raw single-best PRS</code>, retaining the last two inputs as separate columns. Only official-validation performance is reported.</div>
<p><strong>Official metrics shown for every ablation:</strong> AUC; OR90, the odds ratio comparing the top 10% versus the lower 90% of final Stage-2 predicted risk; and Nagelkerke R² from the same final Stage-2 probabilities. Each table reports both arms, the paired difference (<code>challenger − reference</code>), a 95% paired-bootstrap interval, and a two-sided paired p-value.</p>
<ul><li>Official training N: 5,811 (BC eligible-female N: 3,308).</li><li>Official validation N: 2,852 (BC eligible-female N: 1,628).</li><li>Five non-annotation PRSs: P+T, LDpred2, lassosum2, PRS-CS, and SDPR.</li><li>All-eight panel adds MegaPRS no annotation, MegaPRS annotated, and SBayesRC.</li><li>CAD: {settings["cad"]["n_estimators"]} estimators, default checkpoint; BC: {settings["bc"]["n_estimators"]} estimators, binary checkpoint with probability balancing; T2D: {settings["T2D"]["n_estimators"]} estimators, default checkpoint.</li></ul>
<p><strong>Common-baseline rule:</strong> Questions 2, 3, and 5 use <code>TabPFN(single-best PRS + M)</code> as their joint reference arm. Question 4 retains the matched five-PRS input for both TabPFN and XGBoost.</p>
<p><strong>Interpretation guardrail:</strong> the family-specific PRS settings were selected retrospectively using official-validation results. These are explanatory mechanism estimates, not an independent confirmatory benchmark.</p>
</section>
{question_sections}
<section><h2>Official-validation ancestry follow-up</h2><p>Ancestry is used only for post hoc stratification; every model was fit without subgroup-specific retraining. The primary comparison is TabPFN single-best plus metadata versus the raw same single-PRS wrapper.</p><h3>Primary ancestry gain: TabPFN single versus raw single</h3><div class="table-wrap">{_ancestry_gain_table(ancestry_comparisons)}</div><h3>Four-arm AUC ladder</h3><div class="table-wrap">{_ancestry_table(ancestry_metrics)}</div></section>
<section><h2>All final Stage-2 point metrics</h2><div class="table-wrap">{metric_display.to_html(index=False, border=0, classes="data-table", escape=True)}</div></section>
<section><h2>Audited input contract</h2><div class="table-wrap">{contract_display.to_html(index=False, border=0, classes="data-table", escape=True)}</div></section>
<section><h2>Reproducibility</h2><p>Seed: {int(project["seed"])}. Paired bootstrap resamples: {int(_mapping(config, "summary")["paired_bootstrap_resamples"])}. Machine-readable predictions, metrics, comparisons, input contracts, and ancestry results are stored under <code>{html.escape(str(_mapping(config, "paths")["output_root"]))}/summary</code>.</p></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize fixed-Stage-2 official-validation mechanism experiments."
    )
    parser.add_argument("--config", default="configs/official_mechanism_ablations.yaml")
    parser.add_argument(
        "--canonical-html",
        default="outputs/mechanism_ablations/summary/tabpfn_mechanism_ablations.html",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    paths = _mapping(config, "paths")
    output_root = Path(str(paths["output_root"]))
    diseases = _list(_mapping(config, "project"), "diseases")
    for disease in diseases:
        _derive_additive_arm(config, disease, output_root)
    predictions, metrics, contract = _collect_outputs(config, output_root)
    comparisons, bootstrap = _paired_comparisons(
        config=config,
        predictions=predictions,
        metrics=metrics,
    )
    ancestry_predictions, ancestry_metrics, ancestry_comparisons = _ancestry_analysis(
        config=config,
        predictions=predictions,
    )

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_table(predictions, summary_dir / "official_validation_predictions.csv")
    write_table(metrics, summary_dir / "point_metrics.csv")
    write_table(contract, summary_dir / "contract_audit.csv")
    write_table(comparisons, summary_dir / "comparison_point_metrics.csv")
    write_table(bootstrap, summary_dir / "paired_bootstrap_deltas.csv")
    write_table(ancestry_predictions, summary_dir / "ancestry_patient_predictions.csv")
    write_table(ancestry_metrics, summary_dir / "ancestry_group_metrics.csv")
    write_table(
        ancestry_comparisons,
        summary_dir / "ancestry_paired_bootstrap_comparisons.csv",
    )
    report = _render_html(
        config=config,
        metrics=metrics,
        comparisons=comparisons,
        bootstrap=bootstrap,
        contract=contract,
        ancestry_metrics=ancestry_metrics,
        ancestry_comparisons=ancestry_comparisons,
    )
    report_path = summary_dir / "tabpfn_official_mechanism_ablations.html"
    report_path.write_text(report, encoding="utf-8")
    canonical_path = Path(args.canonical_html)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(report, encoding="utf-8")
    (summary_dir / "run_record.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "evaluation": "official validation only",
                "stage1_training_scores": "five-fold OOF within official training",
                "standard_stage2_features": [
                    *_list(_mapping(config, "metadata"), "full"),
                    "stage1_score",
                ],
                "q3_additive_stage2_features": list(
                    q3_additive_stage2_features(_list(_mapping(config, "metadata"), "full"))
                ),
                "q3_components_are_separate": True,
                "question_2_3_5_joint_baseline": "tabpfn_best_prs_full_metadata",
                "prs_panels": {
                    "single_best": _mapping(_mapping(config, "prs"), "canonical_best_single"),
                    "all_nonannotation": _list(_mapping(config, "prs"), "nonannotation"),
                    "all8": _list(_mapping(config, "prs"), "all"),
                },
                "prs_selection_context": str(_mapping(config, "prs")["selection_context"]),
                "old_direct_oof_results_included": False,
                "canonical_html": str(canonical_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote official-validation mechanism summary to {report_path}")
    print(f"Updated canonical HTML at {canonical_path}")


if __name__ == "__main__":
    main()
