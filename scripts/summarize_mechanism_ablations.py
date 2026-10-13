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

from decipehr.config import load_yaml
from decipehr.io import read_table, write_table
from decipehr.mechanism_ablations import (
    align_paired_predictions,
    comparison_specs,
    paired_stratified_bootstrap,
    prediction_metrics,
    validate_prediction_frame,
)
from decipehr.schema import require_columns

QUESTION_TITLES = {
    1: "Is multi-PRS complementarity a core source of gain?",
    2: "Which metadata contribute inside the TabPFN score: PCs, age, or sex?",
    3: "Does joint modeling exceed additive nonlinear main effects?",
    4: "Is the gain TabPFN-specific or generic nonlinearity?",
    5: "Do continuous PCs outperform hard ancestry stratification?",
}

METADATA_SCORE_ARM_ORDER = ["full", "no_age", "no_sex", "no_pcs", "no_age_sex"]
METADATA_SCORE_ARM_DISPLAY = {
    "full": "Full score metadata",
    "no_age": "Score without age",
    "no_sex": "Score without sex",
    "no_pcs": "Score without PC1-PC10",
    "no_age_sex": "Score without age and sex",
}
ANCESTRY_PRIMARY_COMPARISON = "tabpfn3_best_single_vs_best_single"
ANCESTRY_SECONDARY_COMPARISONS = (
    "tabpfn3_full_vs_tabpfn3_best_single",
    "tabpfn3_full_vs_best_single",
)


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


def _load_corrected_metadata_score_summary(
    *,
    summary_dir: Path,
    diseases: list[str],
    final_covariates: list[str],
    n_folds: int,
) -> dict[str, Any]:
    pooled_path = summary_dir / "pooled_metrics.csv"
    bootstrap_path = summary_dir / "paired_bootstrap_deltas.csv"
    contract_path = summary_dir / "contract_audit.csv"
    record_path = summary_dir / "run_record.json"
    for path in [pooled_path, bootstrap_path, contract_path, record_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing corrected metadata-score artifact: {path}")

    pooled = read_table(pooled_path)
    bootstrap = read_table(bootstrap_path)
    contract = read_table(contract_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require_columns(
        pooled,
        [
            "disease",
            "variant",
            "auc",
            "auprc",
            "brier",
            "prevalence",
            "n",
            "nagelkerke_r2",
        ],
        "corrected metadata-score pooled metrics",
    )
    require_columns(
        bootstrap,
        [
            "metric",
            "disease",
            "reference",
            "challenger",
            "delta",
            "ci_lower",
            "ci_upper",
            "p_two_sided",
        ],
        "corrected metadata-score bootstrap",
    )
    require_columns(
        contract,
        [
            "disease",
            "ablation",
            "n_score_covariates",
            "final_covariates",
            "n_final_covariates",
            "outer_folds",
            "inner_oof_folds",
        ],
        "corrected metadata-score contract audit",
    )

    expected_pairs = {(disease, arm) for disease in diseases for arm in METADATA_SCORE_ARM_ORDER}
    pooled_pairs = set(
        zip(
            pooled["disease"].astype(str),
            pooled["variant"].astype(str),
            strict=True,
        )
    )
    contract_pairs = set(
        zip(
            contract["disease"].astype(str),
            contract["ablation"].astype(str),
            strict=True,
        )
    )
    if pooled_pairs != expected_pairs or contract_pairs != expected_pairs:
        raise ValueError("Corrected metadata-score disease/arm coverage is incomplete")
    if (
        pooled.duplicated(["disease", "variant"]).any()
        or contract.duplicated(["disease", "ablation"]).any()
    ):
        raise ValueError("Corrected metadata-score summary contains duplicate disease/arm rows")
    expected_final = ";".join(final_covariates)
    if set(contract["final_covariates"].astype(str)) != {expected_final}:
        raise ValueError("Corrected metadata-score final-wrapper covariates do not match config")
    if set(contract["n_final_covariates"].astype(int)) != {len(final_covariates)}:
        raise ValueError("Corrected metadata-score final-wrapper width does not match config")
    if set(contract["outer_folds"].astype(int)) != {n_folds}:
        raise ValueError("Corrected metadata-score outer-fold count does not match config")
    if int(record.get("n_tasks", -1)) != len(expected_pairs):
        raise ValueError("Corrected metadata-score run record has the wrong task count")

    auc_bootstrap = bootstrap.loc[
        bootstrap["metric"].astype(str).eq("auc") & bootstrap["reference"].astype(str).eq("full")
    ].copy()
    expected_challengers = set(METADATA_SCORE_ARM_ORDER[1:])
    if set(auc_bootstrap["challenger"].astype(str)) != expected_challengers:
        raise ValueError("Corrected metadata-score AUC bootstrap arms are incomplete")
    if auc_bootstrap.duplicated(["disease", "challenger"]).any():
        raise ValueError("Corrected metadata-score bootstrap contains duplicate comparisons")

    pooled_lookup = pooled.set_index(["disease", "variant"])
    contract_lookup = contract.set_index(["disease", "ablation"])
    bootstrap_lookup = auc_bootstrap.set_index(["disease", "challenger"])
    rows = []
    for disease in diseases:
        for arm in METADATA_SCORE_ARM_ORDER:
            metric = pooled_lookup.loc[(disease, arm)]
            audit = contract_lookup.loc[(disease, arm)]
            n = int(metric["n"])
            cases = int(round(float(metric["prevalence"]) * n))
            if arm == "full":
                delta_ci = "reference"
                p_value = "-"
            else:
                interval = bootstrap_lookup.loc[(disease, arm)]
                delta_ci = (
                    f"{_fmt(float(interval['delta']), signed=True)} "
                    f"[{_fmt(float(interval['ci_lower']), signed=True)}, "
                    f"{_fmt(float(interval['ci_upper']), signed=True)}]"
                )
                p = float(interval["p_two_sided"])
                p_value = "<0.001" if p < 0.001 else _fmt(p, digits=3)
            rows.append(
                {
                    "Disease": disease,
                    "Score arm": METADATA_SCORE_ARM_DISPLAY[arm],
                    "Score metadata n": int(audit["n_score_covariates"]),
                    "N/cases": f"{n}/{cases}",
                    "AUC": _fmt(float(metric["auc"])),
                    "Delta AUC vs full [95% CI]": delta_ci,
                    "P (paired bootstrap)": p_value,
                    "AUPRC": _fmt(float(metric["auprc"])),
                    "Brier": _fmt(float(metric["brier"])),
                    "R2": _fmt(float(metric["nagelkerke_r2"])),
                }
            )
    mean_deltas = (
        auc_bootstrap.groupby("challenger", sort=False)["delta"].mean().astype(float).to_dict()
    )
    takeaway = (
        "All final fits restore sex, age, and PC1-PC10. Mean Delta AUC from omitting "
        f"score inputs: PCs {_fmt(mean_deltas['no_pcs'], signed=True)}, "
        f"age {_fmt(mean_deltas['no_age'], signed=True)}, "
        f"sex {_fmt(mean_deltas['no_sex'], signed=True)}, and "
        f"age plus sex {_fmt(mean_deltas['no_age_sex'], signed=True)}."
    )
    return {
        "table": pd.DataFrame(rows),
        "status": "PCs largest; sex minimal",
        "takeaway": takeaway,
        "source": str(summary_dir),
        "n_tasks": int(record["n_tasks"]),
        "n_prediction_rows": int(record["n_prediction_rows"]),
        "n_resamples": int(record["paired_bootstrap_resamples"]),
    }


def _load_ancestry_single_prs_bridge(
    *,
    analysis_dir: Path,
    diseases: list[str],
) -> dict[str, Any]:
    comparison_path = analysis_dir / "paired_three_arm_comparison_summary.csv"
    manifest_path = analysis_dir / "model_selection_manifest.csv"
    for path in [comparison_path, manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing ancestry single-PRS bridge artifact: {path}")

    comparisons = read_table(comparison_path)
    manifest = read_table(manifest_path)
    require_columns(
        comparisons,
        [
            "disease",
            "ancestry_group",
            "n",
            "cases",
            "comparison",
            "metric",
            "reference_value",
            "candidate_value",
            "candidate_minus_reference",
            "ci_low",
            "ci_high",
            "n_bootstraps",
        ],
        "ancestry single-PRS bridge comparisons",
    )
    require_columns(
        manifest,
        ["disease", "single_model"],
        "ancestry single-PRS bridge model manifest",
    )
    comparisons = comparisons.loc[comparisons["metric"].astype(str).eq("auc")].copy()
    required_comparisons = {ANCESTRY_PRIMARY_COMPARISON, *ANCESTRY_SECONDARY_COMPARISONS}
    if set(comparisons["comparison"].astype(str)) != required_comparisons:
        raise ValueError("Ancestry bridge comparison coverage is incomplete")
    if set(comparisons["disease"].astype(str)) != set(diseases):
        raise ValueError("Ancestry bridge disease coverage does not match config")
    if manifest.duplicated("disease").any() or set(manifest["disease"].astype(str)) != set(
        diseases
    ):
        raise ValueError("Ancestry bridge model manifest does not have one row per disease")
    if comparisons.duplicated(["disease", "ancestry_group", "comparison"]).any():
        raise ValueError("Ancestry bridge contains duplicate disease/group comparisons")

    primary = comparisons.loc[
        comparisons["comparison"].astype(str).eq(ANCESTRY_PRIMARY_COMPARISON)
    ].copy()
    primary_groups = primary.groupby("disease")["ancestry_group"].apply(set)
    if any("ALL" not in groups for groups in primary_groups):
        raise ValueError("Ancestry bridge is missing an overall comparison")
    if len({frozenset(groups) for groups in primary_groups}) != 1:
        raise ValueError("Ancestry bridge subgroup coverage differs across diseases")

    model_lookup = manifest.set_index("disease")["single_model"].astype(str)
    disease_order = {disease: index for index, disease in enumerate(diseases)}

    def display_rows(frame: pd.DataFrame, *, include_model: bool) -> pd.DataFrame:
        rows = []
        ordered = frame.assign(
            _disease_order=frame["disease"].astype(str).map(disease_order)
        ).sort_values(["_disease_order", "ancestry_group"])
        for row in ordered.itertuples(index=False):
            rendered = {
                "Disease": row.disease,
                "Group": row.ancestry_group,
                "N/cases": f"{int(row.n)}/{int(row.cases)}",
                "A: single-best AUC": _fmt(float(row.reference_value)),
                "B: TabPFN single-best + meta AUC": _fmt(float(row.candidate_value)),
                "B - A [95% CI]": (
                    f"{_fmt(float(row.candidate_minus_reference), signed=True)} "
                    f"[{_fmt(float(row.ci_low), signed=True)}, "
                    f"{_fmt(float(row.ci_high), signed=True)}]"
                ),
            }
            if include_model:
                rendered = {
                    "Disease": row.disease,
                    "Canonical single PRS": model_lookup.loc[row.disease],
                    **{key: value for key, value in rendered.items() if key != "Disease"},
                }
            rows.append(rendered)
        return pd.DataFrame(rows)

    overall = primary.loc[primary["ancestry_group"].astype(str).eq("ALL")]
    subgroup = primary.loc[~primary["ancestry_group"].astype(str).eq("ALL")]
    secondary_rows = []
    comparison_lookup = comparisons.set_index(["disease", "ancestry_group", "comparison"])
    for disease in diseases:
        c_minus_b = comparison_lookup.loc[(disease, "ALL", ANCESTRY_SECONDARY_COMPARISONS[0])]
        c_minus_a = comparison_lookup.loc[(disease, "ALL", ANCESTRY_SECONDARY_COMPARISONS[1])]
        secondary_rows.append(
            {
                "Disease": disease,
                "C: full multi-PRS TabPFN AUC": _fmt(float(c_minus_b["candidate_value"])),
                "C - B [95% CI]": (
                    f"{_fmt(float(c_minus_b['candidate_minus_reference']), signed=True)} "
                    f"[{_fmt(float(c_minus_b['ci_low']), signed=True)}, "
                    f"{_fmt(float(c_minus_b['ci_high']), signed=True)}]"
                ),
                "C - A [95% CI]": (
                    f"{_fmt(float(c_minus_a['candidate_minus_reference']), signed=True)} "
                    f"[{_fmt(float(c_minus_a['ci_low']), signed=True)}, "
                    f"{_fmt(float(c_minus_a['ci_high']), signed=True)}]"
                ),
            }
        )

    ci_positive = int(subgroup["ci_low"].astype(float).gt(0).sum())
    ci_negative = int(subgroup["ci_high"].astype(float).lt(0).sum())
    negative_groups = [
        f"{row.disease}/{row.ancestry_group}"
        for row in subgroup.loc[subgroup["ci_high"].astype(float).lt(0)].itertuples(index=False)
    ]
    return {
        "overall_table": display_rows(overall, include_model=True),
        "subgroup_table": display_rows(subgroup, include_model=False),
        "secondary_table": pd.DataFrame(secondary_rows),
        "source": str(analysis_dir),
        "n_subgroups": int(len(subgroup)),
        "n_positive": int(subgroup["candidate_minus_reference"].astype(float).gt(0).sum()),
        "ci_positive": ci_positive,
        "ci_negative": ci_negative,
        "negative_groups": negative_groups,
        "n_bootstraps": int(primary["n_bootstraps"].astype(int).min()),
    }


def _load_report_supplements(config: dict[str, Any]) -> dict[str, Any]:
    supplement_config = config.get("supplements")
    if supplement_config is None:
        return {}
    if not isinstance(supplement_config, dict):
        raise ValueError("Config key 'supplements' must be a mapping")
    project = _mapping(config, "project")
    metadata = _mapping(config, "metadata")
    diseases = _list(project, "diseases")
    loaded: dict[str, Any] = {}
    metadata_path = supplement_config.get("corrected_metadata_score_summary")
    if metadata_path:
        loaded["metadata_score"] = _load_corrected_metadata_score_summary(
            summary_dir=Path(str(metadata_path)),
            diseases=diseases,
            final_covariates=_list(metadata, "full"),
            n_folds=int(project["n_folds"]),
        )
    ancestry_path = supplement_config.get("ancestry_single_prs_bridge")
    if ancestry_path:
        loaded["ancestry_bridge"] = _load_ancestry_single_prs_bridge(
            analysis_dir=Path(str(ancestry_path)),
            diseases=diseases,
        )
    return loaded


def _load_completed_tasks(
    *,
    input_root: Path,
    diseases: list[str],
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    prediction_frames = []
    feature_frames = []
    manifest_frames = []
    tuning_frames = []
    records = []
    missing = []
    for disease in diseases:
        for fold in range(n_folds):
            task_dir = input_root / disease / f"fold_{fold}"
            required = [
                task_dir / "predictions.csv",
                task_dir / "feature_audit.csv",
                task_dir / "variant_manifest.csv",
                task_dir / "xgboost_tuning.csv",
                task_dir / "run_record.json",
            ]
            absent = [path for path in required if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            prediction_frames.append(read_table(required[0]))
            feature_frames.append(read_table(required[1]))
            manifest_frames.append(read_table(required[2]).assign(fold=int(fold)))
            tuning = read_table(required[3]).copy()
            tuning["rank"] = np.arange(1, len(tuning) + 1)
            tuning_frames.append(tuning)
            records.append(json.loads(required[4].read_text(encoding="utf-8")))
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(
            f"Mechanism-ablation output is incomplete; {len(missing)} required files are missing:\n{preview}"
        )
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(feature_frames, ignore_index=True),
        pd.concat(manifest_frames, ignore_index=True),
        pd.concat(tuning_frames, ignore_index=True),
        records,
    )


def _metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows = []
    fold_rows = []
    for (disease, variant), frame in predictions.groupby(["disease", "variant"], sort=True):
        pooled_rows.append({"disease": disease, "variant": variant, **prediction_metrics(frame)})
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
    patient_col: str,
    n_resamples: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_lookup = pooled_metrics.set_index(["disease", "variant"])
    jobs = [
        (comparison_index, spec, disease_index, disease)
        for comparison_index, spec in enumerate(comparison_specs())
        if spec.question != 2
        for disease_index, disease in enumerate(diseases)
    ]

    def run_job(job):
        comparison_index, spec, disease_index, disease = job
        print(
            f"Bootstrap question {spec.question}, {disease}: {spec.comparison}",
            flush=True,
        )
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
            metrics=("auc", "brier"),
        ).assign(
            question=spec.question,
            comparison=spec.comparison,
            disease=disease,
            reference=spec.reference,
            challenger=spec.challenger,
            interpretation=spec.interpretation,
        )
        reference_metrics = pooled_lookup.loc[(disease, spec.reference)]
        challenger_metrics = pooled_lookup.loc[(disease, spec.challenger)]
        row: dict[str, Any] = {
            "question": spec.question,
            "comparison": spec.comparison,
            "disease": disease,
            "reference": spec.reference,
            "challenger": spec.challenger,
            "interpretation": spec.interpretation,
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
        row["n"] = int(reference_metrics["n"])
        return row, bootstrap

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(run_job, jobs))
    comparison_rows = [row for row, _ in results]
    bootstrap_table = pd.concat([frame for _, frame in results], ignore_index=True)
    return pd.DataFrame(comparison_rows), bootstrap_table


def _fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _auc_rows_for_html(comparisons: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    auc_bootstrap = bootstrap.loc[bootstrap["metric"] == "auc"].copy()
    keys = ["question", "comparison", "disease", "reference", "challenger"]
    merged = comparisons.merge(
        auc_bootstrap[keys + ["ci_lower", "ci_upper", "p_two_sided"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    out = pd.DataFrame(
        {
            "Comparison": merged["comparison"],
            "Disease": merged["disease"],
            "N": merged["n"].astype(int),
            "Reference AUC": merged["reference_auc"].map(lambda value: _fmt(float(value))),
            "Challenger AUC": merged["challenger_auc"].map(lambda value: _fmt(float(value))),
            "Delta AUC [95% CI]": [
                f"{_fmt(float(delta), signed=True)} "
                f"[{_fmt(float(lower), signed=True)}, {_fmt(float(upper), signed=True)}]"
                for delta, lower, upper in zip(
                    merged["delta_auc"], merged["ci_lower"], merged["ci_upper"], strict=True
                )
            ],
            "P (paired bootstrap)": merged["p_two_sided"].map(
                lambda value: "<0.001" if float(value) < 0.001 else _fmt(float(value), digits=3)
            ),
            "Delta Brier": merged["delta_brier"].map(lambda value: _fmt(float(value), signed=True)),
            "Delta R2": merged["delta_nagelkerke_r2"].map(
                lambda value: _fmt(float(value), signed=True)
            ),
            "Delta OR90": merged["delta_top10_vs_lower90_or"].map(
                lambda value: _fmt(float(value), digits=3, signed=True)
            ),
        }
    )
    return out


def _comparison_subset(comparisons: pd.DataFrame, name: str) -> pd.DataFrame:
    subset = comparisons.loc[comparisons["comparison"] == name]
    if len(subset) != 3:
        raise ValueError(f"Expected three disease rows for comparison {name!r}; got {len(subset)}")
    return subset


def _direction_summary(frame: pd.DataFrame) -> str:
    deltas = frame["delta_auc"].astype(float)
    significant = int((frame["ci_lower"].astype(float) > 0).sum())
    return (
        f"{int((deltas > 0).sum())}/3 diseases; {significant}/3 paired 95% CIs above zero; "
        f"mean Delta AUC {_fmt(float(deltas.mean()), signed=True)}"
    )


def _question_takeaways(
    comparisons: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[int, tuple[str, str]]:
    auc_bootstrap = bootstrap.loc[
        bootstrap["metric"] == "auc",
        [
            "question",
            "comparison",
            "disease",
            "reference",
            "challenger",
            "ci_lower",
            "ci_upper",
        ],
    ]
    comparisons = comparisons.merge(
        auc_bootstrap,
        on=["question", "comparison", "disease", "reference", "challenger"],
        how="left",
        validate="one_to_one",
    )
    selected_best = _comparison_subset(comparisons, "selected PRS panel vs best single PRS")
    all_best = _comparison_subset(comparisons, "all eight PRSs vs best single PRS")
    selected_all = _comparison_subset(comparisons, "selected PRS panel vs all eight PRSs")
    joint_additive = _comparison_subset(
        comparisons, "joint modeling vs separate additive main effects"
    )
    tabpfn_xgb = _comparison_subset(comparisons, "TabPFN vs tuned XGBoost")
    continuous_hard = _comparison_subset(comparisons, "continuous PCs vs one-hot broad ancestry")

    q1_positive = int((selected_best["delta_auc"] > 0).sum())
    q1_significant = int((selected_best["ci_lower"] > 0).sum())
    if q1_significant >= 2:
        q1_status = "Supported"
    elif q1_positive >= 2:
        q1_status = "Point-estimate support"
    else:
        q1_status = "Not a general core source"
    q1_text = (
        f"Selected panel vs best single: {_direction_summary(selected_best)}. "
        f"All eight vs best single: {_direction_summary(all_best)}. "
        f"Selection vs all eight: {_direction_summary(selected_all)}."
    )

    def simple_takeaway(
        frame: pd.DataFrame, positive_label: str, negative_label: str
    ) -> tuple[str, str]:
        positive = int((frame["delta_auc"] > 0).sum())
        significant = int((frame["ci_lower"] > 0).sum())
        if significant >= 2:
            status = positive_label
        elif positive >= 2:
            status = f"Point-estimate {positive_label.lower()}"
        elif positive == 0:
            status = negative_label
        else:
            status = "Mixed"
        return status, _direction_summary(frame)

    q3 = simple_takeaway(joint_additive, "Joint advantage", "No joint advantage")
    q4 = simple_takeaway(tabpfn_xgb, "TabPFN advantage", "XGBoost advantage")
    q5 = simple_takeaway(continuous_hard, "Continuous-PC advantage", "Hard-group advantage")
    return {
        1: (q1_status, q1_text),
        3: q3,
        4: q4,
        5: q5,
    }


def _html_table(frame: pd.DataFrame, classes: str = "data-table") -> str:
    return frame.to_html(index=False, border=0, classes=classes, escape=True)


def _followup_html(supplements: dict[str, Any]) -> str:
    sections = []
    ancestry = supplements.get("ancestry_bridge")
    if ancestry is not None:
        negative_note = (
            "; the only CI entirely below zero was "
            + ", ".join(html.escape(item) for item in ancestry["negative_groups"])
            if ancestry["negative_groups"]
            else ""
        )
        sections.append(
            f"""
  <section id="ancestry-single-prs-bridge"><h2>Ancestry-stratified gain from TabPFN-3 integration of the same single PRS</h2>
    <div class="protocol"><strong>Primary ancestry estimand (A to B):</strong> A is the canonical single-best PRS plus the standard wrapper. B sends that exact same PRS together with metadata through TabPFN-3, then applies the same standard wrapper. This isolates the TabPFN transformation and metadata interaction without changing the underlying PRS.</div>
    <h3>Overall official-validation comparison</h3>
    <div class="table-wrap">{_html_table(ancestry["overall_table"])}</div>
    <h3>Gain within 1000G-reference genetic-similarity groups</h3>
    <p>B had a higher subgroup AUC point estimate in {ancestry["n_positive"]}/{ancestry["n_subgroups"]} disease-by-group comparisons. The paired 95% CI was entirely above zero in {ancestry["ci_positive"]} and entirely below zero in {ancestry["ci_negative"]}{negative_note}. Small groups have wide intervals, so the overall disease estimates remain primary.</p>
    <div class="table-wrap">{_html_table(ancestry["subgroup_table"])}</div>
    <details><summary>Secondary sensitivity comparison: full multi-PRS TabPFN-3</summary>
      <p>The previous single-best to full TabPFN comparison is retained as secondary context. C - B estimates the incremental contribution of the selected multi-PRS panel after introducing the same TabPFN layer.</p>
      <div class="table-wrap">{_html_table(ancestry["secondary_table"])}</div>
    </details>
    <ul>
      <li>Every A/B/C wrapper uses sex, age, and PC1-PC10; breast-cancer evaluation is restricted to eligible female patients.</li>
      <li>Groups are post hoc 1000G-reference genetic-similarity labels, including AMBIGUOUS, not self-identified ancestry.</li>
      <li>Model choices used the official-validation set, so this is an explanatory post hoc analysis rather than an unbiased confirmatory benchmark.</li>
    </ul>
    <p class="small">Source: <code>{html.escape(ancestry["source"])}</code>; {ancestry["n_bootstraps"]:,} paired case-control-stratified bootstrap resamples.</p>
  </section>
"""
        )
    if not sections:
        return ""
    return (
        '<section id="follow-up-analyses"><h2>Official-validation ancestry follow-up</h2>'
        '<p class="lead">This post hoc official-validation analysis is reported separately from the train-validation mechanism questions below.</p>'
        + "".join(sections)
        + "</section>"
    )


def _metric_overview(pooled_metrics: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "tabpfn_best_prs_full_metadata": "TabPFN: best single PRS + full metadata",
        "tabpfn_selected_prs_full_metadata": "TabPFN: selected PRSs + full metadata",
        "tabpfn_all_prs_full_metadata": "TabPFN: all 8 PRSs + full metadata",
        "tabpfn_selected_prs_age_sex": "TabPFN: selected PRSs + age/sex",
        "tabpfn_selected_prs_pcs": "TabPFN: selected PRSs + PCs",
        "tabpfn_selected_prs_only": "TabPFN: selected PRSs only",
        "tabpfn_metadata_only": "TabPFN: metadata only",
        "tabpfn_selected_prs_hard_ancestry": "TabPFN: PRSs + age/sex + hard ancestry",
        "tabpfn_separate_additive": "Separate TabPFN(M) + TabPFN(PRS)",
        "xgboost_selected_prs_full_metadata": "Tuned XGBoost: same joint inputs",
    }
    manifest_compact = manifest.drop_duplicates(["disease", "variant"])[
        ["disease", "variant", "n_prs", "n_covariates_requested"]
    ]
    table = pooled_metrics.merge(
        manifest_compact,
        on=["disease", "variant"],
        how="left",
        validate="one_to_one",
    )
    return pd.DataFrame(
        {
            "Disease": table["disease"],
            "Model": table["variant"].map(labels).fillna(table["variant"]),
            "PRS n": table["n_prs"].fillna(0).astype(int),
            "Metadata n": table["n_covariates_requested"].fillna(0).astype(int),
            "N": table["n"].astype(int),
            "AUC": table["auc"].map(lambda value: _fmt(float(value))),
            "AUPRC": table["auprc"].map(lambda value: _fmt(float(value))),
            "Brier": table["brier"].map(lambda value: _fmt(float(value))),
            "R2": table["nagelkerke_r2"].map(lambda value: _fmt(float(value))),
            "OR90": table["top10_vs_lower90_or"].map(lambda value: _fmt(float(value), digits=3)),
        }
    )


def _build_html(
    *,
    config: dict[str, Any],
    comparisons: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    tuning: pd.DataFrame,
    records: list[dict[str, Any]],
    n_resamples: int,
    supplements: dict[str, Any],
) -> str:
    project = _mapping(config, "project")
    prs_config = _mapping(config, "prs")
    metadata = _mapping(config, "metadata")
    tabpfn = _mapping(config, "tabpfn")
    xgboost = _mapping(config, "xgboost")
    takeaways = _question_takeaways(comparisons, bootstrap)
    auc_table = _auc_rows_for_html(comparisons, bootstrap)
    overview = _metric_overview(pooled_metrics, manifest)
    followup_html = _followup_html(supplements)
    metadata_score = supplements.get("metadata_score")

    cards = []
    sections = []
    question_order = [1, 3, 4, 5]
    if metadata_score is not None:
        question_order.insert(1, 2)
    for question in question_order:
        if question == 2:
            status = str(metadata_score["status"])
            takeaway = str(metadata_score["takeaway"])
        else:
            status, takeaway = takeaways[question]
        cards.append(
            "<article class='card'>"
            f"<div class='eyebrow'>Question {question}</div>"
            f"<h3>{html.escape(QUESTION_TITLES[question])}</h3>"
            f"<span class='status'>{html.escape(status)}</span>"
            f"<p>{html.escape(takeaway)}</p>"
            "</article>"
        )
        if question == 2:
            sections.append(
                f"<section id='q2'><h2>Question 2. "
                f"{html.escape(QUESTION_TITLES[2])}</h2>"
                "<div class='protocol'><strong>Corrected estimand:</strong> only "
                "the metadata entering the intermediate TabPFN-3 score changes. Every "
                "downstream logistic model restores the score plus sex, age, and PC1-PC10. "
                "The score is generated with inner OOF prediction inside each outer-training "
                "fold; outer validation is prediction-only.</div>"
                f"<p class='lead'>{html.escape(takeaway)}</p>"
                "<p>Negative Delta AUC means that omitting the named variable from the "
                "TabPFN score reduced performance even though all 12 metadata variables "
                "were restored in the final wrapper.</p>"
                f"<div class='table-wrap'>{_html_table(metadata_score['table'])}</div>"
                f"<p class='small'>Source: <code>{html.escape(metadata_score['source'])}"
                f"</code>. Audited {metadata_score['n_tasks']} disease-by-arm tasks, "
                f"{metadata_score['n_prediction_rows']:,} OOF prediction rows, and "
                f"{metadata_score['n_resamples']:,} paired bootstrap resamples.</p></section>"
            )
        else:
            question_table = auc_table.loc[comparisons["question"].to_numpy() == question]
            sections.append(
                f"<section id='q{question}'><h2>Question {question}. "
                f"{html.escape(QUESTION_TITLES[question])}</h2>"
                f"<p class='lead'>{html.escape(takeaway)}</p>"
                f"<div class='table-wrap'>{_html_table(question_table)}</div></section>"
            )

    selected_rows = []
    for disease in _list(project, "diseases"):
        selected_rows.append(
            {
                "Disease": disease,
                "Canonical best single": str(
                    _mapping(prs_config, "canonical_best_single")[disease]
                ),
                "Selected PRS panel": ", ".join(
                    str(x) for x in _mapping(prs_config, "selected")[disease]
                ),
            }
        )
    cohort_rows = []
    for record in sorted(records, key=lambda item: (str(item["disease"]), int(item["fold"]))):
        audit = record["data_audit"]
        cohort_rows.append(
            {
                "Disease": record["disease"],
                "Fold": int(record["fold"]),
                "Context N/cases": f"{audit['n_context']}/{audit['context_cases']}",
                "Validation N/cases": f"{audit['n_validation']}/{audit['validation_cases']}",
                "Variants": len(record["variants"]),
            }
        )
    feature_counts = (
        features.groupby(["disease", "variant", "fold"], as_index=False)
        .size()
        .groupby(["disease", "variant"], as_index=False)["size"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "Min encoded features", "max": "Max encoded features"})
    )
    selected_xgboost = tuning.loc[
        tuning["rank"] == 1,
        [
            "disease",
            "fold",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "mean_inner_auc",
        ],
    ].copy()
    selected_xgboost.columns = [
        "Disease",
        "Fold",
        "Max depth",
        "Learning rate",
        "Min child weight",
        "Inner-CV AUC",
    ]
    selected_xgboost["Inner-CV AUC"] = selected_xgboost["Inner-CV AUC"].map(
        lambda value: _fmt(float(value))
    )

    css = """
:root { --ink:#17212b; --muted:#5f6b76; --line:#d8e0e7; --paper:#fff; --bg:#f3f6f8; --accent:#0f766e; --accent-soft:#e4f4f0; --navy:#17365d; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1500px; margin:0 auto; padding:32px 28px 64px; }
h1 { margin:0 0 10px; font-size:32px; line-height:1.15; }
h2 { margin:34px 0 12px; font-size:22px; color:var(--navy); }
h3 { margin:6px 0 10px; font-size:17px; line-height:1.3; }
p { color:var(--muted); line-height:1.55; }
.subtitle { max-width:1050px; font-size:16px; }
.protocol { margin:18px 0; padding:14px 16px; border:1px solid #b9ddd7; border-left:5px solid var(--accent); border-radius:8px; background:var(--accent-soft); }
.grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:22px 0; }
.card { background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:15px; min-height:220px; }
.card p { font-size:13px; margin:10px 0 0; }
.eyebrow { text-transform:uppercase; letter-spacing:.08em; color:var(--accent); font-size:11px; font-weight:800; }
.status { display:inline-block; padding:3px 9px; border-radius:999px; background:var(--accent-soft); color:#075e55; font-size:12px; font-weight:750; }
.lead { max-width:1180px; font-size:14px; }
.table-wrap { overflow:auto; background:var(--paper); border:1px solid var(--line); border-radius:9px; margin:12px 0 24px; }
table.data-table { border-collapse:collapse; width:100%; font-size:12.5px; }
table.data-table th { position:sticky; top:0; background:#eaf0f4; text-align:left; padding:8px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }
table.data-table td { padding:7px 9px; border-bottom:1px solid #edf1f4; vertical-align:top; font-variant-numeric:tabular-nums; }
table.data-table tr:nth-child(even) td { background:#fbfcfd; }
details { background:var(--paper); border:1px solid var(--line); border-radius:9px; padding:12px 14px; margin:12px 0; }
summary { cursor:pointer; font-weight:750; color:var(--navy); }
code { background:#e9eef2; border-radius:4px; padding:1px 4px; }
.small { font-size:12px; color:var(--muted); }
ul { color:var(--muted); line-height:1.55; }
@media (max-width:1100px) { .grid { grid-template-columns:1fr 1fr; } }
@media (max-width:650px) { main { padding:22px 12px 44px; } .grid { grid-template-columns:1fr; } }
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>decipEHR TabPFN Mechanism Ablations</title>
  <style>{css}</style>
</head>
<body><main>
  <h1>TabPFN Mechanism Ablations</h1>
  <p class="subtitle">Nested-CV mechanism experiments with a corrected fixed-wrapper Question 2, plus an official-validation ancestry bridge that holds the single PRS fixed.</p>
  <div class="protocol"><strong>Reading guide:</strong> Question 2 now uses the corrected five-arm metadata-score experiment: only the inputs to the intermediate score change, while every final fit restores sex, age, and PC1-PC10. Use A to B in the ancestry section for gain from TabPFN-3 integration of the same single PRS.</div>
  {followup_html}
  <h2>Mechanism questions</h2>
  <div class="protocol"><strong>Evaluation structure:</strong> Questions 1, 3, 4, and 5 use pooled patient-level direct outer-OOF probabilities. Question 2 uses nested inner-OOF TabPFN score generation followed by the fixed full-covariate wrapper. Every outer validation fold is prediction-only. No final holdout labels were used. AUC uncertainty uses {n_resamples:,} paired case-control-stratified bootstrap resamples.</div>
  <div class="grid">{"".join(cards)}</div>
  <div class="protocol"><strong>Connection to the earlier linear-stacking pattern:</strong> the prior official-validation table found that ridge/elastic-net multi-PRS stacks did not consistently beat the best single PRS. That historical result is context only and is not pooled with these train-validation OOF estimates; it motivated Questions 1, 3, and 4, which separate PRS count and complementarity, joint cross-block modeling, and TabPFN-specific behavior.</div>
  {"".join(sections)}

  <section id="all-models"><h2>Original direct-OOF model arms</h2>
    <p>These raw arms support Questions 1, 3, 4, and 5. The former direct-OOF age/sex-versus-PC comparison is retained only as model provenance and is not used for Question 2. Direct probabilities are used for AUC, AUPRC, Brier score, and Nagelkerke R2; OR90 is a secondary, more volatile endpoint.</p>
    <div class="table-wrap">{_html_table(overview)}</div>
  </section>

  <section id="protocol"><h2>Prespecified protocol and provenance</h2>
    <ul>
      <li>TabPFN: version {html.escape(str(tabpfn["model_version"]))}, {int(tabpfn["n_estimators"])} estimators, probability clipping epsilon {float(tabpfn["clip_eps"])}.</li>
      <li>Full metadata: {html.escape(", ".join(_list(metadata, "full")))}.</li>
      <li>Additive comparator: <code>logit p(M) + logit p(PRS) - logit prevalence(outer training)</code>. It has no learned cross-block interaction and no fitted second-stage weights.</li>
      <li>XGBoost receives the exact same fold-fitted standardized/one-hot matrix as the joint TabPFN arm. Hyperparameters are selected inside each outer training fold with {int(xgboost["inner_folds"])}-fold inner CV; outer validation remains untouched.</li>
      <li>Broad ancestry is the 1000G-reference <code>hard_label</code>, including <code>AMBIGUOUS</code>; it is a genetic-similarity label, not self-identified ancestry.</li>
      <li>Breast-cancer analyses are female-only by eligibility rule; sex is therefore constant and contributes no within-cohort signal for BC.</li>
    </ul>
    <div class="table-wrap">{_html_table(pd.DataFrame(selected_rows))}</div>
  </section>

  <section id="audit"><h2>Completion and leakage audit</h2>
    <p>All {len(records)} disease-fold tasks are present. Patient counts, labels, fold assignments, PRS coverage, ancestry-label coverage, and the ten-arm variant set were checked before aggregation.</p>
    <details><summary>Disease-fold counts</summary><div class="table-wrap">{_html_table(pd.DataFrame(cohort_rows))}</div></details>
    <details><summary>Encoded feature-count audit</summary><div class="table-wrap">{_html_table(feature_counts)}</div></details>
    <details><summary>Selected XGBoost hyperparameters by outer fold</summary><div class="table-wrap">{_html_table(selected_xgboost)}</div></details>
    <details><summary>Fold-level metrics</summary><div class="table-wrap">{_html_table(fold_metrics.round(6))}</div></details>
  </section>

  <section id="limits"><h2>Interpretation guardrails</h2>
    <ul>
      <li>This is a mechanism audit conditional on frozen PRS definitions. The canonical best single PRSs and final selected panels were chosen in earlier project analyses, so these comparisons should not be described as a fresh, unbiased model-selection benchmark.</li>
      <li>A positive AUC delta means the named challenger outranks the reference. For Brier score, a negative delta is better.</li>
      <li>Paired bootstrap intervals quantify patient resampling uncertainty, not uncertainty from retraining TabPFN with different random seeds or changing the PRS construction pipeline.</li>
      <li>OR90 can move sharply when only a few patients cross the top-decile boundary; conclusions prioritize paired AUC and use R2/Brier/OR90 as supporting endpoints.</li>
    </ul>
  </section>
  <p class="small">Generated from machine-readable mechanism, corrected metadata-score, and ancestry-bridge artifacts under ignored <code>outputs/</code> directories.</p>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate completed mechanism-ablation folds and write a self-contained HTML report."
    )
    parser.add_argument("--config", default="configs/mechanism_ablations.yaml")
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    project = _mapping(config, "project")
    paths = _mapping(config, "paths")
    columns = _mapping(config, "columns")
    summary_config = _mapping(config, "summary")
    diseases = _list(project, "diseases")
    n_folds = int(project["n_folds"])
    patient_col = str(columns["patient_id"])
    output_root = Path(str(paths["output_root"]))
    input_root = Path(args.input_root) if args.input_root else output_root / "fold_predictions"
    out_dir = Path(args.out_dir) if args.out_dir else output_root / "summary"
    n_resamples = (
        int(args.bootstrap_resamples)
        if args.bootstrap_resamples is not None
        else int(summary_config["paired_bootstrap_resamples"])
    )
    n_jobs = args.n_jobs if args.n_jobs is not None else int(summary_config.get("n_jobs", 1))
    if n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    supplements = _load_report_supplements(config)

    predictions, features, manifest, tuning, records = _load_completed_tasks(
        input_root=input_root,
        diseases=diseases,
        n_folds=n_folds,
    )
    expected_variants = records[0]["variants"]
    validate_prediction_frame(
        predictions,
        patient_col=patient_col,
        expected_variants=expected_variants,
    )
    if len(records) != len(diseases) * n_folds:
        raise ValueError("The run-record count does not match diseases x folds")
    if any(record["variants"] != expected_variants for record in records):
        raise ValueError("Variant order differs across disease-fold run records")

    pooled_metrics, fold_metrics = _metric_tables(predictions)
    comparisons, bootstrap = _paired_comparisons(
        predictions=predictions,
        pooled_metrics=pooled_metrics,
        diseases=diseases,
        patient_col=patient_col,
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
    write_table(manifest, out_dir / "variant_manifest_by_fold.csv")
    write_table(features, out_dir / "feature_audit_by_fold.csv")
    write_table(tuning, out_dir / "xgboost_tuning_by_fold.csv")

    report = _build_html(
        config=config,
        comparisons=comparisons,
        bootstrap=bootstrap,
        pooled_metrics=pooled_metrics,
        fold_metrics=fold_metrics,
        manifest=manifest,
        features=features,
        tuning=tuning,
        records=records,
        n_resamples=n_resamples,
        supplements=supplements,
    )
    report_path = out_dir / "tabpfn_mechanism_ablations.html"
    report_path.write_text(report, encoding="utf-8")
    (out_dir / "run_record.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "input_root": str(input_root),
                "diseases": diseases,
                "n_folds": n_folds,
                "n_tasks": len(records),
                "n_prediction_rows": len(predictions),
                "variants": expected_variants,
                "paired_bootstrap_resamples": n_resamples,
                "n_jobs": n_jobs,
                "question_2_estimand": "nested_oof_tabpfn_score_fixed_full_covariate_wrapper",
                "legacy_direct_oof_question_2_included": False,
                "supplements": {name: value["source"] for name, value in supplements.items()},
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote mechanism-ablation summary and HTML report to {out_dir}")


if __name__ == "__main__":
    main()
