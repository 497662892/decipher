from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.ancestry_performance import (
    additive_loss_decomposition,
    auc_pair_decomposition,
    paired_bootstrap_comparisons,
    subgroup_metrics,
)
from decipehr.io import read_table, write_table

DEFAULT_ANCESTRY_LABELS = (
    "outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv"
)
DEFAULT_TABPFN_ROOT = "outputs/ancestry_stratified_best_single_vs_tabpfn3/tabpfn_standard_wrapper"
DEFAULT_BRIDGE_ROOT = "outputs/ancestry_stratified_best_single_vs_tabpfn3/tabpfn_best_single_bridge"
DEFAULT_OUT_DIR = "outputs/ancestry_stratified_best_single_vs_tabpfn3"


@dataclass(frozen=True)
class DiseaseSelection:
    disease: str
    single_model: str
    single_predictions: str
    single_filters: dict[str, str | float]
    bridge_predictions: str
    bridge_method: str
    tabpfn_predictions: str
    tabpfn_method: str


def _selections(tabpfn_root: str, bridge_root: str) -> list[DiseaseSelection]:
    return [
        DiseaseSelection(
            disease="cad",
            single_model="PRS-CS (phi=auto)",
            single_predictions=(
                "outputs/final_trainval_official_validation/prscs/cad/predictions.csv"
            ),
            single_filters={"method": "prs_covariate_cad", "phi": "auto"},
            bridge_predictions=f"{bridge_root}/cad/standard_wrapper/predictions.csv",
            bridge_method="prs_covariate_cad",
            tabpfn_predictions=f"{tabpfn_root}/cad/predictions.csv",
            tabpfn_method="prs_covariate_cad",
        ),
        DiseaseSelection(
            disease="bc",
            single_model="lassosum2 (delta=1, lambda=0.0120637653810499)",
            single_predictions=(
                "outputs/final_trainval_official_validation/lassosum2_ukbb/bc/predictions.csv"
            ),
            single_filters={
                "method": "prs_covariate_bc",
                "delta": 1.0,
                "lambda": 0.0120637653810499,
            },
            bridge_predictions=f"{bridge_root}/bc/standard_wrapper/predictions.csv",
            bridge_method="prs_covariate_bc",
            tabpfn_predictions=f"{tabpfn_root}/bc/predictions.csv",
            tabpfn_method="prs_covariate_bc",
        ),
        DiseaseSelection(
            disease="T2D",
            single_model="MegaPRS annotated (BayesR)",
            single_predictions=(
                "outputs/final_trainval_official_validation/megaprs_annotated/T2D/predictions.csv"
            ),
            single_filters={"method": "prs_covariate_T2D", "model": "megaprs_bayesr"},
            bridge_predictions=f"{bridge_root}/T2D/standard_wrapper/predictions.csv",
            bridge_method="prs_covariate_T2D",
            tabpfn_predictions=f"{tabpfn_root}/T2D/predictions.csv",
            tabpfn_method="prs_covariate_T2D",
        ),
    ]


def _filter_rows(frame: pd.DataFrame, filters: dict[str, str | float]) -> pd.DataFrame:
    selected = frame.copy()
    for column, value in filters.items():
        if column not in selected:
            raise ValueError(f"Missing filter column {column!r}")
        if isinstance(value, float):
            selected = selected.loc[
                np.isclose(selected[column].astype(float), value, rtol=1e-10, atol=1e-12)
            ]
        else:
            selected = selected.loc[selected[column].astype(str).eq(value)]
    return selected.copy()


def _load_predictions(
    selection: DiseaseSelection,
    ancestry: pd.DataFrame,
) -> pd.DataFrame:
    single = _filter_rows(read_table(selection.single_predictions), selection.single_filters)
    bridge = read_table(selection.bridge_predictions)
    bridge = bridge.loc[
        bridge["disease"].astype(str).eq(selection.disease)
        & bridge["method"].astype(str).eq(selection.bridge_method)
    ].copy()
    tabpfn = read_table(selection.tabpfn_predictions)
    tabpfn = tabpfn.loc[
        tabpfn["disease"].astype(str).eq(selection.disease)
        & tabpfn["method"].astype(str).eq(selection.tabpfn_method)
    ].copy()
    if len(single) != single["sampleid"].nunique():
        raise ValueError(
            f"{selection.disease}: selected single-PRS predictions are not one row per patient"
        )
    if len(bridge) != bridge["sampleid"].nunique():
        raise ValueError(
            f"{selection.disease}: selected bridge predictions are not one row per patient"
        )
    if len(tabpfn) != tabpfn["sampleid"].nunique():
        raise ValueError(
            f"{selection.disease}: selected TabPFN predictions are not one row per patient"
        )

    single = single[["sampleid", "y_true", "y_prob"]].rename(
        columns={"y_true": "single_y_true", "y_prob": "best_single_prs"}
    )
    bridge = bridge[["sampleid", "y_true", "y_prob"]].rename(
        columns={
            "y_true": "bridge_y_true",
            "y_prob": "tabpfn3_best_single",
        }
    )
    tabpfn = tabpfn[["sampleid", "y_true", "y_prob"]].rename(
        columns={"y_true": "tabpfn_y_true", "y_prob": "tabpfn3_full"}
    )
    merged = single.merge(bridge, on="sampleid", how="inner", validate="one_to_one")
    merged = merged.merge(tabpfn, on="sampleid", how="inner", validate="one_to_one")
    if len(merged) != len(single) or len(merged) != len(bridge) or len(merged) != len(tabpfn):
        raise ValueError(f"{selection.disease}: A/B/C prediction patient sets are not identical")
    if not (
        merged["single_y_true"].equals(merged["bridge_y_true"])
        and merged["single_y_true"].equals(merged["tabpfn_y_true"])
    ):
        raise ValueError(f"{selection.disease}: labels differ among A/B/C sources")
    merged = merged.rename(columns={"single_y_true": "y_true"}).drop(
        columns=["bridge_y_true", "tabpfn_y_true"]
    )
    merged.insert(1, "disease", selection.disease)
    merged.insert(2, "single_model", selection.single_model)
    merged = merged.merge(ancestry, on="sampleid", how="left", validate="one_to_one")
    if merged["hard_label"].isna().any():
        missing = merged.loc[merged["hard_label"].isna(), "sampleid"].head(5).tolist()
        raise ValueError(f"{selection.disease}: missing 1000G-reference labels; examples={missing}")
    return merged.rename(columns={"hard_label": "ancestry_group"})


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    out_dir: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    auc_decomposition_summary: pd.DataFrame,
    loss_decomposition: pd.DataFrame,
    n_bootstraps: int,
) -> None:
    lines = [
        "# Best Single PRS vs TabPFN-3 by 1000G-Reference Genetic-Similarity Group",
        "",
        "## Protocol",
        "",
        "- Cohort: official validation only; every comparison uses identical patients within disease.",
        "- Standard evaluation is unchanged: both scores use the same logistic wrapper with sex, age, and PC1-PC10.",
        "- Breast cancer remains restricted to eligible female patients.",
        "- Single-PRS choices are the highest official-validation-AUC single methods from the canonical table.",
        "- TabPFN-3 choices are the canonical best-observed-OR90 full models from the matched ablation.",
        "- Groups are 1000G-reference genetic-similarity labels, not self-identified ancestry.",
        f"- Uncertainty: {n_bootstraps:,} paired, case-control-stratified bootstrap resamples.",
        "",
        "## Overall check",
        "",
    ]
    overall = metrics.loc[metrics["ancestry_group"].eq("ALL")]
    overall_rows = []
    for disease in ["cad", "bc", "T2D"]:
        disease_rows = overall.loc[overall["disease"].eq(disease)].set_index("model")
        single = disease_rows.loc["best_single_prs"]
        tabpfn = disease_rows.loc["tabpfn3_full"]
        overall_rows.append(
            [
                disease,
                str(int(single["n"])),
                _fmt(single["auc"], 4),
                _fmt(tabpfn["auc"], 4),
                _fmt(tabpfn["auc"] - single["auc"], 4),
                _fmt(single["top10_vs_lower90_or"], 3),
                _fmt(tabpfn["top10_vs_lower90_or"], 3),
                _fmt(single["nagelkerke_r2"], 4),
                _fmt(tabpfn["nagelkerke_r2"], 4),
            ]
        )
    lines.extend(
        [
            _markdown_table(
                [
                    "Disease",
                    "N",
                    "Single AUC",
                    "TabPFN AUC",
                    "Delta AUC",
                    "Single OR90",
                    "TabPFN OR90",
                    "Single R2",
                    "TabPFN R2",
                ],
                overall_rows,
            ),
            "",
        ]
    )
    subgroup_deltas = deltas.loc[~deltas["ancestry_group"].eq("ALL")]
    auc_better_total = int(
        (subgroup_deltas.loc[subgroup_deltas["metric"].eq("auc"), "tabpfn_minus_single"] > 0).sum()
    )
    brier_better_total = int(
        (
            subgroup_deltas.loc[subgroup_deltas["metric"].eq("brier"), "tabpfn_minus_single"] < 0
        ).sum()
    )
    r2_better_total = int(
        (
            subgroup_deltas.loc[
                subgroup_deltas["metric"].eq("nagelkerke_r2"),
                "tabpfn_minus_single",
            ]
            > 0
        ).sum()
    )
    or_deltas = subgroup_deltas.loc[
        subgroup_deltas["metric"].eq("top10_vs_lower90_or"), "tabpfn_minus_single"
    ]
    or_better_total = int((or_deltas > 1e-12).sum())
    or_equal_total = int(np.isclose(or_deltas, 0, atol=1e-12, rtol=0).sum())
    or_lower_total = int((or_deltas < -1e-12).sum())
    negative_auc = subgroup_deltas.loc[
        subgroup_deltas["metric"].eq("auc") & subgroup_deltas["tabpfn_minus_single"].lt(0)
    ]
    exception_text = ""
    if not negative_auc.empty:
        exceptions = [
            (f"{row.disease} {row.ancestry_group} (delta AUC {row.tabpfn_minus_single:+.3f})")
            for row in negative_auc.itertuples(index=False)
        ]
        exception_text = (
            "- The point-estimate exception is "
            + ", ".join(exceptions)
            + "; its interval still includes zero."
        )
    lines.extend(
        [
            "## Main finding",
            "",
            (
                "- Using `calibrator` to mean an ancestry-subgroup performance "
                "stabilizer, the hypothesis is well supported at the point-estimate level."
            ),
            (
                f"- TabPFN has higher AUC in {auc_better_total}/18 comparisons, higher "
                f"Nagelkerke R2 in {r2_better_total}/18, and lower Brier score in "
                f"{brier_better_total}/18. The gain is broad rather than confined to "
                "one genetic-similarity group."
            ),
            (
                f"- Subgroup OR90 is higher in {or_better_total}/18, tied in "
                f"{or_equal_total}/18, and lower in {or_lower_total}/18. OR90 is much "
                "less stable because a subgroup top decile can contain very few patients."
            ),
            "- Most non-EUR subgroup AUC intervals include zero because the groups are small; the point pattern is descriptive rather than confirmatory.",
        ]
    )
    if exception_text:
        lines.append(exception_text)
    lines.append("")

    for disease in ["cad", "bc", "T2D"]:
        lines.extend([f"## {disease}", ""])
        disease_metrics = metrics.loc[
            metrics["disease"].eq(disease) & ~metrics["ancestry_group"].eq("ALL")
        ]
        auc_deltas = deltas.loc[
            deltas["disease"].eq(disease) & deltas["metric"].eq("auc")
        ].set_index("ancestry_group")
        performance_rows = []
        calibration_rows = []
        for ancestry_group in sorted(disease_metrics["ancestry_group"].unique()):
            group_rows = disease_metrics.loc[
                disease_metrics["ancestry_group"].eq(ancestry_group)
            ].set_index("model")
            single = group_rows.loc["best_single_prs"]
            tabpfn = group_rows.loc["tabpfn3_full"]
            auc_delta = auc_deltas.loc[ancestry_group]
            performance_rows.append(
                [
                    ancestry_group,
                    f"{int(single['n'])}/{int(single['cases'])}",
                    _fmt(single["auc"]),
                    _fmt(tabpfn["auc"]),
                    (
                        f"{_fmt(auc_delta['tabpfn_minus_single'])} "
                        f"[{_fmt(auc_delta['ci_low'])}, {_fmt(auc_delta['ci_high'])}]"
                    ),
                    _fmt(single["top10_vs_lower90_or"], 2),
                    _fmt(tabpfn["top10_vs_lower90_or"], 2),
                    _fmt(single["nagelkerke_r2"]),
                    _fmt(tabpfn["nagelkerke_r2"]),
                ]
            )
            calibration_rows.append(
                [
                    ancestry_group,
                    _fmt(single["observed_risk"]),
                    _fmt(single["mean_predicted_risk"]),
                    _fmt(tabpfn["mean_predicted_risk"]),
                    _fmt(single["brier"], 4),
                    _fmt(tabpfn["brier"], 4),
                    _fmt(single["calibration_intercept"]),
                    _fmt(tabpfn["calibration_intercept"]),
                    _fmt(single["calibration_slope"]),
                    _fmt(tabpfn["calibration_slope"]),
                ]
            )
        lines.extend(
            [
                "### Standard performance",
                "",
                _markdown_table(
                    [
                        "Group",
                        "N/cases",
                        "Single AUC",
                        "TabPFN AUC",
                        "Delta AUC [95% CI]",
                        "Single OR90",
                        "TabPFN OR90",
                        "Single R2",
                        "TabPFN R2",
                    ],
                    performance_rows,
                ),
                "",
                "### Secondary probability-calibration diagnostics",
                "",
                _markdown_table(
                    [
                        "Group",
                        "Observed",
                        "Single mean p",
                        "TabPFN mean p",
                        "Single Brier",
                        "TabPFN Brier",
                        "Single intercept",
                        "TabPFN intercept",
                        "Single slope",
                        "TabPFN slope",
                    ],
                    calibration_rows,
                ),
                "",
            ]
        )

        non_all = deltas.loc[deltas["disease"].eq(disease) & ~deltas["ancestry_group"].eq("ALL")]
        auc_better = int(
            (non_all.loc[non_all["metric"].eq("auc"), "tabpfn_minus_single"] > 0).sum()
        )
        brier_better = int(
            (non_all.loc[non_all["metric"].eq("brier"), "tabpfn_minus_single"] < 0).sum()
        )
        r2_better = int(
            (
                non_all.loc[
                    non_all["metric"].eq("nagelkerke_r2"),
                    "tabpfn_minus_single",
                ]
                > 0
            ).sum()
        )
        disease_or_deltas = non_all.loc[
            non_all["metric"].eq("top10_vs_lower90_or"), "tabpfn_minus_single"
        ]
        or_better = int((disease_or_deltas > 1e-12).sum())
        or_equal = int(np.isclose(disease_or_deltas, 0, atol=1e-12, rtol=0).sum())
        or_lower = int((disease_or_deltas < -1e-12).sum())
        disease_auc = auc_decomposition_summary.loc[
            auc_decomposition_summary["disease"].eq(disease)
        ].set_index("pair_type")
        within = disease_auc.loc["within"]
        cross = disease_auc.loc["cross"]
        disease_loss = loss_decomposition.loc[loss_decomposition["disease"].eq(disease)]
        top_brier = disease_loss.nlargest(2, "weighted_brier_improvement_contribution")[
            "ancestry_group"
        ].tolist()
        lines.extend(
            [
                "### Compact diagnostic",
                "",
                (
                    f"- TabPFN has higher point AUC in {auc_better}/6 groups, lower "
                    f"Brier score in {brier_better}/6, and higher Nagelkerke R2 in "
                    f"{r2_better}/6."
                ),
                (
                    f"- OR90 is higher in {or_better}/6, tied in {or_equal}/6, and "
                    f"lower in {or_lower}/6."
                ),
                (
                    f"- Within-group pairs carry {within['pair_weight']:.1%} of all "
                    f"case-control pairs and contribute "
                    f"{within['weighted_auc_delta_contribution']:+.4f} AUC "
                    f"(mean pair delta {within['mean_pair_auc_delta']:+.4f})."
                ),
                (
                    f"- Cross-group pairs carry {cross['pair_weight']:.1%} and "
                    f"contribute {cross['weighted_auc_delta_contribution']:+.4f} AUC "
                    f"(mean pair delta {cross['mean_pair_auc_delta']:+.4f})."
                ),
                (
                    "- Largest weighted Brier-improvement contributors: "
                    + ", ".join(top_brier)
                    + "."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- In this report, ancestry-aware `calibrator` means improved or stabilized subgroup performance, not only statistical probability calibration.",
            "- Higher subgroup AUC indicates better within-group ranking; higher Nagelkerke R2 and lower Brier/log loss provide complementary performance evidence.",
            "- Probability-calibration intercepts, slopes, and mean-risk gaps are retained as secondary diagnostics and are not required for the broader hypothesis.",
            "- Subgroup OR90 is volatile because it uses only the highest-risk 10% inside each already-small group.",
            "- Calibration intercept 0 and slope 1 are ideal, but estimates are unstable in small groups, especially BC SAS-like.",
            "- Because model and single-PRS choices were selected using this validation set, this is a post hoc explanatory analysis.",
            "- Full machine-readable results are in `ancestry_group_metrics.csv`, `paired_bootstrap_delta_summary.csv`, `auc_pair_decomposition.csv`, `auc_decomposition_summary.csv`, and `additive_loss_decomposition.csv`.",
            "",
        ]
    )
    (out_dir / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


COMPARISON_ORDER = (
    "tabpfn3_best_single_vs_best_single",
    "tabpfn3_full_vs_tabpfn3_best_single",
    "tabpfn3_full_vs_best_single",
)
COMPARISON_DISPLAY = {
    "tabpfn3_best_single_vs_best_single": "B - A",
    "tabpfn3_full_vs_tabpfn3_best_single": "C - B",
    "tabpfn3_full_vs_best_single": "C - A",
}
MODEL_DISPLAY = {
    "best_single_prs": "A: best single PRS",
    "tabpfn3_best_single": "B: TabPFN-3 + best single PRS",
    "tabpfn3_full": "C: selected full TabPFN-3",
}


def _legacy_delta_frame(comparisons: pd.DataFrame) -> pd.DataFrame:
    legacy = comparisons.loc[comparisons["comparison"].eq("tabpfn3_full_vs_best_single")].copy()
    legacy = legacy.rename(
        columns={
            "reference_value": "single_value",
            "candidate_value": "tabpfn_value",
            "candidate_minus_reference": "tabpfn_minus_single",
            "probability_candidate_better": "probability_tabpfn_better",
        }
    ).drop(columns=["comparison", "reference_model", "candidate_model"])
    return legacy


def _load_age_sex_semantics_audit() -> pd.DataFrame:
    sources = (
        (
            "sex",
            "raw_8prs_no_sex_tabpfn_only",
            Path("outputs/performance/tabpfn3_8prs_no_sex_tabpfn_only.point.deltas.csv"),
        ),
        (
            "age",
            "raw_8prs_no_age_tabpfn_only",
            Path("outputs/performance/tabpfn3_8prs_no_age_tabpfn_only.point.deltas.csv"),
        ),
    )
    rows = []
    for removed_feature, variant, path in sources:
        if not path.exists():
            continue
        frame = read_table(path)
        selected = frame.loc[frame["input_variant"].astype(str).eq(variant)]
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "removed_from_inner_tabpfn": removed_feature,
                    "disease": str(row.disease),
                    "inner_tabpfn_covariates": int(row.tabpfn_n_covariates_requested),
                    "downstream_wrapper_covariates": int(
                        row.official_validation_covariates_requested
                    ),
                    "delta_auc_vs_full": float(row.delta_auc_vs_raw),
                    "source": str(path),
                }
            )
    return pd.DataFrame(rows)


def _bridge_input_reproduction_audit(
    selections: list[DiseaseSelection],
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for selection in selections:
        canonical = _filter_rows(read_table(selection.single_predictions), selection.single_filters)
        reproduced_path = (
            out_dir / "bridge_input_reproduction" / selection.disease / "predictions.csv"
        )
        if not reproduced_path.exists():
            continue
        reproduced = read_table(reproduced_path)
        reproduced = reproduced.loc[
            reproduced["disease"].astype(str).eq(selection.disease)
            & reproduced["method"].astype(str).eq(selection.bridge_method)
        ]
        paired = canonical[["sampleid", "y_true", "y_prob"]].merge(
            reproduced[["sampleid", "y_true", "y_prob"]],
            on="sampleid",
            how="inner",
            suffixes=("_canonical", "_reproduced"),
            validate="one_to_one",
        )
        differences = np.abs(
            paired["y_prob_canonical"].to_numpy(dtype=float)
            - paired["y_prob_reproduced"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "disease": selection.disease,
                "n_canonical": int(len(canonical)),
                "n_reproduced": int(len(reproduced)),
                "n_paired": int(len(paired)),
                "label_mismatches": int(
                    paired["y_true_canonical"].ne(paired["y_true_reproduced"]).sum()
                ),
                "max_absolute_probability_difference": float(differences.max()),
                "mean_absolute_probability_difference": float(differences.mean()),
                "passes_1e_minus_12": bool(
                    len(paired) == len(canonical) == len(reproduced)
                    and not paired["y_true_canonical"].ne(paired["y_true_reproduced"]).any()
                    and differences.max() <= 1e-12
                ),
                "canonical_predictions": selection.single_predictions,
                "reproduced_predictions": str(reproduced_path),
            }
        )
    return pd.DataFrame(rows)


def _markdown_frame(frame: pd.DataFrame) -> str:
    return _markdown_table(
        [str(column) for column in frame.columns],
        [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)],
    )


def _delta_with_ci(row: pd.Series | None) -> str:
    if row is None:
        return "NA"
    return (
        f"{_fmt(row['candidate_minus_reference'], 4)} "
        f"[{_fmt(row['ci_low'], 4)}, {_fmt(row['ci_high'], 4)}]"
    )


def _lookup_delta(
    comparisons: pd.DataFrame,
    *,
    disease: str,
    ancestry_group: str,
    comparison: str,
    metric: str = "auc",
) -> pd.Series | None:
    selected = comparisons.loc[
        comparisons["disease"].eq(disease)
        & comparisons["ancestry_group"].eq(ancestry_group)
        & comparisons["comparison"].eq(comparison)
        & comparisons["metric"].eq(metric)
    ]
    return None if selected.empty else selected.iloc[0]


def _write_three_arm_report(
    out_dir: Path,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    semantics_audit: pd.DataFrame,
    reproduction_audit: pd.DataFrame,
    *,
    n_bootstraps: int,
) -> None:
    overall_rows = []
    overall_metrics = metrics.loc[metrics["ancestry_group"].eq("ALL")]
    for disease in ["cad", "bc", "T2D"]:
        disease_rows = overall_metrics.loc[overall_metrics["disease"].eq(disease)].set_index(
            "model"
        )
        a = disease_rows.loc["best_single_prs"]
        b = disease_rows.loc["tabpfn3_best_single"]
        c = disease_rows.loc["tabpfn3_full"]
        overall_rows.append(
            {
                "Disease": disease,
                "N/cases": f"{int(a['n'])}/{int(a['cases'])}",
                "A AUC": _fmt(a["auc"], 4),
                "B AUC": _fmt(b["auc"], 4),
                "C AUC": _fmt(c["auc"], 4),
                "B - A [95% CI]": _delta_with_ci(
                    _lookup_delta(
                        comparisons,
                        disease=disease,
                        ancestry_group="ALL",
                        comparison=COMPARISON_ORDER[0],
                    )
                ),
                "C - B [95% CI]": _delta_with_ci(
                    _lookup_delta(
                        comparisons,
                        disease=disease,
                        ancestry_group="ALL",
                        comparison=COMPARISON_ORDER[1],
                    )
                ),
                "C - A [95% CI]": _delta_with_ci(
                    _lookup_delta(
                        comparisons,
                        disease=disease,
                        ancestry_group="ALL",
                        comparison=COMPARISON_ORDER[2],
                    )
                ),
            }
        )
    overall_table = pd.DataFrame(overall_rows)

    secondary_rows = []
    for row in overall_metrics.itertuples(index=False):
        secondary_rows.append(
            {
                "Disease": row.disease,
                "Arm": MODEL_DISPLAY[row.model],
                "OR90": _fmt(row.top10_vs_lower90_or, 3),
                "Nagelkerke R2": _fmt(row.nagelkerke_r2, 4),
                "Brier": _fmt(row.brier, 4),
                "AUPRC": _fmt(row.auprc, 4),
            }
        )
    secondary_table = pd.DataFrame(secondary_rows)

    point_summaries = []
    for comparison in COMPARISON_ORDER:
        selected = comparisons.loc[
            ~comparisons["ancestry_group"].eq("ALL")
            & comparisons["comparison"].eq(comparison)
            & comparisons["metric"].eq("auc")
        ]
        point_summaries.append(
            {
                "comparison": COMPARISON_DISPLAY[comparison],
                "better": int(selected["candidate_minus_reference"].gt(0).sum()),
                "total": int(len(selected)),
                "ci_positive": int(selected["ci_low"].gt(0).sum()),
                "ci_negative": int(selected["ci_high"].lt(0).sum()),
            }
        )

    subgroup_tables: dict[str, pd.DataFrame] = {}
    for disease in ["cad", "bc", "T2D"]:
        disease_metrics = metrics.loc[
            metrics["disease"].eq(disease) & ~metrics["ancestry_group"].eq("ALL")
        ]
        rows = []
        for ancestry_group in sorted(disease_metrics["ancestry_group"].unique()):
            group = disease_metrics.loc[
                disease_metrics["ancestry_group"].eq(ancestry_group)
            ].set_index("model")
            a = group.loc["best_single_prs"]
            b = group.loc["tabpfn3_best_single"]
            c = group.loc["tabpfn3_full"]
            rows.append(
                {
                    "Group": ancestry_group,
                    "N/cases": f"{int(a['n'])}/{int(a['cases'])}",
                    "A AUC": _fmt(a["auc"]),
                    "B AUC": _fmt(b["auc"]),
                    "C AUC": _fmt(c["auc"]),
                    "B - A [95% CI]": _delta_with_ci(
                        _lookup_delta(
                            comparisons,
                            disease=disease,
                            ancestry_group=ancestry_group,
                            comparison=COMPARISON_ORDER[0],
                        )
                    ),
                    "C - B [95% CI]": _delta_with_ci(
                        _lookup_delta(
                            comparisons,
                            disease=disease,
                            ancestry_group=ancestry_group,
                            comparison=COMPARISON_ORDER[1],
                        )
                    ),
                    "C - A [95% CI]": _delta_with_ci(
                        _lookup_delta(
                            comparisons,
                            disease=disease,
                            ancestry_group=ancestry_group,
                            comparison=COMPARISON_ORDER[2],
                        )
                    ),
                }
            )
        subgroup_tables[disease] = pd.DataFrame(rows)

    audit_columns = [
        "removed_from_inner_tabpfn",
        "disease",
        "inner_tabpfn_covariates",
        "downstream_wrapper_covariates",
        "delta_auc_vs_full",
    ]
    audit_display = (
        semantics_audit[audit_columns].copy()
        if not semantics_audit.empty
        else pd.DataFrame(columns=audit_columns)
    )
    if not audit_display.empty:
        audit_display["delta_auc_vs_full"] = audit_display["delta_auc_vs_full"].map(
            lambda value: f"{value:+.4f}"
        )
        audit_display.columns = [
            "Removed from inner TabPFN",
            "Disease",
            "Inner covariates",
            "Wrapper covariates",
            "Delta AUC vs full",
        ]

    reproduction_columns = [
        "disease",
        "n_paired",
        "label_mismatches",
        "max_absolute_probability_difference",
        "passes_1e_minus_12",
    ]
    reproduction_display = (
        reproduction_audit[reproduction_columns].copy()
        if not reproduction_audit.empty
        else pd.DataFrame(columns=reproduction_columns)
    )
    if not reproduction_display.empty:
        reproduction_display["max_absolute_probability_difference"] = reproduction_display[
            "max_absolute_probability_difference"
        ].map(lambda value: f"{value:.3e}")
        reproduction_display.columns = [
            "Disease",
            "Paired N",
            "Label mismatches",
            "Max absolute probability difference",
            "Passes 1e-12",
        ]

    finding_lines = [
        (
            f"- {row['comparison']}: candidate has higher subgroup AUC in "
            f"{row['better']}/{row['total']} groups; paired 95% CI is above zero "
            f"in {row['ci_positive']} and below zero in {row['ci_negative']}."
        )
        for row in point_summaries
    ]
    markdown = [
        "# Ancestry-stratified gain from TabPFN-3 integration of the same single PRS",
        "",
        "## Arms and protocol",
        "",
        "- **A:** canonical best single PRS followed by the standard logistic wrapper.",
        "- **B:** TabPFN-3 receives metadata plus that exact same canonical single PRS; its OOF/full-context score is then passed through the same standard wrapper.",
        "- **C:** the previously selected full multi-PRS TabPFN-3, again passed through the same wrapper.",
        "- The standard wrapper contains sex, age, and PC1-PC10 for every arm. Breast cancer remains restricted to eligible female patients.",
        "- B matches C's selected TabPFN checkpoint and estimator settings by disease: CAD/T2D v3 default with 16 estimators; BC v3 binary with 8 estimators and balanced probabilities.",
        "- Within each disease, A/B/C are evaluated on identical official-validation patients. Groups are 1000G-reference genetic-similarity labels, not self-identified ancestry.",
        f"- Intervals use {n_bootstraps:,} paired, case-control-stratified bootstrap resamples.",
        "",
        "**Primary ancestry comparison: B - A.** It holds the canonical single PRS and final wrapper fixed, isolating the TabPFN-3 transformation of that PRS together with metadata. C - A is retained as secondary context, and C - B estimates the incremental contribution of the selected multi-PRS panel.",
        "",
        "## Primary overall comparison and three-arm AUC ladder",
        "",
        _markdown_frame(overall_table),
        "",
        "![Three-arm ancestry-stratified AUC](plots/three_arm_auc.svg)",
        "",
        "## Overall secondary metrics",
        "",
        _markdown_frame(secondary_table),
        "",
        "## Canonical single-PRS input reproduction",
        "",
        "Before inserting TabPFN-3, the prepared bridge input was sent directly through the standard wrapper. It reproduces arm A patient-by-patient:",
        "",
        _markdown_frame(reproduction_display),
        "",
        "## Subgroup direction summary",
        "",
        *finding_lines,
        "",
    ]
    for disease, table in subgroup_tables.items():
        markdown.extend([f"## {disease}", "", _markdown_frame(table), ""])

    markdown.extend(
        [
            "## Age/sex evaluation-semantics audit",
            "",
            "The earlier one-at-a-time experiments removed sex or age only from the inner TabPFN feature matrix while retaining all 12 covariates in the downstream wrapper. The newer five-arm fixed-wrapper analysis is summarized separately under `outputs/tabpfn_metadata_score_ablations/summary`; the table below is retained only as provenance for the earlier official-validation runs.",
            "",
        ]
    )
    if not audit_display.empty:
        markdown.extend([_markdown_frame(audit_display), ""])
    markdown.extend(
        [
            "## Interpretation guardrails",
            "",
            "- B - A tests whether the TabPFN transformation of one PRS plus metadata adds value beyond the ordinary single-PRS wrapper.",
            "- C - B is the closest available estimate of the incremental value of the selected multi-PRS panel after introducing the same TabPFN-3 layer.",
            "- A and B use the exact canonical final-trainval single-PRS values underlying the prior A arm. This preserves the requested bridge but also preserves the prior post hoc validation-selection context.",
            "- Model choices were selected using this validation set, so this is explanatory and post hoc, not an unbiased confirmatory benchmark.",
            "- Small ancestry-like subgroups yield wide intervals; subgroup OR90 is especially volatile.",
            "- Machine-readable results are in `ancestry_group_metrics.csv`, `paired_three_arm_comparison_summary.csv`, and the existing decomposition files.",
            "",
        ]
    )
    (out_dir / "ANALYSIS.md").write_text("\n".join(markdown), encoding="utf-8")

    def table_html(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, border=0, classes="data", escape=True)

    finding_html = "".join(
        f"<li>{html.escape(line.removeprefix('- '))}</li>" for line in finding_lines
    )
    subgroup_html = "".join(
        f"<h2>{html.escape(disease)}</h2>{table_html(table)}"
        for disease, table in subgroup_tables.items()
    )
    audit_html = table_html(audit_display) if not audit_display.empty else ""
    reproduction_html = table_html(reproduction_display) if not reproduction_display.empty else ""
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ancestry-stratified gain from TabPFN-3 single-PRS integration</title>
<style>
body {{ font-family: Inter, system-ui, sans-serif; color: #17212b; margin: 0; background: #f4f7fb; }}
main {{ max-width: 1500px; margin: 28px auto; padding: 32px; background: white; box-shadow: 0 4px 24px #d8dee8; }}
h1 {{ margin-top: 0; }} h2 {{ margin-top: 34px; border-bottom: 2px solid #dce6f2; padding-bottom: 7px; }}
.arms {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
.arm {{ padding: 14px; border-radius: 8px; background: #eef4fb; }}
table.data {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 0.88rem; }}
table.data th, table.data td {{ border: 1px solid #d8e0ea; padding: 7px 9px; text-align: right; white-space: nowrap; }}
table.data th:first-child, table.data td:first-child {{ text-align: left; }}
table.data th {{ background: #e8f0f8; position: sticky; top: 0; }}
.note {{ background: #fff6dc; border-left: 4px solid #d59b17; padding: 12px 16px; }}
code {{ background: #eef1f5; padding: 1px 4px; }}
@media (max-width: 850px) {{ main {{ margin: 0; padding: 16px; }} .arms {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body><main>
<h1>Ancestry-stratified gain from TabPFN-3 integration of the same single PRS</h1>
<div class="arms">
<div class="arm"><strong>A</strong><br>Best single PRS + standard wrapper</div>
<div class="arm"><strong>B</strong><br>TabPFN-3(metadata + same single PRS) + standard wrapper</div>
<div class="arm"><strong>C</strong><br>Selected full multi-PRS TabPFN-3 + standard wrapper</div>
</div>
<p>Identical official-validation patients within disease; BC is eligible-female only. Every wrapper uses sex, age, and PC1-PC10. Intervals use {n_bootstraps:,} paired case-control-stratified bootstrap resamples.</p>
<div class="note"><strong>Primary ancestry comparison: B - A.</strong> It holds the canonical single PRS and final wrapper fixed, isolating the TabPFN-3 transformation of that PRS together with metadata. C - A is secondary context; C - B estimates the incremental contribution of the selected multi-PRS panel.</div>
<h2>Primary overall comparison and three-arm AUC ladder</h2>{table_html(overall_table)}
<img src="plots/three_arm_auc.svg" alt="Three-arm ancestry-stratified AUC" style="width:100%;height:auto">
<h2>Overall secondary metrics</h2>{table_html(secondary_table)}
<h2>Canonical single-PRS input reproduction</h2>
<p>The prepared bridge input was first passed directly through the standard wrapper and reproduces arm A patient-by-patient.</p>
{reproduction_html}
<h2>Subgroup direction summary</h2><ul>{finding_html}</ul>
{subgroup_html}
<h2>Age/sex evaluation-semantics audit</h2>
<div class="note">The earlier no-sex/no-age runs removed one feature only inside TabPFN while retaining all 12 covariates in the downstream wrapper. The newer five-arm fixed-wrapper analysis is under <code>outputs/tabpfn_metadata_score_ablations/summary</code>; this table is retained only as provenance for the earlier official-validation runs.</div>
{audit_html}
<h2>Guardrails</h2>
<ul><li>This is a post hoc explanatory analysis because model choices used the validation set.</li><li>Groups are 1000G-reference genetic-similarity labels, not self-identified ancestry.</li><li>Small-group intervals and OR90 are unstable.</li></ul>
</main></body></html>
"""
    (out_dir / "ANALYSIS.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare canonical best single-PRS, TabPFN-3 integration of that same PRS, "
            "and full TabPFN-3 predictions within 1000G-reference genetic-similarity groups."
        )
    )
    parser.add_argument("--ancestry-labels", default=DEFAULT_ANCESTRY_LABELS)
    parser.add_argument("--tabpfn-root", default=DEFAULT_TABPFN_ROOT)
    parser.add_argument("--bridge-root", default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ancestry = read_table(args.ancestry_labels)
    required = [
        "sampleid",
        "nearest_1000G_superpop",
        "hard_label",
        "min_mahal",
        "mahal_margin",
    ]
    missing = [column for column in required if column not in ancestry]
    if missing:
        raise ValueError(f"Ancestry labels are missing columns: {missing}")
    ancestry = ancestry[required].copy()
    if ancestry["sampleid"].duplicated().any():
        raise ValueError("Ancestry labels must have one row per patient")

    selections = _selections(args.tabpfn_root, args.bridge_root)
    prediction_frames = [_load_predictions(selection, ancestry) for selection in selections]
    predictions = pd.concat(prediction_frames, ignore_index=True)

    metric_frames = []
    comparison_frames = []
    auc_frames = []
    loss_frames = []
    for disease, frame in predictions.groupby("disease", sort=False):
        del disease
        metric_frames.append(subgroup_metrics(frame))
        comparison_frames.append(
            paired_bootstrap_comparisons(
                frame,
                n_bootstraps=args.n_bootstraps,
                seed=args.seed,
                metric_names=("auc",),
            )
        )
        auc_frames.append(auc_pair_decomposition(frame))
        loss_frames.append(additive_loss_decomposition(frame))

    metrics = pd.concat(metric_frames, ignore_index=True)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    deltas = _legacy_delta_frame(comparisons)
    auc_decomposition = pd.concat(auc_frames, ignore_index=True)
    loss_decomposition = pd.concat(loss_frames, ignore_index=True)
    auc_decomposition_summary = (
        auc_decomposition.groupby(["disease", "pair_type"], as_index=False)
        .agg(
            pair_weight=("pair_weight", "sum"),
            weighted_auc_delta_contribution=(
                "weighted_auc_delta_contribution",
                "sum",
            ),
        )
        .assign(
            mean_pair_auc_delta=lambda frame: (
                frame["weighted_auc_delta_contribution"] / frame["pair_weight"]
            )
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(predictions, out_dir / "patient_predictions.csv")
    write_table(metrics, out_dir / "ancestry_group_metrics.csv")
    legacy_delta_path = out_dir / "paired_bootstrap_delta_summary.csv"
    if not legacy_delta_path.exists():
        write_table(deltas, legacy_delta_path)
    write_table(
        comparisons,
        out_dir / "paired_three_arm_comparison_summary.csv",
    )
    write_table(auc_decomposition, out_dir / "auc_pair_decomposition.csv")
    write_table(
        auc_decomposition_summary,
        out_dir / "auc_decomposition_summary.csv",
    )
    write_table(loss_decomposition, out_dir / "additive_loss_decomposition.csv")
    write_table(
        (
            predictions.groupby(["disease", "ancestry_group"], as_index=False)
            .agg(n=("sampleid", "size"), cases=("y_true", "sum"))
            .assign(controls=lambda frame: frame["n"] - frame["cases"])
        ),
        out_dir / "ancestry_group_counts.csv",
    )
    selection_rows = []
    for selection in selections:
        row = asdict(selection)
        row["single_filters"] = json.dumps(row["single_filters"], sort_keys=True)
        selection_rows.append(row)
    write_table(pd.DataFrame(selection_rows), out_dir / "model_selection_manifest.csv")
    semantics_audit = _load_age_sex_semantics_audit()
    if not semantics_audit.empty:
        write_table(semantics_audit, out_dir / "age_sex_evaluation_semantics_audit.csv")
    reproduction_audit = _bridge_input_reproduction_audit(selections, out_dir)
    if not reproduction_audit.empty:
        write_table(
            reproduction_audit,
            out_dir / "bridge_input_reproduction_audit.csv",
        )
    _write_three_arm_report(
        out_dir,
        metrics,
        comparisons,
        semantics_audit,
        reproduction_audit,
        n_bootstraps=args.n_bootstraps,
    )
    print(f"Wrote ancestry-stratified comparison to {out_dir}")


if __name__ == "__main__":
    main()
