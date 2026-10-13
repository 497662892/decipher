#!/usr/bin/env python3
"""Compile official-validation method and TabPFN ablation tables."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "official_validation_tables"
DISEASES = ("cad", "bc", "T2D")
DISEASE_DISPLAY = {"cad": "CAD", "bc": "BC", "T2D": "T2D"}
COVARIATES = "sex + age_sd + PC1_sd-PC10_sd"
FIVE_PRS = "P+T + LDpred2 + lassosum2 + PRS-CS + SDPR"
ALL_EIGHT_PRS = (
    "P+T + LDpred2 + lassosum2 + PRS-CS + SDPR + MegaPRS no annot + "
    "MegaPRS annotated + SBayesRC 7M"
)

METRIC_COLUMNS = {
    "auc": "AUC",
    "auprc": "AUPRC",
    "brier": "Brier",
    "top10_vs_lower90_or": "OR90",
    "nagelkerke_r2": "Nagelkerke R2",
    "n": "N",
}
OUTPUT_COLUMNS = [
    "category",
    "method",
    "best_parameters",
    "model_input",
    *METRIC_COLUMNS.values(),
    "source_file",
]
TABLE2_COLUMNS = [
    "category",
    "method",
    "best_parameters",
    "model_input",
    "OR90",
    "Nagelkerke R2",
    "AUC",
    "AUPRC",
    "Brier",
    "N",
    "source_file",
]
TABLE1_COLUMNS = [
    "category",
    "method",
    "best_parameters",
    "OR90",
    "Nagelkerke R2",
    "AUC",
    "source_file",
]
ABLATION_AUDIT_COLUMNS = [
    "disease",
    "arm",
    "dataset_protocol",
    "train_context_n",
    "validation_prediction_only_n",
    "n_prs_features",
    "n_metadata_features",
    "prs_panel",
    "train_prediction_mode",
    "n_oof_folds",
    "clip_eps",
    "tabpfn_model_version",
    "n_estimators",
    "fit_summary_source",
    "metrics_source",
]

SINGLE_FAMILIES = (
    "covariates",
    "pt",
    "ldpred2",
    "lassosum2_ukbb",
    "prscs",
    "sdpr",
    "megaprs_noannot",
    "megaprs_annotated",
    "sbayesrc_7m",
)

RIDGE_SPECS = {
    "cad": (
        "C=0.01; L2; class_weight=None; covariate-adjusted stack",
        "5 PRS + metadata",
        "outputs/ensemble_grid_nonannotated_a0p20/cad/"
        "penalty_l2_C0p01_class_none_prs_only_covariate_adjusted/"
        "official_validation_covariate_adjusted/point_metrics.csv",
    ),
    "bc": (
        "C=0.01; L2; class_weight=None; covariate-adjusted stack",
        "5 PRS + metadata",
        "outputs/ensemble_grid_nonannotated_a0p20/bc/"
        "penalty_l2_C0p01_class_none_prs_only_covariate_adjusted/"
        "official_validation_covariate_adjusted/point_metrics.csv",
    ),
    "T2D": (
        "C=0.03; L2; class_weight=None; PRS-only stack",
        "5 PRS",
        "outputs/ensemble_grid_nonannotated_a0p20/T2D/"
        "penalty_l2_C0p03_class_none_prs_only_covariate_adjusted/"
        "official_validation_prs_only/point_metrics.csv",
    ),
}

ELASTIC_SPECS = {
    "cad": (
        "C=0.03; l1_ratio=0.5; class_weight=None; covariate-adjusted stack",
        "5 PRS + metadata",
        "outputs/ensemble_grid_nonannotated_a0p20_elasticnet/cad/"
        "penalty_elasticnet_C0p03_l1r0p5_class_none_prs_only_covariate_adjusted/"
        "official_validation_covariate_adjusted/point_metrics.csv",
    ),
    "bc": (
        "C=0.1; l1_ratio=0.25; class_weight=None; covariate-adjusted stack",
        "5 PRS + metadata",
        "outputs/ensemble_grid_nonannotated_a0p20_elasticnet/bc/"
        "penalty_elasticnet_C0p1_l1r0p25_class_none_prs_only_covariate_adjusted/"
        "official_validation_covariate_adjusted/point_metrics.csv",
    ),
    "T2D": (
        "C=0.1; l1_ratio=0.25; class_weight=None; PRS-only stack",
        "5 PRS",
        "outputs/ensemble_grid_nonannotated_a0p20_elasticnet/T2D/"
        "penalty_elasticnet_C0p1_l1r0p25_class_none_prs_only_covariate_adjusted/"
        "official_validation_prs_only/point_metrics.csv",
    ),
}

TABPFN26_MATCHED_PRS = {
    "cad": "LDpred2 + PRS-CS + SDPR + MegaPRS no annot + MegaPRS annotated",
    "bc": "P+T + LDpred2 + lassosum2 + PRS-CS + MegaPRS annotated + SBayesRC 7M",
    "T2D": "LDpred2 + SDPR + MegaPRS no annot + MegaPRS annotated + SBayesRC 7M",
}
TABPFN26_MATCHED_PRS_INPUT = {
    "cad": "final selected 5 PRS",
    "bc": "final selected 6 PRS",
    "T2D": "final selected 5 PRS",
}

BEST_TABPFN3 = {
    "cad": {
        "prs": "lassosum2 + LDpred2 + PRS-CS + SDPR",
        "input": "best-observed 4 PRS",
        "n_estimators": 16,
        "checkpoint": "v3 default",
        "extra_parameters": "",
        "full_path": (
            "outputs/train_only_tabpfn3_no_pt_no_megaprs_sbayesrc/"
            "clip1e-3_ne16/official_validation_covariate_adjusted/"
            "point_metrics.csv"
        ),
        "prs_only_path": (
            "outputs/tabpfn3_best_or90_matched_prs_only/cad/"
            "official_validation_prs_only/point_metrics.csv"
        ),
    },
    "bc": {
        "prs": FIVE_PRS,
        "input": "best-observed 5 PRS",
        "n_estimators": 8,
        "checkpoint": "v3 binary",
        "extra_parameters": "; balance_probabilities=true",
        "full_path": (
            "outputs/tabpfn3_binary_grid_nonannotated_a0p20/"
            "clip1e-3_ne8_balance/official_validation_covariate_adjusted/"
            "point_metrics.csv"
        ),
        "prs_only_path": (
            "outputs/tabpfn3_binary_grid_nonannotated_a0p20/"
            "clip1e-3_ne8_balance/official_validation_prs_only/"
            "point_metrics.csv"
        ),
    },
    "T2D": {
        "prs": ALL_EIGHT_PRS,
        "input": "best-observed all 8 PRS",
        "n_estimators": 16,
        "checkpoint": "v3 default",
        "extra_parameters": "",
        "full_path": (
            "outputs/tabpfn3_plus_megaprs_sbayesrc/clip1e-3_ne16/"
            "official_validation_covariate_adjusted/point_metrics.csv"
        ),
        "prs_only_path": (
            "outputs/tabpfn3_best_or90_matched_prs_only/T2D/"
            "official_validation_prs_only/point_metrics.csv"
        ),
    },
}


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _metrics(row: pd.Series, prefix: str = "") -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for source, output in METRIC_COLUMNS.items():
        value = row[f"{prefix}{source}"]
        values[output] = int(value) if output == "N" else float(value)
    return values


def _point_row(relative_path: str, disease: str) -> tuple[pd.Series, str]:
    path = ROOT / relative_path
    frame = pd.read_csv(path)
    selected = frame.loc[frame["method"] == f"prs_covariate_{disease}"]
    if len(selected) != 1:
        raise ValueError(f"Expected one official score-plus-covariate row in {path}")
    return selected.iloc[0], _relative(path)


def _record(
    *,
    category: str,
    method: str,
    best_parameters: str,
    model_input: str,
    metrics: dict[str, float | int],
    source_file: str,
) -> dict[str, object]:
    return {
        "category": category,
        "method": method,
        "best_parameters": best_parameters,
        "model_input": model_input,
        **metrics,
        "source_file": source_file,
    }


def _single_records(disease: str) -> list[dict[str, object]]:
    relative_path = (
        "outputs/final_stage_single_prs_performance/"
        "final_single_prs_validation_vs_trainval5fold_best_by_validation_auc.csv"
    )
    path = ROOT / relative_path
    frame = pd.read_csv(path)
    frame = frame.loc[
        (frame["disease"] == disease) & frame["model_family"].isin(SINGLE_FAMILIES)
    ].copy()
    frame["model_family"] = pd.Categorical(
        frame["model_family"], categories=SINGLE_FAMILIES, ordered=True
    )
    frame = frame.sort_values("model_family")
    if len(frame) != len(SINGLE_FAMILIES):
        raise ValueError(f"Missing single-model rows for {disease}")

    records = []
    for row in frame.itertuples(index=False):
        is_covariates = row.model_family == "covariates"
        records.append(
            _record(
                category="Baseline" if is_covariates else "Single PRS",
                method=row.model_display,
                best_parameters=row.setting,
                model_input=COVARIATES if is_covariates else "one PRS score",
                metrics={
                    output: (
                        int(getattr(row, f"validation_{source}"))
                        if output == "N"
                        else float(getattr(row, f"validation_{source}"))
                    )
                    for source, output in METRIC_COLUMNS.items()
                },
                source_file=relative_path,
            )
        )
    return records


def _classical_ensemble_records(disease: str) -> list[dict[str, object]]:
    records = []
    for method, specs in (
        ("Ridge stacking", RIDGE_SPECS),
        ("Elastic-net stacking", ELASTIC_SPECS),
    ):
        parameters, model_input, relative_path = specs[disease]
        row, source_file = _point_row(relative_path, disease)
        records.append(
            _record(
                category="Linear ensemble",
                method=method,
                best_parameters=parameters,
                model_input=model_input,
                metrics=_metrics(row),
                source_file=source_file,
            )
        )
    return records


def _best_linear_ensemble_record(disease: str) -> dict[str, object]:
    records = _classical_ensemble_records(disease)
    return max(records, key=lambda record: float(record["AUC"]))


def _tabpfn_record(
    *,
    disease: str,
    method: str,
    best_parameters: str,
    model_input: str,
    relative_path: str,
) -> dict[str, object]:
    row, source_file = _point_row(relative_path, disease)
    return _record(
        category="TabPFN-based ensemble",
        method=method,
        best_parameters=best_parameters,
        model_input=model_input,
        metrics=_metrics(row),
        source_file=source_file,
    )


def _tabpfn_records(disease: str) -> list[dict[str, object]]:
    best_tabpfn3 = BEST_TABPFN3[disease]
    best_tabpfn3_parameters = (
        f"clip_eps=0.001; n_estimators={best_tabpfn3['n_estimators']}; "
        f"checkpoint={best_tabpfn3['checkpoint']}; 5-fold OOF"
        f"{best_tabpfn3['extra_parameters']}; inputs={best_tabpfn3['prs']}"
    )
    records = [
        _tabpfn_record(
            disease=disease,
            method="TabPFN-2.6 metadata-only",
            best_parameters="clip_eps=0.001; n_estimators=8; 5-fold OOF",
            model_input=COVARIATES,
            relative_path=(
                f"outputs/tabpfn26_final_selection_meta_only/{disease}/"
                "official_validation_covariate_only/point_metrics.csv"
            ),
        ),
        _tabpfn_record(
            disease=disease,
            method="TabPFN-2.6 PRS-only (matched final panel)",
            best_parameters=(
                "clip_eps=0.001; n_estimators=8; 5-fold OOF; "
                f"inputs={TABPFN26_MATCHED_PRS[disease]}"
            ),
            model_input=TABPFN26_MATCHED_PRS_INPUT[disease],
            relative_path=(
                f"outputs/tabpfn26_final_selection_prs_only_no_metadata/{disease}/"
                "official_validation_prs_only/point_metrics.csv"
            ),
        ),
        _tabpfn_record(
            disease=disease,
            method="TabPFN-2.6 full (all 8 PRS)",
            best_parameters=(
                "clip_eps=0.001; n_estimators=8; 5-fold OOF; all 8 PRS"
            ),
            model_input="all 8 PRS + metadata",
            relative_path=(
                "outputs/tabpfn26_plus_megaprs_sbayesrc/clip1e-3_ne8/"
                "official_validation_covariate_adjusted/point_metrics.csv"
            ),
        ),
        _tabpfn_record(
            disease=disease,
            method="TabPFN-3 metadata-only",
            best_parameters="clip_eps=0.001; n_estimators=16; 5-fold OOF",
            model_input=COVARIATES,
            relative_path=(
                f"outputs/tabpfn3_final_selection_meta_only/{disease}/"
                "official_validation_covariate_only/point_metrics.csv"
            ),
        ),
        _tabpfn_record(
            disease=disease,
            method="TabPFN-3 PRS-only (matched best full panel)",
            best_parameters=best_tabpfn3_parameters,
            model_input=str(best_tabpfn3["input"]),
            relative_path=str(best_tabpfn3["prs_only_path"]),
        ),
    ]

    row, source_file = _point_row(str(best_tabpfn3["full_path"]), disease)
    records.append(
        _record(
            category="TabPFN-based ensemble",
            method="TabPFN-3 full (best observed OR90)",
            best_parameters=best_tabpfn3_parameters,
            model_input=f"{best_tabpfn3['input']} + metadata",
            metrics=_metrics(row),
            source_file=source_file,
        )
    )
    return records


def _tabpfn_full_records(disease: str) -> list[dict[str, object]]:
    return [
        record
        for record in _tabpfn_records(disease)
        if str(record["method"]).startswith(("TabPFN-2.6 full", "TabPFN-3 full"))
    ]


def _ablation_records(disease: str) -> list[dict[str, object]]:
    best_tabpfn3 = BEST_TABPFN3[disease]
    best_tabpfn3_parameters = (
        f"clip_eps=0.001; n_estimators={best_tabpfn3['n_estimators']}; "
        f"checkpoint={best_tabpfn3['checkpoint']}; 5-fold OOF"
        f"{best_tabpfn3['extra_parameters']}; inputs={best_tabpfn3['prs']}"
    )
    covariate_baseline = _single_records(disease)[0].copy()
    covariate_baseline.update(
        category="Ablation baseline",
        method="Covariates-only (no TabPFN)",
        best_parameters="logistic regression; no TabPFN",
        model_input=COVARIATES,
    )
    specs = (
        (
            "TabPFN-3 metadata-only",
            COVARIATES,
            f"outputs/tabpfn3_final_selection_meta_only/{disease}/"
            "official_validation_covariate_only/point_metrics.csv",
        ),
        (
            "TabPFN-3 PRS-only",
            str(best_tabpfn3["input"]),
            str(best_tabpfn3["prs_only_path"]),
            best_tabpfn3_parameters,
        ),
        (
            "TabPFN-3 full",
            f"{best_tabpfn3['input']} + metadata",
            str(best_tabpfn3["full_path"]),
            best_tabpfn3_parameters,
        ),
    )
    records = [covariate_baseline]
    for spec in specs:
        method, model_input, relative_path = spec[:3]
        row, source_file = _point_row(relative_path, disease)
        parameters = (
            spec[3]
            if len(spec) == 4
            else "clip_eps=0.001; n_estimators=16; checkpoint=v3 default; "
            "5-fold OOF"
        )
        records.append(
            _record(
                category="TabPFN-3 ablation",
                method=method,
                best_parameters=parameters,
                model_input=model_input,
                metrics=_metrics(row),
                source_file=source_file,
            )
        )
    return records


def _fit_summary_row(
    relative_path: str,
    *,
    disease: str,
    strategy: str,
) -> tuple[pd.Series, str]:
    path = ROOT / relative_path
    frame = pd.read_csv(path)
    selected = frame.loc[
        (frame["disease"] == disease) & (frame["strategy"] == strategy)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {disease}/{strategy} fit-summary row in {path}"
        )
    return selected.iloc[0], _relative(path)


def _ablation_audit_records(
    disease: str,
    metrics_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    best_tabpfn3 = BEST_TABPFN3[disease]
    full_fit_path = str(
        Path(str(best_tabpfn3["full_path"])).parents[1] / "tabpfn_fit_summary.csv"
    )
    prs_only_fit_path = str(
        Path(str(best_tabpfn3["prs_only_path"])).parents[1]
        / "tabpfn_fit_summary.csv"
    )
    specs = (
        (
            "metadata-only",
            "covariate_only",
            f"outputs/tabpfn3_final_selection_meta_only/{disease}/"
            "tabpfn_fit_summary.csv",
            "none",
            16,
        ),
        (
            "prs-only",
            "prs_only",
            prs_only_fit_path,
            str(best_tabpfn3["prs"]),
            int(best_tabpfn3["n_estimators"]),
        ),
        (
            "full",
            "covariate_adjusted",
            full_fit_path,
            str(best_tabpfn3["prs"]),
            int(best_tabpfn3["n_estimators"]),
        ),
    )
    metrics_by_arm = {
        str(record["method"]).removeprefix("TabPFN-3 ").lower(): record
        for record in metrics_records
    }
    records = []
    for arm, strategy, relative_path, prs_panel, n_estimators in specs:
        row, fit_summary_source = _fit_summary_row(
            relative_path,
            disease=disease,
            strategy=strategy,
        )
        records.append(
            {
                "disease": disease,
                "arm": arm,
                "dataset_protocol": (
                    "official training context; official validation prediction-only"
                ),
                "train_context_n": int(row["n_context"]),
                "validation_prediction_only_n": int(row["n_validation_scored"]),
                "n_prs_features": int(row["n_prs_features"]),
                "n_metadata_features": int(row["n_covariates_requested"]),
                "prs_panel": prs_panel,
                "train_prediction_mode": row["train_prediction_mode"],
                "n_oof_folds": int(row["n_oof_folds"]),
                "clip_eps": float(row["clip_eps"]),
                "tabpfn_model_version": row["tabpfn_model_version"],
                "n_estimators": n_estimators,
                "fit_summary_source": fit_summary_source,
                "metrics_source": metrics_by_arm[arm]["source_file"],
            }
        )
    return records


def _write_csv(
    path: Path,
    records: list[dict[str, object]],
    columns: list[str] = OUTPUT_COLUMNS,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _markdown_table(
    records: list[dict[str, object]],
    *,
    challenge_first: bool = False,
) -> str:
    if challenge_first:
        shown = [
            "method",
            "model_input",
            "OR90",
            "Nagelkerke R2",
            "AUC",
            "N",
        ]
    else:
        shown = [
            "category",
            "method",
            "best_parameters",
            "OR90",
            "Nagelkerke R2",
            "AUC",
        ]
    labels = {
        "category": "类别",
        "method": "方法",
        "best_parameters": "最优/固定参数",
        "model_input": "模型输入",
        "OR90": "OR90",
        "Nagelkerke R2": "Nagelkerke R²",
    }
    best = {}
    for metric in ("AUC", "AUPRC", "Brier", "OR90", "Nagelkerke R2"):
        values = [float(record[metric]) for record in records]
        best[metric] = min(values) if metric == "Brier" else max(values)

    lines = [
        "| " + " | ".join(labels.get(column, column) for column in shown) + " |",
        "|" + "|".join("---" for _ in shown) + "|",
    ]
    for record in records:
        cells = []
        for column in shown:
            value = record[column]
            if column in best:
                rendered = f"{float(value):.4f}"
                if abs(float(value) - best[column]) < 1e-12:
                    rendered = f"**{rendered}**"
            elif column == "N":
                rendered = str(int(value))
            else:
                rendered = str(value).replace("|", "/")
            cells.append(rendered)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table1: dict[str, list[dict[str, object]]] = {}
    table2: dict[str, list[dict[str, object]]] = {}
    all_tabpfn_records: dict[str, list[dict[str, object]]] = {}
    ablation_audit = []

    for disease in DISEASES:
        all_tabpfn_records[disease] = _tabpfn_records(disease)
        table1[disease] = [
            *_single_records(disease),
            _best_linear_ensemble_record(disease),
            *_tabpfn_full_records(disease),
        ]
        table2[disease] = _ablation_records(disease)
        ablation_audit.extend(_ablation_audit_records(disease, table2[disease]))
        _write_csv(
            OUT_DIR / f"table1_{disease}.csv",
            table1[disease],
            TABLE1_COLUMNS,
        )
        _write_csv(
            OUT_DIR / f"table2_tabpfn3_ablation_{disease}.csv",
            table2[disease],
            TABLE2_COLUMNS,
        )
    _write_csv(
        OUT_DIR / "table2_tabpfn3_ablation_audit.csv",
        ablation_audit,
        ABLATION_AUDIT_COLUMNS,
    )

    metadata_version_records = [
        record
        for disease in DISEASES
        for record in all_tabpfn_records[disease]
        if record["method"]
        in {"TabPFN-2.6 metadata-only", "TabPFN-3 metadata-only"}
    ]
    _write_csv(
        OUT_DIR / "supplement_tabpfn_metadata_only_versions.csv",
        metadata_version_records,
    )

    sections = [
        "# Official validation method comparison and TabPFN ablation",
        "",
        "生成日期：2026-07-23。",
        "",
        "## 口径",
        "",
        "- 所有数值均来自 official validation 的 `prs_covariate_<disease>` 行；"
        "即先生成方法分数，再使用训练集拟合 `方法分数 + sex + age_sd + PC1_sd-PC10_sd`"
        " 的 logistic evaluation wrapper，validation 仅用于预测和评估。",
        "- 因此 `metadata-only` 和 `PRS-only` 描述的是 TabPFN 内部输入；"
        "official evaluation wrapper 本身仍统一加入协变量。",
        "- BC 按 eligibility 配置限制为女性，N=1,628；CAD/T2D N=2,852。",
        "- Table 1 的单 PRS 参数是各方法在 official validation AUC 最高的已运行设置；"
        "TabPFN-3 则选用 official validation OR90 最高的已完成 full run，"
        "OR90 并列时依次使用 Nagelkerke R² 和 AUC 选择。"
        "每行三个指标始终来自同一 run，不跨配置拼接。"
        "这些选择适合回顾性汇总，但不应解释为独立于 validation 的无偏调参结果。",
        "- Table 1 不包含 ablation arms。每个疾病仅保留 covariates-only baseline、"
        "每个 single-PRS family 的最优设置、AUC 更高的一个 linear ensemble，以及 "
        "TabPFN-2.6 full 和 TabPFN-3 full。",
        "- Table 1 仅展示比赛核心指标 `Top 10% vs lower 90% OR`、"
        "`Nagelkerke R²` 和辅助指标 AUC；完整精度和其他指标保留在对应 CSV。",
        "- Table 2 首先列出不使用 TabPFN 的 covariates-only logistic baseline，随后是"
        " TabPFN-3 输入消融。每个疾病的 PRS-only 与对应最佳 full run 使用完全相同的"
        " PRS panel、checkpoint、clip 和 estimator 设置：CAD 为 4 PRS / v3 default /"
        " 16 estimators，BC 为 5 PRS / v3 binary / 8 estimators（probability balance），"
        "T2D 为 all 8 PRS / v3 default / 16 estimators。metadata-only 是单独完成的"
        " v3-default/16-estimator 参考臂。",
        "- Table 2 严格使用 official training 作为模型 context，official validation "
        "仅用于预测；不使用合并 train+validation 后再切片的 `eval_type=validation_only` "
        "结果。",
        "- `table2_tabpfn3_ablation_audit.csv` 记录三个 TabPFN arms 的实际 "
        "context/validation N、PRS 与 metadata 特征数、TabPFN version、estimators "
        "及 fit-summary 来源；普通 covariates-only baseline 的来源记录在 Table 2 CSV。",
        "- Table 2 按 challenge 口径把 `Top 10% vs lower 90% OR` 和 "
        "`Nagelkerke R²` 作为主指标；AUC 仅作为辅助判别指标。",
        "- 表内加粗表示该疾病、该表中对应指标的最佳值（Brier 越低越好，其余越高越好）。",
        "",
    ]
    for disease in DISEASES:
        display = DISEASE_DISPLAY[disease]
        sections.extend(
            [
                f"## Table 1 — {display}: single methods and ensembles",
                "",
                _markdown_table(table1[disease]),
                "",
                f"完整精度及逐行来源：`table1_{disease}.csv`。",
                "",
            ]
        )
    for disease in DISEASES:
        display = DISEASE_DISPLAY[disease]
        sections.extend(
            [
                f"## Table 2 — {display}: TabPFN-3 input ablation",
                "",
                _markdown_table(table2[disease], challenge_first=True),
                "",
                f"完整精度及逐行来源：`table2_tabpfn3_ablation_{disease}.csv`。",
                "",
            ]
        )

    sections.extend(
        [
            "## TabPFN metadata-only version check",
            "",
            "TabPFN-2.6 和 TabPFN-3 的 metadata-only 结果均已重新以本地模型权重完成；"
            "完整精度见 `supplement_tabpfn_metadata_only_versions.csv`。",
            "",
            "## 主要源文件",
            "",
            "- `outputs/final_stage_single_prs_performance/"
            "final_single_prs_validation_vs_trainval5fold_best_by_validation_auc.csv`",
            "- `outputs/ensemble_grid_nonannotated_a0p20*/.../point_metrics.csv`",
            "- `outputs/tabpfn*_grid_nonannotated_a0p20/.../point_metrics.csv`",
            "- `outputs/tabpfn*_final_selection_meta_only/.../point_metrics.csv`",
            "- `outputs/train_only_tabpfn3_no_pt_no_megaprs_sbayesrc/.../"
            "point_metrics.csv`",
            "- `outputs/tabpfn3_binary_grid_nonannotated_a0p20/.../point_metrics.csv`",
            "- `outputs/tabpfn3_plus_megaprs_sbayesrc/.../point_metrics.csv`",
            "",
            "每个 CSV 的 `source_file` 列提供了该行使用的精确源路径。",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(sections))

    ablation_sections = [
        "# TabPFN-3 Ablation Study — Official Validation",
        "",
        "## Evaluation protocol",
        "",
        "- Strict official validation: the official training cohort supplies model "
        "context and OOF training scores; official-validation subjects are "
        "prediction-only.",
        "- Every arm is evaluated with the same logistic wrapper: the generated score "
        "plus sex, standardized age, and standardized ancestry PCs 1–10.",
        "- `PRS-only` and `full` are exactly matched within disease for PRS panel, "
        "checkpoint, clip, estimators, folds, and score scaling. Their only TabPFN-input "
        "difference is whether metadata is included.",
        "- The separately completed metadata-only reference uses the v3 default "
        "checkpoint, 16 estimators, `clip_eps=0.001`, and 5-fold OOF.",
        "- OR90 and Nagelkerke R² are the challenge-primary metrics; AUC is secondary.",
        "",
    ]
    for disease in DISEASES:
        display = DISEASE_DISPLAY[disease]
        config = BEST_TABPFN3[disease]
        ablation_sections.extend(
            [
                f"## {display}",
                "",
                _markdown_table(table2[disease], challenge_first=True),
                "",
                (
                    f"Matched PRS/full configuration: {config['prs']}; "
                    f"{config['checkpoint']}; n_estimators={config['n_estimators']}; "
                    f"clip_eps=0.001{config['extra_parameters']}."
                ),
                "",
                (
                    "Full-precision values and row-level sources: "
                    f"`table2_tabpfn3_ablation_{disease}.csv`."
                ),
                "",
            ]
        )
    ablation_sections.extend(
        [
            "## Interpretation",
            "",
            "The matched full model is the strongest arm on OR90, Nagelkerke R², and "
            "AUC for all three diseases. Metadata-only exceeds matched PRS-only for "
            "CAD and BC, whereas matched PRS-only exceeds metadata-only for T2D.",
            "",
        ]
    )
    (OUT_DIR / "FINAL_TABPFN3_ABLATION.md").write_text(
        "\n".join(ablation_sections)
    )

    comparison_sections = [
        "# Table 1 — Official Validation Method Comparison",
        "",
        "## Evaluation protocol",
        "",
        "- All rows use strict official validation. Models and evaluation wrappers are "
        "fit on the official training cohort; official-validation subjects are "
        "prediction-only.",
        "- Every score-based method is evaluated using the same logistic wrapper: "
        "method score plus sex, standardized age, and standardized ancestry PCs 1–10.",
        "- Breast cancer is restricted to eligible female participants (N=1,628); "
        "CAD and T2D use N=2,852.",
        "- OR90 and Nagelkerke R² are the challenge-primary metrics. AUC is retained as "
        "a secondary discrimination metric.",
        "- Single-PRS settings are the highest-official-validation-AUC completed settings "
        "for retrospective reporting; they are not independent validation-free tuning "
        "estimates.",
        "- The TabPFN-3 row is the highest-OR90 completed full run for each disease; "
        "ties are resolved by Nagelkerke R² and then AUC. "
        "OR90, Nagelkerke R², and AUC are always copied together from that same run; "
        "metrics are never mixed across configurations.",
        "- For each disease, only the higher-AUC completed linear ensemble is retained: "
        "Ridge stacking for CAD and T2D, and elastic-net stacking for BC.",
        "- TabPFN metadata-only and PRS-only ablations are excluded. Only the TabPFN-2.6 "
        "full and TabPFN-3 full models are retained.",
        f"- Linear ensemble inputs: {FIVE_PRS}.",
        f"- TabPFN-2.6 full inputs: {ALL_EIGHT_PRS}.",
        "- TabPFN-3 best-observed panels: "
        + "; ".join(
            f"{DISEASE_DISPLAY[disease]} = {BEST_TABPFN3[disease]['prs']}"
            for disease in DISEASES
        )
        + ".",
        "- Bold indicates the best value within each disease table; higher is better for "
        "all three displayed metrics.",
        "",
    ]
    for disease in DISEASES:
        display = DISEASE_DISPLAY[disease]
        comparison_sections.extend(
            [
                f"## {display}",
                "",
                _markdown_table(table1[disease]),
                "",
                f"Compact values and row-level sources: `table1_{disease}.csv`.",
                "",
            ]
        )
    (OUT_DIR / "FINAL_METHOD_COMPARISON.md").write_text(
        "\n".join(comparison_sections)
    )
    print(f"Wrote official-validation tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
