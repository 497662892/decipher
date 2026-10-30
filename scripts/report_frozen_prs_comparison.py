"""Independently verify and report the matched frozen-PRS five-fold comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from run_frozen_prs_comparison import DEFAULT_RUN, read_csv, verify_inputs

from decipehr.ancestry_performance import paired_bootstrap_comparisons, prediction_metrics
from decipehr.external_pgs import sha256_file, write_json
from decipehr.frozen_prs_comparison import ARMS, OUTCOMES, paired_predictions, verify_frozen_fold


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    args = parser.parse_args()
    root = Path(args.run_dir)
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    verify_inputs(protocol)
    frames, audits, receipts, versions = [], [], {}, []
    for disease in OUTCOMES:
        for arm in ARMS:
            data = pd.read_parquet(root / "inputs" / disease / f"{arm}.parquet")
            for fold in range(5):
                directory = root / "folds" / disease / arm / str(fold)
                receipt_path = directory / "COMPLETE.json"
                receipt = json.loads(receipt_path.read_text())
                if receipt["status"] != "complete" or receipt["protocol_sha256"] != sha256_file(root / "PROTOCOL.json"):
                    raise ValueError("Incomplete fold or changed protocol")
                for name, digest in receipt["output_sha256"].items():
                    if sha256_file(directory / name) != digest:
                        raise ValueError(f"Changed fold output: {directory / name}")
                pred, inner, fits = [read_csv(directory / name) for name in [
                    "predictions.csv", "wrapper_training_oof.csv", "fit_audit.csv",
                ]]
                audits.append(verify_frozen_fold(data, pred, inner, fits, disease=disease,
                                                 arm=arm, fold=fold, seed=protocol["seed"]))
                frames.append(pred)
                receipts[str(receipt_path)] = sha256_file(receipt_path)
                versions.append(receipt["versions"])
            print(f"Verified {disease}/{arm}: all five folds and independent wrappers", flush=True)
    if any(v != versions[0] for v in versions):
        raise ValueError("Different library versions across comparison arms")
    predictions = pd.concat(frames, ignore_index=True)
    audit = pd.DataFrame(audits)
    for _, frame in audit.groupby(["disease", "fold"]):
        if frame.outer_context_ids_sha256.nunique() != 1 or frame.outer_query_ids_sha256.nunique() != 1:
            raise ValueError("Comparison arms used different outer contexts")
    overall, per_fold, subgroups, contrasts = [], [], [], []
    for scale, column in [("raw", "y_prob"), ("official_wrapper", "wrapper_prob")]:
        wide = paired_predictions(predictions, column)
        for disease in OUTCOMES:
            pair = wide.loc[wide.disease.eq(disease)]
            bootstrap = protocol["bootstrap"]
            contrasts.append(paired_bootstrap_comparisons(
                pair.assign(ancestry_group=pd.NA), n_bootstraps=bootstrap["resamples"],
                seed=bootstrap["seed"], family_size=bootstrap["family_size"],
                comparisons=[("external_minus_internal", "internal", "external")],
                metric_names=["auc"],
            ).assign(prediction_scale=scale, inference_scope="conditional_on_frozen_panels_and_predictions"))
            for arm in ARMS:
                identifiers = {"disease": disease, "arm": arm, "prediction_scale": scale}
                overall.append({**identifiers, **prediction_metrics(pair.y_true.to_numpy(), pair[arm].to_numpy())})
                for fold, part in pair.groupby("fold"):
                    per_fold.append({**identifiers, "fold": int(fold),
                                     **prediction_metrics(part.y_true.to_numpy(), part[arm].to_numpy())})
                for group, part in pair.groupby("ancestry_group"):
                    if part.y_true.nunique() < 2:
                        measured = {"n": len(part), "cases": int(part.y_true.sum()), "auc": np.nan}
                    else:
                        measured = prediction_metrics(part.y_true.to_numpy(), part[arm].to_numpy())
                    subgroups.append({**identifiers, "ancestry_group": group, "exploratory": True, **measured})
            print(f"Computed paired AUC intervals: {disease}/{scale}", flush=True)
    # Historical native predictions are a separate reference, never relabelled as a wrapper.
    historical = read_csv(protocol["paths"]["historical_predictions"])
    old_rows = []
    for disease in OUTCOMES:
        old = historical.loc[historical.disease.eq(disease)].rename(columns={"label": "y_true"})
        current = predictions.loc[predictions.disease.eq(disease) & predictions.arm.eq("external")]
        keys = ["sampleid", "disease", "fold", "y_true"]
        matched = current[keys].merge(old[keys + ["direct_probability"]], on=keys,
                                      how="outer", validate="one_to_one", indicator=True)
        if not matched._merge.eq("both").all():
            raise ValueError("Historical cohort, labels or folds do not match")
        old_rows.append({"disease": disease, "arm": "historical_native_reference",
                         "prediction_scale": "raw", **prediction_metrics(
                             matched.y_true.to_numpy(), matched.direct_probability.to_numpy(),
                         )})
    out = root / "report"
    out.mkdir(exist_ok=False)
    outputs = {
        "oof_predictions.csv": predictions, "cohort_fit_audit.csv": audit,
        "overall_metrics.csv": pd.DataFrame(overall), "fold_metrics.csv": pd.DataFrame(per_fold),
        "ancestry_metrics.csv": pd.DataFrame(subgroups), "paired_auc_differences.csv": pd.concat(contrasts),
        "historical_native_reference.csv": pd.DataFrame(old_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    lines = [
        "# 冻结内部／外部 PRS：train＋development 五折配对比较", "",
        "两边共用原患者五折、年龄／性别／PC1–PC10、TabPFN v3 default（16 estimators）与三折 OOF 训练的 Logistic wrapper。BC 仅纳入符合条件的女性。",
        "", "## Pooled OOF AUC", "",
        "| 疾病 | 输出 | 人数 | 内部 PRS | 外部 PRS | 外部减内部 | 配对 95% CI | 六项同时 CI |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in pd.concat(contrasts).itertuples():
        lines.append(f"| {row.disease} | {row.prediction_scale} | {row.n} | {row.reference_value:.6f} | "
                     f"{row.candidate_value:.6f} | {row.candidate_minus_reference:+.6f} | "
                     f"[{row.ci_low:+.6f}, {row.ci_high:+.6f}] | "
                     f"[{row.familywise_ci_low:+.6f}, {row.familywise_ci_high:+.6f}] |")
    lines.extend([
        "", "## 核查与解释范围", "",
        f"30 个疾病／面板／外层折均完整。wrapper 独立重算最大绝对误差：{audit.wrapper_replay_max_abs_error.max():.3g}。",
        "每个模型的填补、标准化、TabPFN context 和 wrapper 均局限于外层训练患者；wrapper 输入来自该训练集内部的三折 OOF。外层验证标签在预测后才加入输出。",
        "", "这次按用户确认比较已有冻结分数，不重建历史 PRS，也不重新选择旧参数。两套面板均有 development 选择历史，因此这是固定面板条件下的回顾性开发比较，不是完整嵌套 PRS 验证或新的独立测试。",
        "普通配对区间使用 2,000 次病例／对照分层患者 bootstrap、seed 42；Bonferroni 同时区间覆盖三病 × 两种输出共六项 AUC 对比。区间不包含重训练、历史选择及亲缘依赖的不确定性。",
        "亚组仅报告描述性点估计，不据此宣称所有祖源均改善。R²、Brier、log loss、校准及各折指标见相邻 CSV。",
        "", "旧 audited 包的原始输出见 historical_native_reference.csv；本次内部对照重新拟合了当前统一下游流程，所以无需与旧值完全相同。",
        "本次未访问 hold-out 数据、未改变最终 hold-out 预测，也未计算无标签 hold-out 的性能。", "",
    ])
    (out / "REPORT.md").write_text("\n".join(lines))
    verify_inputs(protocol)
    write_json(out / "VERIFIED.json", {
        "status": "complete", "comparison_scope": "matched downstream CV of frozen historical score panels",
        "end_to_end_nested_prs_cv": False, "n_folds_verified": len(audits),
        "n_prediction_rows": len(predictions), "wrapper_replay_max_abs_error": float(audit.wrapper_replay_max_abs_error.max()),
        "protocol_sha256": sha256_file(root / "PROTOCOL.json"), "source_receipts_sha256": receipts,
        "output_sha256": {name: sha256_file(out / name) for name in [*outputs, "REPORT.md"]},
        "versions": versions[0], "holdout_used": False,
    })
    print(outputs["overall_metrics.csv"][["disease", "arm", "prediction_scale", "n", "auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
