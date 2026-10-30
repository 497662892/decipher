"""Aggregate exploratory teacher response probes without publishing individual profiles."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    frames, age = [], []
    for backend in ["tabpfn3", "tabpfn35", "kumo"]:
        directory = root / "constraint_probe" / backend
        receipt = json.loads((directory / "COMPLETE.json").read_text())
        verify_files(directory, receipt["output_sha256"])
        assert receipt["contract"]["script_sha256"] == sha256_file("scripts/probe_prs_foundation_constraints.py")
        frames.append(pd.read_csv(directory / "diagnostics.csv"))
        for trait in ["T2D", "cad", "bc"]:
            p = pd.read_parquet(directory / f"{trait}_private_probes.parquet")
            age.append(p.loc[p.probe_feature.eq("age_sd")].assign(trait=trait, backend=backend))
    detail, curves = pd.concat(frames), pd.concat(age)
    rows = []
    for (backend, trait, dimension), part in detail.groupby(["backend", "trait", "dimension"]):
        rows.append({"backend": backend, "trait": trait, "dimension": dimension,
            "unique_anchor_profiles": int(part.profiles.max()), "features": len(part),
            "profile_feature_pairs": int(part.profiles.sum()), "grid_steps": int(part.steps.sum()),
            "negative_step_fraction_tol_0p0001": part.negative_steps.sum()/part.steps.sum(),
            "step_drop_over_0p1pp_fraction": np.average(part.fraction_steps_drop_over_0p1pp, weights=part.steps),
            "step_drop_over_1pp_fraction": np.average(part.fraction_steps_drop_over_1pp, weights=part.steps),
            "profile_drawdown_over_1pp_fraction": part.profiles_with_drawdown_over_1pp.sum()/part.profiles.sum(),
            "maximum_drawdown": part.maximum_profile_drawdown.max()})
    summary = pd.DataFrame(rows)
    out = root / "report/constraint_probe"
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "per_feature_diagnostics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    names = {"tabpfn3": "TabPFN v3", "tabpfn35": "TabPFN v3.5", "kumo": "Kumo Tabular"}
    for ax, trait in zip(axes, ["T2D", "cad", "bc"], strict=True):
        for backend, name in names.items():
            part = curves.loc[curves.trait.eq(trait) & curves.backend.eq(backend)]
            grouped = part.groupby("probe_value").probability
            middle = grouped.median()
            line, = ax.plot(middle.index, middle, label=name)
            ax.fill_between(middle.index, grouped.quantile(.25), grouped.quantile(.75), color=line.get_color(), alpha=.12)
        ax.set(title=trait.upper(), xlabel="age_sd (not years)", ylabel="Probability across reference profiles")
    axes[0].legend(fontsize=9)
    fig.suptitle("Train-internal teacher age responses · median and IQR across 18 ancestry-balanced profiles")
    for ext in ["svg", "png"]:
        fig.savefig(out / f"age_responses.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    payload = base64.b64encode((out / "age_responses.svg").read_bytes()).decode()
    for lang in ["zh", "en"]:
        def t(zh, en, lang=lang):
            return zh if lang == "zh" else en
        title = t("Teacher 与生物学先验的兼容性诊断", "Compatibility of teacher responses with biological priors")
        text = f'<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
        text += '<style>body{font:16px/1.7 system-ui;color:#18304b;max-width:1220px;margin:auto;padding:25px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{background:#fff4db;padding:20px}section{overflow:auto}</style>'
        text += f'<h1>{title}</h1><p class="note">'+t(
            "这是在主 val 结果出来后新增的探索性诊断。每个 Teacher 只以 U 为上下文；H 只提供特征锚点，未使用 H 标签，也未使用官方 val 构造探针。该诊断不改变已锁定的 Student 或其评估结果。",
            "This exploratory diagnostic was added after the main val results. Each teacher uses U as context; H supplies feature anchors only, without H labels or official val inputs in probe construction. It does not change any locked student or its evaluation.")+'</p>'
        text += '<p>'+t("每个祖源最多随机抽取 3 个 H 特征组合，共 18 个锚点。年龄使用 U 的 5%–95% 分位点网格；单个 PRS 在锚点附近 ±0.5 个 U 标准差内改变，并限制在 U 的边界。其余输入固定。",
            "At most three H feature profiles per ancestry yield 18 anchors. Age sweeps use U's 5th–95th percentile grid; one PRS at a time varies by ±0.5 U standard deviations, bounded by U support. Other inputs stay fixed.")+'</p>'
        text += f'<img alt="Age response medians and IQRs" src="data:image/svg+xml;base64,{payload}">'
        text += '<p>'+t("阴影是 18 个探针特征组合的四分位范围，不是置信区间。中位数曲线可能掩盖个别曲线的下降。1pp 表示概率下降 1 个百分点；drawdown 是在升序网格中，相对之前最高概率的最大回落。",
            "Shading is the IQR across 18 probe profiles, not a confidence interval. Median curves can hide individual decreases. 1pp denotes one percentage point of probability; drawdown is the largest decline from an earlier maximum along the increasing grid.")+'</p><section>'
        text += summary.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)+'</section>'
        text += '<p>'+t("分母按患者特征锚点、特征数和网格步数分别列出；这些比例不是人群发生率。条件响应可能落在真实联合分布支持较弱的区域，单调性违反也可能反映 Teacher 的噪声或偏差。因此，该结果只能指导下一轮约束消融，不能证明生物学先验错误，亦不能证明放宽约束会提高疾病预测。",
            "Denominators separate anchor profiles, features and grid steps; these fractions are not population rates. Conditional probes may have weak support in the real joint distribution, and violations may reflect teacher noise or bias. They motivate constraint ablations but do not establish a biological law as false or prove that relaxing it improves disease prediction.")+'</p></html>'
        (out / f"constraint_probe_{lang}.html").write_text(text)
    write_json(out / "VERIFIED.json", {"status": "aggregate_probe_report_complete", "script_sha256": sha256_file(__file__),
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*")
                          if p.is_file() and p.name != "VERIFIED.json"}})
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
