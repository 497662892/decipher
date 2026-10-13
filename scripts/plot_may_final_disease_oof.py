#!/usr/bin/env python3
"""Render CAD/T2D OOF reports using aggregate tables only."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

COLORS = {"submitted": "#087F8C", "covariates": "#B97918"}
LABELS = {"submitted": "Submitted TabPFN configuration", "covariates": "Covariates-only baseline"}
METRICS = [("top10_vs_lower90_or", "OR90"), ("nagelkerke_r2", "Nagelkerke R2"), ("auc", "AUC")]
CSS = """body{font:16px/1.75 -apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans CJK SC',sans-serif;color:#253445;background:#f3f7fa;margin:0}main{max-width:1120px;margin:30px auto;padding:40px;background:white;border-radius:15px}h1{font-size:32px;line-height:1.35;color:#163343}h2{font-size:23px;margin-top:42px;border-bottom:2px solid #deebed;padding-bottom:9px}.eyebrow{font-size:12px;letter-spacing:.12em;color:#087f8c;font-weight:bold}.lead{font-size:19px;color:#476071}.cards{display:flex;gap:16px;margin:26px 0}.cards div{flex:1;background:#eef6f7;padding:18px;border-radius:9px}.cards b{font-size:29px;display:block}.cards span{font-size:13px}.note{background:#fff8e8;border-left:4px solid #c59e48;padding:16px;font-size:14px}.formula{background:#eef3f7;padding:18px;font-family:monospace}figure{margin:26px 0}figure svg{width:100%;height:auto}figcaption{font-size:13px;color:#617185}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}td,th{border-bottom:1px solid #dce3e9;padding:9px;text-align:left}th{background:#eef3f7}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#f1f5f8;padding:16px}code{font-size:.87em;background:#eff3f6}a{color:#087f8c}.explorer{background:#edf7f8;padding:20px;border-radius:8px}.explorer label{margin-right:22px}select{padding:7px}#result{margin-top:16px;font-size:18px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}@media(max-width:700px){main{padding:18px;margin:0}.cards{display:block}.cards div{margin-bottom:8px}h1{font-size:27px}table{font-size:11px}}"""


def table(frame):
    return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")


def figure(root, name, caption):
    svg = (root / "figures" / f"{name}.svg").read_text()
    svg = svg[svg.index("<svg"):]
    for identifier in set(re.findall(r'id="([^"]+)"', svg)):
        target = f"fig_{root.name}_{name}_{identifier}"
        svg = svg.replace(f'id="{identifier}"', f'id="{target}"')
        svg = svg.replace(f'href="#{identifier}"', f'href="#{target}"')
        svg = svg.replace(f'url(#{identifier})', f'url(#{target})')
    return f"<figure>{svg}<figcaption>{caption}</figcaption></figure>"


def document(title, sections, script=""):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>' + CSS + '</style><main>' + "\n".join(sections) + '</main><script>' + script + '</script></html>'


def render(root: Path, combined: PdfPages) -> dict:
    def read(name):
        return pd.read_csv(root / f"{name}.csv")

    record = json.loads((root / "analysis_record.json").read_text())
    disease = record["disease"].upper()
    points = read("pooled_metrics").set_index("model")
    point, base = points.loc["submitted"], points.loc["covariates"]
    deciles = read("deciles")
    d = deciles.loc[deciles.model.eq("submitted")]
    thresholds, bands = read("thresholds"), read("tail_bands")
    folds, histograms, curves = read("fold_metrics"), read("histograms"), read("roc_pr_curves")
    intervals = read("intervals").set_index(["model", "metric"])
    deltas = read("paired_deltas").set_index("metric")
    membership = read("top10_membership").set_index("group")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    pdf = PdfPages(root / "OOF_ANALYSIS_FIGURES.pdf")
    manifest = []

    def save(fig, name, title):
        fig.suptitle(f"{disease}: {title}", fontsize=15, fontweight="bold")
        fig.text(.5, .005, f"Pooled 5-fold OOF | {int(point.n):,} patients, {int(point.cases):,} cases | Exact May submission scores",
                 ha="center", fontsize=8, color="#64748B")
        fig.tight_layout(rect=[0, .035, 1, .86 if name.startswith("05") else .95])
        for suffix in ["png", "svg", "pdf"]:
            fig.savefig(figures / f"{name}.{suffix}", bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        combined.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"name": name, "title": title})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, scale in zip(axes, ["score", "probability"], strict=True):
        for label, name, color in [(0, "Control", "#4083A5"), (1, "Case", "#C15464")]:
            h = histograms.loc[histograms.model.eq("submitted") & histograms.scale.eq(scale) & histograms.label.eq(label)]
            ax.stairs(h.density, np.r_[h.left, h.right.iloc[-1]], label=name, color=color, lw=1.8)
        if scale == "score":
            cutoff = d.loc[d.decile.eq(10), "score_min"].iloc[0]
            ax.axvline(cutoff, ls="--", color="#334155", lw=1.2, label=f"Top-10% boundary: {cutoff:.3f}")
            ax.set(xlabel="Submitted score (standardized clipped logit)", title="Submitted scores: full range")
        else:
            ax.set(xlim=(0, 1), xlabel="Native TabPFN probability", title="Native probabilities: full range")
        ax.set(ylabel="Density within each outcome group")
        ax.legend(fontsize=8, frameon=False)
    save(fig, "01_prediction_distributions", "Prediction distributions in cases and controls")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].errorbar(d.decile, d.observed, yerr=[d.observed-d.observed_low, d.observed_high-d.observed],
                     fmt="o-", color=COLORS["submitted"], capsize=3, label="Observed case fraction (95% CI)")
    axes[0].plot(d.decile, d.predicted, "s--", color="#475569", ms=4, label="Mean native probability")
    axes[0].axhline(point.prevalence, ls=":", color="#AAA", label="Cohort case fraction")
    for r in d.itertuples():
        axes[0].annotate(f"{r.cases}/{r.n}", (r.decile, r.observed_high), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=7)
    axes[0].set(xticks=range(1, 11), xlabel="Risk decile (1 = lowest, 10 = highest)",
                ylabel="Case fraction", ylim=(0, 1.05), title="Risk stratification")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    for model in LABELS:
        group = deciles.loc[deciles.model.eq(model)]
        axes[1].errorbar(group.predicted, group.observed,
                         yerr=[group.observed-group.observed_low, group.observed_high-group.observed],
                         fmt="o-", color=COLORS[model], capsize=3, label=LABELS[model])
    axes[1].plot([0, 1], [0, 1], ":", color="#AAA")
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed case fraction",
                xlim=(0, 1), ylim=(0, 1.05), title="Calibration (descriptive; no refitting)")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "02_deciles_calibration", "Risk deciles and probability calibration")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, col, title, color in [(axes[0], "ll_gain_null", "Versus prevalence-only null", "#087F8C"),
                                  (axes[1], "ll_gain_covariates", "Versus covariates-only OOF", "#B97918")]:
        low_share = d.loc[d.decile.le(2), col].sum()/d[col].sum()
        high_share = d.loc[d.decile.eq(10), col].sum()/d[col].sum()
        ax.bar(d.decile, d[col], color=color)
        ax.axhline(0, color="#64748B", lw=.8)
        ax.set(xticks=range(1, 11), xlabel="Submitted-score risk decile", ylabel="Log-likelihood gain (nats)",
               title=f"{title}\nLowest 20%: {low_share:.1%}; highest 10%: {high_share:.1%}")
    save(fig, "03_likelihood_decomposition", "Where the whole-distribution R2 gains arise")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for model in LABELS:
        t = thresholds.loc[thresholds.model.eq(model)]
        axes[0].plot(t.top_fraction*100, t["or"], "o-", ms=3, color=COLORS[model], label=LABELS[model])
        axes[0].fill_between(t.top_fraction*100, t.or_low, t.or_high, color=COLORS[model], alpha=.10)
    axes[0].axvline(10, color="#64748B", ls="--", lw=1)
    axes[0].axhline(1, color="#AAA", ls=":")
    axes[0].set(xlabel="Highest-scoring patients selected (%)", ylabel="OR: selected versus remaining (log axis)",
                title="Threshold sensitivity (pointwise Wald 95% CI)", xlim=(0, 50), yscale="log")
    axes[0].legend(fontsize=8, frameon=False)
    x = np.arange(len(bands))
    axes[1].bar(x, bands.observed, color=COLORS["submitted"], alpha=.85)
    axes[1].errorbar(x, bands.observed, yerr=[bands.observed-bands.ci_low, bands.ci_high-bands.observed],
                     fmt="none", color="#334155", capsize=3)
    for i, r in enumerate(bands.itertuples()):
        axes[1].text(i, r.ci_high+.025, f"{r.cases}/{r.n}", ha="center", fontsize=8)
    axes[1].set(xticks=x, xticklabels=[f"{100*r.rank_from:g}–{100*r.rank_to:g}%" for r in bands.itertuples()],
                xlabel="Rank band counted from the highest score", ylabel="Case fraction",
                title="Non-overlapping high-to-low risk bands", ylim=(0, 1.13))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].tick_params(axis="x", labelsize=8)
    save(fig, "04_tail_sensitivity", "High-risk enrichment across selection thresholds")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        for model in LABELS:
            f = folds.loc[folds.model.eq(model)].sort_values("fold")
            ax.plot(f[metric], f.fold, "o-", color=COLORS[model], label=LABELS[model], ms=5)
        ci = intervals.loc[("submitted", metric)]
        ax.axvspan(ci.low, ci.high, color=COLORS["submitted"], alpha=.08)
        ax.axvline(point[metric], color=COLORS["submitted"], ls="--", lw=1)
        ax.set(yticks=range(5), yticklabels=[f"Fold {i+1}" for i in range(5)], ylim=(4.4, -.4), xlabel=title, title=title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(.5, .91), ncol=2)
    save(fig, "05_fold_consistency", "Performance across all five held-out folds")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, curve, metric, title in [(axes[0], "roc", "auc", "ROC (AUC)"),
                                    (axes[1], "pr", "auprc", "Precision-recall (average precision)")]:
        for model in LABELS:
            c = curves.loc[curves.model.eq(model) & curves.curve.eq(curve)]
            ax.plot(c.x, c.y, color=COLORS[model], label=f"{LABELS[model]} ({points.loc[model, metric]:.3f})")
        ax.set(title=title)
        ax.legend(fontsize=8, frameon=False)
    axes[0].plot([0, 1], [0, 1], ":", color="#AAA")
    axes[0].set(xlabel="False positive rate", ylabel="Sensitivity")
    axes[1].axhline(point.prevalence, ls=":", color="#AAA")
    axes[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.04))
    save(fig, "06_roc_pr", "Overall discrimination relative to covariates alone")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        for i, model in enumerate(LABELS):
            ci = intervals.loc[(model, metric)]
            value = points.loc[model, metric]
            ax.errorbar(value, i, xerr=[[value-ci.low], [ci.high-value]], fmt="o", color=COLORS[model], capsize=4)
        ax.set(yticks=[0, 1], yticklabels=["Submitted model", "Covariates only"] if ax is axes[0] else [],
               ylim=(1.5, -.5), title=title, xlabel="Point estimate and 95% bootstrap CI")
    save(fig, "07_pooled_uncertainty", f"Pooled OOF estimates ({record['bootstrap']['replicates']:,} paired bootstrap samples)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    member = membership.loc[["both", "model_only", "baseline_only", "neither"]]
    axes[0].bar(range(4), member.observed, color=["#64748B", COLORS["submitted"], COLORS["covariates"], "#CBD5E1"])
    axes[0].errorbar(range(4), member.observed, yerr=[member.observed-member.ci_low, member.ci_high-member.observed],
                     fmt="none", color="#334155", capsize=3)
    for i, r in enumerate(member.itertuples()):
        axes[0].text(i, r.ci_high+.03, f"{r.cases}/{r.n}", ha="center", fontsize=9)
    axes[0].set(xticks=range(4), xticklabels=["Both", "Model only", "Baseline only", "Neither"],
                xlabel="Membership in each model's highest-scoring 10%", ylabel="Case fraction",
                title="Who enters or leaves the highest-risk group?", ylim=(0, 1.12))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    shared = membership.loc["both"]
    axes[1].bar([0, 1], [shared.cases]*2, label="Cases selected by both", color="#64748B")
    exclusive = [membership.loc[k, "cases"] for k in ["model_only", "baseline_only"]]
    for i, (model, label) in enumerate([("submitted", "Cases selected by submitted model only"),
                                        ("covariates", "Cases selected by baseline only")]):
        axes[1].bar(i, exclusive[i], bottom=shared.cases, color=COLORS[model], label=label)
    for i, count in enumerate([point.top10_cases, base.top10_cases]):
        axes[1].text(i, count+15, f"{int(count)} cases / {int(point.top10_n)} selected", ha="center", fontsize=9)
    axes[1].set(xticks=[0, 1], xticklabels=["Submitted model", "Covariates only"], ylabel="Cases captured",
                title="Same selection capacity, different case yield", ylim=(0, point.top10_n*1.18))
    axes[1].legend(fontsize=8, frameon=False, loc="upper center")
    save(fig, "08_top10_membership", "Top-decile membership changes relative to the baseline")
    pdf.close()
    (root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    rows = []
    for model in LABELS:
        row = {"模型": LABELS[model]}
        for metric, title in METRICS:
            ci = intervals.loc[(model, metric)]
            row[title + " (95% CI)"] = f"{points.loc[model, metric]:.4f} ({ci.low:.4f}–{ci.high:.4f})"
        row["最高一成病例数"] = f"{int(points.loc[model, 'top10_cases'])}/{int(point.top10_n)}"
        rows.append(row)
    low, top = d.loc[d.decile.le(2)], d.loc[d.decile.eq(10)].iloc[0]
    mean_probability = (d.predicted*d.n).sum()/point.n
    low_null = low.ll_gain_null.sum()/d.ll_gain_null.sum()
    high_null = top.ll_gain_null/d.ll_gain_null.sum()
    low_cov = low.ll_gain_covariates.sum()/d.ll_gain_covariates.sum()
    high_cov = top.ll_gain_covariates/d.ll_gain_covariates.sum()
    entered, left = membership.loc["model_only"], membership.loc["baseline_only"]
    fold_main = folds.loc[folds.model.eq("submitted")]
    narrative = (
        f'CAD 的 OR90 与 R² 同时较高。协变量基线本身已有 R²={base.nagelkerke_r2:.4f}、AUC={base.auc:.4f}；最终配置进一步提高至 {point.nagelkerke_r2:.4f} 和 {point.auc:.4f}。较高的总体表现与相对基线的增量是两个不同问题。'
        if disease == "CAD" else
        f'T2D 同时获得整体概率预测和高风险富集的改善：OR90 从 {base.top10_vs_lower90_or:.3f} 提高至 {point.top10_vs_lower90_or:.3f}，R² 从 {base.nagelkerke_r2:.4f} 提高至 {point.nagelkerke_r2:.4f}。最高一成贡献相对协变量似然增益的 {high_cov:.1%}，是新增收益的重要来源。'
    )
    sections = [
        f'<p class="eyebrow">decipEHR · MAY SUBMISSION · {disease} POOLED OOF</p>',
        f'<h1>{disease}：最终提交配置的 OOF 预测分析</h1>',
        f'<p class="lead">{narrative}</p>',
        f'<div class="cards"><div><b>{int(point.n):,}</b><span>OOF 患者</span></div><div><b>{int(point.cases):,}</b><span>病例；样本病例比例 {point.prevalence:.2%}</span></div><div><b>5 折</b><span>每位患者恰有一次折外预测</span></div></div>',
        '<p>直接使用最终 audited 包中的 OOF score 和原始概率，逐人核对提交 TXT/ZIP、标签与共享 fold；未追加新的 logistic 评估层。最终配置包含 5 个 PRS 及 metadata，具体选择见文末。</p>',
        '<h2>1. Pooled OOF 指标与不确定性</h2>', table(pd.DataFrame(rows)),
        f'<p>OR90 按实际提交的标准化 clipped logit 排序；Nagelkerke pseudo-R²、Brier 和 log loss 使用保存的原始概率。主指标合并全部患者计算，不是五折指标的平均。95% CI 为 {record["bootstrap"]["replicates"]:,} 次成对患者 bootstrap 百分位区间；每次重排最高一成，模型不重训。</p>',
        figure(root, "07_pooled_uncertainty", "区间反映固定 OOF 预测下的患者抽样不确定性，不包括模型重训和参数选择不确定性。"),
        '<h2>2. 预测分布、风险分层与校准</h2>',
        figure(root, "01_prediction_distributions", "病例和对照各自归一化为密度；两个面板均显示完整范围。标准化 score 不是概率。"),
        f'<p>最低一成病例比例为 <b>{d.iloc[0].observed:.2%}</b>；最高一成为 <b>{int(top.cases)}/{int(top.n)}（{top.observed:.2%}）</b>，其余 90% 为 <b>{point.lower90_case_rate:.2%}</b>。最高一成覆盖全部病例的 <b>{point.top10_sensitivity:.2%}</b>。较高 OR 不代表覆盖了大多数病例。</p>',
        figure(root, "02_deciles_calibration", "十分位按提交分数定义，第 10 组人数为 ceil(0.1N)。数字是病例数/人数，误差线为 Wilson 95% CI；校准图未在 OOF 标签上重新拟合。病例比例仅指本研究样本，不是人群绝对发病风险。"),
        f'<p>最终模型的平均预测概率为 <b>{mean_probability:.2%}</b>，实际病例比例为 <b>{point.prevalence:.2%}</b>。Brier score 为 <b>{point.brier:.4f}</b>，协变量基线为 {base.brier:.4f}（越低越好）。这些数值描述现有概率输出，未进行再校准。</p>',
        '<h2>3. R² 的似然收益分解</h2>',
        '<div class="formula">OR90 = (最高 10% 病例数 / 对照数) ÷ (其余 90% 病例数 / 对照数)<br>G = LL(model) − LL(prevalence-only null)<br>R²<sub>N</sub> = [1 − exp(−2G/n)] / [1 − exp(2LL<sub>null</sub>/n)]</div>',
        f'<p>相对常数病例比例 null，总似然增益为 <b>{d.ll_gain_null.sum():.2f} nats</b>；最低 20% 贡献 <b>{low_null:.1%}</b>，最高 10% 贡献 <b>{high_null:.1%}</b>。R² 同时反映低风险识别与高风险富集；它不是线性回归的方差解释比例。</p>',
        figure(root, "03_likelihood_decomposition", "分解的是可加的 log-likelihood 增益，不是将非线性 R² 分摊。左右面板分别比较常数基线和协变量模型，不应混用。"),
        f'<p>相对协变量模型的新增似然收益为 <b>{d.ll_gain_covariates.sum():.2f} nats</b>；最低 20% 贡献 <b>{low_cov:.1%}</b>，最高 10% 贡献 <b>{high_cov:.1%}</b>。对 CAD，应同时看到强协变量基线；对 T2D，高风险端的新增收益更突出。这些是当前样本的描述性比较。</p>',
        '<h2>4. 高风险阈值与入选人群变化</h2>',
        figure(root, "04_tail_sensitivity", "左图 OR 使用对数纵轴，阴影为固定分组 2×2 表的点态 Wald CI；右图为不重叠排名区间及 Wilson CI。仅作诊断，不据此重新选择比赛阈值。"),
        '<div class="explorer"><label>模型 <select id="model"><option value="submitted">最终提交模型</option><option value="covariates">协变量基线</option></select></label><label>最高风险比例 <select id="fraction"></select></label><div id="result"></div></div>',
        f'<p>两模型均选取 {int(point.top10_n)} 人。最终模型新纳入的 <b>{int(entered.n)} 人中有 {int(entered.cases)} 个病例（{entered.observed:.1%}）</b>；被替换的 {int(left.n)} 人中有 {int(left.cases)} 个病例（{left.observed:.1%}）。因此最高一成净增加 <b>{int(entered.cases-left.cases)} 个病例</b>，这是 OR90 改善的直接人数解释。</p>',
        figure(root, "08_top10_membership", "Both/Model only/Baseline only/Neither 是两种最高一成集合的交集、差集和补集；这是既有预测的描述，不是用标签优化分组。左图 Wilson CI，右图病例数精确加总。"),
        '<h2>5. 各折一致性与总体判别</h2>',
        f'<p>五折 OR90 范围为 <b>{fold_main.top10_vs_lower90_or.min():.2f}–{fold_main.top10_vs_lower90_or.max():.2f}</b>，R² 为 <b>{fold_main.nagelkerke_r2.min():.4f}–{fold_main.nagelkerke_r2.max():.4f}</b>。各折训练人群有重叠，不能将五折视作五个独立实验。</p>',
        figure(root, "05_fold_consistency", "每折 OR90 在该折内部取最高一成。竖虚线和淡色带是 pooled OOF 的点估计与 bootstrap CI，不是各折置信区间。"),
        '<details><summary>各 fold 的分布与 pooled 最高一成人数</summary>' + table(read("fold_prediction_distributions")) + '</details>',
        figure(root, "06_roc_pr", "曲线按实际提交 score 计算；AUC/AP 已核对与原始概率一致。PR 摘要使用 average precision。"),
        '<p>下表为最终模型减去协变量基线的成对差值。AUC、R²、OR90 越高越好；Brier 与 log loss 越低越好。</p>',
        table(deltas.reset_index()),
        '<h2>6. 评估边界、来源与复现</h2>',
        '<p class="note">每位患者未参与对应折的 TabPFN 拟合；PRS 参数和组合曾利用 validation 信息选择，因此不是全流程嵌套的无偏验证，也不是全 train-val 拟合后的单一模型在 final holdout 上的成绩。基线是仅协变量的 logistic 模型；最终模型包含 PRS、metadata 与 TabPFN 的非线性建模，差值不能全部解释为纯遗传贡献。</p>',
        '<details><summary>5 个 PRS 的选择来源</summary>' + table(pd.DataFrame(record["prs_selection"])) + '</details>',
        '<details><summary>Clipping 敏感性核对</summary>' + table(read("score_transform_audit")) + '</details>',
        '<p>报告和图表不含患者标识；患者级核对表位于本地受保护目录。可使用同目录的聚合 CSV、SVG、PNG、单页 PDF 或合并图集。</p>',
        f'<pre>UV_CACHE_DIR=outputs/uv-cache/prediction_analysis OPENBLAS_NUM_THREADS=1 uv run --offline --no-sync python scripts/analyze_may_final_disease_oof.py --disease {record["disease"]} --eligibility-config configs/disease_eligibility.yaml\n\nUV_PROJECT_ENVIRONMENT=outputs/prediction_analysis_env UV_CACHE_DIR=outputs/uv-cache/prediction_analysis MPLCONFIGDIR=outputs/mpl-cache uv run --offline --no-sync python scripts/plot_may_final_disease_oof.py</pre>',
        '<details><summary>输入来源与审计记录</summary><pre>' + html.escape(json.dumps(record, ensure_ascii=False, indent=2)) + '</pre></details>',
    ]
    script = """const data=DATA;const model=document.getElementById('model'),fraction=document.getElementById('fraction');[...new Set(data.map(x=>x.top_fraction))].forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=(x*100)+'%';o.selected=x===.1;fraction.appendChild(o)});function update(){const r=data.find(x=>x.model===model.value&&Math.abs(x.top_fraction-Number(fraction.value))<1e-8);document.getElementById('result').textContent='选中 '+r.n_top+' 人，病例 '+r.cases_top+' 人；病例比例 '+(100*r.case_rate).toFixed(1)+'%；覆盖全部病例 '+(100*r.sensitivity).toFixed(1)+'%；OR '+r.or.toFixed(2)+'。'}model.onchange=update;fraction.onchange=update;update();""".replace("DATA", thresholds.to_json(orient="records"))
    (root / "OOF_ANALYSIS.html").write_text(document(f"{disease} final submission OOF analysis", sections, script), encoding="utf-8")
    return {"disease": disease, "n": int(point.n), "cases": int(point.cases),
            "or90": point.top10_vs_lower90_or, "r2": point.nagelkerke_r2, "auc": point.auc,
            "delta_or90": deltas.loc["top10_vs_lower90_or", "delta"],
            "delta_r2": deltas.loc["nagelkerke_r2", "delta"],
            "delta_auc": deltas.loc["auc", "delta"], "top10_case_rate": point.top10_case_rate,
            "top10_sensitivity": point.top10_sensitivity, "extra_top10_cases": int(entered.cases-left.cases),
            "low20_ll_gain_null_share": low_null, "top10_ll_gain_null_share": high_null,
            "low20_ll_gain_covariates_share": low_cov, "top10_ll_gain_covariates_share": high_cov}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="outputs/may_final_cad_t2d_oof_prediction_analysis")
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.results_root)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .16, "text.color": "#183445",
                         "axes.labelcolor": "#334155", "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 180})
    with PdfPages(root / "CAD_T2D_OOF_FIGURES.pdf") as combined:
        rows = [render(root / disease, combined) for disease in ["cad", "t2d"]]
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "disease_summary.csv", index=False)
    sections = ['<p class="eyebrow">decipEHR · MAY SUBMISSION · CAD + T2D</p>',
                '<h1>CAD 与 T2D：最终提交配置的 OOF 分析</h1>',
                '<p class="lead">与 BC 分析采用相同口径：冻结的最终提交输出、共享五折、匹配协变量基线、原始概率计算 R²、2,000 次成对 bootstrap。</p>',
                '<p><a href="cad/OOF_ANALYSIS.html">打开 CAD 完整报告</a> · <a href="t2d/OOF_ANALYSIS.html">打开 T2D 完整报告</a> · <a href="CAD_T2D_OOF_FIGURES.pdf">16 页合并图集</a></p>',
                table(summary[["disease", "n", "cases", "or90", "r2", "auc", "delta_or90", "delta_r2", "delta_auc"]]),
                '<p>CAD 的整体表现较高，同时协变量基线已经很强；T2D 的相对基线增量较大，高风险一成是新增收益的重要来源。疾病病例比例不同，这里的跨疾病数值差异不能直接解释为遗传贡献大小。</p>']
    for r in rows:
        sections += [f'<h2>{r["disease"]} 的收益分布</h2>',
                     f'<p>最高一成病例比例 {r["top10_case_rate"]:.1%}，覆盖病例 {r["top10_sensitivity"]:.1%}；同样选取 867 人，相较协变量基线多捕获 {r["extra_top10_cases"]} 个病例。最低 20% / 最高 10% 分别贡献相对协变量似然收益的 {r["low20_ll_gain_covariates_share"]:.1%} / {r["top10_ll_gain_covariates_share"]:.1%}。</p>',
                     figure(root / r["disease"].lower(), "03_likelihood_decomposition", "左：相对常数病例比例。右：相对协变量模型。贡献比例基于可加似然增益。")]
    sections += ['<p class="note">OOF 排除了每位患者自身的 TabPFN 拟合，但 PRS 参数/组合选择并非完全嵌套。这是已选配置的折外表现，不是最终独立 holdout 成绩；bootstrap 仅反映固定预测下的患者抽样不确定性。</p>',
                 '<p>每种疾病均提供 8 张图、可交互 HTML、聚合 CSV、PNG、可编辑文本 SVG 和 PDF。原有 BC 报告保持原样。</p>']
    (root / "OOF_ANALYSIS.html").write_text(document("CAD and T2D OOF analysis", sections), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved CAD/T2D reports and 16 figures to {root.resolve()}")


if __name__ == "__main__":
    main()
