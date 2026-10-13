#!/usr/bin/env python3
"""Render aggregate validation diagnostics as vector figures and a standalone report."""

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

COLORS = {"covariates": "#B97918", "direct": "#7857A5", "adjusted": "#087F8C"}
LABELS = {"covariates": "Covariates only", "direct": "Direct TabPFN score",
          "adjusted": "Score + covariates wrapper"}
MODELS = ["covariates", "direct", "adjusted"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="outputs/may_final_validation_prediction_analysis")
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.results_dir)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    def read(name):
        return pd.read_csv(root / f"{name}.csv")
    metrics = read("metrics")
    points = metrics.loc[metrics.disease.eq("bc")].set_index("model")
    deciles = read("bc_risk_deciles")
    thresholds = read("bc_threshold_sensitivity")
    curves = read("bc_roc_pr_curves")
    histograms = read("bc_probability_histograms")
    intervals = read("bc_metric_intervals")
    deltas = read("bc_paired_deltas")
    moves = read("bc_top10_membership_changes")
    distributions = read("bc_distribution_summary")
    likelihood_change = read("bc_wrapper_likelihood_change")
    age = read("bc_age_quintiles")
    region_age = read("bc_risk_region_age")
    record = json.loads((root / "analysis_record.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12,
                         "axes.labelcolor": "#334155", "text.color": "#182B3A",
                         "axes.grid": True, "grid.alpha": .16,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "savefig.dpi": 180})
    pdf = PdfPages(root / "VALIDATION_ANALYSIS_FIGURES.pdf")
    rendered = []

    def save(fig, name, title):
        fig.suptitle(title, fontsize=15, fontweight="bold")
        footer = ("Original validation | CAD / T2D: n = 2,852; BC: n = 1,628 eligible women"
                  if name == "08_disease_context" else
                  "Original validation | BC: 1,628 eligible women, 320 cases | May final configuration")
        fig.text(.5, .005, footer,
                 ha="center", fontsize=8, color="#64748B")
        fig.tight_layout(rect=[0, .035, 1, .95])
        for suffix in ["png", "svg", "pdf"]:
            fig.savefig(figures / f"{name}.{suffix}", bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        rendered.append((name, title))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, model in zip(axes, ["direct", "adjusted"], strict=True):
        inset = ax.inset_axes([.77, .62, .21, .32])
        for label, color in [("Control", "#4083A5"), ("Case", "#C15464")]:
            h = histograms.loc[histograms.model.eq(model) & histograms.label.eq(label)]
            edges = np.r_[h.left.to_numpy(), h.right.iloc[-1]]
            ax.stairs(h.density, edges, color=color, linewidth=1.8, label=label)
            inset.stairs(h.density, edges, color=color, linewidth=1)
        cutoff = deciles.loc[deciles.model.eq(model) & deciles.decile.eq(10), "score_min"].iloc[0]
        ax.axvline(cutoff, color="#334155", ls="--", lw=1.2, label=f"Top-10% boundary ({cutoff:.3f})")
        above = distributions.loc[distributions.model.eq(model), "above_0p4"].sum()
        ax.set(xlim=(0, .4), xlabel="Predicted probability (main panel: 0 to 0.4)",
               ylabel="Density within each outcome group", title=LABELS[model])
        inset.set(xlim=(0, 1), xticks=[0, .5, 1], yticks=[])
        inset.set_title(f"Full range\n{above} above 0.4", fontsize=7)
        inset.tick_params(labelsize=7)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, "01_prediction_distributions", "BC probability distributions: substantial case-control overlap")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, ["direct", "adjusted"], strict=True):
        d = deciles.loc[deciles.model.eq(model)]
        ax.errorbar(d.decile, d.observed, yerr=[d.observed - d.observed_low, d.observed_high - d.observed],
                    fmt="o-", color=COLORS[model], capsize=3, label="Observed (Wilson 95% CI)")
        ax.plot(d.decile, d.predicted, "s--", color="#444E5E", ms=4, label="Mean predicted probability")
        ax.axhline(points.loc[model, "prevalence"], color="#A0A8B3", ls=":", label="Cohort case fraction")
        for row in d.itertuples():
            ax.annotate(f"{row.cases}/{row.n}", (row.decile, row.observed_high),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        ax.set(xticks=range(1, 11), xlabel="Risk decile (1 = lowest; 10 = highest)",
               ylabel="Case fraction", title=LABELS[model], ylim=(0, .52))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, "02_risk_deciles", "Low-risk exclusion is stronger than extreme-risk enrichment")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, model in zip(axes, ["direct", "adjusted"], strict=True):
        d = deciles.loc[deciles.model.eq(model)]
        ax.bar(d.decile - .18, d.ll_gain_null, width=.36, color=COLORS[model], label="Versus prevalence-only null")
        ax.bar(d.decile + .18, d.ll_gain_covariates, width=.36, color="#C9A55C", label="Versus covariates only")
        share = d.loc[d.decile.le(3), "ll_gain_null"].sum() / d.ll_gain_null.sum()
        ax.axhline(0, color="#4B5563", lw=.8)
        ax.set(xticks=range(1, 11), xlabel="Risk decile", ylabel="Total log-likelihood gain (nats)",
               title=f"{LABELS[model]}\nLowest 30%: {share:.1%} of net gain vs null")
        ax.legend(fontsize=8, frameon=False)
    save(fig, "03_likelihood_decomposition", "R2 uses gains across all patients, including the low-risk tail")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in MODELS:
        t = thresholds.loc[thresholds.model.eq(model)].sort_values("top_fraction")
        axes[0].plot(t.top_fraction * 100, t["or"], "o-", ms=3, color=COLORS[model], label=LABELS[model])
        if model != "covariates":
            axes[0].fill_between(t.top_fraction * 100, t.or_low, t.or_high, color=COLORS[model], alpha=.09)
        axes[1].plot(t.top_fraction * 100, t.sensitivity * 100, "o-", ms=3,
                     color=COLORS[model], label=LABELS[model])
    for ax in axes:
        ax.axvline(10, color="#64748B", ls="--", lw=1)
        ax.set(xlabel="Patients selected from highest risk (%)", xlim=(0, 50))
        ax.legend(fontsize=8, frameon=False)
    axes[0].axhline(1, color="#AAA", ls=":")
    axes[0].set(ylabel="Odds ratio: selected vs remaining", title="Threshold sensitivity (pointwise Wald 95% CI)")
    axes[1].plot([0, 50], [0, 50], ":", color="#AAA")
    axes[1].set(ylabel="Cases captured (%)", title="Cumulative case capture")
    save(fig, "04_tail_thresholds", "The top-10% criterion discards improvements elsewhere in the ranking")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    contrasts = [("covariates", "direct", "Direct minus covariates"),
                 ("covariates", "adjusted", "Wrapper minus covariates"),
                 ("direct", "adjusted", "Wrapper minus direct")]
    for ax, metric, title in zip(axes, ["top10_vs_lower90_or", "nagelkerke_r2", "auc"],
                                 ["Change in OR90", "Change in Nagelkerke R2", "Change in AUC"], strict=True):
        for i, (before, after, _) in enumerate(contrasts):
            row = deltas.loc[deltas.before.eq(before) & deltas.after.eq(after) & deltas.metric.eq(metric)].iloc[0]
            ax.errorbar(row.delta, i, xerr=[[row.delta - row.low], [row.high - row.delta]],
                        fmt="o", capsize=4, color=COLORS[after])
        ax.axvline(0, color="#64748B", ls="--", lw=1)
        ax.set(yticks=range(3), yticklabels=[x[2] for x in contrasts] if ax is axes[0] else [],
               ylim=(2.6, -.6), title=title, xlabel="Difference with paired 95% CI")
    save(fig, "05_paired_uncertainty", "R2 improvement is clearer than OR90 improvement on this validation set")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    movement_rows, movement_labels = [], []
    for raw, adjusted, label in [(True, True, "Stays in top 10%"), (True, False, "Leaves top 10%"),
                                 (False, True, "Enters top 10%")]:
        r = moves.loc[moves.direct_top10.eq(raw) & moves.adjusted_top10.eq(adjusted)].iloc[0]
        movement_rows.append(r)
        movement_labels.append(label)
    for i, r in enumerate(movement_rows):
        axes[0].bar(i, r.cases, color="#C15464", label="Cases" if i == 0 else None)
        axes[0].bar(i, r.n - r.cases, bottom=r.cases, color="#8CB3C8", label="Controls" if i == 0 else None)
        axes[0].text(i, r.n + 2, f"{int(r.cases)} cases / {int(r.n)}", ha="center", fontsize=9)
    axes[0].set(xticks=range(3), xticklabels=movement_labels, ylabel="Patients",
                title="Top-decile membership after wrapper", ylim=(0, max(r.n for r in movement_rows) * 1.2))
    axes[0].legend(frameon=False)
    axes[1].bar(likelihood_change.decile, likelihood_change.ll_adjusted_minus_direct,
                color=np.where(likelihood_change.ll_adjusted_minus_direct.ge(0), "#087F8C", "#C15464"))
    axes[1].axhline(0, color="#64748B", lw=.8)
    axes[1].set(xticks=range(1, 11), xlabel="Fixed wrapper-model risk decile",
                ylabel="Log-likelihood(wrapper) - log-likelihood(direct)",
                title="Whole-cohort fit improves despite top-tail losses")
    save(fig, "06_wrapper_effect", "The evaluation wrapper changes both probability scale and ranking")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for model in MODELS:
        for ax, curve, metric in [(axes[0, 0], "roc", "auc"), (axes[0, 1], "pr", "auprc")]:
            c = curves.loc[curves.model.eq(model) & curves.curve.eq(curve)]
            ax.plot(c.x, c.y, color=COLORS[model], label=f"{LABELS[model]} ({points.loc[model, metric]:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], ":", color="#AAA")
    axes[0, 0].set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC curves (AUC)")
    axes[0, 1].axhline(points.loc["direct", "prevalence"], color="#AAA", ls=":")
    axes[0, 1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (average precision)")
    for ax in axes[0]:
        ax.legend(fontsize=8, frameon=False)
    for ax, model in zip(axes[1], ["direct", "adjusted"], strict=True):
        d = deciles.loc[deciles.model.eq(model)]
        ax.errorbar(d.predicted, d.observed, yerr=[d.observed - d.observed_low, d.observed_high - d.observed],
                    fmt="o-", color=COLORS[model], capsize=3)
        ax.plot([0, .5], [0, .5], ":", color="#AAA")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed case fraction",
               title=f"Calibration: {LABELS[model]}", xlim=(0, .4), ylim=(0, .5))
    save(fig, "07_discrimination_calibration", "Whole-cohort discrimination and probability calibration")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, metric, title in zip(axes, ["top10_vs_lower90_or", "nagelkerke_r2", "auc"],
                                 ["OR90", "Nagelkerke R2", "AUC"], strict=True):
        for i, model in enumerate(MODELS):
            rows = metrics.loc[metrics.model.eq(model)].set_index("disease").loc[["cad", "bc", "T2D"]]
            ax.bar(np.arange(3) + (i - 1) * .24, rows[metric], width=.23,
                   color=COLORS[model], label=LABELS[model])
        ax.set(xticks=range(3), xticklabels=["CAD", "BC", "T2D"], title=title)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_ylim(0, metrics.top10_vs_lower90_or.max() * 1.36)
    save(fig, "08_disease_context", "BC gains are relative to its baseline; absolute R2 remains modest")
    pdf.close()

    def table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")

    def figure(name, caption):
        svg = (figures / f"{name}.svg").read_text()
        svg = svg[svg.index("<svg"):]
        for identifier in set(re.findall(r'id="([^"]+)"', svg)):
            replacement = f"fig_{name}_{identifier}"
            svg = svg.replace(f'id="{identifier}"', f'id="{replacement}"')
            svg = svg.replace(f'href="#{identifier}"', f'href="#{replacement}"')
            svg = svg.replace(f'url(#{identifier})', f'url(#{replacement})')
        return f'<figure>{svg}<figcaption>{caption}</figcaption></figure>'

    ci = intervals.set_index(["model", "metric"])
    rows = []
    for model in MODELS:
        row = {"评估口径": LABELS[model]}
        for metric, label, digits in [("top10_vs_lower90_or", "OR90 (95% CI)", 2),
                                     ("nagelkerke_r2", "Nagelkerke R² (95% CI)", 4),
                                     ("auc", "AUC (95% CI)", 3)]:
            c = ci.loc[(model, metric)]
            row[label] = f"{points.loc[model, metric]:.{digits}f} ({c.low:.{digits}f}–{c.high:.{digits}f})"
        row["最高一成病例数"] = f"{int(points.loc[model, 'top10_cases'])}/163"
        rows.append(row)
    d = deciles.loc[deciles.model.eq("adjusted")]
    low_share = d.loc[d.decile.le(3), "ll_gain_null"].sum() / d.ll_gain_null.sum()
    high_share = d.loc[d.decile.eq(10), "ll_gain_null"].sum() / d.ll_gain_null.sum()
    wrapper_gain = likelihood_change.ll_adjusted_minus_direct.sum()
    bc_metrics_html = table(pd.DataFrame(rows))
    sections = [
        '<p class="eyebrow">decipEHR · MAY FINAL MODEL · ORIGINAL VALIDATION</p>',
        '<h1>为什么 BC 的 OR90 提升有限，R² 却改善明显？</h1>',
        '<p class="lead">主要原因是低风险端的识别收益、整体概率拟合的改善，与最高风险 10% 的病例富集并不完全一致。既有 logistic 评估步骤还会重新排序，使二者出现相反方向的点估计变化。</p>',
        '<div class="cards"><div><b>1,628</b><span>符合条件的女性 validation 患者</span></div><div><b>320</b><span>病例；样本病例比例 19.66%</span></div><div><b>5 月最终配置</b><span>TabPFN-3 · BC 6 PRS + metadata</span></div></div>',
        '<h2>1. 先固定模型、队列和评分口径</h2>',
        '<p>用户确认的提交包为 <code>Go Gene!_audited_pipeline_with_trainval_5cv</code>。本分析使用其相同 PRS 配置在原始 validation 上的保存分数：BC 的模型 context 为原始 training 中的 3,308 名女性，validation 为另 1,628 名女性；患者无重叠。最终 PRS selection manifest 已逐字段核对一致。最终提交时使用全部 train-val 拟合的模型已经见过 validation 标签，因此本报告使用其 validation 阶段版本。</p>',
        '<p>BC 保留 P+T、LDpred2、lassosum2、PRS-CS、annotated MegaPRS、SBayesRC 7M，并包含 metadata。所有比较使用同一批 validation 患者；没有重新选择模型、阈值或训练 TabPFN。评估 wrapper 仅在原始 training 上按既有实现重建，保存的历史指标在 1e−10 精度内复现。</p>',
        '<ul><li><b>Direct：</b>保存的模型分数；排序对应最终提交方式。用于概率指标的数值由 training mean/SD 反变换 logit 得到，保留原有 0.001–0.999 clipping。</li><li><b>Wrapper：</b>原项目标准验证口径：training 上拟合 <code>logistic(score + sex + age_sd + PC1_sd–PC10_sd)</code>，再预测 validation。</li><li><b>Covariates only：</b>相同 training/validation 划分，只使用协变量的 logistic 基线。</li></ul>',
        bc_metrics_html,
        '<p class="note">95% CI：2,000 次成对、非分层患者 bootstrap，每次重新计算最高一成；固定模型分数，不重新拟合。此处 Direct R² 是恢复概率上的本地诊断值；主办方文档未明确标准化 logit 到 R² 的完整实现，因此不将其声称为最终榜单 R² 的复现。</p>',
        '<h2>2. OR90 与 R² 衡量不同的东西</h2>',
        '<div class="formula">OR90 = (最高 10% 病例数 / 对照数) ÷ (其余 90% 病例数 / 对照数)<br>G = Σᵢ [log P(yᵢ | pᵢ) − log P(yᵢ | π)]<br>R²<sub>N</sub> = [1 − exp(−2G/n)] / [1 − exp(2LL<sub>null</sub>/n)]</div>',
        '<p>OR90 只由最高一成的成员和标签决定。R² 使用所有患者的概率与真实标签之间的 log-likelihood；把大量无病者判为低风险可以明显改善它。这里的 R² 是 Nagelkerke pseudo-R²，不能直接解读为普通线性回归中“解释了约 10% 的结局方差”。其定义来自 <a href="https://doi.org/10.1093/biomet/78.3.691">Nagelkerke, 1991</a>。</p>',
        '<p>正的线性标准化及严格单调的 sigmoid 本身不改变 OR90（本数据已核对最高一成成员相同）。因此不能把 OR90 不高单纯归因于概率范围较窄。加入协变量的 wrapper 会改变患者相对次序，才可能改变 OR90。</p>',
        '<h2>3. 分布与十分位：低风险端清楚，高风险端分离有限</h2>',
        figure("01_prediction_distributions", "病例/对照密度分别归一化，避免人数不平衡造成视觉误导。主图放大至 0–0.4，插图保留 0–1 全范围；虚线为各自最高一成边界。"),
        '<p>病例与对照在中高预测区间仍明显重叠。Wrapper 下最低一成只有 <b>4/162（2.47%）</b>为病例，而最高两成中的第 9 与第 10 十分位均为 <b>52/163（31.90%）</b>。这说明该次 validation 上，最顶端的一成并未比紧邻的一成进一步富集。</p>',
        figure("02_risk_deciles", "各模型按自己的分数分组。数字为病例数/组内人数，误差线为 Wilson 95% CI；最高十分位固定 ceil(0.1n)=163 人。组间 CI 有重叠，平台形态属于描述性结果。"),
        '<p>Wrapper 的最高一成病例 52 人，占全部 320 个病例的 <b>16.25%</b>；其余 90% 的病例比例仍为 <b>268/1465=18.29%</b>。因此 OR90 为 2.09。Direct 则是最高一成 58 人、捕获病例 18.13%，OR90 为 2.54；两种口径都有一定富集能力。</p>',
        '<h2>4. 整体拟合收益主要来自哪里？</h2>',
        f'<p>Wrapper 相对 prevalence-only null 的总 log-likelihood 增益为 <b>{d.ll_gain_null.sum():.2f} nats</b>；其中最低风险 30% 贡献 <b>{low_share:.1%}</b>，最高风险 10% 贡献 <b>{high_share:.1%}</b>。这是“低风险者识别得好，但 OR90 不特别高”的直接证据。</p>',
        figure("03_likelihood_decomposition", "分解的是可加的 log-likelihood 增益，不是直接把非线性的 R² 拆成可相加百分比。蓝/紫柱相对样本病例比例 null；金色柱相对同一患者的 covariates-only 预测。负值表示该组降低整体拟合。"),
        '<p>相对协变量基线，Wrapper 的 R² 从 0.0371 增至 0.1009，AUC 从 0.614 增至 0.665。该改进说明新增模型分数有预测信息；由于模型分数内部已包含 metadata，不能把全部增益归为纯遗传效应。</p>',
        '<h2>5. Wrapper 为什么能让 R² 上升、OR90 下降？</h2>',
        f'<p>从 Direct 到 Wrapper，<b>43 人移出</b>最高一成，其中 <b>17 人为病例</b>；另有 <b>43 人移入</b>，其中 <b>11 人为病例</b>。最高一成净少 6 个病例，使 OR90 从 2.54 降至 2.09。同时全体患者 log-likelihood 增加 <b>{wrapper_gain:.2f} nats</b>，R² 从 0.0875 上升至 0.1009。</p>',
        figure("06_wrapper_effect", "这是同一批 validation 患者的成对分解。右图沿用 Wrapper 的十分位；改善整体概率拟合并不保证边界处换入的病例更多。"),
        '<p>因此本次差异同时含有 <b>全体概率拟合改善</b>与<b>最高一成排序变化</b>。它不是数值矛盾，也不应被表述成“R² 高就一定能挑出最危险的 10%”。</p>',
        '<h2>6. 不确定性：整体收益更稳定，尾部增益较不确定</h2>',
        figure("05_paired_uncertainty", "成对 bootstrap 的 percentile 95% CI；竖虚线为零差异。区间只反映固定预测下的患者抽样不确定性，未纳入模型选择与训练不确定性。"),
        '<p>Wrapper 相对 covariates-only：ΔR²=<b>+0.0638（95% CI 0.0435–0.0834）</b>，但 ΔOR90=<b>+0.664（−0.171–1.578）</b>。Direct 到 Wrapper 的 ΔR²=<b>+0.0134（0.0034–0.0240）</b>，而 ΔOR90=<b>−0.444（−1.147–0.278）</b>。R² 改善的区间较明确；OR90 的改善或下降仍有较大抽样不确定性，不能把点估计下降断言为必然损害。</p>',
        '<h2>7. 改变观察阈值后会怎样？</h2>',
        figure("04_tail_thresholds", "不同阈值作为解释性敏感性分析；固定比赛指标仍为最高 10%。OR 曲线区间为固定分组 2×2 表的 Wald 95% CI，未纳入阈值估计不确定性；主要 OR90 区间使用重新排序的患者 bootstrap。"),
        '<p>下面可以查看现有模型在不同选择比例下的实际病例数、OR 与覆盖率。查看阈值不会改变模型或推荐新提交阈值。</p>',
        '<div class="explorer"><label>评分口径 <select id="profile"><option value="direct">Direct TabPFN score</option><option value="adjusted" selected>Score + covariates wrapper</option><option value="covariates">Covariates only</option></select></label><label>最高风险比例 <select id="fraction"></select></label><div id="tailResult"></div></div>',
        '<h2>8. 补充诊断与解释边界</h2>',
        figure("07_discrimination_calibration", "ROC、PR 与分位数校准图共同说明排序与概率拟合表现。校准曲线仅为描述性检查，没有使用 validation 标签重校准预测。"),
        '<p>低风险组也更年轻：Wrapper 最低风险 20% 的平均 age_sd 约 −1.58，最高一成约 +0.56。年龄及其与 PRS 的组合可能参与了低风险端识别，但这是描述性关联，不能取代年龄消融或证明生物学机制；age_sd 不转换为未提供的实际年龄。</p>',
        '<details><summary>按年龄五分位的聚合结果</summary>' + table(age) + table(region_age) + '</details>',
        figure("08_disease_context", "三种疾病均使用相应的 5 月最终 PRS 配置和原始 validation 划分。这里展示点估计用于背景比较；病例比例和疾病定义不同，不宜将跨疾病 R² 当作可直接等量比较的遗传解释率。"),
        '<p><b>“R² 不错”应理解为相对 BC 的基线有明确改善。</b>约 0.10 的 pseudo-R² 与约 0.66 的 AUC 仍提示判别能力有限；BC 在此数据中的 R² 和 OR90 都低于 CAD/T2D，并不存在 R² 绝对很高而 OR90 异常低的证据。</p>',
        '<p class="note">本报告是对已选配置的回顾性 validation 解释；PRS 参数/模型选择曾使用 validation 信息，因此并非全流程独立测试。实际采用原始 training context 的保存分数，未使用 train+validation 混合后再切出的 “validation_only” 历史汇总，也未纳入 holdout 或 train-val OOF 性能。样本病例比例不等于一般人群发病风险。</p>',
        '<h2>复现与审计</h2>',
        '<p>所有患者级数据只在本地受保护 CSV 中保留；本 HTML、图与 PDF 仅含汇总信息。未改动原始预测/提交文件，未提交或推送任何数据。</p>',
        '<pre>UV_CACHE_DIR=outputs/uv-cache/prediction_analysis OPENBLAS_NUM_THREADS=1 uv run --offline --no-sync python scripts/analyze_may_final_validation.py --eligibility-config configs/disease_eligibility.yaml\n\n# Plotting requires matplotlib; this session used a uv environment with the existing system matplotlib.\nUV_PROJECT_ENVIRONMENT=outputs/prediction_analysis_env UV_CACHE_DIR=outputs/uv-cache/prediction_analysis MPLCONFIGDIR=outputs/mpl-cache uv run --offline --no-sync python scripts/plot_may_final_validation.py</pre>',
        '<p>文件：<code>metrics.csv</code>、<code>bc_risk_deciles.csv</code>、<code>bc_metric_intervals.csv</code>、<code>bc_paired_deltas.csv</code>、<code>bc_top10_membership_changes.csv</code>、<code>analysis_record.json</code>。全部图另存 PNG、可编辑文本 SVG 与 PDF；合并 PDF 为 <code>VALIDATION_ANALYSIS_FIGURES.pdf</code>。</p>',
        '<details><summary>输入校验、软件版本与 SHA-256</summary><pre>' + html.escape(json.dumps(record, ensure_ascii=False, indent=2)) + '</pre></details>',
        '<p>指标定义依据：本项目 <code>src/decipehr/metrics.py</code> 与官方本地 <code>docs/Final phase.docx</code>；Nagelkerke R² 原始方法：<a href="https://doi.org/10.1093/biomet/78.3.691">Biometrika 78(3), 691–692 (1991)</a>。</p>',
    ]
    css = """body{font:16px/1.75 -apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans CJK SC',sans-serif;color:#253445;background:#f4f7fa;margin:0}main{max-width:1120px;margin:32px auto;padding:42px;background:white;border-radius:16px}h1{font-size:34px;line-height:1.3;color:#142d3c}h2{font-size:23px;margin-top:48px;border-bottom:2px solid #deebed;padding-bottom:10px}.eyebrow{font-size:12px;letter-spacing:.12em;color:#087f8c;font-weight:700}.lead{font-size:19px;color:#476071}.cards{display:flex;gap:16px;margin:28px 0}.cards div{flex:1;background:#eef6f7;padding:18px;border-radius:9px}.cards b{font-size:27px;display:block}.cards span{font-size:13px}.note{background:#fff8e8;border-left:4px solid #c59e48;padding:16px;font-size:14px}.formula{background:#eef3f7;padding:18px;font-family:monospace}figure{margin:28px 0}figure svg{width:100%;height:auto}figcaption{font-size:13px;color:#627080;margin-top:8px}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}td,th{border-bottom:1px solid #dce3e9;padding:9px;text-align:left}th{background:#eef3f7}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#f1f5f8;padding:16px}code{font-size:.87em;background:#eff3f6}a{color:#087f8c}.explorer{background:#edf7f8;padding:20px;border-radius:8px}.explorer label{margin-right:24px}select{padding:7px;max-width:100%}#tailResult{margin-top:16px;font-size:18px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}@media(max-width:700px){main{padding:18px;margin:0}.cards{display:block}.cards div{margin-bottom:8px}h1{font-size:28px}table{font-size:11px}}"""
    jsdata = thresholds.to_json(orient="records")
    script = """const data=DATA;const profile=document.getElementById('profile');const fraction=document.getElementById('fraction');[...new Set(data.map(x=>x.top_fraction))].forEach(x=>{let o=document.createElement('option');o.value=x;o.textContent=(100*x)+'%';o.selected=x===.1;fraction.appendChild(o)});function update(){let r=data.find(x=>x.model===profile.value&&Math.abs(x.top_fraction-Number(fraction.value))<1e-8);document.getElementById('tailResult').textContent='选中 '+r.n_top+' 人，病例 '+r.cases_top+' 人；病例比例 '+(100*r.case_rate).toFixed(1)+'%；覆盖全部病例 '+(100*r.sensitivity).toFixed(1)+'%；OR '+r.or.toFixed(2)+'。'}profile.onchange=update;fraction.onchange=update;update();""".replace("DATA", jsdata)
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BC validation: OR90 and R²</title><style>' + css + '</style><main>' + "\n".join(sections) + '</main><script>' + script + '</script></html>'
    (root / "VALIDATION_ANALYSIS.html").write_text(document, encoding="utf-8")
    (root / "figure_manifest.json").write_text(json.dumps(rendered, indent=2) + "\n")
    print(f"Wrote {len(rendered)} figures, combined PDF, and {root / 'VALIDATION_ANALYSIS.html'}")


if __name__ == "__main__":
    main()
