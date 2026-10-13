#!/usr/bin/env python3
"""Render the submitted model's OOF diagnostics from aggregate tables."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="outputs/may_final_oof_prediction_analysis")
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.results_dir)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)

    def read(name):
        return pd.read_csv(root / f"{name}.csv")

    pooled = read("pooled_metrics")
    points = pooled.loc[pooled.disease.eq("bc")].set_index("model")
    main_point = points.loc["submitted"]
    deciles = read("bc_deciles")
    d = deciles.loc[deciles.model.eq("submitted")]
    thresholds = read("bc_thresholds")
    bands = read("bc_tail_bands")
    histograms = read("bc_histograms")
    curves = read("bc_roc_pr_curves")
    folds = read("fold_metrics").query("disease == 'bc'")
    fold_distribution = read("fold_prediction_distributions").query("disease == 'bc'")
    intervals = read("bc_intervals").set_index(["model", "metric"])
    deltas = read("bc_paired_deltas").set_index("metric")
    transform = read("score_transform_audit")
    record = json.loads((root / "analysis_record.json").read_text())
    n_boot = record["bootstrap"]["replicates"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .16, "text.color": "#183445",
                         "axes.labelcolor": "#334155", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "savefig.dpi": 180})
    pdf = PdfPages(root / "OOF_ANALYSIS_FIGURES.pdf")
    manifest = []

    def save(fig, name, title):
        fig.suptitle(title, fontsize=15, fontweight="bold")
        scope = "Pooled 5-fold OOF | BC: 4,936 women, 922 cases | Exact May submission scores"
        if name.startswith("08"):
            scope = "Pooled 5-fold OOF | CAD / T2D: n = 8,663; BC: n = 4,936 eligible women"
        fig.text(.5, .005, scope, ha="center", fontsize=8, color="#64748B")
        layout_top = .86 if name.startswith("05") else .95
        fig.tight_layout(rect=[0, .035, 1, layout_top])
        for suffix in ["png", "svg", "pdf"]:
            fig.savefig(figures / f"{name}.{suffix}", bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"name": name, "title": title})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, scale in zip(axes, ["score", "probability"], strict=True):
        inset = ax.inset_axes([.77, .63, .21, .32]) if scale == "probability" else None
        for label, name, color in [(0, "Control", "#4083A5"), (1, "Case", "#C15464")]:
            h = histograms.loc[histograms.model.eq("submitted") & histograms.scale.eq(scale) & histograms.label.eq(label)]
            edges = np.r_[h.left, h.right.iloc[-1]]
            ax.stairs(h.density, edges, label=name, color=color, lw=1.8)
            if inset is not None:
                inset.stairs(h.density, edges, color=color, lw=1)
        if scale == "score":
            cutoff = d.loc[d.decile.eq(10), "score_min"].iloc[0]
            ax.axvline(cutoff, ls="--", color="#334155", lw=1.2, label=f"Top-10% boundary: {cutoff:.3f}")
            ax.set(xlabel="Submitted score (standardized clipped logit)", title="Actual submitted score: full range")
        else:
            ax.set(xlim=(0, .5), xlabel="Native TabPFN probability (main panel: 0 to 0.5)",
                   title="Native probabilities used for probability metrics")
            inset.set(xlim=(0, 1), xticks=[0, .5, 1], yticks=[])
            inset.set_title("Full range", fontsize=7)
            inset.tick_params(labelsize=7)
        ax.set(ylabel="Density within each outcome group")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    save(fig, "01_bc_prediction_distributions", "BC OOF predictions: cases and controls still overlap substantially")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].errorbar(d.decile, d.observed, yerr=[d.observed-d.observed_low, d.observed_high-d.observed],
                     fmt="o-", color=COLORS["submitted"], capsize=3, label="Observed case fraction (95% CI)")
    axes[0].plot(d.decile, d.predicted, "s--", color="#475569", ms=4, label="Mean native probability")
    axes[0].axhline(main_point.prevalence, ls=":", color="#AAA", label="Cohort case fraction")
    for row in d.itertuples():
        axes[0].annotate(f"{row.cases}/{row.n}", (row.decile, row.observed_high),
                         xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    axes[0].set(xticks=range(1, 11), xlabel="Risk decile (1 = lowest, 10 = highest)",
                ylabel="Case fraction", ylim=(0, .49), title="Risk stratification by the submitted score")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    for model in ["submitted", "covariates"]:
        group = deciles.loc[deciles.model.eq(model)]
        axes[1].errorbar(group.predicted, group.observed,
                         yerr=[group.observed-group.observed_low, group.observed_high-group.observed],
                         fmt="o-", color=COLORS[model], capsize=3, label=LABELS[model])
    axes[1].plot([0, .5], [0, .5], ":", color="#AAA")
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed case fraction",
                xlim=(0, .4), ylim=(0, .49), title="Calibration (descriptive; no refitting)")
    axes[1].legend(fontsize=8, frameon=False)
    save(fig, "02_bc_deciles_calibration", "Low-risk recognition and high-risk enrichment measure different strengths")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, col, title, color in [(axes[0], "ll_gain_null", "Compared with prevalence-only null", "#087F8C"),
                                  (axes[1], "ll_gain_covariates", "Compared with covariates-only OOF", "#B97918")]:
        ax.bar(d.decile, d[col], color=color)
        ax.axhline(0, color="#64748B", lw=.8)
        share = d.loc[d.decile.le(2), col].sum() / d[col].sum()
        ax.set(xticks=range(1, 11), xlabel="Submitted-score risk decile", ylabel="Log-likelihood gain (nats)",
               title=f"{title}\nLowest 20%: {share:.1%} of net gain")
    save(fig, "03_bc_likelihood_decomposition", "R2 incorporates the predictive gains in the low-risk population")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for model in ["submitted", "covariates"]:
        t = thresholds.loc[thresholds.model.eq(model)]
        axes[0].plot(t.top_fraction*100, t["or"], "o-", ms=3, color=COLORS[model], label=LABELS[model])
        axes[0].fill_between(t.top_fraction*100, t.or_low, t.or_high, color=COLORS[model], alpha=.10)
    axes[0].axvline(10, color="#64748B", ls="--", lw=1)
    axes[0].axhline(1, color="#AAA", ls=":")
    axes[0].set(xlabel="Highest-scoring patients selected (%)", ylabel="OR: selected versus remaining",
                title="Threshold sensitivity (pointwise Wald 95% CI)", xlim=(0, 50))
    axes[0].legend(fontsize=8, frameon=False)
    x = np.arange(len(bands))
    axes[1].bar(x, bands.observed, color="#087F8C", alpha=.85)
    axes[1].errorbar(x, bands.observed, yerr=[bands.observed-bands.ci_low, bands.ci_high-bands.observed],
                     fmt="none", color="#334155", capsize=3)
    for i, row in enumerate(bands.itertuples()):
        axes[1].text(i, row.ci_high+.025, f"{row.cases}/{row.n}", ha="center", fontsize=8)
    axes[1].set(xticks=x, xticklabels=[f"{100*r.rank_from:g}–{100*r.rank_to:g}%" for r in bands.itertuples()],
                xlabel="Rank band counted from the highest score", ylabel="Case fraction",
                title="The extreme tail becomes diluted in the top 10%", ylim=(0, .85))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].tick_params(axis="x", labelsize=8)
    save(fig, "04_bc_tail_sensitivity", "The strongest enrichment is confined to a small high-score subgroup")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    for ax, metric, title in zip(axes, ["top10_vs_lower90_or", "nagelkerke_r2", "auc"],
                                 ["OR90", "Nagelkerke R2", "AUC"], strict=True):
        for model in ["submitted", "covariates"]:
            f = folds.loc[folds.model.eq(model)].sort_values("fold")
            ax.plot(f[metric], f.fold, "o-", color=COLORS[model], label=LABELS[model], ms=5)
        ci = intervals.loc[("submitted", metric)]
        ax.axvspan(ci.low, ci.high, color=COLORS["submitted"], alpha=.08)
        ax.axvline(main_point[metric], color=COLORS["submitted"], ls="--", lw=1)
        ax.set(yticks=range(5), yticklabels=[f"Fold {i+1}" for i in range(5)], ylim=(4.4, -.4),
               xlabel=title, title=title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False, loc="upper center",
               bbox_to_anchor=(.5, .91), ncol=2)
    save(fig, "05_bc_fold_consistency", "All five held-out folds show the same broad performance pattern")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, curve, metric, title in [(axes[0], "roc", "auc", "ROC (AUC)"),
                                    (axes[1], "pr", "auprc", "Precision-recall (average precision)")]:
        for model in ["submitted", "covariates"]:
            c = curves.loc[curves.model.eq(model) & curves.curve.eq(curve)]
            ax.plot(c.x, c.y, color=COLORS[model], label=f"{LABELS[model]} ({points.loc[model, metric]:.3f})")
        ax.set(title=title)
        ax.legend(fontsize=8, frameon=False)
    axes[0].plot([0, 1], [0, 1], ":", color="#AAA")
    axes[0].set(xlabel="False positive rate", ylabel="Sensitivity")
    axes[1].axhline(main_point.prevalence, ls=":", color="#AAA")
    axes[1].set(xlabel="Recall", ylabel="Precision")
    save(fig, "06_bc_roc_pr", "The submitted model improves overall discrimination over covariates alone")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metric, title in zip(axes, ["top10_vs_lower90_or", "nagelkerke_r2", "auc"],
                                 ["OR90", "Nagelkerke R2", "AUC"], strict=True):
        for i, model in enumerate(["submitted", "covariates"]):
            ci = intervals.loc[(model, metric)]
            point = points.loc[model, metric]
            ax.errorbar(point, i, xerr=[[point-ci.low], [ci.high-point]], fmt="o",
                        color=COLORS[model], capsize=4)
        ax.set(yticks=[0, 1], yticklabels=["Submitted model", "Covariates only"] if ax is axes[0] else [],
               ylim=(1.5, -.5), title=title, xlabel="Point estimate and 95% bootstrap CI")
    save(fig, "07_bc_pooled_uncertainty", f"Pooled OOF estimates from {n_boot:,} paired patient bootstrap samples")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, metric, title in zip(axes, ["top10_vs_lower90_or", "nagelkerke_r2", "auc"],
                                 ["OR90", "Nagelkerke R2", "AUC"], strict=True):
        for i, model in enumerate(["submitted", "covariates"]):
            group = pooled.loc[pooled.model.eq(model)].set_index("disease").loc[["cad", "bc", "T2D"]]
            ax.bar(np.arange(3)+(i-.5)*.32, group[metric], width=.3, color=COLORS[model], label=LABELS[model])
        ax.set(xticks=range(3), xticklabels=["CAD", "BC", "T2D"], title=title)
    axes[0].set_ylim(0, pooled.top10_vs_lower90_or.max()*1.32)
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "08_disease_context", "BC has useful relative gains, but more limited absolute predictive performance")
    pdf.close()

    def table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")

    def figure(name, caption):
        svg = (figures / f"{name}.svg").read_text()
        svg = svg[svg.index("<svg"):]
        for identifier in set(re.findall(r'id="([^"]+)"', svg)):
            target = f"fig_{name}_{identifier}"
            svg = svg.replace(f'id="{identifier}"', f'id="{target}"')
            svg = svg.replace(f'href="#{identifier}"', f'href="#{target}"')
            svg = svg.replace(f'url(#{identifier})', f'url(#{target})')
        return f"<figure>{svg}<figcaption>{caption}</figcaption></figure>"

    metric_rows = []
    for model in ["submitted", "covariates"]:
        row = {"模型": LABELS[model]}
        for metric, label, precision in [("top10_vs_lower90_or", "OR90 (95% CI)", 3),
                                         ("nagelkerke_r2", "Nagelkerke R² (95% CI)", 4),
                                         ("auc", "AUC (95% CI)", 4)]:
            ci = intervals.loc[(model, metric)]
            row[label] = f"{points.loc[model, metric]:.{precision}f} ({ci.low:.{precision}f}–{ci.high:.{precision}f})"
        row["最高一成病例数"] = f"{int(points.loc[model, 'top10_cases'])}/{int(points.loc[model, 'top10_n'])}"
        metric_rows.append(row)
    low = d.loc[d.decile.le(2)]
    top = d.loc[d.decile.eq(10)].iloc[0]
    bottom = d.loc[d.decile.eq(1)].iloc[0]
    low_share = low.ll_gain_null.sum()/d.ll_gain_null.sum()
    low_increment = low.ll_gain_covariates.sum()/d.ll_gain_covariates.sum()
    high_increment = top.ll_gain_covariates/d.ll_gain_covariates.sum()
    first_band = bands.iloc[0]
    middle_top_n = int(top.n-first_band.n)
    middle_top_cases = int(top.cases-first_band.cases)
    fold_main = folds.loc[folds.model.eq("submitted")]
    delta_r2 = deltas.loc["nagelkerke_r2"]
    delta_or = deltas.loc["top10_vs_lower90_or"]
    sections = [
        '<p class="eyebrow">decipEHR · MAY SUBMISSION · POOLED OOF ANALYSIS</p>',
        '<h1>最终提交模型的 OOF 表现：为什么 OR90 与 R² 不完全同步？</h1>',
        '<p class="lead">本报告直接使用最终 audited 提交包的 5 折 OOF 输出。每位患者由未使用其标签拟合的那一折 TabPFN 预测；不追加新的 logistic 评估层。</p>',
        '<div class="cards"><div><b>4,936</b><span>BC 女性 OOF 患者</span></div><div><b>922</b><span>病例；样本病例比例 18.68%</span></div><div><b>5 折</b><span>每位患者恰有一次 OOF 预测</span></div></div>',
        '<p>模型为 5 月最终提交的配置：TabPFN-3 融合 6 个 BC PRS（P+T、LDpred2、lassosum2、PRS-CS、annotated MegaPRS、SBayesRC 7M）及 metadata。OOF score 已与提交 TXT 和 ZIP 中对应患者的分数逐人核对一致。</p>',
        '<h2>1. 最终模型的 pooled OOF 指标</h2>',
        table(pd.DataFrame(metric_rows)),
        f'<p>OR90 按实际提交的标准化 logit 分数排序；R²、Brier 和 log loss 使用包中保存的原始 <code>direct_probability</code>。全体 OOF 患者合并计算主指标；不是把 5 个 fold 指标取平均。基线是此前保存、患者和 fold 完全匹配的 covariates-only OOF 预测。95% CI 来自 {n_boot:,} 次成对患者 bootstrap。</p>',
        '<h2>2. 预测分布：低风险一端分离更清楚</h2>',
        figure("01_bc_prediction_distributions", "病例与对照分别归一化密度。左图是提交分数的完整范围；右图放大原始概率 0–0.5，并用插图保留 0–1 完整范围。标准化分数不是概率，不能直接代入概率型 R²。"),
        f'<p>最低风险一成只有 <b>{int(bottom.cases)}/{int(bottom.n)}（{bottom.observed:.2%}）</b>为病例，而最高风险一成是 <b>{int(top.cases)}/{int(top.n)}（{top.observed:.2%}）</b>。最高一成捕获全部病例的 <b>{main_point.top10_sensitivity:.2%}</b>；其余 90% 的病例比例仍为 <b>{main_point.lower90_case_rate:.2%}</b>，因此 OR90 为 <b>{main_point.top10_vs_lower90_or:.3f}</b>。</p>',
        figure("02_bc_deciles_calibration", "十分位按实际提交分数定义；第 10 组人数为 ceil(0.1N)=494。数字为病例数/人数；误差线为 Wilson 95% CI。校准图只描述已有预测，没有在 OOF 标签上重新拟合或校准。"),
        '<h2>3. R² 的收益来自整个风险分布</h2>',
        '<div class="formula">OR90 = (最高 10% 病例数 / 对照数) ÷ (其余 90% 病例数 / 对照数)<br>G = LL(model) − LL(prevalence-only null)<br>R²<sub>N</sub> = [1 − exp(−2G/n)] / [1 − exp(2LL<sub>null</sub>/n)]</div>',
        f'<p>模型相对 prevalence-only null 的总 log-likelihood 增益为 <b>{d.ll_gain_null.sum():.2f} nats</b>。最低风险 20% 贡献 <b>{low_share:.1%}</b>，最高风险 10% 贡献 <b>{top.ll_gain_null/d.ll_gain_null.sum():.1%}</b>。大量无病者被赋予低概率能够改善 R²，即使最高一成没有捕获很多病例。</p>',
        figure("03_bc_likelihood_decomposition", "分解的是可加的 log-likelihood 增益，不是将非线性的 R² 直接分摊。左图相对常数病例比例 null；右图相对相同患者的 covariates-only 预测，两个参照回答不同问题。"),
        f'<p>相对 covariates-only 的新增收益则更分散：最低 20% 贡献 <b>{low_increment:.1%}</b>，最高 10% 贡献 <b>{high_increment:.1%}</b>。因此不能把模型相对协变量的增量解释为“只在低风险端有效”，也不能将包含 metadata 的模型收益全部归于纯遗传信息。</p>',
        '<p>这里是 <a href="https://doi.org/10.1093/biomet/78.3.691">Nagelkerke pseudo-R²</a>，并非普通线性回归的方差解释比例。约 0.095 的值说明相对本任务基线有改善，并不意味着绝对预测能力很强。</p>',
        '<h2>4. 极高分小群体有效，但扩展到最高一成后被稀释</h2>',
        f'<p>最高 1% 中有 <b>{int(first_band.cases)}/{int(first_band.n)}（{first_band.observed:.1%}）</b>为病例；第 1%–10% 区间是 <b>{middle_top_cases}/{middle_top_n}（{middle_top_cases/middle_top_n:.1%}）</b>。这一结构解释了更极端的小尾部有明显富集，而固定 top-10% 指标仍相对有限。</p>',
        figure("04_bc_tail_sensitivity", "阈值曲线只用于解释，不改变比赛预先指定的 top-10% 指标，也不据此选择新的阈值。左图为固定分组 2×2 表的点态 Wald CI；主 OR90 CI 使用每次重排的患者 bootstrap。右图为不重叠风险排名区间，误差线为 Wilson CI。"),
        '<div class="explorer"><label>模型 <select id="model"><option value="submitted">最终提交模型</option><option value="covariates">协变量基线</option></select></label><label>最高风险比例 <select id="fraction"></select></label><div id="result"></div></div>',
        '<h2>5. 各折一致性与不确定性</h2>',
        f'<p>5 个留出 fold 的模型 OR90 为 <b>{fold_main.top10_vs_lower90_or.min():.2f}–{fold_main.top10_vs_lower90_or.max():.2f}</b>，R² 为 <b>{fold_main.nagelkerke_r2.min():.4f}–{fold_main.nagelkerke_r2.max():.4f}</b>。这不是由单个 fold 独占的现象；各折不是独立重复实验，其训练患者有重叠。</p>',
        figure("05_bc_fold_consistency", "点为每个留出 fold 的指标；每折 OR90 在该折内部取最高一成。竖虚线和淡色区域是模型 pooled OOF 的点估计与 bootstrap CI，并非各折置信区间。"),
        '<details><summary>检查 pooled 最高一成是否集中于个别 fold</summary>' + table(fold_distribution) + '</details>',
        figure("07_bc_pooled_uncertainty", "同一患者 bootstrap 抽样同时作用于提交模型和协变量基线。仅反映固定 OOF 预测下的患者抽样不确定性，不包括模型重训与配置选择的不确定性。"),
        f'<p>相对协变量基线，ΔR²=<b>{delta_r2.delta:+.4f}（95% CI {delta_r2.low:.4f}–{delta_r2.high:.4f}）</b>；ΔOR90=<b>{delta_or.delta:+.3f}（{delta_or.low:.3f}–{delta_or.high:.3f}）</b>。这里应区分“OR90 绝对值不很高”和“OR90 没有改善”：是否支持改善应依据这一成对差值区间。</p>',
        figure("06_bc_roc_pr", "ROC 与 PR 曲线使用最终提交 score；审计确认其 AUC 与原始概率的 AUC 一致。图中 PR 摘要为 average precision。"),
        '<h2>6. 三种疾病背景与本次评估的边界</h2>',
        figure("08_disease_context", "所有疾病均来自同一个最终 audited 包和同一套患者级 fold。跨疾病患病比例不同，R² 不宜被直接等量比较为遗传解释率；这里只提供各疾病预测难度的背景。"),
        table(pooled.loc[pooled.model.eq("submitted"), ["disease", "n", "cases", "prevalence", "auc", "top10_vs_lower90_or", "nagelkerke_r2"]]),
        '<p class="note"><b>可作出的结论：</b>这是已提交配置在各折未参与 TabPFN 拟合患者上的 OOF 表现，适合当前没有 final holdout 标签时进行模型评估与分布解释。PRS 参数和 panel 曾利用 validation 信息选择，因此不是全流程嵌套的无偏评估；也不是那个使用全部 train-val 拟合后的单一模型在独立 holdout 上的成绩。</p>',
        '<details><summary>原始概率与提交 logit clipping 的敏感性审计</summary>' + table(transform) + '<p>主 R² 使用保存的原始概率；从提交的 clipped logit 恢复概率作为敏感性结果单列。两个评分范围不混用，OR90 排序一致已经核对。</p></details>',
        '<h2>文件与复现</h2>',
        '<p>本报告、图和聚合 CSV 不含患者标识。患者级核对表只保存在本地受保护目录。原来的 validation 报告保留为另一种队列/评估口径，未混入本报告指标。</p>',
        '<pre>UV_CACHE_DIR=outputs/uv-cache/prediction_analysis OPENBLAS_NUM_THREADS=1 uv run --offline --no-sync python scripts/analyze_may_final_oof.py --eligibility-config configs/disease_eligibility.yaml\n\nUV_PROJECT_ENVIRONMENT=outputs/prediction_analysis_env UV_CACHE_DIR=outputs/uv-cache/prediction_analysis MPLCONFIGDIR=outputs/mpl-cache uv run --offline --no-sync python scripts/plot_may_final_oof.py</pre>',
        '<p>主要数据：<code>pooled_metrics.csv</code>、<code>fold_metrics.csv</code>、<code>bc_deciles.csv</code>、<code>bc_intervals.csv</code>、<code>bc_paired_deltas.csv</code>、<code>cohort_audit.csv</code>。图提供 PNG、可编辑文本 SVG、单页 PDF 及合并 <code>OOF_ANALYSIS_FIGURES.pdf</code>。</p>',
        '<details><summary>输入来源、方法与校验记录</summary><pre>' + html.escape(json.dumps(record, ensure_ascii=False, indent=2)) + '</pre></details>',
    ]
    css = """body{font:16px/1.75 -apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans CJK SC',sans-serif;color:#253445;background:#f3f7fa;margin:0}main{max-width:1120px;margin:30px auto;padding:40px;background:white;border-radius:15px}h1{font-size:33px;line-height:1.35;color:#163343}h2{font-size:23px;margin-top:44px;border-bottom:2px solid #deebed;padding-bottom:9px}.eyebrow{font-size:12px;letter-spacing:.12em;color:#087f8c;font-weight:bold}.lead{font-size:19px;color:#476071}.cards{display:flex;gap:16px;margin:26px 0}.cards div{flex:1;background:#eef6f7;padding:18px;border-radius:9px}.cards b{font-size:29px;display:block}.cards span{font-size:13px}.note{background:#fff8e8;border-left:4px solid #c59e48;padding:16px;font-size:14px}.formula{background:#eef3f7;padding:18px;font-family:monospace}figure{margin:26px 0}figure svg{width:100%;height:auto}figcaption{font-size:13px;color:#617185}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}td,th{border-bottom:1px solid #dce3e9;padding:9px;text-align:left}th{background:#eef3f7}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#f1f5f8;padding:16px}code{font-size:.87em;background:#eff3f6}a{color:#087f8c}.explorer{background:#edf7f8;padding:20px;border-radius:8px}.explorer label{margin-right:22px}select{padding:7px}#result{margin-top:16px;font-size:18px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}@media(max-width:700px){main{padding:18px;margin:0}.cards{display:block}.cards div{margin-bottom:8px}h1{font-size:27px}table{font-size:11px}}"""
    script = """const data=DATA;const model=document.getElementById('model'),fraction=document.getElementById('fraction');[...new Set(data.map(x=>x.top_fraction))].forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=(x*100)+'%';o.selected=x===.1;fraction.appendChild(o)});function update(){const r=data.find(x=>x.model===model.value&&Math.abs(x.top_fraction-Number(fraction.value))<1e-8);document.getElementById('result').textContent='选中 '+r.n_top+' 人，病例 '+r.cases_top+' 人；病例比例 '+(100*r.case_rate).toFixed(1)+'%；覆盖全部病例 '+(100*r.sensitivity).toFixed(1)+'%；OR '+r.or.toFixed(2)+'。'}model.onchange=update;fraction.onchange=update;update();""".replace("DATA", thresholds.to_json(orient="records"))
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final submission OOF analysis</title><style>' + css + '</style><main>' + "\n".join(sections) + '</main><script>' + script + '</script></html>'
    (root / "OOF_ANALYSIS.html").write_text(document, encoding="utf-8")
    (root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(manifest)} OOF figures and {root / 'OOF_ANALYSIS.html'}")


if __name__ == "__main__":
    main()
