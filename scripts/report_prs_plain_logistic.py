"""Build a compact offline bilingual report for the matched logistic comparison."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import zipfile
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_prs_plain_logistic import ARMS, COHORTS, TRAITS
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files

NAMES = ["Logistic: no penalty", "Logistic: L2", "M0: labels only", "M0: distillation", "Teacher: TabPFN v3"]
COLORS = ["#bc6c25", "#d59f52", "#94a3b8", "#087f8c", "#334e68"]
METRICS = {"auroc": "AUROC ↑", "log_loss": "Log loss ↓", "nagelkerke_r2": "Nagelkerke R² ↑"}


def read(path):
    return json.loads(Path(path).read_text())


def image_uri(path):
    return "data:image/svg+xml;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def formula(expression):
    fig = plt.figure(figsize=(11.5, .65))
    fig.text(.015, .4, expression, fontsize=17, color="#17344c", va="center")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=.12, transparent=True)
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buffer.getvalue()).decode()


def figures(metrics, out):
    for cohort in COHORTS:
        fig, axes = plt.subplots(3, 3, figsize=(13.2, 8.5), layout="constrained")
        for col, trait in enumerate(TRAITS):
            part = metrics.loc[metrics.cohort.eq(cohort) & metrics.trait.eq(trait)].set_index("arm").loc[ARMS]
            for row, (metric, title) in enumerate(METRICS.items()):
                ax = axes[row, col]
                values = part[metric].to_numpy()
                y = np.arange(len(ARMS))
                ax.scatter(values, y, c=COLORS, s=55, zorder=3)
                span = max(float(np.ptp(values)), .004)
                ax.set_xlim(values.min() - .16 * span, values.max() + .58 * span)
                for pos, value in zip(y, values, strict=True):
                    ax.annotate(f"{value:.4f}", (value, pos), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
                ax.set(yticks=y, yticklabels=NAMES if col == 0 else [""] * len(ARMS), xlabel=title)
                ax.invert_yaxis()
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
                ax.grid(axis="x", alpha=.14)
                ax.tick_params(axis="y", length=0)
                if row == 0:
                    ax.set_title(f"{'Female BC' if trait == 'bc' else trait.upper()} · N = {int(part.n.iloc[0]):,}", pad=13)
        cohort_name = {"original_val": "Original validation", "exclude_shared_group": "Validation: exclude shared training kinship groups",
                       "exclude_direct_0177": "Validation: exclude direct train–val KING ≥ 0.177"}[cohort]
        fig.suptitle(cohort_name + "\nMatched inputs · frozen models · point estimates; axes differ by panel", fontsize=13)
        for ext in ["svg", "png", "pdf"]:
            fig.savefig(out / f"performance_{cohort}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)


def html_reports(out, metrics, gaps, counts, training):
    expressions = [
        r"$\eta_i=\alpha+\sum_{k=1}^{K}\beta_k Z_{ik}+\beta_a a_i+\boldsymbol{\gamma}^{T} A_i+\boldsymbol{\delta}^{T} C_i,\qquad p_i=\frac{1}{1+e^{-\eta_i}}$",
        r"$\mathcal{L}_{\mathrm{LR}}=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]$",
        r"$\mathcal{L}_{\mathrm{L2}}=\mathcal{L}_{\mathrm{LR}}+\frac{\|\boldsymbol{\beta}_{\mathrm{all}}\|_2^2}{2Cn},\qquad C=0.1$",
        r"$\eta_i^{\mathrm{M0}}=\alpha+\boldsymbol{\gamma}^{T}A_i+\boldsymbol{\delta}^{T}C_i+f_{\uparrow}(a_i)+\sum_{k=1}^{K}\pi_k(A_i)e^{\rho_k}Z_{ik}$",
        r"$\pi(A_i)=\mathrm{softmax}(WA_i+c),\qquad \widetilde{y}_i=(1-\lambda)y_i+\lambda q_i^{\mathrm{OOF}},\qquad \mathcal{L}_{\mathrm{KD}}=\mathrm{BCE}(\widetilde{y},p)+\mathcal{R}$",
    ]
    rendered = [formula(x) for x in expressions]
    for language in ["zh", "en"]:
        def t(zh, en, lang=language):
            return zh if lang == "zh" else en

        title = t("普通多变量 Logistic 与蒸馏对照", "Multivariable logistic regression vs. distillation")
        html = [f'<!doctype html><html lang="{language}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>',
                "<style>body{margin:0;background:#f3f6f8;color:#17344c;font:16px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1180px;margin:auto;padding:26px}h1{font-size:30px;line-height:1.25}h2{font-size:22px;margin-top:0}h3{font-size:17px}section{background:white;border:1px solid #dbe4e9;border-radius:12px;padding:24px;margin:18px 0}.muted{color:#536779;font-size:14px}.banner{border-left:5px solid #087f8c;padding:12px 16px;background:#ecf7f7}a{color:#087f8c}button{font:inherit;padding:8px 14px;background:white;border:1px solid #b5c5d0;border-radius:7px;cursor:pointer;margin:4px}button[aria-selected=true]{color:white;background:#087f8c;border-color:#087f8c}img.plot{width:100%;height:auto}.formula{overflow:auto;padding:8px 0}.formula img{display:block;max-width:100%;height:auto;min-width:440px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e2e9ee;white-space:nowrap}th{background:#f2f6f8}details{margin-top:14px}summary{cursor:pointer;font-weight:600}.pills{display:flex;gap:10px;flex-wrap:wrap}.pill{padding:6px 12px;background:#eef4f7;border-radius:5px}[hidden]{display:none!important}@media(max-width:650px){main{padding:12px}section{padding:15px}h1{font-size:25px}}@media print{button,nav{display:none}.panel[hidden]{display:block!important}section{break-inside:avoid}}</style><main>",
                f'<nav><a href="logistic_{"en" if language == "zh" else "zh"}.html">{t("English version", "Chinese version")}</a></nav>',
                f'<h1>{title}</h1><p class="muted">{t("全量 train 拟合 → 官方 val 评估 · 三个疾病 · 2026-10-27", "Full train → official validation · three endpoints · 2026-10-27")}</p>',
                '<div class="banner">' + t("同一批患者、同一组输入。主对照是无惩罚的普通 Logistic；L2 Logistic 是训练内部选择正则化的补充对照。M0_SUP 帮助区分结构差异与蒸馏增益。",
                    "Matched patients and input variables. Unpenalized logistic is the primary baseline; train-selected L2 logistic is a secondary comparator. M0_SUP helps distinguish model structure from distillation gains.") + '</div>',
                '<section><h2>' + t("当前结论", "Current findings") + '</h2><ul><li>' + t(
                    "T2D：M0_KD 相对普通 Logistic 的 AUROC 提升约 0.0085，log loss 降低约 0.0054。同结构 M0_SUP/KD 比较也支持蒸馏收益。",
                    "T2D: M0_KD improves AUROC by about 0.0085 and lowers log loss by about 0.0054 versus ordinary logistic. The matched M0_SUP/KD comparison also supports a distillation benefit.") + '</li><li>' + t(
                    "CAD：AUROC 差异很小（约 +0.0015，区间跨 0），主要改善 log loss。M0_KD 与 M0_SUP 很接近，不能把全部收益归于蒸馏。",
                    "CAD: the AUROC difference is small (about +0.0015, with an interval spanning zero); the main improvement is log loss. M0_KD and M0_SUP are close, so the full benefit cannot be attributed to distillation.") + '</li><li>' + t(
                    "女性乳腺癌：M0_KD 相对普通 Logistic 的 AUROC 提升约 0.0216；但 M0_KD 与 M0_SUP 的差值区间跨 0，单独的蒸馏增益仍不确定。",
                    "Female BC: M0_KD improves AUROC by about 0.0216 versus ordinary logistic. Its difference from M0_SUP spans zero, so the incremental distillation benefit remains uncertain.") + '</li></ul><p class="muted">' + t(
                    "以上数值来自原始 val。两种亲缘筛选后，T2D 和乳腺癌的 AUROC 优势仍在；L2 与无惩罚 Logistic 的表现接近。所有统计区间均为未校正多重比较的名义区间。",
                    "These values use original val. The T2D and BC AUROC advantages persist under both kinship filters. L2 and unpenalized logistic perform similarly. All statistical intervals are nominal and unadjusted for multiplicity.") + '</p></section>',
                '<section><h2>' + t("模型与训练", "Models and training") + '</h2><div class="pills"><span class="pill">T2D: 26 PRS</span><span class="pill">CAD: 7 PRS</span><span class="pill">Female BC: 5 PRS</span><span class="pill">age_sd + sex + PC1–PC10</span></div>',
                '<h3>' + t("普通 Logistic：所有变量一起进入线性 logit", "Ordinary logistic: all variables enter a linear logit") + '</h3>']
        for index in range(3):
            html.append(f'<div class="formula"><img alt="{escape(expressions[index])}" src="{rendered[index]}"></div>')
        html += ['<p>' + t("Z 为各 PRS，a 为 age_sd，A 为 PC1–PC10，Cᵢ 为性别哑变量；L2 公式中的 C 是正则化参数，截距不惩罚。所有标准化仅在对应训练范围内拟合。无惩罚版不调参、不用软标签或交互项；L2 版 C∈{0.1,1,10} 在 train 内 U/H 上选定，三个疾病均选择 0.1。",
            "Z denotes the PRS inputs, a is age_sd, A contains PC1–PC10, and Cᵢ denotes sex indicators. C in the L2 formula is the regularization parameter; the intercept is unpenalized. Preprocessing is fitted only on each training context. The unpenalized model has no tuning, soft labels or interactions. L2 C∈{0.1,1,10} was selected on train-internal U/H; all endpoints selected 0.1.") + '</p>',
            '<p class="muted">' + t("女性乳腺癌仅纳入符合配置的女性，性别为常量而省略。无惩罚版删除一个性别参考哑变量；L2 版保留历史编码。两者的信息来源相同。优化采用全量数据的 L-BFGS，最多 20,000 次迭代、tol=10⁻⁹；实际迭代数见下方训练记录。",
            "BC includes eligible females only; constant sex is omitted. Unpenalized logistic drops one reference sex dummy; L2 retains its original encoding. Both use the same input information. Unpenalized optimization uses full-batch L-BFGS, max_iter=20,000 and tol=10⁻⁹; actual iterations are listed below.") + '</p><details><summary>' + t("M0 对照及蒸馏 loss", "M0 comparator and distillation loss") + '</summary>']
        for index in [3, 4]:
            html.append(f'<div class="formula"><img alt="{escape(expressions[index])}" src="{rendered[index]}"></div>')
        html += ['<p>' + t("M0 使用祖源门控、正的 PRS 尺度和三段单调年龄基线。M0_SUP 固定 λ=0，M0_KD 使用 train 内选定的 λ>0；q 是训练范围内的 Teacher 折外概率，ℛ 是既有正则项。二者各自按 train 内规则选择优化步数与正则化，因此 SUP/KD 是训练流程比较。与普通 LR 的差异同时涉及结构、约束及训练目标，不能全部归为蒸馏。",
            "M0 uses an ancestry gate, positive PRS scales and a three-segment monotone age baseline. M0_SUP fixes λ=0; M0_KD uses a positive train-selected λ. q is the training-context teacher OOF probability and ℛ contains the existing penalties. SUP and KD independently select steps and regularization within train, so their comparison concerns training procedures. Differences from ordinary LR combine architecture, constraints and training targets.") + '</p></details></section>',
            '<section><h2>' + t("Validation 性能", "Validation performance") + '</h2><p class="muted">' + t("Teacher 为实际蒸馏使用的 TabPFN V3。主表是全 train → val，OOF 仅作为下方训练内部诊断。",
            "Teacher is the TabPFN V3 used for distillation. Main results use full train → val; OOF is a separate training diagnostic below.") + '</p><div role="tablist">']
        labels = [t("原始 validation", "Original validation"), t("排除共享亲缘组", "Exclude shared groups"), t("排除直接一级近亲", "Exclude direct close relatives")]
        for index, label in enumerate(labels):
            html.append(f'<button role="tab" aria-selected="{str(index == 0).lower()}" onclick="selectCohort({index})">{label}</button>')
        html.append('</div>')
        for index, cohort in enumerate(COHORTS):
            subset = metrics.loc[metrics.cohort.eq(cohort)]
            html.append(f'<div class="panel" id="panel{index}" {"hidden" if index else ""}>')
            count = counts.loc[counts.cohort.eq(cohort)].set_index("trait")
            html.append('<p class="muted">' + ' · '.join(f'{"Female BC" if trait == "bc" else trait.upper()}: N={int(count.loc[trait, "n"]):,}, {t("病例", "cases")}={int(count.loc[trait, "events"]):,}' for trait in TRAITS) + '</p>')
            html.append(f'<img class="plot" alt="{labels[index]}: AUROC, log loss, Nagelkerke R squared" src="{image_uri(out / f"performance_{cohort}.svg")}">')
            html.append('<h3>' + t("M0_KD − 对照：配对差值与 95% 区间", "M0_KD − comparator: paired differences and 95% intervals") + '</h3><p class="muted">' + t("ΔAUROC > 0、Δlog loss < 0 有利于 M0_KD。区间跨 0 不代表已经证明等效。", "Positive ΔAUROC and negative Δlog loss favor M0_KD. An interval crossing zero does not establish equivalence.") + '</p>')
            table = []
            for trait in TRAITS:
                for reference in ["LR_UNPENALIZED", "LR_L2", "M0_SUP"]:
                    part = gaps.loc[gaps.trait.eq(trait) & gaps.cohort.eq(cohort) & gaps.reference.eq(reference)].set_index("metric")
                    row = {t("疾病", "Endpoint"): "Female BC" if trait == "bc" else trait.upper(), t("对照", "Comparator"): NAMES[ARMS.index(reference)]}
                    for metric in ["auroc", "log_loss"]:
                        p = part.loc[metric]
                        row["Δ " + metric] = f'{p.delta:+.4f} [{p.ci_low:+.4f}, {p.ci_high:+.4f}]'
                    table.append(row)
            html.append('<div class="table">' + pd.DataFrame(table).to_html(index=False, border=0) + '</div>')
            html.append('<details><summary>' + t("完整指标表（含 Brier、AP 和校准）", "Complete metrics, including Brier, AP and calibration") + '</summary><div class="table">')
            html.append(subset.drop(columns=["cohort", "calibration_status"]).to_html(index=False, border=0, float_format=lambda x: f"{x:.5f}"))
            html.append('</div></details></div>')
        html += ['<p class="muted">' + t("筛选只改变 val：共享组由 KING≥0.0884 的连通关系定义；直接近亲要求与训练患者直接 KING≥0.177。两种方案均保留亲缘未知者，不重新训练。",
            "Filters change val only. Shared groups use KING≥0.0884 connected components; direct close relatives require a direct KING≥0.177 edge to a training patient. Both retain unresolved kinship; models are not refitted.") + '</p></section>',
            '<section><h2>' + t("解释与边界", "Interpretation and limits") + '</h2><p>' + t("以真实疾病标签的 log loss、AUROC、Brier 和 Nagelkerke 伪 R² 判断预测性能。这里的 R² 不是对 Teacher 的拟合 R²，也不是连续结局的解释方差。AP 随患病比例变化；校准截距和斜率仅用于诊断，没有用 val 重新校准预测。",
            "Disease-prediction metrics use true labels: log loss, AUROC, Brier and Nagelkerke pseudo-R². This R² is neither teacher fidelity nor explained variance for a continuous outcome. AP depends on prevalence. Calibration intercept and slope are diagnostic only; validation does not recalibrate predictions.") + '</p><p class="muted">' + t("差值区间使用 1,000 次可用亲缘组配对 bootstrap，条件于固定预测，未做多重比较校正。当前 val 已被多次查看，本次属于开发性、事后比较；亲缘估计和未知亲缘均有局限。",
            "Differences use 1,000 paired available-kinship-group bootstrap resamples, conditional on fixed predictions, without multiplicity adjustment. Validation has been inspected previously; this is a post-hoc development comparison. Kinship estimation and unresolved relationships remain limitations.") + '</p>',
            '<details><summary>' + t("共享五折的训练 OOF 诊断", "Training OOF diagnostics on the shared five folds") + '</summary><div class="table">' + pd.read_csv(out / "training_oof_metrics.csv").drop(columns="calibration_status").to_html(index=False, border=0, float_format=lambda x: f"{x:.5f}") + '</div></details>',
            '<details><summary>' + t("无惩罚模型训练记录", "Unpenalized model training records") + '</summary><div class="table">' + training[["trait", "scope", "n_context", "n_query", "iterations", "gradient_max_abs", "design_rank", "design_columns"]].to_html(index=False, border=0, float_format=lambda x: f"{x:.3g}") + '</div></details>',
            '<p><a href="metrics.csv">Metrics CSV</a> · <a href="paired_differences.csv">Paired differences CSV</a> · <a href="analysis_verification.json">Verification</a></p></section>',
            '<p class="muted">' + t("所有公式均已渲染为内嵌矢量图，离线打开即可显示。页面不包含患者标识。", "All equations are embedded vector graphics and display offline. This page contains no patient identifiers.") + '</p></main>',
            '<script>function selectCohort(i){document.querySelectorAll(".panel").forEach((p,j)=>p.hidden=i!==j);document.querySelectorAll("[role=tab]").forEach((b,j)=>b.setAttribute("aria-selected",String(i===j)));}</script></html>']
        (out / f"logistic_{language}.html").write_text(''.join(html))


class CheckHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = 0
        self.panels = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        assert not (tag == "script" and a.get("src"))
        if tag == "img":
            assert a["src"].startswith("data:image/svg+xml;base64,") and a.get("alt")
            ElementTree.fromstring(base64.b64decode(a["src"].split(",", 1)[1], validate=True))
            self.images += 1
        if a.get("class") == "panel":
            self.panels += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_dir, args.run_dir / "report"
    receipt = read(root / "VERIFIED.json")
    verify_files(root, receipt["output_sha256"])
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    metrics, gaps, counts, training = [pd.read_csv(out / n) for n in
        ["metrics.csv", "paired_differences.csv", "cohort_counts.csv", "training_details.csv"]]
    figures(metrics, out)
    html_reports(out, metrics, gaps, counts, training)
    (out / "analysis_verification.json").write_bytes((root / "VERIFIED.json").read_bytes())
    for language in ["zh", "en"]:
        p = out / f"logistic_{language}.html"
        checker = CheckHTML()
        checker.feed(p.read_text())
        assert checker.images == 8 and checker.panels == 3
        if language == "en":
            assert not re.search("[\u4e00-\u9fff]", p.read_text())
    members = [p for p in out.iterdir() if p.suffix in {".html", ".svg", ".png", ".pdf", ".csv"}]
    with zipfile.ZipFile(out / "logistic_bilingual.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in members:
            if p.suffix == ".csv":
                assert not {"sampleid", "kinship_group", "IID1", "IID2"} & set(pd.read_csv(p, nrows=0).columns)
            z.write(p, p.name)
        z.writestr("analysis_verification.json", (root / "VERIFIED.json").read_bytes())
    with zipfile.ZipFile(out / "logistic_bilingual.zip") as z:
        assert z.testzip() is None
    (root / "source_snapshot" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(root / "REPORT_VERIFIED.json", {"status": "generated_and_structurally_verified",
        "embedded_formula_svgs_per_language": 5, "cohort_panels_per_language": 3,
        "browser_render_tested": False, "offline_resources": True,
        "analysis_receipt_sha256": sha256_file(root / "VERIFIED.json"),
        "script_sha256": sha256_file(__file__),
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}})
    print(f"Bilingual report: {out}")


if __name__ == "__main__":
    main()
