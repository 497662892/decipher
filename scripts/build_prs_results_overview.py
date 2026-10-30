"""Concise bilingual overview of sealed PRS distillation experiments; no fitting."""
from __future__ import annotations

import json
import os
import re
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
from prs_report_math import check_latex, render_latex
from report_prs_plain_logistic import image_uri

from decipehr.external_pgs import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "outputs/prs_distillation_bio/exploration_20261026_3569648f99a8"
CENT = ROOT / "outputs/prs_distillation_bio/centroid_init_20261027"
LR = ROOT / "outputs/prs_distillation_bio/plain_logistic_20261027"
KIN = ROOT / "outputs/prs_distillation_bio/validation_kinship_filters_20261027"
V1 = ROOT / "outputs/prs_distillation_v1/presentation_20261026"
V3 = ROOT / "outputs/prs_distillation_v3/signaware_20261026_95dfc04c8c06/report"
COHORT = ROOT / "outputs/prs_distillation_v1/phase_a_fulltrain_20261026_3d4334d2_0161ad36d950/cohort_comparison"
OUT = ROOT / "outputs/prs_distillation_bio/summary_20261027"
ORIGINAL = ROOT / "outputs/prs_distillation_bio/original_proposal_comparison_20261027"
METRICS = ["auroc", "log_loss", "nagelkerke_r2", "brier", "average_precision"]
ORDER = ["LR_UNPENALIZED", "LR_L2", "V1_KD_MIX", "M0_SUP", "M0_KD", "M0_CENTROID", "V3_M2_KD",
         "BIO_POLICY", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"]
NAMES = ["Logistic: unpenalized", "Logistic: L2", "V1 mixed KD", "M0 supervised", "M0 KD",
         "M0 KD: centroid init", "V3 M2 KD", "Train-selected policy", "TabPFN V3 (teacher)",
         "TabPFN 3.5 (control)", "Kumo Tabular (control)"]
EQUATIONS = [
    r"$\begin{aligned}p&=\sigma(\eta),\\[4pt]\eta&=\alpha+\gamma^\top A+\delta^\top C+\sum_{j=1}^{3}b_jB_j(a)\\&\quad+\sum_{k=1}^{K}\pi_k(A)e^{\rho_k}Z_k,\qquad b_j\geq0.\end{aligned}$",
    r"$\pi(A)=\operatorname{softmax}(WA+c),\qquad r_k=e^{\rho_k}>0$",
    r"$\begin{aligned}\widetilde y_i&=(1-\lambda)y_i+\lambda q_i^{\mathrm{OOF}},\\[5pt]\mathcal{L}&=\frac{1}{N}\sum_i\operatorname{BCEWithLogits}(\eta_i,\widetilde y_i)+\Omega.\end{aligned}$",
    r"$\begin{aligned}\Omega&=\omega_g\!\left(\lVert\bar W\rVert_F^2+\lVert\bar c\rVert_2^2\right)+\frac{\kappa_r}{K}\sum_k\rho_k^2\\&\quad+\omega_b\!\left(\lVert\gamma\rVert_2^2+\lVert\delta\rVert_2^2+\lVert b\rVert_2^2\right).\end{aligned}$",
    r"$R_N^2=\frac{1-\exp\{2(\ell_0-\ell_p)/n\}}{1-\exp(2\ell_0/n)}$",
]


def compact_view(lang):
    def t(zh, en):
        return zh if lang == "zh" else en

    html = '<section><h2>'+t("相比最初方案：有小幅改善", "Compared with the original proposal: modest gains")+'</h2>'
    html += '<p>'+t(
        "这里比较的是 10 月 26 日最初的 v1（共同年龄缩放）与当前 M0_KD。二者使用相同的全量 train、相同 val 和相同 Teacher。",
        "This compares the October 26 original v1, with a shared age scale, against current M0_KD using the same full train, val patients and teacher.")+'</p>'
    html += '<p class="callout">'+t(
        "三个疾病的 AUROC、log loss 和 R² 点估计均改善，三个种子的方向一致。但三项 AUROC 的 95% 区间均跨 0；CAD 的 log loss 改善更明确。现阶段是小幅进步，尚不能称为稳定的大幅提升。",
        "AUROC, log loss and R² point estimates improve for all endpoints, with the same direction across three seeds. All three AUROC intervals still cross zero; CAD log loss has clearer evidence of improvement. These are modest development gains.")+'</p>'
    html += '<p class="muted">'+t(
        "主要结构变化：共同年龄缩放 → 单调年龄基线＋每个 PRS 的独立正尺度；祖源门控和蒸馏保留。不能把这一比较单独解释为蒸馏收益。",
        "Main structural change: shared age scaling → a monotone age baseline plus a positive scale for each PRS. Ancestry gating and KD remain. This contrast does not isolate the effect of KD.")+'</p></section>'
    html += '<section><h2>'+t("只看四个主要参照", "Four main reference models")+'</h2><label>'+t("指标 ", "Metric ")
    html += '<select id="compact-metric" onchange="updateCompact()"><option value="auroc">AUROC ↑</option><option value="log_loss">Log loss ↓</option><option value="nagelkerke_r2">Nagelkerke R² ↑</option></select></label><div id="compact-table" class="scroll"></div>'
    html += '<p class="muted">'+t(
        "原始 val：T2D / CAD 各 2,852 人，女性 BC 1,628 人；主种子 17。Teacher 使用全 train 作为 context 直接预测 val。R² 指 Nagelkerke pseudo-R²。",
        "Original val: 2,852 T2D/CAD and 1,628 female BC patients; primary seed 17. The teacher predicts val with all train as context. R² is Nagelkerke pseudo-R².")+'</p>'
    html += '<details><summary>'+t("当前 − 原始方案：差值与 95% 区间", "Current − original: differences and 95% intervals")+'</summary><div id="compact-deltas" class="scroll"></div><p class="muted">'+t(
        "1,000 次可用亲缘组配对 bootstrap，固定预测，未校正多重比较。AUROC / R² 的正差值、log loss 的负差值有利于当前模型。",
        "1,000 paired available-kinship-group bootstrap resamples of fixed predictions; no multiplicity adjustment. Positive AUROC/R² and negative log-loss differences favor the current model.")+'</p><a href="original_proposal_seed_differences.csv">'+t("查看三个种子的差值", "Differences across all three seeds")+'</a></details></section>'
    html += '<section><h2>'+t("其他实验，三句话", "Other experiments in three points")+'</h2><ul>'
    for zh, en in [
        ("更复杂的年龄修饰、PRS 变换和交互：尚无稳定的额外收益，先保留 M0_KD。", "More complex age modifiers, PRS transforms and interactions: no stable additional gain; retain M0_KD as the reference."),
        ("祖源中心初始化：乳腺癌的一次提升未在其他种子复现，暂不替换基线。", "Centroid initialization: the BC improvement at one seed did not repeat at the others; keep it exploratory."),
        ("亲缘筛选和新 Teacher：不同患者子集的差距会改变，3.5 / Kumo 也没有跨疾病统一赢家。", "Kinship filters and newer teachers: gaps depend on the patient subset; 3.5/Kumo have no consistent winner across endpoints.")]:
        html += '<li>'+t(zh, en)+'</li>'
    html += '</ul><p class="muted">'+t("以上为开发集直接概率结果；val 已多次观察，正式 Phase B 尚未运行。", "These are developmental direct-probability results; val has been examined repeatedly and formal Phase B has not run.")+'</p></section>'
    config = {"arms": ["LR_UNPENALIZED", "V1_KD_MIX", "M0_KD", "TabPFN_v3"],
        "names": [t("普通回归", "Plain logistic"), t("原始 v1", "Original v1"), t("当前 M0_KD", "Current M0_KD"), "Teacher V3"],
        "traits": ["T2D", "cad", "bc"], "trait_names": ["T2D", "CAD", t("女性 BC", "Female BC")],
        "model_label": t("模型", "Model"), "endpoint_label": t("疾病", "Endpoint"),
        "delta_label": t("差值", "Difference"), "interval_label": t("名义 95% 区间", "Nominal 95% interval")}
    return html, config


def main():
    os.umask(0o077)
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {}
    archives = []

    def read_table(base, relative, receipt="VERIFIED.json", key="output_sha256"):
        path, seal = base / relative, base / receipt
        record = json.loads(seal.read_text())
        assert sha256_file(path) == record[key][relative], path
        for p in [path, seal]:
            provenance[str(p.relative_to(ROOT))] = sha256_file(p)
        return pd.read_csv(path)

    def archive(frame, family, assessment=None):
        d = frame.copy().rename(columns={"cohort": "assessment"})
        if assessment:
            d["assessment"] = assessment
        if "seed" not in d:
            d["seed"] = 17
        d["family"] = family
        archives.append(d)
        return d

    val = archive(read_table(BIO, "report/val/metrics.csv", "COMPLETE.json", "aggregate_output_sha256"), "BIO", "original_val")
    nested = archive(read_table(BIO, "report/nested/metrics.csv", "COMPLETE.json", "aggregate_output_sha256"), "BIO", "train_oof")
    archive(read_table(V1, "metrics.csv"), "V1", "original_val")
    archive(read_table(V3, "metrics.csv"), "V3", "original_val")
    logistic = archive(read_table(LR, "report/metrics.csv"), "Plain logistic comparison")
    logistic_oof = archive(read_table(LR, "report/training_oof_metrics.csv"), "Plain logistic comparison", "train_oof")
    kinship = archive(read_table(KIN, "report/metrics.csv"), "Validation kinship filters")
    centroid = archive(read_table(CENT, "report/metrics.csv"), "Centroid initialization")
    for scope, assessment in [("val", "original_val"), ("nested", "train_oof")]:
        archive(read_table(BIO / f"report/constraint_ablation_{scope}", "metrics.csv"), "Constraint ablation", assessment)
    probes = read_table(BIO / "report/constraint_probe", "summary.csv")
    cohorts = read_table(KIN, "report/cohort_counts.csv")
    old_cohorts = read_table(COHORT, "teacher_metrics.csv")
    budgets = read_table(CENT, "report/scope_budgets.csv")
    centroid_gaps = read_table(CENT, "report/paired_differences.csv")
    plain_gaps = read_table(LR, "report/paired_differences.csv")
    bio_gaps = read_table(BIO, "report/val/paired_differences.csv", "COMPLETE.json", "aggregate_output_sha256")
    kinship_gaps = read_table(KIN, "report/paired_differences.csv")
    original_gaps = read_table(ORIGINAL, "paired_differences.csv")
    original_metrics = read_table(ORIGINAL, "metrics.csv")
    original_seeds = read_table(ORIGINAL, "seed_differences.csv")

    main_frames = [val, nested, logistic, logistic_oof,
                   kinship.assign(arm=kinship.arm.replace({"PRIMARY_SELECTED": "BIO_POLICY"})),
                   centroid.loc[centroid.seed.eq(17)].assign(arm=lambda d: d.arm.replace({"M0_RANDOM": "M0_KD"}))]
    main_rows = pd.concat(main_frames, ignore_index=True)
    main_rows["arm"] = main_rows.arm.replace({"LR": "LR_L2"})
    main_rows = main_rows.loc[main_rows.arm.isin(ORDER)]
    for _, part in main_rows.groupby(["assessment", "trait", "arm"]):
        for column in ["n", "events", *METRICS]:
            values = part[column].dropna().to_numpy()
            np.testing.assert_allclose(values, values[0], atol=1e-12, rtol=0)
    main_rows = main_rows.drop_duplicates(["assessment", "trait", "arm"])
    for _, part in main_rows.groupby(["assessment", "trait"]):
        assert part.n.nunique() == part.events.nunique() == 1
    main_rows = main_rows[["assessment", "trait", "arm", "n", "events", *METRICS]]
    main_rows.to_csv(OUT / "main_metrics.csv", index=False)
    for row in original_metrics.loc[original_metrics.seed.eq(17)].itertuples():
        match = main_rows.loc[main_rows.assessment.eq("original_val") & main_rows.trait.eq(row.trait) & main_rows.arm.eq(row.arm)].iloc[0]
        np.testing.assert_allclose([getattr(row, m) for m in ["auroc", "log_loss", "nagelkerke_r2"]],
                                   match[["auroc", "log_loss", "nagelkerke_r2"]].to_numpy(float), atol=1e-12, rtol=0)
    all_metrics = pd.concat(archives, ignore_index=True)
    all_metrics.to_csv(OUT / "all_metrics.csv", index=False)
    for name, frame in [("cohort_counts", cohorts), ("historical_cohort_control", old_cohorts),
                        ("teacher_constraint_probe", probes), ("centroid_paired_differences", centroid_gaps),
                        ("plain_logistic_paired_differences", plain_gaps), ("bio_paired_differences", bio_gaps),
                        ("kinship_paired_differences", kinship_gaps), ("training_budgets", budgets),
                        ("original_proposal_paired_differences", original_gaps), ("original_proposal_seed_differences", original_seeds)]:
        frame.to_csv(OUT / f"{name}.csv", index=False)
    for p in [ROOT / "src/decipehr/prs_distillation_bio.py", ROOT / "src/decipehr/prs_distillation_v3.py",
              ROOT / "src/decipehr/metrics.py", ROOT / "src/decipehr/prs_distillation/student.py",
              ROOT / "scripts/report_prs_plain_logistic.py", CENT / "report/seed_stability.svg", CENT / "REPORT_VERIFIED.json"]:
        provenance[str(p.relative_to(ROOT))] = sha256_file(p)
    centroid_report = json.loads((CENT / "REPORT_VERIFIED.json").read_text())
    assert sha256_file(CENT / "report/seed_stability.svg") == centroid_report["output_sha256"]["seed_stability.svg"]

    math_style, formulas, math_metadata = render_latex(EQUATIONS)
    provenance.update(math_metadata["source_sha256"])
    for lang in ["zh", "en"]:
        def t(zh, en, lang=lang):
            return zh if lang == "zh" else en

        def link(path, label):
            assert path.is_file(), path
            return f'<a href="{escape(os.path.relpath(path, OUT))}">{escape(label)}</a>'

        def eq(i):
            return formulas[i]

        def table(d):
            return '<div class="scroll">'+d.to_html(index=False, border=0, float_format=lambda v: f"{v:.5f}")+'</div>'

        title = t("PRS 蒸馏：简明结果", "PRS distillation: concise results")
        html = ['<!doctype html><html lang="'+lang+'"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>'+title+'</title><style>body{margin:0;background:#f2f6f8;color:#18354b;font:16px/1.6 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:26px}section{background:white;padding:24px;margin:18px 0;border:1px solid #d9e3e9;border-radius:12px}h1{font-size:32px}h2{font-size:22px;margin-top:0}h3{font-size:18px}.muted{font-size:14px;color:#516a7b}.callout{padding:14px;background:#edf7f7;border-left:4px solid #0b7e87}.scroll,.formula{overflow:auto}.formula{margin:13px 0}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #e1e7eb;white-space:nowrap}th{background:#f3f6f8}a{color:#067985}select{font:inherit;padding:7px;margin:3px 10px 3px 0;border:1px solid #b7c9d4;border-radius:7px}summary{cursor:pointer;font-weight:600}details{margin:14px 0}.plot{width:100%;height:auto}svg{max-width:100%;height:auto}.pipeline{display:flex;flex-wrap:wrap;gap:8px}.pipeline span{background:#f1f6f9;padding:10px;border-radius:7px;flex:1;min-width:150px}@media(max-width:650px){main{padding:10px}section{padding:15px}#compact-table table{font-size:13px;table-layout:fixed}#compact-table th,#compact-table td{padding:7px 4px;white-space:normal}#compact-table th:first-child{width:32%}}@media print{select{display:none}section{break-inside:avoid}}</style><main>',
            '<a href="overview_'+('en' if lang == 'zh' else 'zh')+'.html">'+t("English version", "Chinese version")+'</a>',
            '<h1>'+title+'</h1><p class="muted">2026-10-27 · '+t("train 内开发，val 查看效果；全部为直接概率评估。", "Development within train; evaluation on val. All scores use direct probabilities.")+'</p>',
            '<section><h2>'+t("现在可以得出的结论", "What the results support")+'</h2><p><strong>'+t("保留 M0_KD 作为简单、可解释的主要参照。", "Retain M0_KD as the simple, interpretable reference.")+'</strong> '+t("相对无正则多变量 logistic regression，T2D 和女性乳腺癌的原始 val AUROC 较高；CAD 的 AUROC 接近。复杂结构尚未表现出跨疾病的稳定优势。", "Compared with unpenalized multivariable logistic regression, original-val AUROC is higher for T2D and female BC, while CAD AUROC is similar. More complex structures have not shown a stable advantage across endpoints.")+'</p><p>'+t("祖源中心初始化已试：T2D/CAD 几乎不变，乳腺癌只在 seed 17 明显提高，另两个种子未复现。它目前是探索性结果，不能直接替代基线。", "Centroid initialization has now been tested: T2D/CAD barely change; BC improves materially only at seed 17, without replication at the other two seeds. It remains exploratory and does not establish a replacement for the baseline.")+'</p><p class="muted">'+t("普通回归与 M0_KD 的差异包含结构和蒸馏两部分。判断蒸馏本身，应比较 M0_KD 与同结构 M0_SUP。所有结果仍属于开发证据，val 已被反复观察。", "The logistic–M0_KD contrast combines architecture and distillation effects. M0_KD versus the same-architecture M0_SUP isolates the KD contrast. These remain development results; val has been examined repeatedly.")+'</p></section>',
            '<section><h2>'+t("主要性能：同一批患者横向比较", "Main performance: matched evaluation patients")+'</h2>',
            '<label>'+t("疾病 ", "Endpoint ")+'<select id="trait" onchange="update()"><option value="T2D">T2D</option><option value="cad">CAD</option><option value="bc">'+t("女性乳腺癌", "Female BC")+'</option></select></label>',
            '<label>'+t("评估集 ", "Assessment ")+'<select id="assessment" onchange="update()">',
        ]
        assessments = [("original_val", t("原始 val", "Original val")), ("exclude_shared_group", t("val：排除共享亲缘组", "Val: exclude shared groups")),
                       ("exclude_direct_0177", t("val：排除直接近亲", "Val: exclude direct relatives")), ("train_oof", t("训练集五折 OOF", "Training five-fold OOF"))]
        html += [f'<option value="{key}">{label}</option>' for key, label in assessments]
        html += ['</select></label><label>'+t("图中指标 ", "Plot metric ")+'<select id="metric" onchange="update()"><option value="auroc">AUROC ↑</option><option value="log_loss">Log loss ↓</option><option value="nagelkerke_r2">Nagelkerke R² ↑</option></select></label>',
            '<p id="cohort-note" class="muted"></p><div id="chart"></div><div id="metric-table" class="scroll"></div><p class="muted">'+t("图为点估计，横轴局部放大；缺少对应评估的版本不显示。主种子为 17，中心初始化的三种子结果见下方。名义配对置信区间在详细报告及 CSV 中；区间跨 0 不等于证明等效。", "The plot shows point estimates on a restricted axis. Versions without a matching assessment are omitted. Seed 17 is primary; the centroid comparison below shows all three seeds. Nominal paired intervals are in detailed reports and CSVs; crossing zero does not establish equivalence.")+'</p></section>',
            '<section><h2>'+t("M0_KD 怎么设计和训练", "How M0_KD is designed and trained")+'</h2>'+eq(0)+eq(1),
            '<p>'+t("Z 是已经计算好的 PRS（T2D 26、CAD 7、BC 5 个），不更新 SNP 权重。A 是 train 标准化的 PC1–PC10；C 为非年龄协变量（性别；女性 BC 中为常量）。祖源既进入基线，也控制 PRS 门控。年龄用 age_sd，三段非负斜率保证年龄基线不下降；M0 没有年龄×PRS 放大项。", "Z contains fixed PRSs (26 T2D, 7 CAD, 5 BC); SNP weights are not updated. A is train-standardized PC1–PC10. C contains non-age covariates (sex, constant in female BC). Ancestry enters both the baseline and the PRS gate. Age is age_sd, with three nonnegative piecewise slopes. M0 has no age-by-PRS amplification term.")+'</p>',
            '<div class="pipeline"><span>'+t("① train 内分组五折，生成 Teacher OOF 软标签", "1. Generate teacher targets using grouped folds within train")+'</span><span>'+t("② train 内选择 λ、正则化及步数", "2. Select lambda, penalties and steps within train")+'</span><span>'+t("③ 全 train 重拟合 Student；冻结后预测 val", "3. Refit student on all train; freeze and predict val")+'</span></div>'+eq(2),
            '<p>'+t("软标签 q 不含该患者所在折的训练标签。Teacher 的正式 val 分数则用全 train 作为 context，直接预测 val；训练 OOF 是另一个评估口径。主蒸馏 Teacher 一直是 TabPFN V3；3.5 和 Kumo 是独立对照，未用于本轮蒸馏。", "Each soft target excludes the patient’s fold from teacher context. For official-val scoring, the teacher uses all train as context and predicts val directly. Training OOF is a separate assessment. TabPFN V3 remains the distillation teacher; 3.5 and Kumo are independent controls and did not generate these KD targets.")+'</p><details><summary>'+t("正则化与实际训练预算", "Regularization and actual training budgets")+'</summary>'+eq(3),
            '<p>'+t("W̄、c̄ 为按 PRS 通道居中的门控参数。当前全 train M0_KD 使用 Adam，学习率 0.01，全批次训练，梯度范数上限 10；下表为训练内部选定的固定步数。中心初始化沿用相同预算。", "W-bar and c-bar are centered across PRS channels. Full-train M0_KD uses Adam, learning rate 0.01, full batches and gradient-norm clipping at 10. Fixed step counts below were selected within train and reused for centroid initialization.")+'</p>'+table(budgets.loc[budgets.scope.eq("val"), ["trait", "train_n", "prs_n", "lambda", "steps", "gate_ridge", "kappa_r"]]),
            '<p class="muted">'+t("batch size：T2D/CAD 5,811，BC 3,308；种子 17/29/43。嵌套 OOF 按各外折重算中心、软标签和训练内部选择；其 gate ridge 固定为 0.001，和最终全 train 流程的历史设置不同。", "Batch size: 5,811 for T2D/CAD and 3,308 for BC; seeds 17/29/43. Nested OOF rebuilds centers, targets and internal selection within each outer fold. Its gate ridge is fixed at 0.001, unlike the historical settings of the full-train procedure.")+'</p></details></section>',
            '<section><h2>'+t("祖源中心初始化：最新结果", "Centroid initialization: latest result")+'</h2><p>'+t("只更改 W、c 的初始值；中心由 train 内明确祖源组的 PC 均值估计，是原 GWAS 中心的代理。明确映射比例为 T2D 25/26、CAD 2/7、BC 4/5。", "Only initial W and c change. Centers are PC means of ancestry-assigned training patients, serving as proxies for the original GWAS centers. Mapping coverage is 25/26 T2D, 2/7 CAD and 4/5 BC PRSs.")+'</p>',
            '<img class="plot" alt="All three paired seeds on original validation" src="'+image_uri(CENT / "report/seed_stability.svg")+'">',
            '<p>'+t("BC seed 17：AUROC 0.70747→0.71470；Δ 的名义 95% 区间 [−0.00015, +0.01389]。原随机初始化 seed 29 已达 0.71465。训练五折 OOF 的全部 AUROC 差值绝对值小于 0.00008；两种亲缘筛选不能消除种子间不一致。", "BC seed 17: AUROC 0.70747 to 0.71470, with a nominal 95% interval of [-0.00015, +0.01389] for the difference. Random-init seed 29 already reaches 0.71465. All training five-fold OOF AUROC differences have absolute magnitude below 0.00008; kinship filtering does not resolve seed inconsistency.")+'</p>'+link(CENT / f"report/centroid_{lang}.html", t("查看初始化公式、全部指标和门控变化", "Initialization equations, all metrics and gate changes"))+'</section>',
            '<section><h2>'+t("已完成的实验，各自说明什么", "Completed experiments and their implications")+'</h2>']
        catalog = [
            (t("早期 QC 与全量训练", "Early QC versus full train"), t("匹配相同 QC val 时，全量训练 Teacher 得分恢复到历史记录；提示训练队列变化很重要，不能把差距都归因于版本。", "On the same QC-val patients, the full-train teacher recovers historical scores. Training-cohort changes matter; version alone does not explain the gap."), COHORT / "teacher_metrics.csv"),
            ("V1 / Phase A", t("固定 PRS、祖源门控和共同年龄尺度。T2D 的蒸馏改善更明确；共同尺度不保证总风险随年龄单调。", "Fixed PRSs, ancestry gate and a shared age scale. The KD benefit is clearest for T2D; a shared scale does not guarantee age-monotone total risk."), V1 / f"distillation_{lang}.html"),
            ("V3 sign-aware", t("M0 无年龄修饰；M1 共用、M2 逐 PRS 的符号感知年龄修饰。增加修饰未显示稳定的普遍收益。", "M0 has no age modifier; M1 shares one and M2 uses per-PRS sign-aware modifiers. Added modifiers do not show a stable general benefit."), V3 / f"signaware_v3_{lang}.html"),
            ("M0 / A6 / G / G_A6", t("比较三段/六段年龄基线与单调非线性 PRS。复杂度增加并未稳定优于简单 M0_KD。", "Compare three/six age segments and monotone nonlinear PRS transforms. Added complexity does not consistently outperform M0_KD."), BIO / f"report/val/bio_distillation_{lang}.html"),
            (t("嵌套 OOF 与低阶交互", "Nested OOF and low-order interactions"), t("完整 train 内重做选择；共同/逐 PRS 年龄修饰及更复杂选择流程，对 M0 的优势仍不稳定。", "Selection is repeated within train. Shared/per-PRS age modifiers and more complex selection policies do not show a stable advantage over M0."), BIO / f"report/nested/bio_distillation_{lang}.html"),
            (t("约束放松消融", "Constraint relaxation"), t("严格单调、仅年龄单调、仅 PRS 单调和无约束的低阶模型；未获得跨疾病稳定改善证据。", "Strict, age-only, PRS-only and unconstrained low-order models; no stable improvement across endpoints."), BIO / f"report/constraint_ablation_nested/constraint_ablation_{lang}.html"),
            (t("Teacher 对照与先验探查", "Teacher controls and prior probes"), t("V3、3.5、Kumo 无统一赢家。局部扰动探查揭示预测与单调先验的差异，不是生物因果验证。", "V3, 3.5 and Kumo have no universal winner. Local perturbations probe predictive monotonicity, not biological causality."), BIO / f"report/constraint_probe/constraint_probe_{lang}.html"),
            (t("仅删除 val 近亲", "Remove relatives from val only"), t("模型不重训；两种规则都保留亲缘未知者。Teacher–Student 差距随人群改变，不能据此证明泄漏或因果。", "No refitting; both filters retain unresolved patients. Teacher–student gaps change with the cohort, without establishing leakage or causality."), KIN / f"report/validation_filters_{lang}.html"),
            (t("普通多变量 logistic", "Plain multivariable logistic"), t("无正则 MLE 与 L2 对照，输入相同 PRS、年龄、性别和 PC1–PC10。当前所有方案都使用祖源变量。", "Unpenalized MLE and L2 controls use the same PRSs, age, sex and PC1–PC10. All current comparisons include ancestry variables."), LR / f"report/logistic_{lang}.html"),
        ]
        for name, description, path in catalog:
            html += ['<p><strong>'+escape(name)+'</strong> — '+description+' '+link(path, t("详情", "Details"))+'</p>']
        html += ['<p class="callout">'+t("命名澄清：此前加入年龄缩放的版本属于 Student / Phase A 的结构变化；正式 Phase B / Stage-2 评估尚未运行。不能把直接预测分数当成 Phase B 官方评分。", "Terminology: the age-scaling versions change the student within Phase A. Formal Phase B / Stage-2 evaluation has not run. Direct prediction scores are not official Phase B scores.")+'</p></section>',
            '<section><h2>'+t("亲缘规则和指标口径", "Kinship rules and metric definitions")+'</h2><p>'+t("规则 1：从 val 排除与 train 共享 KING≥0.0884 连通组的患者。规则 2：仅排除与 train 有直接 KING≥0.177 配对的 val 患者。连通组允许经中间患者连接，不表示组内每一对都是近亲。训练患者和模型均保持不变；未知亲缘者保留（T2D/CAD 42，BC 26）。", "Rule 1 removes val patients sharing a connected KING≥0.0884 group with train. Rule 2 removes only val patients directly paired with train at KING≥0.177. Group membership can be transitive, so not every pair within a group is close. Training patients and models stay fixed. Unresolved patients remain (42 T2D/CAD, 26 BC).")+'</p>',
            table(cohorts[["trait", "cohort", "training_n_unchanged", "n", "removed", "unresolved_retained"]]),
            '<p>'+t("AUROC 衡量排序；log loss 与 Brier 衡量概率预测质量；AP 辅助观察类别不平衡。R² 使用 Nagelkerke pseudo-R²，ℓp 为当前预测的总对数似然，ℓ0 为同一评估人群患病率基线的总对数似然。", "AUROC measures ranking; log loss and Brier assess probability quality; AP complements them under class imbalance. R-squared is Nagelkerke pseudo-R-squared: ell-p is total log likelihood of predictions and ell-0 is that of the prevalence-only baseline on the same assessment cohort.")+'</p>'+eq(4),
            '<p class="muted">'+t("Nagelkerke R² 不是遗传力或解释方差百分比；Teacher logit R² / probability R² 衡量蒸馏保真度，和疾病预测 R² 分开。所有年龄仍为 age_sd。置信区间使用可用亲缘组配对 bootstrap，条件于已固定的预测；未作多重比较校正。", "Nagelkerke R-squared is neither heritability nor a percentage of explained variance. Teacher logit/probability R-squared measures fidelity and is separate from disease-prediction R-squared. Age remains age_sd. Paired intervals resample available kinship groups, conditional on fixed predictions, without multiplicity adjustment.")+'</p></section>',
            '<section><h2>'+t("完整结果和核验", "Complete results and verification")+'</h2><p>'+t("所有已纳入版本、评估范围和种子保留在完整表中；历史 QC 队列单列，不混入当前排行榜。", "All included versions, assessment scopes and seeds remain in the complete table. Historical QC cohorts are separate from the current comparison.")+'</p><p><a href="all_metrics.csv">All metrics CSV</a> · <a href="main_metrics.csv">Main comparison CSV</a> · <a href="historical_cohort_control.csv">Historical cohort control</a> · <a href="VERIFIED.json">Verification receipt</a></p><details><summary>'+t("普通回归、蒸馏及初始化的配对区间", "Paired intervals for logistic, KD and initialization contrasts")+'</summary><p><a href="plain_logistic_paired_differences.csv">Logistic / M0 contrasts</a> · <a href="bio_paired_differences.csv">Architecture / KD contrasts</a> · <a href="centroid_paired_differences.csv">Centroid contrasts</a> · <a href="kinship_paired_differences.csv">Kinship contrasts</a></p></details><p class="muted">'+t("本总览只汇总已封存结果，没有新拟合。公式使用 KaTeX 排版 LaTeX，字体已内嵌，可离线显示；详细报告链接指向项目内原文件。", "This overview aggregates sealed results without new fits. Equations use KaTeX to typeset LaTeX, with embedded fonts for offline display; detail links point to the original project files.")+'</p></section></main>']
        data = json.dumps(json.loads(main_rows.to_json(orient="records", double_precision=15)), allow_nan=False)
        config = {"order": ORDER, "names": dict(zip(ORDER, NAMES, strict=True)), "notes": {
            "original_val": t("全 train → 原始 val。", "All train → original val."),
            "exclude_shared_group": t("只筛选 val；排除共享已知亲缘组，保留未知者。", "Val filtering only: exclude shared known groups; retain unresolved patients."),
            "exclude_direct_0177": t("只筛选 val；排除直接 KING≥0.177 亲缘对，保留未知者。", "Val filtering only: exclude direct KING≥0.177 pairs; retain unresolved patients."),
            "train_oof": t("训练集五折 OOF；不是官方 val 分数。不同流程的内层选择和正则化设置有区别。", "Five-fold training OOF, not official-val scores. Internal selection and regularization differ between procedures.")},
            "metrics": {"auroc": "AUROC ↑", "log_loss": "Log loss ↓", "nagelkerke_r2": "Nagelkerke R² ↑"}}
        compact, compact_config = compact_view(lang)
        comparisons = original_gaps.to_json(orient="records", double_precision=15)
        html += ['<script>const DATA='+data+';const CONFIG='+json.dumps(config, ensure_ascii=False)+';'
                 +'const COMPARISONS='+comparisons+';const COMPACT='+json.dumps(compact_config, ensure_ascii=False)+';'
                 +JS+COMPACT_JS+'</script></html>']
        page = ''.join(html).replace('<main>', math_style+'<main>', 1)
        start, end = page.index('<section>'), page.index('</main>')
        details_label = t("展开完整结果、公式和敏感性分析", "Expand all results, equations and sensitivity analyses")
        page = page[:start]+compact+'<details id="full-results"><summary>'+details_label+'</summary>'+page[start:end]+'</details>'+page[end:]
        check_latex(page, len(EQUATIONS))
        if lang == "en":
            assert not re.search(r"[\u4e00-\u9fff]", page)
        (OUT / f"overview_{lang}.html").write_text(page)
    # Inputs are unchanged by presentation generation.
    for relative, digest in provenance.items():
        assert sha256_file(ROOT / relative) == digest
    (OUT / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(OUT / "VERIFIED.json", {"status": "aggregate_metrics_verified", "new_fits": 0,
        "main_rows": len(main_rows), "archived_rows": len(all_metrics), "archived_families": sorted(all_metrics.family.unique()),
        "overlapping_metrics_agree_atol": 1e-12, "same_n_and_events_within_main_comparisons": True,
        "input_sha256": provenance, "script_sha256": sha256_file(__file__), "browser_render_tested": False,
        "math_renderer": math_metadata,
        "compact_default_models": 4, "full_results_collapsed_by_default": True,
        "output_sha256": {p.name: sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name not in {"VERIFIED.json", "FINAL_REVIEW.json"}}})
    print(f"Overview completed: {OUT}; {len(main_rows)} main rows / {len(all_metrics)} archived rows")


JS = r"""
function update(){
 const trait=document.getElementById('trait').value, assessment=document.getElementById('assessment').value, metric=document.getElementById('metric').value;
 const rows=DATA.filter(r=>r.trait===trait&&r.assessment===assessment).sort((a,b)=>CONFIG.order.indexOf(a.arm)-CONFIG.order.indexOf(b.arm));
 if(!rows.length)throw Error('Missing assessment');
 document.getElementById('cohort-note').textContent=CONFIG.notes[assessment]+' N='+rows[0].n.toLocaleString()+'; events='+rows[0].events.toLocaleString();
 const values=rows.map(r=>r[metric]), low=Math.min(...values), high=Math.max(...values), span=Math.max(high-low,.003), min=low-.12*span, max=high+.12*span;
 const x=v=>265+(v-min)/(max-min)*620, h=rows.length*32+58;
 let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 ${h}" role="img" aria-label="${CONFIG.metrics[metric]}"><rect width="1000" height="${h}" fill="white"/>`;
 for(let i=0;i<4;i++){const v=min+(max-min)*i/3;svg+=`<line x1="${x(v)}" y1="8" x2="${x(v)}" y2="${h-39}" stroke="#e4ebef"/><text x="${x(v)}" y="${h-15}" text-anchor="middle" font-size="13" fill="#516a7b">${v.toFixed(4)}</text>`;}
 rows.forEach((r,i)=>{const y=24+i*32, color=r.arm==='M0_KD'?'#087f8c':r.arm==='M0_CENTROID'?'#bf6d10':r.arm.startsWith('TabPFN')||r.arm==='Kumo_Tabular'?'#69569f':'#708393';svg+=`<text x="8" y="${y+5}" fill="#18354b" font-size="14">${CONFIG.names[r.arm]}</text><circle cx="${x(r[metric])}" cy="${y}" r="5.5" fill="${color}"/><text x="${x(r[metric])+11}" y="${y+5}" fill="#18354b" font-size="14">${r[metric].toFixed(4)}</text>`;});
 document.getElementById('chart').innerHTML=svg+'</svg>';
 const cols=['auroc','log_loss','nagelkerke_r2','brier','average_precision'];
 document.getElementById('metric-table').innerHTML='<table><thead><tr><th>Model</th><th>AUROC ↑</th><th>Log loss ↓</th><th>Nagelkerke R² ↑</th><th>Brier ↓</th><th>AP ↑</th></tr></thead><tbody>'+rows.map(r=>'<tr><td>'+CONFIG.names[r.arm]+'</td>'+cols.map(k=>'<td>'+r[k].toFixed(5)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
}
update();
"""


COMPACT_JS = r"""
function updateCompact(){
 const metric=document.getElementById('compact-metric').value;
 let table='<table><thead><tr><th>'+COMPACT.model_label+'</th>'+COMPACT.trait_names.map(n=>'<th>'+n+'</th>').join('')+'</tr></thead><tbody>';
 COMPACT.arms.forEach((arm,i)=>{
   const strong=arm==='M0_KD';
   table+='<tr><td>'+(strong?'<strong>':'')+COMPACT.names[i]+(strong?'</strong>':'')+'</td>';
   COMPACT.traits.forEach(trait=>{
     const row=DATA.find(r=>r.assessment==='original_val'&&r.trait===trait&&r.arm===arm);
     table+='<td>'+(strong?'<strong>':'')+row[metric].toFixed(4)+(strong?'</strong>':'')+'</td>';
   });table+='</tr>';
 });
 document.getElementById('compact-table').innerHTML=table+'</tbody></table>';
 const signed=x=>(x>=0?'+':'')+x.toFixed(5);
 document.getElementById('compact-deltas').innerHTML='<table><thead><tr><th>'+COMPACT.endpoint_label+'</th><th>'+COMPACT.delta_label+'</th><th>'+COMPACT.interval_label+'</th></tr></thead><tbody>'+COMPACT.traits.map((trait,i)=>{
   const row=COMPARISONS.find(r=>r.trait===trait&&r.metric===metric);
   return '<tr><td>'+COMPACT.trait_names[i]+'</td><td>'+signed(row.delta)+'</td><td>['+signed(row.ci_low)+', '+signed(row.ci_high)+']</td></tr>';
 }).join('')+'</tbody></table>';
}
updateCompact();
"""


class ResourceCheck(HTMLParser):
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img":
            assert a.get("alt") and a["src"].startswith("data:image/svg+xml;base64,")
        if tag == "a":
            assert (OUT / a["href"]).is_file()
        assert not (tag == "script" and a.get("src"))


if __name__ == "__main__":
    main()
    for lang in ["zh", "en"]:
        ResourceCheck().feed((OUT / f"overview_{lang}.html").read_text())
