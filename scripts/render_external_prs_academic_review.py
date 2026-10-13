"""Render the revised trait-specific PRS review from verified aggregate evidence."""
from __future__ import annotations

import base64
import html
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from decipehr.external_pgs import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/external_pgs/progress_review_20261012"
DATA = BASE / "academic_revision_20261012"
PRIMARY = BASE / "primary_disco_revision_20261012"
OUT = BASE / "html"
FIG = OUT / "figures"
CAD = ROOT / "outputs/external_pgs/cad_expansion_20261012"
NAMES = {"T2D": "T2D", "cad": "CAD", "bc": "女性乳腺癌"}
GROUPS = ["AFR-like", "AMR-like", "EAS-like", "EUR-like", "SAS-like", "AMBIGUOUS"]
COLORS = {"T2D": "#16786c", "cad": "#315f9c", "bc": "#9c4d70"}
SOURCES = {}
FIGURES = []


def load(name, folder=DATA):
    path = folder / name
    SOURCES[str(path.relative_to(ROOT))] = sha256_file(path)
    return pd.read_csv(path, float_precision="round_trip")


def table(headers, rows, identity="", caption=""):
    head = "".join(f"<th scope='col'>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table id='{identity}'><caption>{caption}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def small(text):
    return f"<small>{text}</small>"


def num(value, digits=6):
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def ci(row, adjusted=False, digits=4):
    prefix = "familywise_" if adjusted else ""
    return f"[{row[prefix + 'ci_low']:+.{digits}f}, {row[prefix + 'ci_high']:+.{digits}f}]"


def effect(row, adjusted=False):
    return f"{row.candidate_minus_reference:+.6f}{small(ci(row, adjusted))}"


def link_score(sid, label=None):
    return f"<a href='https://www.pgscatalog.org/score/{sid}/'>{html.escape(label or sid)}</a>"


def paper(doi, label="论文"):
    return f"<a href='https://doi.org/{doi}'>{label}</a>"


def download(path, title):
    SOURCES[str(path.relative_to(ROOT))] = sha256_file(path)
    payload = base64.b64encode(path.read_bytes()).decode()
    return f"<a class='download' download='{path.name}' href='data:application/octet-stream;base64,{payload}'>{title}</a>"


def figure(path, title, caption):
    SOURCES[str(path.relative_to(ROOT))] = sha256_file(path)
    payload = base64.b64encode(path.read_bytes()).decode()
    FIGURES.append(path.name)
    return f"<figure><h3>{title}</h3><img alt='{html.escape(caption)}' src='data:image/svg+xml;base64,{payload}'><figcaption>{caption} <a download='{path.name}' href='data:image/svg+xml;base64,{payload}'>SVG</a></figcaption></figure>"


def section(identity, title, content):
    return f"<section id='{identity}'><h2>{title}</h2>{content}</section>"


def details(title, content):
    return f"<details><summary>{title}</summary><div class='detail-body'>{content}</div></details>"


def make_figures(metrics, pairs):
    font = Path("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
    if font.exists():
        fm.fontManager.addfont(font)
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Droid Sans Fallback"], "font.size": 10,
                         "svg.fonttype": "path", "pdf.fonttype": 42, "svg.hashsalt": "prs-academic-20261012"})
    FIG.mkdir(exist_ok=True)
    methods = [("covariates_lr", "协变量 Logistic"), ("best_single_lr", "Single-best PRS Logistic"),
               ("full_lr", "Multi-PRS Logistic"), ("tabpfn_full", "Multi-PRS TabPFN")]
    fig, axes = plt.subplots(3, 2, figsize=(12.3, 8.5))
    for i, disease in enumerate(NAMES):
        for j, key in enumerate(["auc", "nagelkerke_r2"]):
            ax = axes[i, j]
            sub = metrics.loc[metrics.disease.eq(disease) & metrics.ancestry_group.eq("ALL")].set_index("method")
            vals = [sub.loc[m, key] for m, _ in methods]
            span = max(max(vals) - min(vals), .012)
            for y, (method, _), value in zip(range(3, -1, -1), methods, vals, strict=True):
                color = COLORS[disease] if method == "tabpfn_full" else "#657681"
                ax.scatter(value, y, color=color, s=48, zorder=3)
                ax.annotate(f"{value:.4f}", (value, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
            ax.set(yticks=range(3, -1, -1), yticklabels=[n for _, n in methods] if j == 0 else [""]*4,
                   xlim=(min(vals)-span*.12, max(vals)+span*.45), ylim=(-.6, 3.6),
                   title=f"{NAMES[disease]} · {'AUC' if j == 0 else 'Nagelkerke R²'}")
            ax.grid(axis="x", alpha=.17)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=.23, right=.98, top=.95, bottom=.06, hspace=.54, wspace=.17)
    save(fig, "07_primary_baselines")
    for comparison, name in [("full_vs_single_lr", "08_ancestry_single"), ("full_vs_disco", "09_ancestry_disco")]:
        fig, axes = plt.subplots(3, 2, figsize=(12.3, 10.0))
        for i, disease in enumerate(NAMES):
            for j, metric in enumerate(["auc", "nagelkerke_r2"]):
                ax = axes[i, j]
                ax.axvline(0, color="#6e7d87", lw=1)
                rows = pairs.loc[pairs.disease.eq(disease) & pairs.comparison.eq(comparison) & pairs.metric.eq(metric)].set_index("ancestry_group")
                for y, group in zip(range(5, -1, -1), GROUPS, strict=True):
                    row = rows.loc[group]
                    ax.plot([row.ci_low, row.ci_high], [y, y], color=COLORS[disease], lw=1.5)
                    ax.scatter(row.candidate_minus_reference, y, color=COLORS[disease], s=29)
                labels = []
                for group in GROUPS:
                    r = rows.loc[group]
                    flag = "*" if min(r.cases, r.n-r.cases) < 20 else ""
                    labels.append(f"{group}{flag}  ({int(r.n)}/{int(r.cases)})")
                ax.set(yticks=range(5, -1, -1), yticklabels=labels if j == 0 else [""]*6,
                       ylim=(-.7, 5.7), title=f"{NAMES[disease]} · {'ΔAUC' if j == 0 else 'ΔNagelkerke R²'}")
                ax.set_xlabel("TabPFN − " + ("Single-best PRS Logistic" if comparison == "full_vs_single_lr" else "Disco"))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.grid(axis="x", alpha=.17)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.tick_params(axis="y", length=0)
        fig.subplots_adjust(left=.24, right=.98, top=.96, bottom=.06, hspace=.48, wspace=.16)
        save(fig, name)


def save(fig, name):
    for ext in ["svg", "png", "pdf"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight", metadata={"Creator": "decipEHR"})
    plt.close(fig)


def resource_section():
    for path in [BASE / "external_resource_populations.json", CAD / "weights.new.lock.json",
                 CAD / "resource_review.yaml",
                 ROOT / "outputs/external_pgs/cad_population_candidates_20261012/shim2023_arab.xml"]:
        SOURCES[str(path.relative_to(ROOT))] = sha256_file(path)
    inventory = load("external_resource_inventory.csv", BASE)
    qc = load("score_coverage.csv", CAD / "report")
    coverage = dict(zip(inventory.pgs_id, inventory.coverage, strict=True))
    coverage.update(dict(zip(qc.model, qc.coverage, strict=True)))
    def model(sid, name, method, source, tuning, reason, doi):
        return [f"<b>{name}</b>{small(link_score(sid))}", method, source, tuning,
                reason + small(f"覆盖率 {coverage[sid]:.2%} · {paper(doi)}")]
    headers = ["外部模型", "模型方法", "GWAS 发现人群", "模型调优 / 参考人群", "纳入理由与覆盖率"]
    t2d = [
        ["<b>D-PRISM：25 个成分</b>" + small(link_score("PGS005353", "PGS005353–PGS005377")),
         "PRS-CSx；5 个来源祖源 × 5 个调优目标",
         "AFR 128,695；AMR 82,013；EAS 400,683；EUR 1,486,934；SAS 87,223。包括 MVP、BioMe、SIGMA、BBJ、UKB、FinnGen、LOLIPOP 等。",
         "AFR：BioVU；AMR：SIGMA；EAS：BBJ；EUR：BioVU；SAS：LOLIPOP。每个目标模型使用全部五类来源。",
         "提供可直接输入的跨祖源分量。" + small("覆盖率 94.83%–98.86% · <a href='https://github.com/ahuertach/D-PRISM-T2D-PRS'>作者代码</a>")],
        model("PGS002308", "PRScsx_T2D", "PRS-CSx auto/meta，多祖源综合分数",
              "EUR 898,130；日本 EAS 177,415；AFR 23,827。", "Catalog 未单列独立表型调优队列。",
              "补充另一套外部综合模型；形成主面板第 26 个输入。", "10.1186/s13073-022-01074-2"),
    ]
    targets = [["AFR", "PGS005353–PGS005357", "BioVU，非洲裔美国人", "6,779 / 1,409"],
               ["AMR", "PGS005358–PGS005362", "SIGMA，Hispanic American / Mexican", "5,747 / 2,964"],
               ["EAS", "PGS005363–PGS005367", "BBJ，日本人群", "2,837 / 906"],
               ["EUR", "PGS005368–PGS005372", "BioVU，欧洲裔美国人", "21,706 / 3,694"],
               ["SAS", "PGS005373–PGS005377", "LOLIPOP，British Indian", "5,715 / 2,019"]]
    cad = [
        model("PGS003725", "GPS_Mult", "LDpred2 后整合多祖源、多性状信息",
              "AFR、AMR、EAS、EUR、SAS；Catalog 登记 1,443,386。", "UKB 欧洲祖源 116,649（4,412 病例）。",
              "原面板：多祖源发现、欧洲人群调优的综合模型。", "10.1038/s41591-023-02429-x"),
        model("PGS004696", "multi_anc_hg37CSx", "PRS-CSx，多祖源综合模型",
              "EUR 773,268；AFR 76,709；AMR 30,648；EAS 212,453；BBJ、CARDIoGRAMplusC4D、MVP、UKB。",
              "MVP 87,724：EUR、AFR、AMR 和 EAS/SAS。", "原面板：补充混合祖源调优的模型。", "10.1161/circgen.123.004272"),
        model("PGS012558", "CAD_EAS_MIXPRSplus", "MIXPRS+；综合 JointPRS、SDPRX、SBayesRC、ClinicalPRS 等",
              "EUR 86,995；BBJ EAS 212,453。", "面向 EAS；Catalog 未单列独立调优队列。",
              "原面板：东亚目标综合模型。", "10.1038/s41588-026-02637-4"),
        model("PGS003727", "GPS_CADEUR（新增）", "LDpred2，单一 EUR 来源模型",
              "CARDIoGRAMplusC4D 等 EUR 504,636（86,847 病例）；发现数据排除 UKB。",
              "UKB 欧洲祖源 116,649（4,412 病例）。", "加入与 GPS_Mult 对应的 EUR 模型。", "10.1038/s41591-023-02429-x"),
        model("PGS005143", "CAD PRS-CS EAS（新增）", "PRS-CS auto",
              "东亚 GWAS，51,442；GCST010480。", "未单列表型调优；韩国 HEXA 属于评估人群。",
              "加入 EAS GWAS 来源的独立分数。", "10.1038/s42003-025-07767-9"),
        model("PGS000296", "GPS_CAD_SA（新增）", "LDpred；ρ=0.003，1000G EUR LD",
              "多祖源 GWAS 187,599，约 75.3% EUR、13.6% SAS。",
              "UKB 南亚 7,244（398 病例）调参；另有 GenomeAsia100K 1,522 人作参考。",
              "南亚优化模型；不等于纯 SAS GWAS 权重。", "10.1016/j.jacc.2020.06.024"),
        model("PGS003866", "CAD_lassosum2_ARB（新增）", "lassosum2；1000G EUR LD",
              "外部 GWAS GCST005195，547,261；Catalog 未标注其祖源比例。",
              "沙特 Arab 队列训练集 2,700；验证集 2,699，实际可用人数随缺失值变化。",
              "Arab 目标优化模型；不等于 Arab GWAS 来源。", "10.1038/s41467-023-41985-1"),
    ]
    bc = [
        model("PGS005378", "Overall_Prop_M1", "前向选择 + PRSice-2 + elastic net",
              "AFR 女性 25,334（12,174 病例）；AABC、AABCG、AMBER、BCAC、GBHS、ROOT。",
              "AFR 女性 28,953（13,913 病例）；来源队列同上。", "提供非洲祖源女性模型。", "10.1038/s41588-026-02501-5"),
        model("PGS004869", "INTERVENE_MegaPRS", "MegaPRS auto",
              "欧洲祖源乳腺癌 GWAS；Catalog 登记 139,274。",
              "404 名非芬兰欧洲祖源 1000G 样本为遗传参考，不是表型训练病例。",
              "提供欧洲祖源 genome-wide 模型。", "10.1038/s41467-024-48938-2"),
        model("PGS018538", "TPMI PRS-CS", "PRS-CS；φ=10⁻⁵",
              "台湾汉族 TPMI 326,227（12,566 病例）。",
              "TPMI 汉族 73,949（4,385 病例）。", "提供东亚来源和调优模型。", "10.1038/s41586-025-09350-y"),
    ]
    body = "<p>三种性状分别构建外部分数面板。区分 <b>GWAS 发现人群</b>、<b>模型调优人群</b>及<b>遗传参考人群</b>；样本量按各资源登记，不能跨模型相加为独立参与者总数。</p>"
    body += "<h3>2.1 T2D：26 个输入</h3>" + table(headers, t2d, "resources-t2d")
    body += "<p>D-PRISM 的 25 个输入为 <b>5 个目标调优设置 × 5 个 GWAS 来源分量</b>。它们不是 25 个独立 GWAS。主模型直接接收全部成分；发表的五个 metascores 仅用于单分数候选、Disco 适配和压缩分析。</p>"
    body += details("D-PRISM 五个调优目标及样本量", table(["调优目标", "成分编号", "队列与人群", "人数 / 病例"], targets, "dprism-targets"))
    body += "<h3>2.2 CAD：7 个输入（原 3 个 + 新增 4 个）</h3>" + table(headers, cad, "resources-cad")
    body += "<p>本轮保留已核验的 7 个模型；此前讨论的 AoU 来源候选组未纳入。SAS 和 Arab 优化分数扩展了目标人群覆盖，但当前 CAD 面板仍不是完整的祖源特异 GWAS 分量矩阵。</p>"
    body += "<h3>2.3 女性乳腺癌：3 个输入</h3>" + table(headers, bc, "resources-bc")
    body += "<p>所有乳腺癌结果限制于符合资格的女性。原候选 PRS313（PGS000004，67.73%）和 PGS005379（74.98%）未达到预先固定的 75% 变异匹配门槛，因此未参与比较。各分数使用已核验的 GRCh38 对齐与等位基因匹配；覆盖率反映已发表变异的匹配比例，不代表效应权重或预测信号的保留比例。</p>"
    return body


def history_section(metrics):
    historical = load("historical_wrapper_metrics.csv", BASE)
    audit = load("cohort_and_source_audit.csv", ROOT / "outputs/external_pgs/run_20261010/comparison_with_previous_official_validation")
    assert audit.identical_patients_and_labels.all()
    auc_pairs = load("historical_wrapper_auc_comparisons.csv", BASE)
    r2_pairs = load("paired_r2.csv", OUT)
    rows, deltas = [], []
    local = {"T2D": "P+T、LDpred2、lassosum2、PRS-CS、SDPR、MegaPRS（有 / 无注释）、SBayesRC",
             "cad": "lassosum2、LDpred2、PRS-CS、SDPR",
             "bc": "P+T、LDpred2、lassosum2、PRS-CS、SDPR"}
    for disease in NAMES:
        for method, label in [("previous_wrapper", "此前本地构建 PRS + TabPFN + wrapper"),
                              ("external_pc_wrapper", "外部 PRS 原面板 + TabPFN + wrapper")]:
            r = historical.loc[historical.disease.eq(disease) & historical.model.eq(method)].iloc[0]
            panel = local[disease] if method == "previous_wrapper" else {"T2D": "26 分数", "cad": "原 3 分数", "bc": "3 分数"}[disease]
            rows.append([NAMES[disease], label, panel, num(r.auc), num(r.nagelkerke_r2)])
        r = metrics.loc[metrics.disease.eq(disease) & metrics.method.eq("tabpfn_full") & metrics.ancestry_group.eq("ALL")].iloc[0]
        rows.append([NAMES[disease], "<b>当前完整外部 PRS + TabPFN（直接概率）</b>", {"T2D": "26 分数", "cad": "扩展 7 分数", "bc": "3 分数"}[disease], f"<b>{r.auc:.6f}</b>", f"<b>{r.nagelkerke_r2:.6f}</b>"])
        au = auc_pairs.loc[auc_pairs.disease.eq(disease) & auc_pairs.ancestry_group.eq("ALL") & auc_pairs.comparison.eq("external_pc_wrapper_vs_previous")].iloc[0]
        r2 = r2_pairs.loc[r2_pairs.disease.eq(disease) & r2_pairs.scope.eq("historical_wrapper") & r2_pairs.comparison.eq("external_pc_wrapper_vs_previous")].iloc[0]
        deltas.append([NAMES[disease], effect(au), ci(au, True), effect(r2), ci(r2, True)])
    body = "<p>此前的本地方案是使用 P+T、LDpred2、PRS-CS 等方法在本地构建 PRS，再用 TabPFN 整合。与直接使用作者已发表的冻结权重相比，PRS 构建来源与下游设置均有变化。保存的核验表确认三种性状的新旧预测使用相同 development 患者和结局，女性乳腺癌仍为 1,628 人。</p>"
    body += table(["性状", "预测流程", "PRS 输入", "AUC", "Nagelkerke R²"], rows, "history-main", "并列保留本地历史、外部原面板 wrapper、当前完整模型三种版本")
    body += "<p><b>如何读取：</b>加粗行与第 3、4 节主模型完全相同。当前完整流程的 AUC / R² 点估计均高于此前本地流程；但历史流程有额外协变量校准 wrapper，不能将差值解释成仅替换 PRS 权重的效应。下表沿用双方均经过 wrapper 的已完成配对实验，其 CAD 输入为原 3 分数。</p>"
    body += table(["性状", "ΔAUC [95% CI]", "AUC：原 6 项同时区间", "ΔR² [95% CI]", "R²：原 29 项同时区间"], deltas, "history-deltas", "外部原面板 wrapper − 本地历史 wrapper；两项指标沿用各自已封存检验族")
    body += "<p>T2D 的历史 AUC 增量在同时区间下仍跨零。历史模型由过去 development 表现回顾性选出；wrapper 的 OOF 折数不同，女性乳腺癌的 checkpoint、estimators 和概率平衡设置也不同。该证据回答整套流程是否改善，不隔离权重来源或祖源机制。</p>"
    return body


def ablation_section():
    current = load("current_auc_comparisons.csv", BASE)
    hard = load("hard_ancestry_comparisons.csv", BASE)
    soft = load("soft_ancestry_comparisons.csv", BASE)
    r2 = load("paired_r2.csv", OUT)
    definitions = [
        ("full_vs_covariates", "加入 PRS − 仅协变量 TabPFN", "current_development", "外部分数提供增量预测信息。"),
        ("full_vs_single_tabpfn", "完整面板 − 最佳单 PRS TabPFN", "current_development", "三病 AUC 区间均跨零，未证明多个分数本身必需；不等同于与 single-best Logistic 的主比较。"),
        ("continuous_vs_no_pc", "连续 PC − 无 PC 的 TabPFN", "current_development", "支持保留连续 PC；无 PC 的 PRS 本身仍可能携带祖源信息。"),
        ("varying_vs_fixed", "祖源可变系数 − 固定系数", "current_development", "相同灵活基础项下未见稳定收益，不能据此断言 TabPFN 没有交互。"),
        ("hard_vs_continuous", "Hard ancestry 替换连续 PC", "current_development", "三病性能下降；参考标签与原 PC 来源不同，并非单纯数值分箱。"),
        ("full_vs_pc", "PC 之外加入 soft ancestry", "soft_ancestry", "未发现额外稳定收益；soft 为参考相似性，不是祖源比例。"),
    ]
    body = "<p>下面完整列出已完成的六类机制消融，均为原 <b>T2D / CAD / 女性乳腺癌 = 26 / 3 / 3</b> 面板。所有差值为表中左侧变更减右侧参照；展示名义 95% 配对区间。CAD 7 分数的同系列消融尚未运行。</p>"
    body += table(["已完成问题", "能够说明什么"], [[label, meaning] for _, label, _, meaning in definitions], "ablation-interpretation")
    body += figure(FIG / "05_ablations.svg", "图 4 · 六类消融：AUC 与 R²", "原 26 / 3 / 3 面板，名义 95% 配对区间。AUC 沿用各冻结实验；R² 为已完成的 29 项探索性补充。不同检验族不合并为一次确认性检验。")
    for disease in NAMES:
        rows = []
        for comparison, label, scope, _ in definitions:
            source = hard if comparison == "hard_vs_continuous" else soft if comparison == "full_vs_pc" else current
            source_comparison = "hard_minus_pc" if comparison == "hard_vs_continuous" else comparison
            au = source.loc[source.disease.eq(disease) & source.comparison.eq(source_comparison)].iloc[0].copy()
            au["candidate_minus_reference"] = au.get("candidate_minus_reference", au.get("delta"))
            rs = r2.loc[r2.disease.eq(disease) & r2.comparison.eq(comparison) & r2.scope.eq(scope)].iloc[0]
            rows.append([label, effect(au), effect(rs), ci(rs, True)])
        body += f"<h3>{NAMES[disease]}：六类消融完整结果</h3>"
        body += table(["变更 − 参照", "ΔAUC [95% CI]", "ΔR² [95% CI]", "R²：29 项同时区间"], rows, f"ablation-{disease}")
    p25 = load("dprism25_auc_comparisons.csv", BASE)
    candidates = p25.loc[p25.evaluation.eq("development") & p25.ancestry_group.eq("ALL")]
    # Use the stored candidate/reference identities to identify the compression contrast.
    au = candidates.loc[candidates.candidate_model.eq("tabpfn_dprism25") & candidates.reference_model.eq("tabpfn_matched")].iloc[0]
    comparison = au.comparison
    rs = r2.loc[r2.scope.eq("dprism25_development") & r2.comparison.eq(comparison)].iloc[0]
    body += "<h3>第七类：T2D 的 25 成分是否可以压成 5 个 metascores？</h3>"
    body += table(["同预处理 TabPFN 比较", "ΔAUC [95% CI]", "ΔR² [95% CI]"], [["直接 25 成分 − 5 个发表 metascores", effect(au), effect(rs)]], "ablation-compression")
    body += "<p>支持直接保留 25 成分：特定发表压缩会损失当前 TabPFN 可利用的信息。原 26 分数主模型与这个 25 分数模型的 PC / 预处理不同，二者不是单独删去 PGS002308 的消融。</p>"
    body += "<h3>第八类与补充敏感性</h3><p><b>CAD 3→7 分数扩展：</b>完整结果与配对区间见第 7 节；不能把 development 的提升写成已经跨折稳定。<b>QC / 亲缘敏感性：</b>此前额外过滤版本的 training OOF AUC 为 T2D 0.7369、CAD 0.7863、女性 BC 0.6827；人群及训练上下文发生变化，不能放入同队列主表或当作纯单因素消融。<b>尚未完成：</b>CAD 7 分数的完整机制消融、无预设五维压缩的 Disco 扩展、学生蒸馏及独立外部队列验证。</p>"
    all_metrics = load("current_development_metrics.csv", BASE)
    other_rows = []
    for comparison, label in [("fixed_genetic_increment", "灵活基础项中加入固定 PRS 项"),
                              ("full_vs_lr", "完整 TabPFN − Multi-PRS Logistic"),
                              ("full_vs_catboost", "完整 TabPFN − CatBoost"),
                              ("matched_vs_fixed", "匹配 TabPFN − 匹配固定系数")]:
        for disease in NAMES:
            r = current.loc[current.disease.eq(disease) & current.comparison.eq(comparison)].iloc[0].copy()
            r["candidate_minus_reference"] = r.delta
            m = all_metrics.loc[all_metrics.disease.eq(disease)].set_index("method")
            delta_r2 = m.loc[r.candidate, "nagelkerke_r2"] - m.loc[r.reference, "nagelkerke_r2"]
            other_rows.append([label, NAMES[disease], effect(r), f"{delta_r2:+.6f}"])
    body += "<h3>其他已完成对照：整合器与固定遗传项</h3>"
    body += table(["变更 − 参照", "性状", "ΔAUC [95% CI]", "ΔR² 点估计"], other_rows, "ablation-other", "原 26 / 3 / 3 面板；这些附加对照的 R² 配对区间尚未计算，不作相应显著性判断")
    rows = [[NAMES[r.disease], r.method, num(r.auc), num(r.nagelkerke_r2)] for r in all_metrics.itertuples()]
    body += details("附录：原面板全部 18 个配置 × 3 性状的性能", table(["性状", "保存的配置名称", "AUC", "R²"], rows, "all-original-configurations"))
    return body


def build(metrics, pairs, choices):
    def point(disease, method, group="ALL"):
        rows = metrics.loc[metrics.disease.eq(disease) & metrics.method.eq(method) & metrics.ancestry_group.eq(group)]
        assert len(rows) == 1
        return rows.iloc[0]
    def pair(disease, comparison, metric, group="ALL"):
        rows = pairs.loc[pairs.disease.eq(disease) & pairs.comparison.eq(comparison) & pairs.metric.eq(metric) & pairs.ancestry_group.eq(group)]
        assert len(rows) == 1
        return rows.iloc[0]
    selected_names = {"T2D": "D-PRISM EUR 目标 metascore", "cad": "GPS_Mult · PGS003725", "bc": "MegaPRS · PGS004869"}
    intro = """<header id='overview'><p class='eyebrow'>decipEHR · 2026 年 10 月 12 日更新</p><h1>External PRS 整合研究</h1><p class='subtitle'>模型来源、单分数基线、Disco 对照与祖源分层评价</p><p>研究问题：冻结已发表的 PRS 权重后，多分数 TabPFN 能否超越传统单分数风险模型；其表现相对现有 Disco 适配如何；总体结果是否在不同祖源相似性亚组中一致。</p><div class='note'><b>本版重点：</b>主结果采用 T2D / CAD / 女性乳腺癌的 26 / 7 / 3 分数面板，并将不使用 TabPFN 的 single-best PRS Logistic 设为明确参照。主表统一使用当前完整模型比较 single-best Logistic 和 Disco；另列本地历史流程、完整消融及探索性亚组。</div></header>"""
    design = """<p>外部 SNP 权重固定；本项目标签仅用于下游整合、校准与训练内选择。主要评价采用完整 training 拟合后预测既定 development。Development 已参与历史研究开发，因此本报告属于内部开发评价。</p>"""
    design += table(["性状", "Training：人数 / 病例", "Development：人数 / 病例", "当前完整面板"], [
        ["T2D", "5,811 / 1,701", "2,852 / 847", "26"],
        ["CAD", "5,811 / 2,313", "2,852 / 1,072", "7"],
        ["女性乳腺癌", "3,308 / 602", "1,628 / 320", "3"]], "cohorts")
    design += """<p><b>Single-best PRS Logistic：</b>一个外部分数 + 年龄 + 性别 + PC1–PC10，使用正则化 Logistic 回归；女性队列的常量性别列去除。PRS 身份和 C 均仅由 training 内三折 pooled AUC 选择，log loss 用作并列判据。T2D 候选包含 26 个输入及 5 个发表 metascores；CAD 在全部 7 个候选中重新选择；乳腺癌在 3 个候选中选择。</p><p><b>Multi-PRS TabPFN：</b>完整外部分数面板 + 相同协变量，使用固定 TabPFN v3 checkpoint、16 estimators、seed 42。主结果使用直接预测概率。Single-PRS TabPFN 是另一个消融对照，不能替代上述传统单分数基线。</p><p><b>指标：</b>AUC 衡量排序；Nagelkerke R² 衡量预测概率相对截距模型的似然改善，包含协变量贡献，不是遗传率或 PRS 独立解释率，且可为负。Brier、log loss 和校准指标见下载表。所有概率指标直接在保留评估集计算，不在该集合重新拟合校准器。</p><p><b>不确定性：</b>主比较为完整 TabPFN 对 single-best Logistic / Disco，两类比较 × 三个性状 × 两项指标共 12 项总体差异，采用 2,000 次结局分层配对患者 bootstrap，提供普通 95% 区间及 Bonferroni 同时区间。亚组区间为未经多重校正的探索性结果。区间条件于保存预测，未覆盖重训练、历史开发选择及家系依赖的不确定性。</p>"""
    parts = [section("design", "1 · 研究设计与评价口径", design), section("resources", "2 · 按性状整理外部 PRS 模型", resource_section())]
    primary_rows, delta_rows, model_rows = [], [], []
    for disease in NAMES:
        a, b = point(disease, "tabpfn_full"), point(disease, "best_single_lr")
        primary_rows.append([NAMES[disease], selected_names[disease], num(b.auc), num(b.nagelkerke_r2), f"<b>{a.auc:.6f}</b>", f"<b>{a.nagelkerke_r2:.6f}</b>"])
        au, r2 = pair(disease, "full_vs_single_lr", "auc"), pair(disease, "full_vs_single_lr", "nagelkerke_r2")
        delta_rows.append([NAMES[disease], effect(au), ci(au, True), effect(r2), ci(r2, True)])
        for method, label in [("covariates_lr", "协变量 Logistic"), ("best_single_lr", "Single-best PRS Logistic"),
                              ("full_lr", "Multi-PRS Logistic"), ("tabpfn_full", "Multi-PRS TabPFN")]:
            r = point(disease, method)
            model_rows.append([NAMES[disease], label, num(r.auc), num(r.nagelkerke_r2), num(r.brier), num(r.log_loss)])
    body = "<p>表中最佳单分数由 training 选择，在 development 固定评估。T2D 的一个 metascore 已融合五种来源，仍是一个进入 Logistic 的分数；本比较不将其解释为单一祖源模型。</p>"
    body += table(["性状", "Training 选定的单分数", "Single AUC", "Single R²", "TabPFN AUC", "TabPFN R²"], primary_rows, "primary-main", "主要参照：Single-best PRS Logistic（不使用 TabPFN）")
    body += table(["性状", "ΔAUC [95% CI]", "AUC 同时区间", "ΔR² [95% CI]", "R² 同时区间"], delta_rows, "primary-deltas", "Δ = 完整面板 TabPFN − Single-best PRS Logistic；同时区间覆盖本版 12 项总体比较")
    body += "<p><b>主要结果：</b>相对不使用 TabPFN 的最佳单 PRS Logistic，三个性状的 AUC 和 R² 均提高，且本版 12 项比较的同时区间均高于零。该结果支持当前 development 中完整预测流程的增量价值。</p>"
    body += figure(FIG / "07_primary_baselines.svg", "图 1 · 单分数与多分数整合的整体性能", "同一性状使用相同 development 患者。图为点估计；配对差异区间见上表。CAD 使用扩展 7 分数，其余为 26 / 3 分数。")
    body += details("补充：协变量、线性整合与 TabPFN 的完整指标", table(["性状", "模型", "AUC", "R²", "Brier", "Log loss"], model_rows, "baseline-all"))
    body += "<p>此比较回答完整预测流程相对传统单分数模型的增量，增量同时包含整合器的非线性能力及输入数量差异。多 PRS 是否必要，仍须结合相同 TabPFN 下的单分数消融解释。</p>"
    parts.append(section("single", "3 · 主要比较：Single-best PRS Logistic", body))
    disco_rows, disco_deltas = [], []
    for disease in NAMES:
        a, b = point(disease, "tabpfn_full"), point(disease, "selected_disco")
        scope = {"T2D": "TabPFN：完整 26 分数；Disco：25 成分形成的 5 个发表 metascores", "cad": "TabPFN：完整 7 分数；Disco：GPS_Mult + EAS MIXPRS+ 两分数", "bc": "双方均 3 分数：AFR / EUR / EAS；PC 与预处理不同"}[disease]
        disco_rows.append([NAMES[disease], scope, num(b.auc), num(b.nagelkerke_r2), num(a.auc), num(a.nagelkerke_r2)])
        au, r2 = pair(disease, "full_vs_disco", "auc"), pair(disease, "full_vs_disco", "nagelkerke_r2")
        disco_deltas.append([NAMES[disease], effect(au), ci(au, True), effect(r2), ci(r2, True)])
    body = "<p>DiscoDivas 根据个体与模型参考人群在共同 PC 空间中的距离组合多个 PRS。当前结果复用官方插值实现，采用 1000G 代理中心；风险链接（线性 / 样条）及正则化由 training 内选择。" + paper("10.1016/j.ajhg.2026.05.006", "AJHG 2026 论文") + " · <a href='https://github.com/YunfengRuan/DiscoDivas'>官方代码</a>。</p>"
    body += table(["性状", "输入来源及表示", "Disco AUC", "Disco R²", "TabPFN AUC", "TabPFN R²"], disco_rows, "disco-main")
    body += table(["性状", "ΔAUC [95% CI]", "AUC 同时区间", "ΔR² [95% CI]", "R² 同时区间"], disco_deltas, "disco-deltas", "Δ = 表中对应版本 TabPFN − 训练内选定 Disco；本版重新计算统一检验族")
    conclusions = []
    for disease in NAMES:
        outcomes = []
        for metric, label in [("auc", "AUC"), ("nagelkerke_r2", "R²")]:
            r = pair(disease, "full_vs_disco", metric)
            outcomes.append(label + ("同时区间高于零" if r.familywise_ci_low > 0 else "同时区间跨零"))
        conclusions.append(NAMES[disease] + "：" + "、".join(outcomes))
    body += "<p><b>结果解释：</b>" + "；".join(conclusions) + "。所有区间已按当前完整模型的保存预测重新计算。</p>"
    body += "<div class='note'><b>主比较使用当前完整模型：</b>与第 3 节完全相同的 26 / 7 / 3 分数 TabPFN，对比 training 内在 linear / spline 和 C 中选定的 Disco。这里评估整套流程，双方输入数、PC 和预处理并非全部一致；不能单独归因于整合算法。5 个 metascores 并非 Disco 输入上限，CAD 7 分数 Disco 尚未运行。</div><p>原文 AoU 调优成品模型仍未运行；当前为复用官方插值实现、采用 1000G 代理中心的本项目适配。下方保留控制来源 / 输入的辅助比较，不再用其替换完整主模型。</p>"
    aux_rows = []
    for disease in NAMES:
        a, b = point(disease, "tabpfn_disco_comparator"), point(disease, "selected_disco")
        aux_rows.append([NAMES[disease], {"T2D": "直接 25 成分 / 5 metascores", "cad": "共同 2 分数", "bc": "共同 3 分数"}[disease], num(b.auc), num(b.nagelkerke_r2), num(a.auc), num(a.nagelkerke_r2)])
    body += details("辅助分析：相同来源或匹配输入，参考 PC 与残差化一致", table(["性状", "TabPFN / Disco 输入", "Disco AUC", "Disco R²", "TabPFN AUC", "TabPFN R²"], aux_rows, "disco-matched-aux"))
    m25 = load("dprism25_metrics.csv", BASE)
    rows = []
    for method, label in [("tabpfn_dprism25", "直接 25 成分 TabPFN"), ("tabpfn_matched", "5 metascores TabPFN"), ("selected_disco", "5 metascores Disco")]:
        a = m25.loc[m25.evaluation.eq("development") & m25.method.eq(method)].iloc[0]
        b = m25.loc[m25.evaluation.eq("training_oof") & m25.method.eq(method)].iloc[0]
        rows.append([label, num(a.auc), num(a.nagelkerke_r2), num(b.auc), num(b.nagelkerke_r2)])
    body += details("T2D：25 成分、5 metascores 与共享五折 OOF", table(["流程", "Dev AUC", "Dev R²", "OOF AUC", "OOF R²"], rows, "dprism-oof") + "<p>OOF 与 development 为不同患者集合及训练上下文。直接 25 成分和五 metascores 的结果不能作为原 26 分数主模型的单列消融，因为 PC 与预处理也不同。</p>")
    parts.append(section("disco", "4 · 与 DiscoDivas 的比较", body))
    parts.append(section("history", "5 · 与此前本地构建 PRS 的比较", history_section(metrics)))
    body = "<p>对同一套总体训练模型进行分层评价，不在各亚组重新训练或重新选择最佳 PRS。AFR / AMR / EAS / EUR / SAS-like 为 1000G 遗传相似性标签，AMBIGUOUS 单独保留；它们不是自报种族或祖源比例。SAS 指南亚。当前没有单独的 Middle Eastern / Arab 评估标签。</p><p>每个亚组均报告总人数、病例及对照；R² 的截距参照按该亚组患病率计算。<b>★</b> 表示病例或对照少于 20，估计不稳定。不同亚组的 R² 受患病率与人群构成影响，不宜据大小作直接排名。</p>"
    for comparison, title, fig_name, reference, candidate in [
        ("full_vs_single_lr", "6.1 完整面板 TabPFN 对 Single-best PRS Logistic", "08_ancestry_single", "best_single_lr", "tabpfn_full"),
        ("full_vs_disco", "6.2 完整面板 TabPFN 对 Disco", "09_ancestry_disco", "selected_disco", "tabpfn_full"),
    ]:
        body += f"<h3>{title}</h3>"
        body += figure(FIG / f"{fig_name}.svg", "图 2 · 祖源分层：传统单分数参照" if comparison == "full_vs_single_lr" else "图 3 · 祖源分层：Disco 参照", "横线为名义 95% 配对区间；括号为人数 / 病例，* 表示任一结局类别少于 20。横轴范围按面板设定；亚组均为探索性分析。")
        for disease in NAMES:
            rows = []
            for group in GROUPS:
                a, b = point(disease, candidate, group), point(disease, reference, group)
                au, r2 = pair(disease, comparison, "auc", group), pair(disease, comparison, "nagelkerke_r2", group)
                label = group + (" ★" if bool(a.small_group) else "")
                rows.append([label, f"{int(a['n'])} / {int(a.cases)} / {int(a.controls)}",
                             f"{b.auc:.4f} / {b.nagelkerke_r2:.4f}", f"{a.auc:.4f} / {a.nagelkerke_r2:.4f}", effect(au), effect(r2)])
            ref_label = "Single-best LR" if comparison == "full_vs_single_lr" else "Disco"
            sub = table(["相似性分组", "人数 / 病例 / 对照", f"{ref_label} AUC / R²", "TabPFN AUC / R²", "ΔAUC [95% CI]", "ΔR² [95% CI]"], rows, f"sub-{comparison}-{disease}")
            body += details(NAMES[disease] + "：各亚组的完整数值", sub)
    negative = pairs.loc[pairs.comparison.eq("full_vs_disco") & pairs.metric.eq("auc") & pairs.ancestry_group.ne("ALL") & pairs.candidate_minus_reference.lt(0)]
    labels = [NAMES[r.disease] + " " + r.ancestry_group + f"（ΔAUC {r.candidate_minus_reference:+.4f}）" for r in negative.itertuples()]
    body += "<p>完整 TabPFN 相对 Disco 的 AUC 反向点差包括：" + "；".join(labels) + "。这些亚组的区间见表。乳腺癌 AMR-like 和 SAS-like 分别只有 17 和 8 例。总体优势不能扩写为每个祖源组均获益；组内显著性差异也不构成组间效应差异的正式检验。</p>"
    parts.append(section("ancestry", "6 · 祖源亚组分析", body))
    expansion = load("overall_metrics.csv", CAD / "report")
    cad_pairs = load("paired_differences.csv", CAD / "report")
    nested = load("cad_nested_single_metrics.csv")
    rows = []
    for scope, label in [("development", "Development"), ("oof", "训练五折 OOF")]:
        for panel, label2 in [("baseline", "原 3 PRS TabPFN"), ("expanded", "扩展 7 PRS TabPFN")]:
            r = expansion.loc[expansion.scope.eq(scope) & expansion.panel.eq(panel) & expansion.method.eq("tabpfn_prs_pc")].iloc[0]
            rows.append([label, label2, num(r.auc), num(r.nagelkerke_r2)])
        r = nested.loc[nested.scope.eq("development" if scope == "development" else "training_oof")].iloc[0]
        rows.append([label, "7 候选内选择 Single-best LR", num(r.auc), num(r.nagelkerke_r2)])
    body = table(["评估范围", "流程", "AUC", "R²"], rows, "cad-expansion")
    r = cad_pairs.loc[cad_pairs.scope.eq("development") & cad_pairs.method.eq("tabpfn_prs_pc") & cad_pairs.metric.eq("nagelkerke_r2")].iloc[0]
    body += f"<p>CAD 扩展后 development 的 ΔR² 为 {r.candidate_minus_reference:+.6f}，该实验原八项比较的同时区间为 {ci(r, True, 6)}；AUC 以及训练 OOF 的 TabPFN 增量尚不明确。Multi-PRS Logistic 的 OOF 表现下降。新增分数尚未证明具有跨划分稳定的额外收益。</p>"
    body += "<p>本版补跑 CAD 单分数的嵌套选择：每个外层训练集独立做三折选择，然后预测相应外层留出集。五折均选择 GPS_Mult；C 可随折变化。该 OOF 结果未用全体训练 OOF 表现先挑分数，避免把选择集同时当作无偏评估集。</p>"
    cad_rows = []
    for scope, scope_label in [("development", "Development"), ("oof", "训练五折 OOF")]:
        for method, label in [("tabpfn_prs_pc", "TabPFN"), ("multi_prs_pc_lr", "Multi-PRS Logistic")]:
            frame = cad_pairs.loc[cad_pairs.scope.eq(scope) & cad_pairs.method.eq(method)]
            au = frame.loc[frame.metric.eq("auc")].iloc[0]
            rs = frame.loc[frame.metric.eq("nagelkerke_r2")].iloc[0]
            cad_rows.append([scope_label, label, effect(au), effect(rs), ci(rs, True)])
    body += table(["评估", "整合器", "扩展 ΔAUC [95% CI]", "扩展 ΔR² [95% CI]", "R²：原 8 项同时区间"], cad_rows, "cad-expansion-deltas", "7 分数 − 原 3 分数；原实验区间")
    parts.append(section("robustness", "7 · CAD 扩展与共享五折 OOF", body))
    parts.append(section("ablations", "8 · 已完成的消融与敏感性分析", ablation_section()))
    body = """<p>当前证据应分为三个层次：完整外部 PRS + TabPFN 相对传统单分数 Logistic 的预测增量；完整主模型相对 training 选定 Disco 的流程差异（另附匹配分析）；以及祖源亚组中的不确定性。三者不能相互替代，也不足以证明祖源调权机制或跨队列迁移成功。</p><ol><li><b>固定比较协议。</b>保留 single-best Logistic、Multi-PRS Logistic、Single-PRS TabPFN 和 Disco 四类作用不同的对照。CAD 7 分数的公平 Disco 扩展需要先明确各分数的参考中心、重复中心及混合调优模型的处理。</li><li><b>优先研究单分数与多分数的增量。</b>扩展 CAD 的 OOF 增量较小，应在相同外层折内检验删减或正则化，而不是依据 development 反复挑选有利子集。</li><li><b>随后开展蒸馏。</b>在训练折内生成教师 OOF 软标签，比较同结构学生的硬标签训练与蒸馏训练；再比较固定系数和祖源可变系数。评价 AUC、R²、校准、亚组稳定性及推理成本。</li><li><b>保留独立外部验证。</b>当前 development 已被多轮使用；未来用独立队列检验迁移与校准。Arab 优化模型的加入不等于已经拥有 Arab 亚组验证。</li></ol>"""
    parts.append(section("discussion", "9 · 讨论与后续研究", body))
    source_rows = [[NAMES[r.disease], r.selected_score, str(r.candidate_scores), num(r.C, 2), num(r.training_inner_auc)] for r in choices.itertuples()]
    evidence = table(["性状", "选中分数", "候选分数数", "C", "Training 内层 AUC"], source_rows, "selection-evidence")
    evidence += "<p>本版核对各模型患者集合、结局与祖源标签，复算 AUC、R²、Brier 和 log loss。CAD 单分数新增 510 次低成本 Logistic 拟合，含 1 个完整训练及 5 个外层上下文中的选择；TabPFN 与 Disco 使用已保存预测。外部来源独立性为队列层面核查，尚未完成跨库参与者连接或家系隔离核验。</p>"
    evidence += "<div class='downloads'>" + "".join(download(path, title) for path, title in [
        (DATA / "development_metrics.csv", "整体与亚组指标 CSV"), (PRIMARY / "primary_paired_comparisons.csv", "主比较差异与区间 CSV"),
        (DATA / "single_best_selection.csv", "Single-best 选择 CSV"), (DATA / "cad_nested_single_metrics.csv", "CAD 嵌套 OOF CSV"),
        (DATA / "VERIFIED.json", "单分数分析核验凭证"), (PRIMARY / "DISCO_PRIMARY_VERIFIED.json", "完整模型 Disco 核验"),
        (BASE / "historical_wrapper_metrics.csv", "本地历史对照 CSV"), (BASE / "historical_wrapper_auc_comparisons.csv", "历史 AUC 配对 CSV"),
        (OUT / "paired_r2.csv", "历史与消融 R² 配对 CSV"), (BASE / "current_development_metrics.csv", "原面板 54 配置 CSV"),
        (BASE / "current_auc_comparisons.csv", "原面板配对消融 CSV"), (BASE / "hard_ancestry_comparisons.csv", "Hard ancestry CSV"),
        (BASE / "soft_ancestry_comparisons.csv", "Soft ancestry CSV"), (CAD / "report/paired_differences.csv", "CAD 扩展配对结果"),
        (OUT / "External_PRS_Figures.pdf", "图表 PDF"),
    ]) + "</div>"
    original_doc = ROOT / "docs/1012/External_PRS_Progress_Review_20261012_CN.md"
    evidence += f"<p class='muted'>下载仅包含汇总统计与核验信息；个体预测未嵌入 HTML。旧版报告及其核验文件已保留于 before_academic_revision_20261012。完整历史口径见 <a href='{os.path.relpath(original_doc, OUT)}'>原文字报告</a>。</p>"
    parts.append(section("evidence", "10 · 可复核结果与下载", evidence))
    nav_items = [("overview", "摘要"), ("design", "1 研究设计"), ("resources", "2 三种性状的 PRS"), ("single", "3 单分数主对照"), ("disco", "4 Disco 比较"), ("history", "5 本地历史对照"), ("ancestry", "6 祖源亚组"), ("robustness", "7 CAD 扩展与 OOF"), ("ablations", "8 完整消融"), ("discussion", "9 讨论与下一步"), ("evidence", "10 数据与核验")]
    nav = "<aside><b>EXTERNAL PRS</b><p>研究评价报告</p><nav>" + "".join(f"<a href='#{i}'>{label}</a>" for i, label in nav_items) + "</nav><button id='expand'>展开全部明细</button><button id='print'>打印 / PDF</button></aside>"
    js = """document.getElementById('expand').onclick=function(){const open=this.dataset.open!=='true';document.querySelectorAll('details').forEach(d=>d.open=open);this.dataset.open=String(open);this.textContent=open?'收起明细':'展开全部明细';};document.getElementById('print').onclick=()=>{document.querySelectorAll('details').forEach(d=>d.open=true);window.print();};"""
    return "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>External_PRS_Review_20261012_CN · 学术修订版</title><style>" + CSS + "</style></head><body><div class='layout'>" + nav + "<main>" + intro + "".join(parts) + "<footer>研究性内部开发评价 · 冻结外部 SNP 权重 · 不含独立外部验证结论</footer></main></div><script>" + js + "</script></body></html>"


CSS = """
:root{--ink:#243847;--muted:#617481;--line:#dce5e9;--accent:#24645d}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f6f7;color:var(--ink);font:15px/1.8 system-ui,-apple-system,'Noto Sans CJK SC','Microsoft Yahei',sans-serif}.layout{max-width:1530px;margin:auto;display:grid;grid-template-columns:185px minmax(0,1fr);gap:30px;padding:34px 30px}aside{position:sticky;top:28px;align-self:start;font-size:13px}aside>b{letter-spacing:.12em;color:var(--accent)}aside p{margin:5px 0 18px}nav{display:flex;flex-direction:column;gap:5px}nav a{padding:5px 0;text-decoration:none;color:var(--muted)}nav a:hover{color:var(--accent)}button{display:block;border:1px solid var(--line);border-radius:3px;background:white;color:var(--ink);padding:8px 12px;margin-top:12px;cursor:pointer}main{min-width:0}header,section{background:white;border:1px solid var(--line);padding:28px 34px;margin-bottom:22px;border-radius:5px}header{border-top:5px solid var(--accent)}h1{font-size:31px;line-height:1.35;margin:8px 0 10px}h2{font-size:23px;line-height:1.4;margin:0 0 20px;font-weight:650}h3{font-size:17px;line-height:1.5;margin:25px 0 12px}p{margin:11px 0}.eyebrow{font-size:12px;letter-spacing:.05em;color:var(--muted)}.subtitle{font-size:19px;color:var(--muted)}.note{padding:16px 20px;margin:19px 0;background:#eff5f4;border-left:3px solid var(--accent)}a{color:#285f98;text-underline-offset:3px}.muted,small,figcaption,caption{color:var(--muted)}small{display:block;font-size:11px;line-height:1.6;margin-top:4px}.table-wrap{overflow-x:auto;margin:17px 0 22px}table{border-collapse:collapse;width:100%;font-size:12.5px;line-height:1.65}caption{text-align:left;margin:0 0 8px;font-size:12px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px 11px;vertical-align:top}th{background:#edf2f5;font-weight:650}td{min-width:90px}td:first-child{min-width:104px}table[id^=resources-] td{min-width:135px}table[id^=resources-] td:nth-child(3),table[id^=resources-] td:nth-child(4){min-width:180px}tbody tr:hover{background:#fafcfc}figure{margin:22px 0;padding:10px 0}figure h3{margin:0 0 12px}figure img{width:100%;height:auto;display:block}figcaption{font-size:12px;padding-top:10px}details{border-top:1px solid var(--line);margin:17px 0;padding:11px 0}summary{cursor:pointer;font-weight:600}.detail-body{padding:10px 0 0}.downloads{display:flex;gap:9px;flex-wrap:wrap}.download{padding:8px 12px;background:#edf2f5;text-decoration:none;border-radius:3px;font-size:12px}li{margin:12px 0}footer{font-size:12px;color:var(--muted);padding:8px 0 28px}section{scroll-margin-top:18px}
@media(max-width:1000px){.layout{display:block;padding:18px}aside{position:static;margin-bottom:18px}aside>p,aside>b{display:none}nav{flex-direction:row;overflow-x:auto;gap:18px}nav a{white-space:nowrap}button{display:inline-block;margin:9px 8px 0 0}header,section{padding:22px}.subtitle{font-size:17px}}
@media(max-width:500px){.layout{padding:10px}header,section{padding:20px 16px}h1{font-size:26px}h2{font-size:21px}body{font-size:14px}.note{padding:12px}table{font-size:12px}}
@media print{body{background:white;font-size:10pt}.layout{display:block;padding:0;max-width:none}aside,button,.download,figcaption a{display:none!important}header,section{padding:15px 0;border:0;border-radius:0;margin:0}section{break-before:auto;border-top:1px solid var(--line)}h1{font-size:25pt}h2{font-size:18pt}h2,h3,summary{break-after:avoid}table{font-size:8pt;table-layout:fixed;width:100%}thead{display:table-header-group}th,td{min-width:0!important;padding:6px;overflow-wrap:anywhere}tr{break-inside:avoid}.table-wrap{overflow:visible}figure{break-inside:avoid}figure img{max-height:220mm;object-fit:contain}details{border:0}a{color:inherit;text-decoration:none}.note{background:#f1f4f4}footer{break-inside:avoid}@page{size:A4;margin:14mm}
"""


def main():
    receipt = json.loads((DATA / "VERIFIED.json").read_text())
    assert receipt["status"] == "verified"
    for name, digest in receipt["output_sha256"].items():
        assert sha256_file(DATA / name) == digest
    primary_receipt = json.loads((PRIMARY / "DISCO_PRIMARY_VERIFIED.json").read_text())
    assert primary_receipt["status"] == "verified"
    for name, digest in primary_receipt["output_sha256"].items():
        assert sha256_file(PRIMARY / name) == digest
    metrics, pairs, choices = load("development_metrics.csv"), load("primary_paired_comparisons.csv", PRIMARY), load("single_best_selection.csv")
    make_figures(metrics, pairs)
    original_figures = BASE / "before_academic_revision_20261012/html/External_PRS_Figures.pdf"
    SOURCES[str(original_figures.relative_to(ROOT))] = sha256_file(original_figures)
    subprocess.run(["pdfseparate", "-f", "5", "-l", "5", str(original_figures), str(DATA / "ablation_page_%d.pdf")], check=True)
    subprocess.run(["pdfunite", *[str(FIG / f"{name}.pdf") for name in
                    ["07_primary_baselines", "08_ancestry_single", "09_ancestry_disco"]],
                    str(DATA / "ablation_page_5.pdf"), str(OUT / "External_PRS_Figures.pdf")], check=True)
    document = build(metrics, pairs, choices)
    target = OUT / "External_PRS_Review_20261012_CN.html"
    target.write_text(document)
    write_json(OUT / "BUILD_MANIFEST.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "edition": "primary_disco_history_ablation_revision_20261012",
        "source_sha256": SOURCES, "script_sha256": sha256_file(Path(__file__)),
        "html": target.name, "html_sha256": sha256_file(target), "figures": FIGURES,
        "patient_data_embedded": False, "previous_html_archived": "../before_primary_disco_revision_20261012",
        "new_model_fits": receipt["new_model_fits"], "frozen_tabpfn_and_disco_predictions_reused": True,
    })
    print(json.dumps({"html": str(target), "bytes": target.stat().st_size, "figures": FIGURES}, ensure_ascii=False))


if __name__ == "__main__":
    main()
