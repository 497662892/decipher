"""Create a portable Chinese HTML review and vector figures from frozen PRS evidence.

Run after supplement_external_prs_review_r2.py. No patient-level data are embedded.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from markdown_it import MarkdownIt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/external_pgs/progress_review_20261012"
OUT = BASE / "html"
FIG = OUT / "figures"
DOC = ROOT / "docs/1012/External_PRS_Progress_Review_20261012_CN.md"
NAMES = {"T2D": "T2D", "cad": "CAD", "bc": "女性乳腺癌"}
COLORS = {"T2D": "#0f766e", "cad": "#3765b0", "bc": "#b5527c"}
INK, MUTED, GRID = "#18323c", "#647780", "#e5ecee"
FIGURES = []
SOURCES = {}


def load(name, folder=BASE):
    path = folder / name
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, float_precision="round_trip")


M = load("current_development_metrics.csv")
P = load("current_auc_comparisons.csv")
M25 = load("dprism25_metrics.csv")
P25 = load("dprism25_auc_comparisons.csv")
H = load("historical_wrapper_metrics.csv")
HP = load("historical_wrapper_auc_comparisons.csv")
HARD = load("hard_ancestry_comparisons.csv")
SOFT = load("soft_ancestry_comparisons.csv")
SUB = load("dprism25_ancestry_metrics.csv")
RESOURCES = load("external_resource_inventory.csv")
R2 = load("paired_r2.csv", OUT)
CAL = load("calibration_deciles.csv", OUT)
SOFT_METRICS = load("soft_ancestry_metrics.csv", OUT)


def metric(disease, method):
    return M.loc[(M.disease == disease) & (M.method == method)].iloc[0]


def metric25(method, scope="development"):
    return M25.loc[(M25.evaluation == scope) & (M25.method == method)].iloc[0]


def paired(disease, comparison, metric_name="auc", scope="current_development"):
    if metric_name == "nagelkerke_r2":
        rows = R2.loc[(R2.disease == disease) & (R2.comparison == comparison) & (R2.scope == scope)]
    elif scope == "dprism25_development":
        rows = P25.loc[(P25.evaluation == "development") & (P25.ancestry_group == "ALL") & (P25.comparison == comparison)]
    elif scope == "historical_wrapper":
        rows = HP.loc[(HP.disease == disease) & (HP.ancestry_group == "ALL") & (HP.comparison == comparison)]
    elif scope == "soft_ancestry":
        rows = SOFT.loc[SOFT.disease == disease]
    elif comparison == "hard_vs_continuous":
        rows = HARD.loc[HARD.disease == disease]
    else:
        rows = P.loc[(P.disease == disease) & (P.comparison == comparison)]
    assert len(rows) == 1, (disease, comparison, metric_name, scope)
    r = rows.iloc[0].copy()
    r["effect"] = r.get("candidate_minus_reference", r.get("delta"))
    return r


def interval(r, adjusted=False):
    prefix = "familywise_" if adjusted else ""
    return f"[{r[prefix + 'ci_low']:+.6f}, {r[prefix + 'ci_high']:+.6f}]"


def effect_cell(r):
    return f"<b>{r.effect:+.6f}</b><small>{interval(r)}</small>"


def table(headers, rows, caption="", table_id=""):
    head = "".join(f"<th scope='col'>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
    identity = f" id='{table_id}'" if table_id else ""
    return f"<div class='table-wrap'><table class='results'{identity}><caption>{caption}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


MD = MarkdownIt("commonmark", {"html": True}).enable("table")


def md(text):
    # Preserve repo-relative evidence links when the report moves into outputs/.
    def relink(match):
        label, target = match.groups()
        if "://" in target or target.startswith("#"):
            return match.group(0)
        resolved = (DOC.parent / target).resolve()
        assert resolved.is_file(), resolved
        SOURCES[str(resolved.relative_to(ROOT))] = hashlib.sha256(resolved.read_bytes()).hexdigest()
        payload = base64.b64encode(resolved.read_bytes()).decode()
        return f"<a download='{html.escape(resolved.name)}' href='data:application/octet-stream;base64,{payload}'>{label}</a>"
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", relink, text)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", text)
    rendered = MD.render(text)
    return re.sub(r"<table>.*?</table>", lambda match: "<div class='table-wrap'>" + match.group() + "</div>", rendered, flags=re.S)


def save_figure(fig, name, title, caption, pdf):
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(FIG / f"{name}.png", dpi=140, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append({"name": name, "title": title, "caption": caption})


def figure(name):
    info = next(f for f in FIGURES if f["name"] == name)
    encoded = base64.b64encode((FIG / f"{name}.svg").read_bytes()).decode()
    return f"<figure id='{name}'><div class='figure-label'>{info['title']}</div><img alt='{html.escape(info['caption'])}' src='data:image/svg+xml;base64,{encoded}'><figcaption>{info['caption']} <a download='{name}.svg' href='data:image/svg+xml;base64,{encoded}'>下载 SVG</a></figcaption></figure>"


def style_axis(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=7)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


BASELINES = [
    ("tabpfn_full", "TabPFN 完整面板"), ("tabpfn_single", "TabPFN 最佳单 PRS"),
    ("tabpfn_covariates", "TabPFN 仅协变量"), ("full_lr", "完整面板 Logistic"),
    ("flex_fixed", "灵活基础项 + 固定系数"), ("catboost", "CatBoost"),
    ("selected_baseline", "Training 内选定常规基线"),
]
ABLATIONS = [
    ("full_vs_covariates", "加入 External PRS", "current_development"),
    ("full_vs_single_tabpfn", "完整面板 − 最佳单 PRS", "current_development"),
    ("continuous_vs_no_pc", "连续 PC − 无显式祖源输入", "current_development"),
    ("varying_vs_fixed", "祖源可变系数 − 固定系数", "current_development"),
    ("hard_vs_continuous", "Hard labels 替换连续 PC", "current_development"),
    ("full_vs_pc", "PC 之外加入 soft ancestry", "soft_ancestry"),
]


def make_figures():
    FIG.mkdir(exist_ok=True)
    font = Path("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
    if font.exists():
        fm.fontManager.addfont(font)
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Droid Sans Fallback"],
                         "font.size": 11, "axes.labelcolor": INK, "text.color": INK,
                         "axes.titleweight": "bold", "axes.titlesize": 13,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "svg.fonttype": "path", "pdf.fonttype": 42,
                         "svg.hashsalt": "decipehr-review-20261012"})
    with PdfPages(OUT / "External_PRS_Figures.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(12.8, 5.4))
        ax.set(xlim=(-1.55, 10.3), ylim=(-1.3, 5.8))
        ax.axis("off")
        groups = ["AFR", "AMR", "EAS", "EUR", "SAS"]
        shades = ["#e3f1ed", "#e6eef8", "#f5e9ed", "#f2efde", "#ebe8f4"]
        ax.text(1.95, 5.65, "GWAS 来源成分", ha="center", weight="bold", fontsize=13)
        ax.text(-1.4, 2.4, "目标\n调优\n祖源", ha="center", va="center", fontsize=12)
        for col, group in enumerate(groups):
            ax.text(col, 5.1, group, ha="center", fontsize=12)
        for row_index, group in enumerate(groups):
            y = 4 - row_index
            ax.text(-0.8, y, group, ha="center", va="center")
            for col in range(5):
                ax.add_patch(FancyBboxPatch((col - .43, y - .37), .86, .74,
                                           boxstyle="round,pad=0.03,rounding_size=0.07", fc=shades[row_index], ec="white"))
                ax.text(col, y, f"PGS\n{5353 + row_index * 5 + col:06d}", ha="center", va="center", fontsize=9)
        ax.text(1.9, -.75, "5 个调优目标 × 5 个来源 = 25 个成分", ha="center", fontsize=12, weight="bold")
        for x, y, width, title, sub, color in [
            (5.7, 3.1, 3.7, "25 成分直接输入 TabPFN", "T2D 与 Disco 的主比较版本", "#e3f1ed"),
            (5.7, .5, 3.7, "25 → 5 个发表 metascores", "现有 Disco 适配 / 压缩消融", "#edf0f3"),
        ]:
            ax.add_patch(FancyBboxPatch((x, y), width, 1.25, boxstyle="round,pad=.12", fc=color, ec="none"))
            ax.text(x + width / 2, y + .83, title, ha="center", va="center", weight="bold", fontsize=12)
            ax.text(x + width / 2, y + .32, sub, ha="center", va="center", fontsize=10, color=MUTED)
            ax.add_patch(FancyArrowPatch((4.5, 2.1), (x - .15, y + .63), arrowstyle="-|>", mutation_scale=17, color=MUTED, lw=1.5))
        save_figure(fig, "01_dprism25", "图 1 · 25 个 D-PRISM 成分究竟是什么", "行是目标调优祖源，列是 GWAS 来源；每人均可计算全部 25 个成分。固定的 25→5 组合保留来源贡献，但限制组内系数。完整 T2D 面板另加 PGS002308，共 26 个输入。", pdf)

        fig, axes = plt.subplots(3, 2, figsize=(12.8, 10.7))
        for row_index, disease in enumerate(NAMES):
            for col, key in enumerate(["auc", "nagelkerke_r2"]):
                ax = axes[row_index, col]
                ys = np.arange(len(BASELINES))[::-1]
                values = [metric(disease, method)[key] for method, _ in BASELINES]
                spread = max(values) - min(values)
                pad = max(spread * .18, .006)
                ax.set_xlim(min(values) - pad, max(values) + pad * 3.5)
                for y, value, (method, _) in zip(ys, values, BASELINES, strict=True):
                    color = COLORS[disease] if method == "tabpfn_full" else "#8da0a9"
                    ax.scatter(value, y, s=65 if method == "tabpfn_full" else 32, color=color, zorder=3)
                    ax.annotate(f"{value:.4f}", (value, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9, color=color)
                ax.set_yticks(ys, [name for _, name in BASELINES] if col == 0 else [""] * len(BASELINES), fontsize=9)
                ax.set_ylim(-.6, len(BASELINES) - .35)
                ax.set_title(f"{NAMES[disease]} · {'AUC' if col == 0 else 'Nagelkerke R²'}", loc="left", pad=14)
                style_axis(ax)
        fig.subplots_adjust(left=.22, right=.98, top=.96, bottom=.055, hspace=.43, wspace=.14)
        save_figure(fig, "02_current_models", "图 2 · 当前整合器：AUC 与 R² 并列看", "原定 development；完整面板为 26 / 3 / 3。圆点为冻结预测的点估计，无误差线。横轴按疾病与指标分别设定范围；精确值见表格。选定基线可能与某个已展示算法重合。", pdf)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.0))
        for col, key in enumerate(["auc", "nagelkerke_r2"]):
            ax = axes[col]
            for i, disease in enumerate(NAMES):
                y = 2 - i
                old = H.loc[(H.disease == disease) & (H.model == "previous_wrapper"), key].iloc[0]
                new = H.loc[(H.disease == disease) & (H.model == "external_pc_wrapper"), key].iloc[0]
                ax.plot([old, new], [y, y], color=COLORS[disease], lw=2.5)
                ax.scatter(old, y, s=62, facecolor="white", edgecolor=COLORS[disease], lw=1.8, zorder=3)
                ax.scatter(new, y, s=62, color=COLORS[disease], zorder=3)
                ax.annotate(f"{old:.4f}", (old, y), xytext=(-5, -19), textcoords="offset points", ha="right", fontsize=10)
                ax.annotate(f"{new:.4f}", (new, y), xytext=(5, 12), textcoords="offset points", ha="left", fontsize=10)
            ax.set_yticks([2, 1, 0], list(NAMES.values()) if col == 0 else ["", "", ""])
            ax.set_ylim(-.55, 2.55)
            ax.set_xlim((.64, .85) if col == 0 else (.065, .415))
            ax.set_title("AUC" if col == 0 else "Nagelkerke R²", loc="left")
            style_axis(ax)
        fig.legend([Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED), Line2D([], [], marker="o", ls="", color=MUTED)],
                   ["历史自行构建 PRS + wrapper", "当前冻结 External PRS + wrapper"], loc="lower center", ncol=2, frameon=False, fontsize=10)
        fig.subplots_adjust(left=.11, right=.98, bottom=.2, top=.9, wspace=.18)
        save_figure(fig, "03_historical", "图 3 · 与历史流程相比，两个指标的点估计均提高", "双方均为同患者的协变量 wrapper 概率；不能与直接概率结果交叉拼接。PRS 面板、wrapper OOF 折数及女性 BC checkpoint 有变化，故为整套流程的历史比较。", pdf)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.1))
        labels = ["T2D：直接 25 成分", "CAD：共同 2 分数", "女性 BC：共同 3 分数"]
        for col, key in enumerate(["auc", "nagelkerke_r2"]):
            ax = axes[col]
            for i, disease in enumerate(NAMES):
                scope, comparison = ("dprism25_development", "same_25_sources") if disease == "T2D" else ("current_development", "matched_vs_selected_disco")
                r = paired(disease, comparison, key, scope)
                ax.errorbar(r.effect, 2 - i, xerr=[[r.effect - r.ci_low], [r.ci_high - r.effect]], fmt="o", color=COLORS[disease], capsize=4, ms=7, lw=2)
                ax.annotate(f"{r.effect:+.4f}", (r.effect, 2 - i), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=10)
            ax.axvline(0, color=MUTED, lw=1, ls="--")
            ax.set_yticks([2, 1, 0], labels if col == 0 else ["", "", ""])
            ax.set_ylim(-.5, 2.6)
            ax.set_title("ΔAUC" if col == 0 else "ΔNagelkerke R²", loc="left")
            ax.set_xlabel("TabPFN − 当前 Disco 适配")
            style_axis(ax)
        fig.subplots_adjust(left=.18, right=.98, top=.88, bottom=.22, wspace=.18)
        save_figure(fig, "04_disco", "图 4 · 与当前 Disco 适配相比的性能差值", "误差线为普通 95% 配对区间。AUC 沿用冻结分析；R² 为本次新增的探索性分析。T2D 共享原始 25 成分，TabPFN 直接整合，现有 Disco 采用 25→5 发表组合；CAD/BC 中间输入匹配。", pdf)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.9))
        offsets = {"T2D": .22, "cad": 0, "bc": -.22}
        for col, key in enumerate(["auc", "nagelkerke_r2"]):
            ax = axes[col]
            for j, (comparison, _, scope) in enumerate(ABLATIONS):
                for disease in NAMES:
                    r = paired(disease, comparison, key, scope)
                    ax.errorbar(r.effect, len(ABLATIONS) - 1 - j + offsets[disease],
                                xerr=[[r.effect - r.ci_low], [r.ci_high - r.effect]], fmt="o",
                                color=COLORS[disease], capsize=2.5, ms=4.7, lw=1.4)
            ax.axvline(0, color=MUTED, lw=1, ls="--")
            ax.set_yticks(np.arange(len(ABLATIONS))[::-1], [x[1] for x in ABLATIONS] if col == 0 else [""] * len(ABLATIONS), fontsize=10)
            ax.set_ylim(-.65, len(ABLATIONS) - .35)
            ax.set_title("ΔAUC" if col == 0 else "ΔNagelkerke R²", loc="left", pad=17)
            ax.set_xlabel("左侧变更 − 参照方案；右侧表示指标提高", fontsize=10)
            style_axis(ax)
        fig.legend([Line2D([], [], color=c, marker="o", lw=1.5) for c in COLORS.values()], list(NAMES.values()), loc="lower center", ncol=3, frameon=False)
        fig.subplots_adjust(left=.24, right=.98, top=.93, bottom=.13, wspace=.17)
        save_figure(fig, "05_ablations", "图 5 · 六个消融问题，用 AUC 与 R² 共同判断", "普通 95% 配对区间；各行均使用已保存的相应模型。Hard/soft 的历史 AUC 随机种子和检验族不同；新增 R² 使用统一的 2,000 次结局分层配对 bootstrap，29 项同时区间见明细。", pdf)

        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
        for ax, disease in zip(axes, NAMES, strict=True):
            f = CAL.loc[CAL.disease == disease]
            ax.plot([0, 1], [0, 1], "--", color="#a8b4ba", lw=1)
            ax.errorbar(f.predicted, f.observed,
                        yerr=[f.observed - f.observed_low, f.observed_high - f.observed],
                        fmt="o-", color=COLORS[disease], capsize=3, lw=1.5, ms=5)
            bound = min(1, max(f.observed_high.max(), f.predicted.max()) + .08)
            ax.set(xlim=(0, bound), ylim=(0, bound), xlabel="组内平均预测概率", ylabel="实际病例比例")
            ax.set_title(NAMES[disease], loc="left")
            ax.set_aspect("equal")
            style_axis(ax)
        fig.subplots_adjust(left=.06, right=.98, top=.9, bottom=.16, wspace=.33)
        save_figure(fig, "06_calibration", "图 6 · 为什么 R² 需要结合校准来看", "当前完整面板的直接概率；按风险等人数分成 10 组，纵向误差线为病例比例的 Wilson 95% 区间。虚线为理想校准。女性 BC 存在整体低估，因此 AUC 改善不等于绝对概率已经准确。", pdf)

CSS = """
:root{--ink:#19333e;--muted:#617780;--teal:#087b71;--line:#dce6e8;--paper:#fff;--bg:#f3f6f6;--blue:#3765b0;--rose:#b5527c}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:26px}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans CJK SC','Microsoft YaHei','PingFang SC',sans-serif;font-size:15px;line-height:1.8}a{color:var(--teal);text-underline-offset:3px}button{font:inherit;cursor:pointer}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #edae43;outline-offset:3px}.layout{max-width:1580px;margin:auto;display:grid;grid-template-columns:220px minmax(0,1fr);gap:32px;padding:32px 36px 70px}.sidebar{position:sticky;top:28px;align-self:start}.brand{font-size:12px;letter-spacing:.16em;color:var(--teal);font-weight:800}.sidebar h2{font-size:20px;margin:10px 0 5px}.sidebar .date{color:var(--muted);font-size:12px}.sidebar nav{display:grid;gap:5px;margin:24px 0}.sidebar nav a{text-decoration:none;color:var(--muted);padding:9px 12px;border-radius:7px;font-size:14px;border-left:3px solid transparent}.sidebar nav a:hover,.sidebar nav a.active{background:#e2eeea;color:var(--teal);border-left-color:var(--teal)}.tools{display:flex;flex-wrap:wrap;gap:9px;margin-top:20px}.tools button,.download{border:1px solid var(--line);background:white;border-radius:6px;color:var(--ink);padding:8px 11px;text-decoration:none;font-size:13px}.small{font-size:12px;color:var(--muted)}main{min-width:0}.hero{padding:34px 38px;background:#153b41;color:#eff9f7;border-radius:16px;position:relative;overflow:hidden}.eyebrow{font-size:12px;letter-spacing:.14em;font-weight:700;color:#9cd5c9}.hero h1{font-size:clamp(28px,3.2vw,42px);line-height:1.25;letter-spacing:-.02em;margin:14px 0}.hero p{max-width:820px;margin:10px 0;color:#d3e6e4}.hero .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px}.chip{font-size:12px;border:1px solid #638a8c;border-radius:20px;padding:3px 11px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:22px 0}.metric-card{background:white;border:1px solid var(--line);border-radius:12px;padding:20px 22px;border-top:4px solid var(--accent)}.metric-card h3{font-size:18px;margin:0 0 3px}.metric-card p{margin:0;color:var(--muted);font-size:12px}.metric-pair{display:flex;justify-content:space-between;gap:15px;margin-top:16px}.metric-pair span{display:block;font-size:12px;color:var(--muted)}.metric-pair b{font-size:27px;line-height:1.45;font-weight:750;font-variant-numeric:tabular-nums;color:var(--accent)}.summary-box{background:#e8f1ee;border-left:4px solid var(--teal);padding:18px 23px;margin:22px 0;border-radius:0 9px 9px 0}.summary-box p{margin:6px 0}.section{background:white;border:1px solid var(--line);padding:30px 34px;border-radius:14px;margin:26px 0;min-width:0}.section-title{display:flex;align-items:baseline;gap:15px;border-bottom:1px solid var(--line);padding-bottom:15px;margin-bottom:21px}.section-title .number{font-size:30px;color:#95b6b1;font-weight:750;line-height:1}.section-title h2{font-size:25px;line-height:1.4;margin:0}.section-title .kicker{margin-left:auto;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.section h3{font-size:19px;margin:26px 0 12px}.section p{margin:14px 0}.section strong{font-weight:750}.table-wrap{overflow-x:auto;margin:17px 0 23px;border:1px solid var(--line);border-radius:8px;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6}caption{text-align:left;padding:10px 13px;font-weight:600;background:#fafcfc}caption:empty{display:none}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:11px 13px;min-width:70px}th{background:#edf3f3;font-weight:700;color:#35515b;font-size:12px}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafcfc}td b,td small{font-variant-numeric:tabular-nums}td small{display:block;color:var(--muted);font-size:11px;white-space:nowrap}td:first-child{font-weight:600;min-width:98px}#disco-main td:nth-child(2),#disco-main td:nth-child(3){min-width:140px}figure{margin:24px 0 28px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 18px 12px;break-inside:avoid}.figure-label{font-weight:750;font-size:16px;margin:0 6px 12px}figure img{display:block;width:100%;height:auto}figcaption{font-size:12px;color:var(--muted);line-height:1.7;padding:8px 8px 0}figcaption a{white-space:nowrap}details{border:1px solid var(--line);border-radius:9px;margin:18px 0;background:#fcfdfd}summary{padding:13px 17px;cursor:pointer;font-weight:700;color:#34585c}details>.details-body{padding:0 18px 16px}details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:15px}.note{padding:15px 19px;background:#f1f5f8;border-left:3px solid #7293aa;border-radius:0 7px 7px 0;margin:20px 0;font-size:14px}.note.warm{background:#fbf3e8;border-left-color:#b68135}.findings{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:20px 0}.finding{padding:17px 19px;background:#f7faf9;border:1px solid var(--line);border-radius:9px}.finding h3{font-size:16px;margin:0 0 7px}.finding p{font-size:14px;margin:0}.status{display:inline-block;border-radius:12px;padding:1px 8px;font-size:11px;font-weight:700;background:#e1efea;color:#1d6c5b}.status.pending{background:#f6ebd9;color:#8a642b}.workflow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:22px 0}.workflow div{position:relative;background:#eef5f3;border:1px solid #cadeda;border-radius:8px;padding:17px 13px;min-height:125px}.workflow div+div:before{content:'→';position:absolute;left:-13px;top:39%;color:var(--teal);font-weight:800}.workflow b,.workflow span{display:block}.workflow b{font-size:14px}.workflow span{font-size:12px;color:var(--muted);margin-top:8px}.formula{padding:18px;background:#edf3f3;border-radius:8px;font-family:Georgia,'Times New Roman',serif;font-size:19px;overflow-x:auto}.metric-guide{display:grid;grid-template-columns:1fr 1fr;gap:16px}.metric-guide>div{padding:17px;background:#f7faf9;border-radius:8px}.metric-guide h3{margin:0 0 6px;font-size:16px}.metric-guide p{font-size:14px;margin:0}.foot{font-size:12px;color:var(--muted);padding:8px 5px 25px}code{font-size:12px;background:#edf2f4;border-radius:3px;padding:2px 5px;overflow-wrap:anywhere}li{margin:7px 0}ol,ul{padding-left:24px}.divider{height:1px;background:var(--line);margin:26px 0}.skip-link{position:absolute;left:20px;top:-70px;background:white;padding:8px 16px;z-index:10}.skip-link:focus{top:10px}
@media(max-width:1120px){.layout{grid-template-columns:180px minmax(0,1fr);gap:20px;padding:24px 22px}.section{padding:26px 24px}.metric-pair b{font-size:23px}.metric-card{padding:16px}.hero{padding:28px}}
@media(max-width:820px){.layout{display:block;padding:15px}.sidebar{position:static}.sidebar h2,.sidebar .date,.sidebar>.small{display:none}.sidebar nav{display:flex;overflow:auto;margin:10px 0 16px;gap:4px}.sidebar nav a{white-space:nowrap;padding:6px 10px;font-size:12px}.sidebar .tools{display:none}.hero{padding:25px}.hero h1{font-size:30px}.section{padding:23px 18px}.section-title h2{font-size:22px}.section-title .kicker{display:none}.metric-pair{display:block;margin-top:9px}.metric-pair>div+div{margin-top:7px}.metric-pair b{font-size:24px}.cards{gap:10px}.metric-card{padding:12px}.metric-card h3{font-size:16px}.findings{grid-template-columns:1fr}.workflow{grid-template-columns:1fr 1fr}.workflow div+div:before{display:none}.metric-guide{grid-template-columns:1fr}figure{padding:12px 5px 9px}.figure-label{font-size:14px}.section-title .number{font-size:26px}}
@media(max-width:440px){.cards{grid-template-columns:1fr}.metric-card{padding:14px 18px}.metric-pair{display:flex;margin-top:8px}.metric-pair>div+div{margin:0}.metric-card h3{display:inline-block;margin-right:10px}.metric-card p{display:inline}.hero h1{font-size:27px}}
@media print{body{background:white;font-size:10pt}.layout{display:block;padding:0;max-width:none}.sidebar,.tools,.download,.skip-link,figcaption a{display:none!important}.hero{background:white;color:var(--ink);border:1px solid var(--line);padding:20px}.hero p,.eyebrow{color:var(--muted)}.hero .chip{color:var(--ink);border-color:var(--line)}.section{padding:20px 0;border:0;border-top:2px solid var(--line);border-radius:0;break-before:page}.section-title{break-after:avoid}.section-title h2{font-size:19pt}h3,summary{break-after:avoid}figure{break-inside:avoid;max-height:none}figure img{max-height:220mm;object-fit:contain}table{font-size:8pt;table-layout:fixed;width:100%}td,th{overflow-wrap:anywhere}td small{white-space:normal}thead{display:table-header-group}tr{break-inside:avoid}.table-wrap{overflow:visible}th,td{min-width:0!important;padding:6px}.cards{grid-template-columns:repeat(3,1fr)}.metric-pair b{font-size:22px}.finding{break-inside:avoid}.foot{break-inside:avoid}a{color:inherit;text-decoration:none}@page{size:A4;margin:14mm}}
"""


def replace_table(text, first_header, replacement):
    pattern = re.compile(r"(?:^\|[^\n]*\n)+", re.M)
    matches = [m for m in pattern.finditer(text) if m.group().startswith(first_header)]
    assert len(matches) == 1, first_header
    item = matches[0]
    return text[:item.start()] + replacement + "\n" + text[item.end():]


def section(number, title, body, kicker=""):
    return f"<section class='section' id='s{number}'><div class='section-title'><span class='number'>{number:02d}</span><h2>{title}</h2><span class='kicker'>{kicker}</span></div>{body}</section>"


def source_sections():
    source = DOC.read_text()
    SOURCES[str(DOC.relative_to(ROOT))] = hashlib.sha256(DOC.read_bytes()).hexdigest()
    chunks = re.split(r"^\*\*([1-5]) (.*?)\*\*\s*$", source, flags=re.M)
    return {int(chunks[i]): chunks[i + 2].strip() for i in range(1, len(chunks), 3)}


def build_html():
    chunks = source_sections()
    cards = ""
    for disease in NAMES:
        r = metric(disease, "tabpfn_full")
        count = 26 if disease == "T2D" else 3
        cards += f"<article class='metric-card' style='--accent:{COLORS[disease]}'><h3>{NAMES[disease]}</h3><p>完整面板 · {count} 个分数 · n = {int(r['n']):,}</p><div class='metric-pair'><div><span>AUC</span><b>{r.auc:.4f}</b></div><div><span>Nagelkerke R²</span><b>{r.nagelkerke_r2:.4f}</b></div></div></article>"
    intro = f"""<header class='hero' id='overview'><div class='eyebrow'>DECIPHER · EXTERNAL PRS · RESEARCH REVIEW</div><h1>External PRS 整合<br>从模型来源到预测证据</h1><p>两日实验的系统整理：资源选择、整合模型、Disco 比较、消融结果与下一步。AUC 与 Nagelkerke R² 并列评价，所有结果对应明确的输入和人群。</p><div class='chips'><span class='chip'>实验：2026.10.10—10.11</span><span class='chip'>R² 补充：2026.10.12</span><span class='chip'>原定 development · 内部科研汇总</span></div></header><div class='cards'>{cards}</div>
<div class='summary-box'><p><strong>目前最稳妥的结论：</strong>冻结 External PRS 配合连续 PC 的 TabPFN 整合，在当前开发集有预测价值。加入 PRS 的收益同时获得 AUC 与 R² 支持；祖源显式调权和 soft ancestry 尚无稳定增量。</p><p><strong>T2D 对 Disco 的主比较使用直接 25 分数：</strong>AUC 0.763921，R² 0.245696；上方 26 分数卡片属于完整面板，两者的 PC／预处理方案也不同。</p><p><strong>验证边界：</strong>已有外部 PRS 权重，尚无独立外部验证队列。这里的 development 是历史上已用于开发的官方 validation。</p></div>"""

    metrics_guide = """<div class='metric-guide'><div><h3>AUC：排序是否更好</h3><p>判断病例是否倾向于排在对照前面。它评价区分能力，不直接检验预测概率是否准确。</p></div><div><h3>Nagelkerke R²：概率是否更有信息</h3><p>衡量保存的预测概率相对截距模型的似然改善，同时受区分能力和概率校准影响。本报告与 AUC 使用相同患者及相同预测版本。</p></div></div>
<details id='r2-definition'><summary>R² 的具体口径、公式与解释边界</summary><div class='details-body'><div class='formula'>R²<sub>N</sub> = {1 − exp[2(ℓ<sub>0</sub> − ℓ<sub>model</sub>)/n]} / {1 − exp(2ℓ<sub>0</sub>/n)}</div><p>ℓ<sub>model</sub> 是冻结预测概率的 Bernoulli 对数似然；ℓ<sub>0</sub> 来自本次评价人群的平均病例比例。计算前沿用已有评估的概率裁剪 [10<sup>−6</sup>, 1−10<sup>−6</sup>]，不在 development 重拟合模型或校准器。</p><p>这里报告的是<strong>包括 PRS、年龄、性别及 PC 贡献的整体预测 pseudo-R²</strong>，不是线性回归 R²、遗传率或 PRS 单独解释的遗传方差。样本外 R² 可以为负；不同疾病的患病比例不同，不用绝对 R² 跨疾病排名。ΔR² 是两个同口径值的绝对差，不等同于“解释了多少百分比遗传率”。</p><p>直接概率与协变量 wrapper 概率分别报告；也不将本指标与 Disco 原文的 covariate-adjusted full/null R² 直接比较。R² 同时依赖概率质量，因此配套展示 Brier、log loss 和校准。</p><p>定义来源：<a href='https://doi.org/10.1093/biomet/78.3.691'>Nagelkerke, Biometrika 1991</a>；实现依据：项目 <code>src/decipehr/metrics.py::nagelkerke_r2</code>。</p></div></details>"""
    intro += metrics_guide

    s1 = chunks[1]
    split = s1.index("D-PRISM 的 25 个成分表示")
    resources = md(s1[:split])
    rest = md(s1[split:])
    one = section(1, "选了哪些 External PRS，为什么", "<p>按多祖源覆盖、方法互补、资源可用性和 SNP 对齐覆盖选择。<strong>34 个候选中 32 个通过既定门槛；T2D / CAD / 女性 BC 的完整输入数为 26 / 3 / 3。</strong></p>" + figure("01_dprism25") + "<details open><summary>八类资源：方法、发现人群、调优人群与选择理由</summary><div class='details-body'>" + resources + "</div></details><details><summary>D-PRISM 调优队列、25 个成分含义与 SNP 覆盖明细</summary><div class='details-body'>" + rest + "</div></details>", "resources")

    s2 = chunks[2]
    workflow = """<div class='workflow' aria-label='当前模型流程'><div><b>01 · 外部 SNP 权重</b><span>作者发表并冻结<br>本项目不重新做 GWAS</span></div><div><b>02 · 每人全部 PRS</b><span>PLINK2 评分<br>T2D 26 / CAD 3 / BC 3</span></div><div><b>03 · TabPFN 整合</b><span>PRS + age / sex + 连续 PC<br>training 内拟合与选择</span></div><div><b>04 · Development</b><span>输出疾病状态概率<br>AUC / R² / OR90 / 校准</span></div></div>"""
    s2 = re.sub(r"```mermaid\n.*?```", workflow, s2, flags=re.S)
    main_rows = []
    for disease in NAMES:
        r, cov = metric(disease, "tabpfn_full"), metric(disease, "tabpfn_covariates")
        main_rows.append([NAMES[disease], f"{r.auc:.6f}", f"<b>{r.nagelkerke_r2:.6f}</b>", f"{r.nagelkerke_r2 - cov.nagelkerke_r2:+.6f}", f"{r.top10_vs_lower90_or:.4f}", f"{r.brier:.6f}", f"{r.log_loss:.6f}"])
    s2 = replace_table(s2, "| 疾病 | AUC |", table(["疾病", "AUC ↑", "Nagelkerke R² ↑", "ΔR² vs 仅协变量", "OR90 ↑", "Brier ↓", "Log loss ↓"], main_rows, "完整面板 · 直接概率 · 相同 development", "main-metrics"))
    s2 = s2.replace("“之前自己训练”", "<h3>2.1 与此前自行构建 PRS 的整套流程相比</h3>\n\n“之前自己训练”")
    hist_rows = []
    for disease in NAMES:
        for model, name in [("previous_wrapper", "历史自行构建 PRS"), ("external_pc_wrapper", "当前作者冻结 PRS")]:
            r = H.loc[(H.disease == disease) & (H.model == model)].iloc[0]
            hist_rows.append([NAMES[disease], name, f"{r.auc:.6f}", f"{r.nagelkerke_r2:.6f}", f"{r.top10_vs_lower90_or:.4f}"])
    s2 = replace_table(s2, "| 疾病 | 完整流程及 wrapper |", figure("03_historical") + table(["疾病", "流程（双方均加 wrapper）", "AUC", "Nagelkerke R²", "OR90"], hist_rows))
    hist_effects = []
    for disease in NAMES:
        a = paired(disease, "external_pc_wrapper_vs_previous", "auc", "historical_wrapper")
        r = paired(disease, "external_pc_wrapper_vs_previous", "nagelkerke_r2", "historical_wrapper")
        hist_effects.append([NAMES[disease], effect_cell(a), effect_cell(r), interval(r, True)])
    s2 = replace_table(s2, "| 疾病 | 当前 − 历史 ΔAUC |", table(["疾病", "ΔAUC [95% CI]", "ΔR² [95% CI]", "新增 R²：29 项同时区间"], hist_effects, "当前 − 历史；AUC 使用原分析，R² 为本次探索性补充"))
    s2 = s2.replace("在当前冻结 External PRS 面板内", "<div class='note'>历史比较中，CAD 和女性 BC 的 ΔR² 在本次 29 项校正后仍高于零；T2D 的普通区间高于零，但同时区间跨零。不能将点估计一致提高表述为三病均已确证改善。</div>\n\n<h3>2.2 同一 External PRS 面板，比较下游整合器</h3>\n\n在当前冻结 External PRS 面板内")
    baseline_rows = []
    for method, label in BASELINES:
        values = [label]
        for disease in NAMES:
            r = metric(disease, method)
            values += [f"{r.auc:.6f}", f"{r.nagelkerke_r2:.6f}"]
        baseline_rows.append(values)
    s2 = replace_table(s2, "| 当前 External PRS 下游模型 |", figure("02_current_models") + table(["整合方式", "T2D AUC", "T2D R²", "CAD AUC", "CAD R²", "女性 BC AUC", "女性 BC R²"], baseline_rows))
    s2 += "\n\n<div class='note'>补充 R² 后需收窄结论：完整 TabPFN 相对训练内选定常规基线，T2D / CAD 的 ΔR² 为 +0.038660 / +0.035803，29 项同时区间高于零；女性 BC 为 +0.014467，普通 95% CI [−0.003205, +0.033701]。女性 BC 的优势目前主要由 AUC 支持。</div>"
    before_history, rest = s2.split("<h3>2.1 与此前自行构建 PRS 的整套流程相比</h3>")
    history, baselines = rest.split("<h3>2.2 同一 External PRS 面板，比较下游整合器</h3>")
    s2 = (before_history + "<h3>2.1 同一 External PRS 面板，比较下游整合器</h3>" + baselines
          + "<h3>2.2 与此前自行构建 PRS 的整套流程相比</h3>" + history)
    two = section(2, "当前整合方式，以及与历史方案的比较", md(s2), "integration")

    s3 = chunks[3]
    disco_rows, disco_effects = [], []
    for disease in NAMES:
        ours = metric25("tabpfn_dprism25") if disease == "T2D" else metric(disease, "tabpfn_matched")
        disco = metric25("selected_disco") if disease == "T2D" else metric(disease, "selected_disco")
        ours_input = "<b>直接 25 成分</b>" if disease == "T2D" else ("共同 2 分数" if disease == "cad" else "共同 3 分数")
        disco_input = "同 25 成分 → 5 metascores" if disease == "T2D" else ours_input
        disco_rows.append([NAMES[disease], ours_input, disco_input, f"<b>{ours.auc:.6f}</b>", f"<b>{ours.nagelkerke_r2:.6f}</b>", f"{disco.auc:.6f}", f"{disco.nagelkerke_r2:.6f}"])
        scope, comparison = ("dprism25_development", "same_25_sources") if disease == "T2D" else ("current_development", "matched_vs_selected_disco")
        a, r = paired(disease, comparison, "auc", scope), paired(disease, comparison, "nagelkerke_r2", scope)
        disco_effects.append([NAMES[disease], effect_cell(a), effect_cell(r), interval(r, True)])
    s3 = replace_table(s3, "| 疾病 | TabPFN 输入 |", table(["疾病", "我们的输入", "Disco 输入", "TabPFN AUC", "TabPFN R²", "Disco AUC", "Disco R²"], disco_rows, "T2D 使用直接 25 分数主比较；CAD / BC 使用匹配面板", "disco-main") + figure("04_disco") + table(["疾病", "ΔAUC [95% CI]", "ΔR² [95% CI]", "新增 R²：29 项同时区间"], disco_effects))
    s3 = s3.replace("为解释 T2D 主比较的增益", "<div class='note warm'><strong>R² 带来的关键补充：</strong>T2D 和 CAD 相对现有 Disco 的 ΔR² 在本次 29 项同时区间下仍高于零。女性 BC 的 ΔR² = +0.018320，普通区间 [−0.002365, +0.042815] 跨零；其 AUC 优势不能扩写为概率质量全面改善。</div>\n\n<h3>3.1 T2D：把 25 成分与 5 metascores 的作用拆开</h3>\n\n为解释 T2D 主比较的增益")
    rows25 = []
    for method, label in [("tabpfn_dprism25", "<b>25 成分直接进入 TabPFN</b>"), ("tabpfn_matched", "5 metascores 进入 TabPFN（辅助）"), ("selected_disco", "5 metascores 进入 Disco")]:
        d, o = metric25(method), metric25(method, "training_oof")
        rows25.append([label, f"{d.auc:.6f}", f"{d.nagelkerke_r2:.6f}", f"{d.brier:.6f}", f"{o.auc:.6f}", f"{o.nagelkerke_r2:.6f}"])
    s3 = replace_table(s3, "| 同一 25 成分来源的流程 |", table(["同一 25 成分来源", "Dev AUC", "Dev R²", "Dev Brier", "Training OOF AUC", "Training OOF R²"], rows25))
    rows25p = []
    for name, label in [("same_25_sources", "25 成分 TabPFN − Disco"), ("direct25_vs_meta5", "25 成分 − 5 metascores TabPFN"), ("same_5_metascores", "5 metascores TabPFN − Disco")]:
        a = paired("T2D", name, "auc", "dprism25_development")
        r = paired("T2D", name, "nagelkerke_r2", "dprism25_development")
        rows25p.append([label, effect_cell(a), effect_cell(r)])
    s3 = replace_table(s3, "| Development 比较 |", table(["Development 比较", "ΔAUC [95% CI]", "ΔR² [95% CI]"], rows25p))
    subgroup_rows = []
    for group in ["AFR-like", "AMR-like", "EAS-like", "EUR-like", "SAS-like", "AMBIGUOUS"]:
        f = SUB.loc[(SUB.evaluation == "development") & (SUB.ancestry_group == group)]
        if f.empty:
            continue
        a = f.loc[f.method == "tabpfn_dprism25"].iloc[0]
        b = f.loc[f.method == "selected_disco"].iloc[0]
        subgroup_rows.append([group, f"{int(a['n'])} / {int(a.cases)}", f"{a.auc:.6f} / {b.auc:.6f}", f"{a.nagelkerke_r2:.6f} / {b.nagelkerke_r2:.6f}", f"{a.auc - b.auc:+.6f}", f"{a.nagelkerke_r2 - b.nagelkerke_r2:+.6f}"])
    s3 += "\n\n<details><summary>查看 T2D 祖源亚组：样本数、AUC 与 R²</summary><div class='details-body'>" + table(["1000G 相似性分组", "人数 / 病例", "AUC：25 TabPFN / Disco", "R²：25 TabPFN / Disco", "ΔAUC", "ΔR²"], subgroup_rows, "探索性分组点估计；小样本结果不代表祖源间普遍改善") + "</div></details>"
    three = section(3, "对比 DiscoDivas：使用正确的输入版本", md(s3), "disco comparison")

    ablation_rows = []
    for comparison, label, scope in ABLATIONS:
        for disease in NAMES:
            a, r = paired(disease, comparison, "auc", scope), paired(disease, comparison, "nagelkerke_r2", scope)
            ablation_rows.append([label, NAMES[disease], effect_cell(a), effect_cell(r), interval(r, True)])
    findings = [
        ("01 · External PRS 提供额外信号", "三病相对仅协变量模型的 AUC 和 R² 均提高；新增 ΔR² 的 29 项同时区间均高于零。这支持外部分数有可利用的预测信息。"),
        ("02 · 多 PRS 的必要性仍未确证", "完整面板相对最佳单 PRS 的 AUC 区间均跨零。T2D 的 ΔR² 为 +0.007148，普通区间高于零，但 29 项同时区间跨零；CAD / BC 的普通 ΔR² 区间也跨零。"),
        ("03 · 连续 PC 值得保留", "三病相对无显式祖源输入的 AUC 和 R² 普通区间均支持改善。ΔR² 在 29 项校正后，T2D / CAD 保留支持，女性 BC 区间跨零；不能把 PC 的作用直接解释为祖源调权机制。"),
        ("04 · Hard labels 不能替代连续 PC", "三个疾病的 AUC 和 R² 均下降，ΔR² 的 29 项同时区间均低于零。标签来自 1000G 相似性，与原临床 PC 来源不同，故不只是把同一 PC 数值分箱。"),
        ("05 · Soft ancestry 尚无额外收益", "在 PC 之外加入 soft ancestry，三病的 AUC 与 R² 普通区间均跨零。soft 是参考祖源相似度，不是祖源比例；删除 soft 属于已看过 development 后的开发决定。"),
        ("06 · 显式祖源调权尚无支持", "共享灵活基础项后，可变 PRS 系数相对固定系数的 ΔAUC 与 ΔR² 都接近零。这不能说明 TabPFN 不含交互，也不能证明更复杂 gate 有益。"),
    ]
    cards4 = "<div class='findings'>" + "".join(f"<article class='finding'><h3>{title}</h3><p>{body}</p></article>" for title, body in findings) + "</div>"
    four_body = "<p>每项消融只回答一个具体问题。下面将原 AUC 证据与新增 R² 分析并列，避免仅凭一个指标形成机制结论。</p>" + figure("05_ablations") + cards4
    four_body += "<div class='summary-box'><strong>第七个问题：25 成分能否无损压成五个发表组合？</strong><p>当前结果不支持。直接 25 成分相对五 metascores TabPFN 的 ΔAUC = +0.007220、ΔR² = +0.021022，各自的调整区间均高于零。它只说明这一固定压缩在当前系统有预测代价，不代表所有降维无效。26 输入与补充 25 输入同时改变 PC / 预处理，其差值不能单独归因于 PGS002308。</p></div>"
    four_body += "<details><summary>展开 18 组消融数值：ΔAUC、ΔR² 与区间</summary><div class='details-body'>" + table(["变更 − 参照", "疾病", "ΔAUC [95% CI]", "ΔR² [95% CI]", "R²：29 项同时区间"], ablation_rows) + "</div></details>"
    four_body += "<p>最佳单分数均由 training 选择：T2D 为 EUR 目标 D-PRISM metascore，CAD 为 PGS003725，女性 BC 为 PGS004869。T2D 的这个“单分数”已融合五种来源；未检出优势也不等于证明等效。</p>"
    four_body += "<h3>4.1 概率质量：R² 与校准一起解释</h3>" + figure("06_calibration")
    calibration_rows = []
    for disease in NAMES:
        r = metric(disease, "tabpfn_full")
        calibration_rows.append([NAMES[disease], f"{r.observed_risk:.4f}", f"{r.mean_predicted_risk:.4f}", f"{r.calibration_intercept:.4f}", f"{r.calibration_slope:.4f}", f"{r.brier:.6f}", f"{r.log_loss:.6f}"])
    four_body += table(["完整面板", "实际病例比例", "平均预测概率", "校准截距", "校准斜率", "Brier", "Log loss"], calibration_rows)
    four_body += "<p>女性 BC 平均预测概率约 0.1560，低于实际病例比例 0.1966；斜率接近 1 不代表整体风险水平准确。校准线仅是评价诊断，本次没有在 development 拟合并应用新的校准器。</p><p class='small'>此前额外 QC / 亲缘版本改变了人群及 context，其 training OOF 结果属于单独敏感性分析，不替换本报告原定 development 结果，也不复用为当前蒸馏软标签。</p>"
    four = section(4, "消融说明了什么，哪些结论仍需保留", four_body, "ablation & calibration")

    s5 = chunks[5].split("**证据与完成状态**")[0]
    five = section(5, "下一步：先验证蒸馏，再判断是否需要调权", "<div class='note'><strong>状态：</strong>以下是后续实验方案，教师软标签缓存与学生蒸馏尚未运行。本次仅整理报告、补算冻结预测的 R² 区间并作图。</div>" + md(s5), "next steps")

    status_rows = [
        ["External PRS 来源、对齐与评分", "<span class='status'>已完成</span>", "34 候选，32 个有效分数；来源人群与覆盖见第 1 节"],
        ["原定人群的三疾病比较", "<span class='status'>已完成</span>", "54 个配置；外部 SNP 权重不变"],
        ["T2D 直接 25 成分与 Disco 比较", "<span class='status'>已完成</span>", "10 行 development / pooled OOF 性能；共享患者五折"],
        ["本次 R² 补充与报告", "<span class='status'>已完成</span>", "76 行点估计复核；29 项配对 ΔR²；6 幅矢量图"],
        ["Disco 原文冻结模型 / 直接 25 维扩展", "<span class='status pending'>未运行</span>", "前者资源不足；后者按既定决定暂缓"],
        ["学生蒸馏 / 独立外部队列", "<span class='status pending'>未运行</span>", "已有蒸馏设计；目前暂无独立外部队列"],
    ]
    all_r2_rows = []
    for _, r in R2.iterrows():
        all_r2_rows.append([NAMES[r.disease], r.scope, r.comparison, f"{r.candidate_minus_reference:+.6f}", interval(r), interval(r, True)])
    download_files = [BASE / "current_development_metrics.csv", BASE / "historical_wrapper_metrics.csv", BASE / "dprism25_metrics.csv", OUT / "paired_r2.csv", OUT / "r2_recalculation.csv", OUT / "calibration_deciles.csv", OUT / "External_PRS_Figures.pdf"]
    downloads = "<div class='tools'>"
    for path in download_files:
        encoded = base64.b64encode(path.read_bytes()).decode()
        mime = "application/pdf" if path.suffix == ".pdf" else "text/csv;charset=utf-8"
        title = {"current_development_metrics.csv": "54 行当前性能 CSV", "historical_wrapper_metrics.csv": "历史比较 CSV", "dprism25_metrics.csv": "T2D 25 分数 CSV", "paired_r2.csv": "29 项 ΔR² CSV", "r2_recalculation.csv": "R² 复核 CSV", "calibration_deciles.csv": "校准分组 CSV", "External_PRS_Figures.pdf": "六幅图 PDF"}[path.name]
        downloads += f"<a class='download' download='{path.name}' href='data:{mime};base64,{encoded}'>{title}</a>"
    downloads += "</div>"
    appendix = """<section class='section' id='evidence'><div class='section-title'><span class='number'>+</span><h2>证据、统计口径与下载</h2></div>""" + table(["工作", "状态", "范围"], status_rows)
    appendix += "<details><summary>区间如何计算，哪些不确定性没有被覆盖</summary><div class='details-body'><p><strong>AUC：</strong>沿用各冻结实验的 2,000 次配对 bootstrap。当前修正比较是患者 bootstrap；D-PRISM25 / hard / soft / 历史比较使用结局分层的配对患者 bootstrap。既有 AUC 的随机种子与检验族不同：修正主比较六项 Holm；T2D 补充三项 Bonferroni；不将本报告汇总的行当成一次共同检验。</p><p><strong>新增 R²：</strong>seed = 20261012，2,000 次结局分层配对患者 bootstrap，使用已存概率，不重训模型。正文图中显示普通 95% 百分位区间；29 项新增总体比较统一提供 Bonferroni 同时区间，属于事后探索性分析。严格校正的尾部分位在 2,000 次重复下精度有限。</p><p>所有区间条件于已保存预测，不包括模型重训、历史开发选择或家系依赖的不确定性。患者级五折不等于已经核验的家系隔离。不同 evaluation、概率 wrapper 或 PRS 输入版本的指标不能拼接。外部权重来源与本项目的独立性沿用项目确认记录，尚无跨库参与者级连接审计。</p><p><strong>核验：</strong>在相同患者集合上，从保存概率复算 76 行 AUC / R²，并用已存 log loss 独立核对 R² 公式。原 Markdown 及其 16 个封存文件哈希均保持一致；没有修改旧实验结果。</p></div></details>"
    appendix += "<details><summary>全部 29 项新增 R² 比较与同时区间</summary><div class='details-body'>" + table(["疾病", "概率 / 实验范围", "比较", "ΔR²", "普通 95% CI", "29 项同时区间"], all_r2_rows) + "</div></details>"
    appendix += "<p>下列下载内容均已嵌入本 HTML，可离线保存；图表以 SVG 显示并另附 PDF。表格只含汇总统计。</p>" + downloads
    appendix += f"<p class='small'>原始完整文字报告：<a href='{os.path.relpath(DOC, OUT)}'>External_PRS_Progress_Review_20261012_CN.md</a>。新增统计凭证：<a href='R2_VERIFICATION.json'>R2_VERIFICATION.json</a>；页面核验凭证：<a href='HTML_VERIFICATION.json'>HTML_VERIFICATION.json</a>。</p></section>"
    nav = """<aside class='sidebar'><div class='brand'>DECIPHER / PRS</div><h2>研究进展报告</h2><div class='date'>2026 年 10 月 12 日</div><nav aria-label='报告目录'><a href='#overview'>概览 · 主要结果</a><a href='#s1'>01 · External PRS 来源</a><a href='#s2'>02 · 整合与历史比较</a><a href='#s3'>03 · Disco 对照</a><a href='#s4'>04 · 消融与概率质量</a><a href='#s5'>05 · 下一步方案</a><a href='#evidence'>证据 · 下载</a></nav><div class='tools'><button type='button' id='print'>打印 / PDF</button><button type='button' id='expand'>展开明细</button></div><p class='small'>主指标：AUC + Nagelkerke R²<br>补充：OR90、Brier、log loss、校准</p></aside>"""
    js = """document.getElementById('expand').addEventListener('click',function(){const open=this.dataset.open!=='true';document.querySelectorAll('details').forEach(d=>d.open=open);this.dataset.open=String(open);this.textContent=open?'收起明细':'展开明细';});document.getElementById('print').addEventListener('click',()=>{document.querySelectorAll('details').forEach(d=>d.open=true);window.print();});const links=[...document.querySelectorAll('nav a')];const obs=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){links.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));}});},{rootMargin:'-8% 0px -70% 0px'});document.querySelectorAll('main>section,header.hero').forEach(s=>obs.observe(s));"""
    return "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><meta name='description' content='External PRS 整合研究报告：外部模型、AUC与Nagelkerke R²、Disco比较、消融与蒸馏计划。'><title>External PRS 整合研究进展 · AUC 与 R² · 2026-10-12</title><style>" + CSS + "</style></head><body><a class='skip-link' href='#overview'>跳到报告正文</a><div class='layout'>" + nav + "<main>" + intro + one + two + three + four + five + appendix + "<footer class='foot'>研究范围：2026.10.10—10.11 已完成实验，附 2026.10.12 冻结预测 R² 补充分析。独立外部验证及学生蒸馏仍待完成。</footer></main></div><script>" + js + "</script></body></html>"


def main():
    receipt = json.loads((OUT / "R2_VERIFICATION.json").read_text())
    assert receipt["status"] == "verified" and receipt["point_rows_verified"] == 76
    for name, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected
    original = json.loads((BASE / "REPORT_VERIFICATION.json").read_text())
    for name, expected in original["files_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    make_figures()
    document = build_html()
    target = OUT / "External_PRS_Review_20261012_CN.html"
    target.write_text(document)
    assert document.count("<figure ") == 6
    assert "mermaid" not in document
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": SOURCES,
                "html": target.name, "figures": FIGURES, "patient_data_embedded": False,
                "source_report_unchanged": True, "new_model_fits": 0}
    (OUT / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"html": str(target.relative_to(ROOT)), "bytes": target.stat().st_size, "figures": len(FIGURES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
