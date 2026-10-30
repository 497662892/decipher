"""Build bilingual, aggregate-only presentations from sealed distillation predictions.

No model training, model selection, calibration correction, or job submission occurs.
Run from the repository root with the existing uv environment.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from decipehr.external_pgs import sha256_file, write_json
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.reporting import basic_metrics, calibration, fidelity

RUN = Path("outputs/prs_distillation_v1/phase_a_fulltrain_20261026_3d4334d2_0161ad36d950")
LR_RUN = Path("outputs/prs_distillation_v1/joint_logistic_comparison_fulltrain_20261026")
DEFAULT_OUT = Path("outputs/prs_distillation_v1/presentation_20261026")
TRAITS = ["T2D", "cad", "bc"]
ARMS = ["LR", "S_SUP", "S_KD_PURE", "S_KD_MIX", "T_MATCHED"]
STUDENTS = ARMS[1:4]
COLORS = dict(zip(ARMS, ["#8793A4", "#C78926", "#3786B8", "#087F70", "#293F70"], strict=True))
INK, MUTED, BG, LINE = "#172B44", "#5D7086", "#F4F7FB", "#DAE3ED"
NAMES = {
    "en": {"LR": "Joint logistic", "S_SUP": "Supervised student", "S_KD_PURE": "Pure distillation",
           "S_KD_MIX": "Mixed distillation", "T_MATCHED": "TabPFN v3 teacher"},
    "zh": {"LR": "全变量 Logistic", "S_SUP": "同结构监督学生", "S_KD_PURE": "纯蒸馏学生",
           "S_KD_MIX": "混合蒸馏学生", "T_MATCHED": "TabPFN v3 教师"},
}
SOURCES = [
    ("Scikit-learn: evaluation metrics", "https://scikit-learn.org/stable/modules/model_evaluation.html"),
    ("Scikit-learn: probability calibration", "https://scikit-learn.org/stable/modules/calibration.html"),
    ("SAS: Nagelkerke / max-rescaled R-squared", "https://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/statug_logistic_details13.htm"),
]


def tx(lang, zh, en):
    return zh if lang == "zh" else en


def trait_name(trait, lang):
    return {"T2D": "T2D", "cad": "CAD", "bc": tx(lang, "女性 BC", "Female BC")}[trait]


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_seal(directory, name):
    seal = read_json(directory / name)
    for relative, expected in seal["output_sha256"].items():
        if sha256_file(directory / relative) != expected:
            raise ValueError(f"Frozen artifact changed: {directory / relative}")
    return len(seal["output_sha256"])


def r2_intervals(y, predictions, groups):
    """Cluster bootstrap of fixed probabilities; recompute the null prevalence per draw."""
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = inverse.max() + 1
    group_n = np.bincount(inverse)
    group_y = np.bincount(inverse, weights=y)
    p = np.clip(np.column_stack([predictions[a] for a in ARMS]), 1e-15, 1-1e-15)
    row_ll = y[:, None] * np.log(p) + (1-y[:, None]) * np.log1p(-p)
    group_ll = np.zeros((n_groups, len(ARMS)))
    np.add.at(group_ll, inverse, row_ll)
    rng = np.random.default_rng(20261027)
    values = []
    for _ in range(1000):
        weights = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
        n = weights @ group_n
        prevalence = (weights @ group_y) / n
        if not 0 < prevalence < 1:
            continue
        null_mean_ll = prevalence * np.log(prevalence) + (1-prevalence) * np.log1p(-prevalence)
        mean_ll = weights @ group_ll / n
        values.append(-np.expm1(2*(null_mean_ll-mean_ll)) / -np.expm1(2*null_mean_ll))
    return np.array(values)


def load_evidence(root, lr_root):
    counts = {"core": verify_seal(root, "COMPLETE.json"), "logistic": verify_seal(lr_root, "VERIFIED.json")}
    complete = read_json(root / "COMPLETE.json")
    assert not complete["phase_b_executed"] and complete["training_only_development"]
    audit = read_json(root / "audit_report.json")
    label_path = Path(audit["paths"]["evaluation"])
    assert sha256_file(label_path) == audit["source_sha256"][str(label_path)]
    labels = pd.read_csv(label_path, dtype={"sampleid": str}, usecols=["sampleid", *TRAITS])
    assert labels.sampleid.is_unique
    split = pd.read_parquet(root / "split_manifest.parquet").set_index("sampleid")
    old = pd.read_csv(lr_root / "validation_metrics.csv").set_index(["trait", "arm"])
    old_core = pd.read_csv(root / "all_direct_metrics.csv").set_index(["trait", "arm"])
    metrics, cohorts, settings, curves, bins, histograms, ages = [], [], [], [], [], [], []
    predictions, locks, r2_effects = {}, {}, []
    patient_ids = set()
    for trait in TRAITS:
        directory = root / "core" / trait
        stored = pd.read_parquet(directory / "direct_predictions.parquet",
                                 columns=["sampleid", "kinship_group", "arm", "seed", "probability", "p_teacher"])
        lr = pd.read_parquet(lr_root / trait / "validation_predictions.parquet")
        assert lr.sampleid.is_unique
        joined = lr.merge(labels[["sampleid", trait]], on="sampleid", validate="one_to_one")
        assert len(joined) == len(lr) and joined[trait].isin([0, 1]).all()
        y = joined[trait].to_numpy()
        arrays = {"LR": joined.probability.to_numpy()}
        patient_ids.update(joined.sampleid)
        for arm in ARMS[1:]:
            part = stored.loc[stored.arm.eq(arm) & stored.seed.eq(17)].set_index("sampleid")
            assert part.index.is_unique and set(part.index) == set(joined.sampleid)
            part = part.loc[joined.sampleid]
            assert np.array_equal(part.kinship_group.to_numpy(), joined.kinship_group.to_numpy())
            arrays[arm] = part.probability.to_numpy()
            if arm == "T_MATCHED":
                np.testing.assert_array_equal(arrays[arm], part.p_teacher.to_numpy())
        train = pd.read_parquet(root / trait / "training.parquet", columns=["sampleid", "y", "sex"])
        eligible = pd.read_parquet(root / trait / "evaluation_features.parquet", columns=["sampleid", "sex"])
        assert set(eligible.sampleid) == set(joined.sampleid)
        assert not set(train.sampleid) & set(joined.sampleid)
        if trait == "bc":
            assert train.sex.eq("Female").all() and eligible.sex.eq("Female").all()
        roles = split.loc[train.sampleid, "internal_role"]
        lock = read_json(directory / "selection_lock.json")
        locks[trait] = lock
        assert lock["final_seed"] == 17 and lock["SELECTED_POLICY"] == "S_KD_MIX"
        pre = read_json(directory / "preprocessing.json")
        cohorts.append({"trait": trait, "train_n": len(train), "train_events": int(train.y.sum()),
                        "u_n": int(roles.eq("U").sum()), "h_n": int(roles.eq("H").sum()),
                        "val_n": len(y), "val_events": int(y.sum()), "val_prevalence": float(y.mean()),
                        "n_prs": len(pre["prs_columns"])})
        r2_samples = r2_intervals(y, arrays, joined.kinship_group.to_numpy())
        r2_point = {a: nagelkerke_r2(y, arrays[a]) for a in ARMS}
        for arm in ARMS:
            p = arrays[arm]
            measured = basic_metrics(y, p)
            # Independently reproduce the sealed endpoint metrics with sklearn.
            independent = {"auroc": roc_auc_score(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1-1e-6)),
                           "brier": brier_score_loss(y, p)}
            for key in independent:
                np.testing.assert_allclose(measured[key], independent[key], rtol=0, atol=1e-12)
                np.testing.assert_allclose(measured[key], old.loc[(trait, arm), key], rtol=0, atol=1e-12)
            cal, fid = calibration(y, p), fidelity(p, arrays["T_MATCHED"])
            pl, ql = [logit(np.clip(v, 1e-6, 1-1e-6)) for v in (p, arrays["T_MATCHED"])]
            np.testing.assert_allclose(fid["logit_r2"], 1-np.mean((pl-ql)**2)/np.var(ql), atol=1e-12)
            if arm != "LR":
                for key in [*fid, "calibration_intercept", "calibration_slope"]:
                    np.testing.assert_allclose((fid | cal)[key], old_core.loc[(trait, arm), key], rtol=0, atol=1e-10)
            prevalence = float(y.mean())
            null_ll = prevalence*np.log(prevalence)+(1-prevalence)*np.log1p(-prevalence)
            frozen_ll = -log_loss(y, np.clip(p, 1e-15, 1-1e-15))
            np.testing.assert_allclose(r2_point[arm], -np.expm1(2*(null_ll-frozen_ll)) / -np.expm1(2*null_ll), atol=1e-12)
            lo, hi = np.quantile(r2_samples[:, ARMS.index(arm)], [.025, .975])
            metrics.append({"trait": trait, "arm": arm, **measured, "average_precision": average_precision_score(y, p),
                            "nagelkerke_r2": r2_point[arm], "nagelkerke_ci_low": lo, "nagelkerke_ci_high": hi,
                            **cal, **fid})
            edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
            assignment = np.searchsorted(edges[1:-1], p)
            prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            local = []
            for k in np.unique(assignment):
                mask = assignment == k
                local.append({"trait": trait, "arm": arm, "bin": int(k), "n": int(mask.sum()),
                              "mean_prediction": float(p[mask].mean()), "event_fraction": float(y[mask].mean())})
            np.testing.assert_allclose([r["mean_prediction"] for r in local], prob_pred)
            np.testing.assert_allclose([r["event_fraction"] for r in local], prob_true)
            bins.extend(local)
            if arm in STUDENTS:
                meta = read_json(directory / "models" / arm / "seed17" / "model_metadata.json")
                selected = lock["arms"][arm]
                logs = meta["training"]["logs"]
                assert logs[-1]["epoch"] == selected["refit_epochs"] == meta["training"]["best_epoch"]
                settings.append({"trait": trait, "arm": arm, **selected, "batch_size": len(train),
                                 "learning_rate": meta["training"]["learning_rate"], "primary_seed": 17,
                                 "numerical_retries": len(meta["training"]["numerical_failures"])})
                curves.extend({"trait": trait, "arm": arm, "step": r["epoch"], "soft_target_bce": r["loss"],
                               "gradient_norm": r["gradient_norm"], "lambda": selected["lambda"]} for r in logs)
        for arm, reference in [(a, b) for a in STUDENTS for b in ["S_SUP", "LR", "T_MATCHED"] if a != b]:
            samples = r2_samples[:, ARMS.index(arm)] - r2_samples[:, ARMS.index(reference)]
            lo, hi = np.quantile(samples, [.025, .975])
            r2_effects.append({"trait": trait, "arm": arm, "reference": reference, "metric": "nagelkerke_r2",
                               "delta": r2_point[arm]-r2_point[reference], "ci_low": lo, "ci_high": hi,
                               "n_requested_replicates": 1000, "n_valid_replicates": len(samples),
                               "resampling_unit": "available_kinship_group", "refit_models": False,
                               "metric_source": "new_presentation_supplement"})
        h, xe, ye = np.histogram2d(arrays["T_MATCHED"], arrays["S_KD_MIX"], bins=25, range=[[0, 1], [0, 1]])
        assert int(h.sum()) == len(y)
        histograms.extend({"trait": trait, "teacher_left": xe[i], "student_left": ye[j],
                           "bin_width": xe[1] - xe[0], "n": int(h[i, j])} for i in range(25) for j in range(25))
        ages.append(pd.read_csv(directory / "age_scaling_by_seed.csv").assign(trait=trait))
        predictions[trait] = {"y": y, **arrays}
        print(f"Verified {trait}: matched patients, frozen metrics, calibration and training logs", flush=True)
    effects = pd.concat([pd.read_csv(root / "all_paired_differences.csv"),
                         pd.read_csv(lr_root / "paired_differences.csv"), pd.DataFrame(r2_effects)], ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    indexed = metric_frame.set_index(["trait", "arm"])
    for r in effects.itertuples():
        if r.arm in ARMS and r.reference in ARMS:
            np.testing.assert_allclose(r.delta, indexed.loc[(r.trait, r.arm), r.metric] -
                                       indexed.loc[(r.trait, r.reference), r.metric], atol=1e-12, rtol=0)
    return {"metrics": metric_frame, "cohorts": pd.DataFrame(cohorts), "settings": pd.DataFrame(settings),
            "training_curves": pd.DataFrame(curves), "calibration_bins": pd.DataFrame(bins),
            "fidelity_bins": pd.DataFrame(histograms), "age_scaling": pd.concat(ages, ignore_index=True),
            "paired_effects": effects, "predictions": predictions, "locks": locks,
            "verified_files": counts, "patient_ids": patient_ids, "label_path": label_path}


def style():
    font = Path("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
    if font.exists():
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Droid Sans Fallback"], "font.size": 11,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": LINE,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 14,
                         "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "path",
                         "savefig.facecolor": BG, "figure.facecolor": BG})


def slide(lang, number, title, subtitle, footer=None):
    fig = plt.figure(figsize=(13.333, 7.5))
    fig.text(.045, .949, tx(lang, "PRS 蒸馏  /  初步机制与性能", "PRS DISTILLATION  /  PRELIMINARY METHODS & RESULTS"),
             fontsize=10, color=MUTED, weight="bold")
    fig.text(.045, .883, title, fontsize=25, weight="bold")
    fig.text(.045, .837, subtitle, fontsize=11.5, color=MUTED)
    fig.add_artist(plt.Line2D([.045, .955], [.807, .807], color=LINE, lw=1))
    fig.text(.045, .035, footer or tx(lang, "Phase A · 原始 train 内开发，原始 val 评估 · 全量队列敏感性实验",
                                    "Phase A · Development within original train; evaluation on original val · Full-cohort sensitivity"),
             fontsize=8.5, color=MUTED)
    fig.text(.955, .035, f"{tx(lang, '图', 'FIG.')} {number:02d}", ha="right", fontsize=9, color=MUTED)
    return fig


def canvas(fig):
    ax = fig.add_axes([.045, .10, .91, .68])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, title, body, color=INK, size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                               fc="white", ec=LINE, lw=1))
    ax.text(x+.018, y+h-.042, title, fontsize=size+1, weight="bold", color=color, va="top")
    ax.text(x+.018, y+h-.115, body, fontsize=size, va="top", linespacing=1.7)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 color=MUTED, lw=1.5, connectionstyle="arc3,rad=0"))


def table(ax, rows, headers, widths=None, fontsize=11):
    ax.axis("off")
    obj = ax.table(cellText=rows, colLabels=headers, colWidths=widths, loc="center", cellLoc="left")
    obj.auto_set_font_size(False)
    obj.set_fontsize(fontsize)
    obj.scale(1, 1.85)
    for (r, _), cell in obj.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_linewidth(.5)
        cell.set_facecolor("#E7EEF6" if r == 0 else ("white" if r % 2 else "#F2F6FA"))
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
    return obj


def legend(fig, lang, arms=ARMS, y=.10):
    handles = [plt.Line2D([], [], marker="o", color=COLORS[a], lw=2, label=NAMES[lang][a]) for a in arms]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=len(arms), frameon=False, fontsize=10)


def metric_value(evidence, trait, arm, metric):
    d = evidence["metrics"]
    return float(d.loc[d.trait.eq(trait) & d.arm.eq(arm), metric].iloc[0])


def effect(evidence, trait, reference="S_SUP", metric="auroc", arm="S_KD_MIX"):
    d = evidence["paired_effects"]
    return d.loc[d.trait.eq(trait) & d.arm.eq(arm) & d.reference.eq(reference) & d.metric.eq(metric)].iloc[0]


def figures(e, lang):
    fig = slide(lang, 1, tx(lang, "Student：可分解的遗传贡献", "Student: an explicit genetic contribution"),
                tx(lang, "每个疾病独立训练；固定 PRS 输入，学习组合权重、年龄尺度与基础项。",
                   "One model per disease; frozen PRS inputs with learned combination weights, age scale and baseline."))
    ax = canvas(fig)
    rows = [
        (.70, tx(lang, "PRS + 祖源 PC", "PRS + ancestry PCs"), r"$Z_1,\ldots,Z_K$; $A=\mathrm{PC1\!:\!PC10}$",
         tx(lang, "祖源门控 → 组合 PRS", "Ancestry gate → combined PRS"),
         r"$\pi(A)=\mathrm{softmax}(WA+c)$" + "\n" + r"$G=\sum_k\pi_k(A)Z_k$"),
        (.40, tx(lang, "标准化年龄", "Standardized age"), r"$a=\mathrm{age\_sd}$",
         tx(lang, "年龄 → 正值尺度", "Age → positive scale"), r"$s(a)=\exp(u_0+\theta_s^T\phi(a))$"),
        (.10, tx(lang, "年龄 + PC + 性别", "Age + PCs + sex"), r"$\phi(a), A, C$",
         tx(lang, "协变量 → 基础项", "Covariates → baseline"), r"$b=\alpha+\theta_b^T\phi(a)+\gamma^T A+\delta^T C$"),
    ]
    for y, title, body, middle, formula in rows:
        box(ax, .01, y, .23, .23, title, body, size=11)
        box(ax, .31, y, .40, .23, middle, formula, color=COLORS["S_KD_MIX"], size=12)
        arrow(ax, (.25, y+.115), (.30, y+.115))
        arrow(ax, (.72, y+.115), (.80, .51))
    box(ax, .81, .34, .17, .35, tx(lang, "最终预测", "Prediction"),
        r"$\eta=b+sG$" + "\n" + r"$p=\sigma(\eta)$", color=COLORS["S_KD_MIX"], size=14)
    yield "architecture", fig

    fig = slide(lang, 2, tx(lang, "软标签与模型选择都在 train 内完成", "Soft labels and model selection stay inside train"),
                tx(lang, "Teacher = TabPFN v3，16 个 estimators，seed 42；冻结预训练权重，仅改变训练 context。",
                   "Teacher = TabPFN v3, 16 estimators, seed 42; fixed pretrained weights, different training contexts."))
    ax = canvas(fig)
    titles = [tx(lang, "① 内部开发 U / H", "1  Develop within U / H"), tx(lang, "② 锁定配置", "2  Lock the configuration"),
              tx(lang, "③ 全 train 重拟合", "3  Refit on all train"), tx(lang, "④ val 一次评估", "4  Evaluate on val")]
    bodies = [tx(lang, "U 内五折 OOF → qᵁ\nTeacher(U) → qᴴ\nStudent 在 U 训练\nH 选择与早停",
                 "5-fold OOF in U → qᵁ\nTeacher(U) → qᴴ\nTrain students on U\nSelect / early-stop on H"),
              tx(lang, "选择 λ 与正则强度\n锁定最终训练步数\n比较 3 个初始化\n主结果固定 seed 17",
                 "Choose λ and ridge\nLock refit steps\nCheck 3 initializations\nPrimary seed fixed at 17"),
              tx(lang, "D 内重新五折 OOF\n得到每人的 qᴰ\n全 D 训练 Student\n不再 early stop",
                 "Fresh 5-fold OOF in D\nOne qᴰ per patient\nTrain student on all D\nNo refit early stopping"),
              tx(lang, "Teacher(D) → val\nStudent(D) → val\n预测锁定后计算指标\n不按 val 改选模型",
                 "Teacher(D) → val\nStudent(D) → val\nScore locked predictions\nNo selection using val")]
    for i in range(4):
        box(ax, .005+i*.25, .40, .215, .47, titles[i], bodies[i], size=10.5)
        if i < 3:
            arrow(ax, (.228+i*.25, .63), (.25+i*.25, .63))
    ax.plot([.738, .738], [.32, .96], color="#B65D45", ls="--", lw=1.5)
    ax.text(.37, .96, tx(lang, "原始 train：D = U ∪ H", "Original train: D = U ∪ H"), ha="center", weight="bold")
    rows = [[trait_name(r.trait, lang), r.train_n, r.u_n, r.h_n, r.val_n] for r in e["cohorts"].itertuples()]
    table(fig.add_axes([.12, .14, .76, .16]), rows,
          [tx(lang, "疾病", "Trait"), "Train D", "U", "H", "Val E"], fontsize=10)
    yield "training_flow", fig

    fig = slide(lang, 3, tx(lang, "优化过程：实际记录的训练 BCE", "Optimization: recorded training BCE"),
                tx(lang, "固定最终步数的全 train 重拟合；每一步使用整个训练集，曲线不包含正则项。",
                   "Full-train refits with locked step counts; one full batch per step. Curves exclude the penalty."),
                tx(lang, "各条曲线的软目标不同，绝对 loss 高低不能作为跨模型性能排名；没有 val 逐步调参。",
                   "Targets differ across curves: absolute losses are not a cross-model performance ranking. No per-step val tuning."))
    axes = fig.subplots(1, 3, gridspec_kw={"left": .07, "right": .97, "bottom": .25, "top": .73, "wspace": .26})
    for ax, trait in zip(axes, TRAITS, strict=True):
        for arm in STUDENTS:
            d = e["training_curves"].query("trait == @trait and arm == @arm")
            ax.plot(d.step, d.soft_target_bce, color=COLORS[arm], lw=2)
            ax.scatter(d.step.iloc[-1], d.soft_target_bce.iloc[-1], color=COLORS[arm], s=24)
        ax.set(xscale="log", xlabel=tx(lang, "梯度更新步数（对数轴）", "Gradient steps (log scale)"),
               ylabel="BCE", title=trait_name(trait, lang))
        ax.grid(alpha=.15)
        settings = e["settings"].query("trait == @trait").set_index("arm")
        ax.text(.04, .94, tx(lang, "结束步数：", "Final steps: ") + " / ".join(str(int(settings.loc[a, "refit_epochs"])) for a in STUDENTS),
                transform=ax.transAxes, va="top", fontsize=9)
    legend(fig, lang, STUDENTS, y=.13)
    yield "training_loss", fig

    for number, key, metrics, title, subtitle in [
        (4, "ranking", [("auroc", "AUROC ↑"), ("average_precision", "Average precision (AP) ↑")],
         tx(lang, "区分能力：AUROC 与 Average Precision", "Discrimination: AUROC and average precision"),
         tx(lang, "同一疾病内使用完全相同的 val 患者；AP 虚线为该队列事件率。",
            "Identical val patients within each disease. The dashed AP line is the cohort event rate.")),
        (5, "probability_quality", [("log_loss", "Log loss ↓"), ("brier", "Brier score ↓")],
         tx(lang, "概率预测质量：不能只看 AUC", "Probability quality: beyond AUC"),
         tx(lang, "Log loss 与 Brier 都越低越好；它们同时涉及区分与校准，不能单独当作校准误差。",
            "Lower is better for both. These scores reflect discrimination and calibration, not calibration alone.")),
    ]:
        fig = slide(lang, number, title, subtitle)
        axes = fig.subplots(2, 3, gridspec_kw={"left": .18, "right": .95, "bottom": .17, "top": .73,
                                             "wspace": .38, "hspace": .62})
        for row, (metric, label) in enumerate(metrics):
            for col, trait in enumerate(TRAITS):
                ax = axes[row, col]
                values = [metric_value(e, trait, a, metric) for a in ARMS]
                width = max(values)-min(values)
                ax.set_xlim(min(values)-max(width*.30, .004), max(values)+max(width*.65, .012))
                for i, (arm, value) in enumerate(zip(ARMS, values, strict=True)):
                    ax.scatter(value, i, color=COLORS[arm], s=45, zorder=3)
                    ax.annotate(f"{value:.4f}", (value, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
                ax.set(yticks=range(5), yticklabels=[NAMES[lang][a] for a in ARMS] if col == 0 else [""]*5,
                       ylim=(4.6, -.6), xlabel=label)
                ax.tick_params(axis="y", length=0, labelsize=10)
                ax.grid(axis="x", alpha=.16)
                if row == 0:
                    n = e["cohorts"].query("trait == @trait").iloc[0]
                    ax.set_title(f"{trait_name(trait, lang)} · n={n.val_n:,}", pad=12)
                if metric == "average_precision":
                    prevalence = float(e["cohorts"].query("trait == @trait").val_prevalence.iloc[0])
                    lo, hi = ax.get_xlim()
                    ax.set_xlim(min(lo, prevalence-.015), hi)
                    ax.axvline(prevalence, color=MUTED, ls="--", lw=1)
        yield key, fig

    fig = slide(lang, 6, tx(lang, "结局预测的 Nagelkerke R²", "Nagelkerke R² for outcome prediction"),
                tx(lang, "新增计算：冻结的直接预测概率 vs val 事件率常数模型；越高越好。",
                   "New calculation: frozen direct probabilities versus a constant val-prevalence reference. Higher is better."),
                tx(lang, "1,000 次可用亲缘组 bootstrap，名义 95% CI；不是遗传解释度，也不是 Phase B 的正式分数。",
                   "1,000 available-family-group bootstrap draws; nominal 95% CIs. Not genetic variance explained or a Phase B score."))
    axes = fig.subplots(1, 3, gridspec_kw={"left": .18, "right": .96, "bottom": .25, "top": .71, "wspace": .25})
    for j, (ax, trait) in enumerate(zip(axes, TRAITS, strict=True)):
        for i, arm in enumerate(ARMS):
            r = e["metrics"].query("trait == @trait and arm == @arm").iloc[0]
            ax.errorbar(r.nagelkerke_r2, i, xerr=[[r.nagelkerke_r2-r.nagelkerke_ci_low], [r.nagelkerke_ci_high-r.nagelkerke_r2]],
                        fmt="o", color=COLORS[arm], capsize=3)
            ax.annotate(f"{r.nagelkerke_r2:.4f}", (r.nagelkerke_r2, i), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=9, color=COLORS[arm])
        ax.set(title=trait_name(trait, lang), xlabel="Nagelkerke R² ↑", ylim=(4.7, -.8),
               yticks=range(5), yticklabels=[NAMES[lang][a] for a in ARMS] if j == 0 else [""]*5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=.15)
    yield "nagelkerke_r2", fig

    fig = slide(lang, 7, tx(lang, "主比较：保持结构不变，蒸馏带来了什么？", "Primary comparison: what does distillation add?"),
                tx(lang, "混合蒸馏 − 同结构监督学生；模型各自在 train 内选择超参数。",
                   "Mixed distillation − the same-structure supervised student; each arm uses train-only hyperparameter selection."),
                tx(lang, "固定预测下的配对 bootstrap；名义 95% CI，未做多重比较校正，不含重新训练的不确定性。",
                   "Paired bootstrap of fixed predictions; nominal 95% CIs, no multiplicity correction or retraining uncertainty."))
    axes = fig.subplots(1, 3, gridspec_kw={"left": .105, "right": .96, "bottom": .28, "top": .71, "wspace": .40})
    for ax, metric, label in zip(axes, ["auroc", "log_loss", "nagelkerke_r2"],
                                 ["Δ AUROC →", "← Δ Log loss", "Δ Nagelkerke R² →"], strict=True):
        ax.axvline(0, color=MUTED, ls="--", lw=1)
        for i, trait in enumerate(TRAITS):
            r = effect(e, trait, metric=metric)
            ax.errorbar(r.delta, i, xerr=[[r.delta-r.ci_low], [r.ci_high-r.delta]], fmt="o",
                        color=COLORS["S_KD_MIX"], capsize=4, ms=7)
            ax.text(.5, .84-i*.30, f"{r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
                    transform=ax.transAxes, ha="center", fontsize=9)
        ax.set(yticks=range(3), yticklabels=[trait_name(t, lang) for t in TRAITS], ylim=(2.65, -.65), xlabel=label)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=.15)
        ax.margins(x=.20)
    fig.text(.5, .16, tx(lang, "T2D：有改善信号     |     CAD：变化很小     |     女性 BC：区间仍跨零",
                        "T2D: evidence of improvement     |     CAD: little change     |     Female BC: intervals still include zero"),
             ha="center", fontsize=13, weight="bold")
    yield "paired_distillation_effect", fig

    fig = slide(lang, 8, tx(lang, "校准：预测概率与实际事件率是否吻合？", "Calibration: do probabilities match event rates?"),
                tx(lang, "每个模型按预测概率分成 10 个分位数箱；点为各箱均值，虚线为理想校准。",
                   "Ten prediction-quantile bins per model; points are bin means. The diagonal indicates ideal calibration."),
                tx(lang, "这是描述性诊断，没有在 val 上校正预测；曲线未画置信区间，分箱计数另存于 CSV。",
                   "Descriptive diagnostics, without recalibrating predictions on val. No curve CIs; bin counts are exported in CSV."))
    axes = fig.subplots(1, 3, gridspec_kw={"left": .07, "right": .97, "bottom": .25, "top": .72, "wspace": .25})
    for ax, trait in zip(axes, TRAITS, strict=True):
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
        for arm in ARMS:
            d = e["calibration_bins"].query("trait == @trait and arm == @arm")
            ax.plot(d.mean_prediction, d.event_fraction, "o-", color=COLORS[arm], ms=3,
                    lw=2 if arm in ["S_KD_MIX", "T_MATCHED"] else 1, alpha=.9)
        ax.set(title=trait_name(trait, lang), xlim=(0, 1), ylim=(0, 1),
               xlabel=tx(lang, "平均预测概率", "Mean predicted probability"),
               ylabel=tx(lang, "观测事件率", "Observed event fraction"))
        ax.grid(alpha=.12)
    legend(fig, lang, y=.11)
    yield "calibration", fig

    fig = slide(lang, 9, tx(lang, "教师拟合度：Student 学到了多少？", "Teacher fidelity: how much does the student retain?"),
                tx(lang, "上排：混合学生与教师的二维概率分箱计数；下排：各学生的 Teacher-logit R²。",
                   "Top: binned probability counts, mixed student vs teacher. Bottom: teacher-logit R² for each student."),
                tx(lang, "Teacher-logit R² 衡量模仿程度；它不等于结局预测的 Nagelkerke R²，也不等于相关系数的平方。",
                   "Teacher-logit R² measures imitation, not Nagelkerke outcome R² or squared correlation."))
    for j, trait in enumerate(TRAITS):
        ax = fig.add_axes([.075+j*.31, .38, .245, .36])
        d = e["fidelity_bins"].query("trait == @trait")
        h = d.n.to_numpy().reshape(25, 25)
        ax.imshow(np.ma.masked_equal(h.T, 0), origin="lower", extent=(0, 1, 0, 1), cmap="Blues",
                  norm=LogNorm(vmin=1, vmax=max(2, h.max())), aspect="auto")
        ax.plot([0, 1], [0, 1], "--", color=COLORS["S_KD_MIX"], lw=1)
        r = e["metrics"].query("trait == @trait and arm == 'S_KD_MIX'").iloc[0]
        ax.set(title=trait_name(trait, lang), xlabel=tx(lang, "Teacher 概率", "Teacher probability"),
               ylabel=tx(lang, "混合 Student 概率", "Mixed student probability"))
        ax.text(.03, .95, f"MAE = {r.probability_mae:.3f}\nKL(q || p) = {r.bernoulli_kl:.4f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        low = fig.add_axes([.075+j*.31, .16, .245, .12])
        for i, arm in enumerate(STUDENTS):
            value = metric_value(e, trait, arm, "logit_r2")
            low.scatter(value, i, color=COLORS[arm], s=30)
            low.annotate(f"{value:.3f}", (value, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
        low.set(xlim=(.45, 1.02), ylim=(2.6, -.6), yticks=range(3),
                yticklabels=[tx(lang, "监督", "Supervised"), tx(lang, "纯蒸馏", "Pure KD"), tx(lang, "混合", "Mixed KD")],
                xlabel="Teacher-logit R² ↑")
        low.tick_params(axis="y", length=0, labelsize=9)
        low.grid(axis="x", alpha=.15)
    yield "teacher_fidelity", fig

    fig = slide(lang, 10, tx(lang, "当前年龄尺度：正值，但未施加单调先验", "Current age scale: positive, without monotonicity"),
                tx(lang, "每个疾病内所有 PRS 共用一条年龄尺度；年龄同时也进入基础项。",
                   "All PRS share one age-scale curve within a disease; age also enters the baseline term."),
                tx(lang, "横轴单位为 age_sd，不是岁。单调年龄新版本已暂停，尚无相应性能结果。",
                   "The x-axis is age_sd, not years. The proposed monotonic-age version is paused and has no performance results."))
    axes = fig.subplots(1, 3, gridspec_kw={"left": .075, "right": .97, "bottom": .26, "top": .72, "wspace": .30})
    for ax, trait in zip(axes, TRAITS, strict=True):
        for arm in STUDENTS:
            d = e["age_scaling"].query("trait == @trait and arm == @arm and seed == 17")
            ax.plot(d.age_sd, d.age_scale, color=COLORS[arm], lw=2)
        ax.set(title=trait_name(trait, lang), xlabel="age_sd", ylabel="$s(a)$")
        ax.grid(alpha=.15)
    legend(fig, lang, STUDENTS, y=.12)
    yield "age_scale", fig


FORMULAS = {
    "student": r"$\eta_i=b_i+s(a_i)\sum_{k=1}^{K}\pi_k(A_i)Z_{ik},\qquad p_i=\sigma(\eta_i)$",
    "baseline": r"$b_i=\alpha+\theta_b^{\mathsf{T}}\phi(a_i)+\gamma^{\mathsf{T}}A_i+\delta^{\mathsf{T}}C_i$",
    "gate": r"$\pi_k(A_i)=\frac{\exp(w_k^{\mathsf{T}}A_i+c_k)}{\sum_{j=1}^{K}\exp(w_j^{\mathsf{T}}A_i+c_j)},\quad \pi_k\geq0,\quad\sum_k\pi_k=1$",
    "scale": r"$s(a_i)=\exp\left(u_0+\theta_s^{\mathsf{T}}\phi(a_i)\right)>0,\qquad\beta_{ik}=s(a_i)\pi_k(A_i)$",
    "oof": r"$q_i=T\!\left(x_i;D\setminus F(i)\right),\qquad i\in D$",
    "target": r"$\widetilde y_i=(1-\lambda)y_i+\lambda q_i,\qquad q_i\ \mathrm{is\ fixed}$",
    "loss": r"$\mathcal{L}=\frac{1}{N}\sum_i\left[(1-\lambda)\,\mathrm{BCE}(y_i,p_i)+\lambda\,\mathrm{BCE}(q_i,p_i)\right]+\Omega$",
    "bce": r"$\mathrm{BCE}(t,p)=-t\log p-(1-t)\log(1-p)$",
    "regularizer": r"$\Omega=10^{-4}\Vert\beta_{b,\mathrm{nonint}}\Vert_2^2+\rho(\Vert W_c\Vert_F^2+\Vert c_c\Vert_2^2)+10^{-3}\Vert\theta_s\Vert_2^2$",
    "kl": r"$\mathrm{BCE}(q,p)=H(q)+D_{\mathrm{KL}}\!\left(\mathrm{Bern}(q)\,\Vert\,\mathrm{Bern}(p)\right)$",
    "nagelkerke": r"$R_N^2=\frac{1-\exp\left[\frac{2}{N}(\ell_0-\ell_p)\right]}{1-\exp\left(\frac{2}{N}\ell_0\right)},\quad \ell_p=\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]$",
    "fidelity_r2": r"$R_{\mathrm{Teacher\!\! -\!\! logit}}^2=1-\frac{\sum_i[\mathrm{logit}(p_i)-\mathrm{logit}(q_i)]^2}{\sum_i[\mathrm{logit}(q_i)-\overline{\mathrm{logit}(q)}]^2}$",
}


def save_formula(formula, path):
    fig = plt.figure(figsize=(12, .8), facecolor="white")
    fig.text(.015, .45, formula, fontsize=20, va="center")
    fig.savefig(path, bbox_inches="tight", pad_inches=.16, facecolor="white")
    plt.close(fig)


def inline_svg(path, alt, css=""):
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img class="{css}" src="data:image/svg+xml;base64,{encoded}" alt="{html.escape(alt, quote=True)}">'


def html_table(rows, headers):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in row) +
                                        '</tr>' for row in rows) + '</tbody></table></div>'


def report_html(e, lang, out):
    def t(zh, en):
        return tx(lang, zh, en)

    def formula(name):
        return '<div class="equation">' + inline_svg(out / "formulas" / f"{name}.svg", FORMULAS[name]) + '</div>'

    def figure(name, zh, en):
        caption = t(zh, en)
        return '<figure class="figure">' + inline_svg(out / lang / "figures" / f"{name}.svg", caption) + \
            f'<figcaption>{caption}</figcaption></figure>'

    def section(key, number, title):
        return f'<section id="{key}"><div class="kicker">{number}</div><h2>{title}</h2>'

    names = NAMES[lang]
    title = t("将 TabPFN 蒸馏为可解释的 PRS 学生模型", "Distilling TabPFN into an interpretable PRS student")
    nav = [("overview", t("概览", "Overview")), ("design", t("模型设计", "Design")),
           ("training", t("训练流程", "Training")), ("loss", t("损失函数", "Loss")),
           ("results", t("性能对比", "Results")), ("r2", "R²"),
           ("fidelity", t("教师拟合度", "Fidelity")), ("interpretation", t("结论与边界", "Interpretation"))]
    body = f'<header><div class="eyebrow">DECIP EHR · PHASE A · 26 OCT 2026</div><h1>{title}</h1>'
    body += '<p>' + t("从 Teacher 软标签到显式预测公式：模型怎么设计、怎么训练，以及蒸馏是否改善初步预测表现。",
                     "From teacher soft labels to an explicit prediction formula: the design, the training process, and the preliminary benefit of distillation.") + '</p>'
    badges = ["TabPFN v3", t("T2D / CAD / 女性 BC", "T2D / CAD / female BC"), t("train 内开发 · val 评估", "Develop on train · evaluate on val"),
              t("直接预测 · Phase B 未运行", "Direct predictions · Phase B not run"), t("新模型实验已暂停", "New model experiments paused")]
    body += '<div class="badges">' + ''.join(f'<span class="badge">{b}</span>' for b in badges) + '</div></header><nav>'
    body += ''.join(f'<a href="#{key}">{name}</a>' for key, name in nav)
    other = "en" if lang == "zh" else "zh"
    body += f'<a class="language" href="distillation_{other}.html">{t("English version", "中文版")}</a></nav><main>'
    body += section("overview", "01 / CONTEXT", t("先看我们正在比较什么", "What is being compared?"))
    body += '<p class="lead">' + t("核心问题是：能否用一个结构透明的小模型，学习 TabPFN Teacher 的预测规律，并在相同输入上优于仅用真实标签训练的同结构学生？",
        "Can a small, transparent model learn the predictive behavior of a TabPFN teacher and outperform the same student architecture trained only on true labels?") + '</p><div class="grid">'
    for r in e["cohorts"].itertuples():
        body += f'<div class="card"><h3>{trait_name(r.trait, lang)}</h3><div class="stat">{r.n_prs} <small>PRS</small></div>'
        body += f'<p>Train: {r.train_n:,} · Val: {r.val_n:,}<br>{t("val 事件", "Val events")}: {r.val_events:,} ({r.val_prevalence:.1%})</p></div>'
    body += '</div><div class="note">' + t("主比较：混合蒸馏学生 vs 同结构监督学生。与普通 Logistic 的差值同时包含模型结构和训练目标的变化，不能全部归因于蒸馏。",
        "Primary contrast: mixed distillation versus the same-structure supervised student. Differences versus joint logistic regression combine architecture and objective changes; they cannot all be attributed to distillation.") + '</div>'
    body += '<p class="muted">' + t("本次是原始全量队列的敏感性分析。val 此前已用于开发；保留了跨 train/val 的近亲关系，因此不属于新的独立外部验证。",
        "This is a sensitivity analysis on the original full cohort. Val has been used for earlier development, and relatives across train/val are retained; this is not a new independent external validation.") + '</p></section>'

    body += section("design", "02 / STUDENT", t("把预测拆成基础风险与遗传贡献", "Decompose the prediction into baseline and genetic terms"))
    body += formula("student")
    body += '<p class="formula-caption">' + t("Z 是训练范围内标准化的 PRS；A 是 PC1–PC10；a 是 age_sd；C 是性别编码；σ 是 sigmoid。最终概率由基础项与遗传项共同决定。",
        "Z denotes PRS standardized within the training scope; A is PC1–PC10, a is age_sd, C is the sex encoding, and σ is sigmoid. Baseline and genetic terms jointly determine the probability.") + '</p>'
    body += figure("architecture", "实际训练的 v1 结构。T2D、CAD、女性 BC 分别使用 26、7、5 个固定 PRS。",
                   "The v1 architecture actually trained, using 26, 7 and 5 frozen PRS for T2D, CAD and female BC, respectively.")
    body += '<h3>' + t("① 基础项：年龄、PC 与性别", "1. Baseline: age, PCs and sex") + '</h3>' + formula("baseline")
    body += '<p>' + t("年龄通过三个二次样条基函数进入模型；节点取当前训练范围的最小值、中位数和最大值，并将基函数中心化。女性 BC 的性别为常数，去掉该项。",
        "Age enters through three quadratic spline basis functions, with knots at the training-scope minimum, median and maximum and centered basis columns. The constant sex term is removed for female BC.") + '</p>'
    body += '<h3>' + t("② 祖源门控：哪些 PRS 更重要？", "2. Ancestry gate: which PRS receive more weight?") + '</h3>' + formula("gate")
    body += '<p>' + t("门控只读取连续 PC。每位患者的 PRS 权重非负且总和为 1。这些是模型组合权重，不是祖源比例，也不是各 PRS 的因果贡献。",
        "The gate reads continuous PCs only. PRS weights are nonnegative and sum to one for each patient. They are model-combination weights, not ancestry proportions or causal contributions.") + '</p>'
    body += '<h3>' + t("③ 年龄尺度：组合 PRS 的作用有多强？", "3. Age scale: how strongly does the combined PRS enter?") + '</h3>' + formula("scale")
    body += '<p>' + t("同一疾病内所有 PRS 共用一条 s(a)。u₀ 与三个年龄系数一起学习；初始化为 s(a)=0.25。当前仅约束尺度为正，没有约束随年龄递增，也没有保证总风险随年龄单调。",
        "All PRS share one s(a) within a disease. The global intercept u₀ and three age coefficients are learned jointly, starting at s(a)=0.25. The current model enforces positivity, not an increasing age scale or monotonic total risk.") + '</p>'
    body += '<details><summary>' + t("输入对齐与模型对照", "Input alignment and model controls") + '</summary>'
    rows = [[names["LR"], t("PRS + 线性 age_sd + sex + PC1–10；L2 Logistic，无交互，系数可正可负", "PRS + linear age_sd + sex + PC1–10; L2 logistic, no interactions, unrestricted coefficient signs")],
            [names["S_SUP"], t("上述完整 Student 结构，只用真实标签，λ=0", "Full student architecture above; true labels only, λ=0")],
            [names["S_KD_PURE"], t("相同结构，只拟合 Teacher OOF 概率，λ=1", "Same architecture; teacher OOF probabilities only, λ=1")],
            [names["S_KD_MIX"], t("相同结构，真实标签与 Teacher OOF 概率混合", "Same architecture; a mixture of true labels and teacher OOF probabilities")],
            [names["T_MATCHED"], t("相同 PRS、年龄、性别与连续 PC 输入；历史最佳配置的原始概率", "Same PRS, age, sex and continuous-PC inputs; raw probabilities from the adopted historical configuration")]]
    body += html_table(rows, [t("方法", "Method"), t("含义", "Definition")])
    body += '<p>' + t("所有方法都使用祖源 PC。这里的匹配 Logistic 不含历史 T2D 普通回归额外使用的 5 个 soft ancestry 列。",
        "All methods use ancestry PCs. The matched logistic control here does not include the five extra soft-ancestry columns used by the older T2D logistic baseline.") + '</p></details></section>'

    body += section("training", "03 / TRAINING", t("Teacher prediction 不能用训练内拟合值替代", "Teacher targets require out-of-fold predictions"))
    body += formula("oof")
    body += '<p>' + t("F(i) 是包含患者 i 的预先固定训练折。Teacher 只用另外四折作为 context，预测该患者，保证软标签的 context 不含该患者自己的结局。已知亲缘组在 train 内保持同折。",
        "F(i) is the fixed training fold containing patient i. The teacher conditions on the other four folds, so the context generating that patient's soft label excludes their own outcome. Known relatives remain together within train folds.") + '</p>'
    body += figure("training_flow", "U/H 用于内部选择；最终全 train 的 qᴰ 重新做五折 OOF。val 只在预测锁定后用于评分。",
                   "U/H is used for internal selection. Final qᴰ is regenerated by five-fold OOF on all train. Val is scored only after predictions are locked.")
    body += '<div class="note">' + t("这里的 OOF 是 Teacher 训练软标签。现有 Student 的全 train 拟合值不是严格的 Student pipeline OOF；若以后运行 Phase B，需要额外的嵌套 Student OOF。",
        "These OOF predictions are teacher training targets. The final student's fitted train predictions are not strict student-pipeline OOF predictions; Phase B would require additional nested student OOF fits.") + '</div>'
    body += '<p>' + t("Teacher 生成训练软标签时约使用 80% 的 train 作为 context；最终 val 的 Teacher 预测使用 100% train。两者 context 不同，因此不能把教师拟合误差简单解释成模型容量差距。",
        "The teacher uses about 80% of train to generate each training soft label, but 100% of train to predict val. This context difference means teacher-fidelity error is not solely a model-capacity gap.") + '</p></section>'

    body += section("loss", "04 / OBJECTIVE", t("蒸馏目标、正则项与实际优化过程", "Distillation objective, regularization and optimization"))
    body += formula("target") + formula("loss") + formula("bce")
    body += '<p>' + t("qᵢ 是冻结并缓存的 Teacher OOF 概率，不接收梯度。上述加权损失等价于直接对软目标 ỹᵢ 计算 BCE。纯蒸馏 λ=1；监督学生 λ=0；混合学生的 λ 只在 train 内选择。没有另加蒸馏温度。",
        "qᵢ is a frozen, cached teacher OOF probability and receives no gradient. The weighted objective equals BCE against the soft target ỹᵢ. Pure distillation uses λ=1 and supervised training uses λ=0; mixed λ is selected within train. No additional distillation temperature is applied.") + '</p>'
    body += formula("regularizer")
    body += '<p class="formula-caption">' + t("基础项非截距参数使用 L2；W、c 在 PRS 维度中心化后正则化；不惩罚基础截距或 u₀。ρ 只在 train 内选择。",
        "L2 applies to non-intercept baseline coefficients and the gate parameters centered across PRS. The baseline intercept and u₀ are unpenalized. ρ is selected within train.") + '</p>'
    body += '<div class="grid two"><div class="card"><h3>' + t("优化器", "Optimizer") + '</h3><p>' + t(
        "Adam，学习率 0.01，float64；full batch，一步更新使用整个训练集；梯度范数裁剪为 10。Teacher 权重保持冻结。",
        "Adam, learning rate 0.01, float64; full-batch updates over the entire training set; gradient-norm clipping at 10. Teacher weights remain frozen.") + '</p></div><div class="card"><h3>' + t("内部选择", "Internal selection") + '</h3><p>' + t(
        "最多 2,000 步，每 10 步检查 H，patience=200。纯蒸馏按 H 上 Teacher BCE 选择；监督/混合按 H 真实标签 log loss 选择。最终步数取三个初始化的中位最佳值。",
        "Up to 2,000 steps, checking H every 10 steps, patience 200. Pure KD uses teacher BCE on H; supervised/mixed use true-label log loss on H. Final steps are the median best step across three initializations.") + '</p></div></div>'
    rows = []
    for trait in TRAITS:
        d = e["settings"].query("trait == @trait").set_index("arm")
        rows.append([trait_name(trait, lang), int(d.loc["S_KD_MIX", "batch_size"]),
                     *[int(d.loc[a, "refit_epochs"]) for a in STUDENTS], d.loc["S_KD_MIX", "lambda"],
                     f'{d.loc["S_KD_MIX", "gate_ridge"]:g}'])
    body += html_table(rows, [t("疾病", "Trait"), "Batch size", t("监督步数", "Supervised steps"), t("纯蒸馏步数", "Pure KD steps"),
                             t("混合步数", "Mixed KD steps"), "Mixed λ", "Mixed ρ"])
    body += '<p class="muted">' + t("初始化 seeds 为 17、29、43，主结果预先固定使用 17，不按 val 选择 seed，也不做三个 seed 的预测平均。CAD 达到预设 2,000 步上限；尚未检验更长训练。",
        "Initialization seeds are 17, 29 and 43. Primary results use prespecified seed 17, without selecting or averaging seeds on val. CAD reached the 2,000-step cap; longer training has not been tested.") + '</p>'
    body += figure("training_loss", "曲线来自已保存的最终 refit 日志，展示各自软目标的 BCE，不是共同真实标签的验证 loss，也不包含正则项。",
                   "Curves come from saved final-refit logs: BCE against each arm's own target, not a shared validation-label loss, and excluding regularization.")
    body += '<details><summary>' + t("为什么 Teacher BCE 可以作为蒸馏 loss？", "Why does teacher BCE implement distillation?") + '</summary>' + formula("kl")
    body += '<p>' + t("Teacher q 固定时，H(q) 不随 Student 改变，所以最小化软标签 BCE 等价于最小化 Bernoulli KL(q‖p)。混合目标则在模仿 Teacher 与贴近真实标签之间折中。",
        "With fixed q, H(q) does not depend on the student. Minimizing soft-label BCE therefore minimizes Bernoulli KL(q‖p). The mixed objective balances teacher imitation with agreement to the observed labels.") + '</p></details></section>'

    body += section("results", "05 / PERFORMANCE", t("同一 val 上的初步性能对比", "Preliminary performance on the same val cohort"))
    body += '<p>' + t("下面全部使用冻结的直接预测概率。AUROC 衡量排序；AP 补充观察病例不均衡下的检出表现；log loss 与 Brier 衡量概率预测质量。未指定临床决策阈值，因此不以 accuracy 或 F1 作为主指标。",
        "All results use frozen direct probabilities. AUROC measures ranking; AP complements it under class imbalance; log loss and Brier assess probability quality. With no prespecified decision threshold, accuracy and F1 are not primary endpoints.") + '</p>'
    metrics_ui = {"auroc": "AUROC ↑", "average_precision": "Average precision (AP) ↑", "log_loss": "Log loss ↓",
                  "brier": "Brier score ↓", "nagelkerke_r2": "Nagelkerke R² ↑"}
    body += '<div class="controls"><label>' + t("疾病", "Disease") + '<select id="trait-select">' + ''.join(
        f'<option value="{tr}">{trait_name(tr, lang)}</option>' for tr in TRAITS) + '</select></label><label>' + t("指标", "Metric")
    body += '<select id="metric-select">' + ''.join(f'<option value="{k}">{v}</option>' for k, v in metrics_ui.items()) + '</select></label><label>'
    body += t("配对差值的参照模型", "Reference for paired differences") + '<select id="reference-select">' + ''.join(
        f'<option value="{a}">{names[a]}</option>' for a in ["S_SUP", "LR", "T_MATCHED"]) + '</select></label></div>'
    body += '<p id="explorer-cohort" class="muted"></p><div id="explorer-chart"></div><p id="metric-description" class="formula-caption"></p>'
    body += '<div class="table-wrap"><table id="explorer-table"></table></div><p class="muted">' + t(
        "Δ 为当前方法减参照方法；区间为配对差值的名义 95% CI。破折号表示没有该区间，不表示差值为零。AP 是 average precision，不是梯形积分的 PR-AUC。",
        "Δ is the method minus the reference. Intervals are nominal 95% CIs for paired differences. A dash means no interval is available, not a zero difference. AP is average precision, not trapezoidal PR-AUC.") + '</p>'
    body += figure("ranking", "AP 的无信息参考为同一 val 队列的事件率；不同疾病的 AP 不宜直接当作难度一致的排名。",
                   "The no-information AP reference is the event rate in the same val cohort; AP across diseases reflects differing prevalences.")
    body += figure("probability_quality", "Brier 是概率平方误差；低 Brier 不必然表示校准更好，下面另展示校准曲线与参数。",
                   "Brier is squared probability error. Lower Brier alone does not establish better calibration; calibration curves and parameters are shown separately.")
    body += figure("paired_distillation_effect", "保持 Student 结构相同的主比较。区间沿用已封存的 AUROC/log loss 结果；Nagelkerke R² 区间为本次补算。",
                   "The primary architecture-matched contrast. AUROC/log-loss intervals are reused from sealed results; Nagelkerke R² intervals are newly calculated.")
    body += '<details><summary>' + t("展开完整数值表", "Show the full numerical table") + '</summary>'
    rows = [[trait_name(r.trait, lang), names[r.arm], *[f'{getattr(r, m):.4f}' for m in metrics_ui]] for r in e["metrics"].itertuples()]
    body += html_table(rows, [t("疾病", "Trait"), t("方法", "Method"), *metrics_ui.values()]) + '</details></section>'

    body += section("r2", "06 / TWO DIFFERENT R² VALUES", t("R² 要区分结局预测与教师拟合", "Separate outcome R² from teacher-fidelity R²"))
    body += '<h3>' + t("A. Nagelkerke R²：针对真实二分类结局", "A. Nagelkerke R²: against the observed binary outcome") + '</h3>' + formula("nagelkerke")
    body += '<p>' + t("ℓₚ 是固定预测概率的验证集 log-likelihood；ℓ₀ 是用 val 事件率作为常数概率的 null log-likelihood。复用项目既有 Nagelkerke 实现，概率裁剪至 [10⁻¹⁵, 1−10⁻¹⁵]。这里只计算评分参考，不重新校正 Student 预测。",
        "ℓₚ is the validation log-likelihood of fixed predictions; ℓ₀ uses the val event rate as a constant-probability null reference. The existing project implementation clips probabilities to [10⁻¹⁵, 1−10⁻¹⁵]. This defines a scoring reference and does not recalibrate student predictions.") + '</p>'
    body += '<p>' + t("这是 pseudo-R²，不是遗传力或线性回归中解释方差的百分比。对冻结的验证集预测，它可以为负；不应截断为零。同一批患者上，它与 log loss 使用相同的似然信息。",
        "This is a pseudo-R², not heritability or a percentage of variance explained as in linear regression. For frozen validation predictions it can be negative and is not truncated at zero. Within the same cohort it uses the same likelihood information as log loss.") + '</p>'
    body += figure("nagelkerke_r2", "每个点附带 1,000 次可用亲缘组 bootstrap 的名义 95% CI；组内患者一同重采样，null 事件率每次重新计算。",
                   "Each point has a nominal 95% CI from 1,000 available-family-group bootstrap draws. Group members are resampled together and null prevalence is recomputed in every draw.")
    body += '<h3>' + t("B. Teacher-logit R²：针对 Teacher 输出", "B. Teacher-logit R²: against the teacher output") + '</h3>' + formula("fidelity_r2")
    body += '<p>' + t("这里 q 是同一 val 患者的 Teacher 概率。R²=1 表示 logit 完全一致；0 表示与预测 Teacher 平均 logit 的平方误差相同，也允许负值。它不是相关系数平方，概率取 logit 前裁剪至 [10⁻⁶, 1−10⁻⁶]。",
        "Here q is the teacher probability for the same val patient. R²=1 means identical logits; 0 matches the squared error of predicting the mean teacher logit, and negative values are possible. This is not squared correlation. Probabilities are clipped to [10⁻⁶, 1−10⁻⁶] before taking logits.") + '</p></section>'

    body += section("fidelity", "07 / PROBABILITIES & FIDELITY", t("概率是否可靠，Student 是否接近 Teacher？", "Are probabilities reliable, and is the student close to the teacher?"))
    body += figure("calibration", "理想校准对应截距 0、斜率 1。下表通过 val 上 logit(p) 的诊断性 logistic 拟合得到，不用于修改任何预测。",
                   "Ideal calibration has intercept 0 and slope 1. The diagnostic logistic fit on val logit(p) below is not used to modify any prediction.")
    rows = [[trait_name(r.trait, lang), names[r.arm], f'{r.calibration_intercept:.3f}', f'{r.calibration_slope:.3f}'] for r in e["metrics"].itertuples()]
    body += '<details><summary>' + t("校准截距与斜率", "Calibration intercepts and slopes") + '</summary>'
    body += html_table(rows, [t("疾病", "Trait"), t("方法", "Method"), t("截距，理想值 0", "Intercept, ideal 0"), t("斜率，理想值 1", "Slope, ideal 1")]) + '</details>'
    body += figure("teacher_fidelity", "二维分箱为聚合计数，颜色越深计数越高；MAE / RMSE / KL 越低越接近 Teacher。只模仿得更像，不能单独证明疾病预测更好。",
                   "The two-dimensional bins contain aggregate counts; darker bins contain more patients. Lower MAE/RMSE/KL means closer teacher agreement. Better imitation alone does not establish better outcome prediction.")
    rows = [[trait_name(r.trait, lang), names[r.arm], f'{r.probability_mae:.4f}', f'{r.probability_rmse:.4f}',
             f'{r.bernoulli_kl:.4f}', f'{r.logit_r2:.4f}'] for r in e["metrics"].query("arm in @STUDENTS").itertuples()]
    body += html_table(rows, [t("疾病", "Trait"), t("方法", "Method"), "MAE ↓", "RMSE ↓", "KL(q || p) ↓", "Teacher-logit R² ↑"]) + '</section>'

    body += section("interpretation", "08 / INTERPRETATION", t("目前可以得出什么结论？", "What can be concluded at this stage?"))
    for trait, zh, en in [
        ("T2D", "相对同结构监督学生，混合蒸馏的 AUROC、log loss 与 Nagelkerke R² 都有改善，名义配对区间排除零。",
         "Versus the same-structure supervised student, mixed distillation improves AUROC, log loss and Nagelkerke R², with nominal paired intervals excluding zero."),
        ("cad", "混合蒸馏与同结构监督学生的差异很小，尚无明确的性能改善证据。",
         "Mixed distillation differs little from the supervised student, without clear evidence of a performance gain."),
        ("bc", "AUROC 的点估计提高，但配对区间仍跨零；log loss 与 Nagelkerke R² 未改善。相对普通 Logistic 的提升包含结构差异。",
         "The AUROC point estimate rises, but the paired interval includes zero; log loss and Nagelkerke R² do not improve. Gains over joint logistic also include architecture changes."),
    ]:
        a, r = effect(e, trait), effect(e, trait, metric="nagelkerke_r2")
        body += f'<div class="finding"><strong>{trait_name(trait, lang)}</strong><div>{t(zh, en)}<br><span class="muted">'
        body += f'ΔAUROC {a.delta:+.4f} [{a.ci_low:+.4f}, {a.ci_high:+.4f}] · ΔR²<sub>N</sub> {r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]</span></div></div>'
    body += '<p>' + t("Teacher 在三种疾病上仍有更高的 AUROC、较低的 log loss 和更高的 Nagelkerke R²。当前结果支持部分知识转移，但还不能宣称 Student 已达到 Teacher 性能。",
        "The teacher still has higher AUROC, lower log loss and higher Nagelkerke R² in all three diseases. These results support partial knowledge transfer, but do not establish teacher-level student performance.") + '</p>'
    body += figure("age_scale", "展示的是当前已完成模型；单调年龄先验属于待研究的新版本，此次仅改进展示和补充指标。",
                   "These are the completed models. The monotonic-age prior belongs to a future version; this work changes presentation and supplements evaluation metrics only.")
    limitations = [
        t("完整原始队列保留 115 个跨 train/val 的已知亲缘组；train 112 人、val 42 人亲缘关系未确定。bootstrap 无法消除这类依赖或潜在乐观偏差。",
          "The full cohort retains 115 known family groups crossing train/val; kinship is unresolved for 112 train and 42 val patients. Bootstrap intervals cannot remove this dependence or possible optimism."),
        t("所有新增预处理、软标签、参数选择与拟合都在 train 内完成；但冻结 PRS 面板与该 val 曾参与历史开发，因此仍是探索性开发结果。",
          "All new preprocessing, soft labels, model selection and fitting stay inside train. However, the frozen panels and this val cohort have a development history, so results remain exploratory."),
        t("区间条件于固定模型与预测，未包含完整重训练的不确定性，未做多重比较校正。初始化稳定性不能替代置信区间。",
          "Intervals condition on fixed models and predictions; they exclude full retraining uncertainty and multiplicity adjustment. Initialization stability is not a confidence interval."),
        t("当前年龄单位是 age_sd；没有单调年龄约束。Phase B / Stage-2 尚未运行，未使用 blind holdout 标签。",
          "Age is measured in age_sd, with no monotonic-age constraint. Phase B / Stage-2 has not run, and no blind holdout labels were used."),
    ]
    body += '<details open><summary>' + t("解释范围", "Interpretation boundaries") + '</summary><ul>' + ''.join(f'<li>{x}</li>' for x in limitations) + '</ul></details>'
    csv_data = base64.b64encode(e["metrics"].to_csv(index=False).encode()).decode()
    body += f'<p><a download="metrics.csv" href="data:text/csv;base64,{csv_data}">{t("下载全部聚合指标 CSV", "Download all aggregate metrics (CSV)")}</a></p>'
    body += '<details><summary>' + t("来源与复现", "Sources and reproducibility") + '</summary><p>'
    body += t("本报告核对了核心运行 925 个文件、Logistic 对照 90 个文件的封存哈希；逐一对齐验证患者并复算原有指标。两种语言版本使用完全相同的数据。",
        "Sealed hashes were checked for 925 core-run files and 90 logistic-control files. Val patients were aligned and original metrics independently reproduced. Both language versions use exactly the same data.")
    body += '</p><p><code>' + str(RUN) + '</code><br><code>' + str(LR_RUN) + '</code></p><p>'
    body += t("补充指标：AP、直接预测 Nagelkerke R² 及其 1,000 次组 bootstrap；LR 校准与拟合度。原有模型和预测未修改。",
        "Supplementary calculations: AP, direct-prediction Nagelkerke R² and its 1,000 group-bootstrap draws, plus LR calibration and fidelity. Existing models and predictions were not modified.") + '</p>'
    body += '<ul class="sources">' + ''.join(f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">{name}</a></li>' for name, url in SOURCES) + '</ul></details></section></main>'
    body += '<footer>' + t("离线独立 HTML · 仅含聚合图表与指标 · 公式和图形已内嵌，无需联网加载",
                           "Standalone offline HTML · Aggregate charts and metrics only · Equations and graphics embedded; no network loading required") + '</footer>'
    descriptions = {
        "auroc": t("AUROC：越高越好，衡量病例相对非病例的排序。图为点估计；下表显示可用的配对差值区间。",
                   "AUROC: higher is better; measures case/non-case ranking. The plot shows point estimates; paired-difference intervals appear below where available."),
        "average_precision": t("AP：越高越好，使用 sklearn average_precision_score。虚线为 val 事件率；AP 本次仅报告点估计。",
                               "AP: higher is better, computed with sklearn average_precision_score. The dashed line is val prevalence; only point estimates are reported for AP."),
        "log_loss": t("Log loss：越低越好；对自信但错误的预测惩罚更大。使用自然对数，概率裁剪至 [10⁻⁶, 1−10⁻⁶]。",
                      "Log loss: lower is better; penalizes confident errors. Natural logarithms, with probabilities clipped to [10⁻⁶, 1−10⁻⁶]."),
        "brier": t("Brier：越低越好，平均 (p−y)²；不是纯粹的校准指标。", "Brier: lower is better; mean (p−y)². It is not a pure calibration metric."),
        "nagelkerke_r2": t("Nagelkerke R²：越高越好，基于直接预测的验证集似然。点的误差线与差值区间均为 1,000 次可用亲缘组 bootstrap 的名义 95% CI。",
                            "Nagelkerke R²: higher is better, based on direct-prediction validation likelihood. Point error bars and difference intervals are nominal 95% CIs from 1,000 available-family-group bootstrap draws."),
    }
    data = {"metrics": json.loads(e["metrics"].to_json(orient="records", double_precision=15)),
            "cohorts": json.loads(e["cohorts"].to_json(orient="records", double_precision=15)),
            "effects": json.loads(e["paired_effects"].loc[e["paired_effects"].arm.isin(ARMS) & e["paired_effects"].reference.isin(ARMS),
                                                          ["trait", "arm", "reference", "metric", "delta", "ci_low", "ci_high"]].to_json(orient="records", double_precision=15))}
    ui = {"names": names, "colors": COLORS, "traits": {tr: trait_name(tr, lang) for tr in TRAITS}, "metrics": metrics_ui,
          "descriptions": descriptions, "events": t("事件", "events"), "method": t("方法", "Method"),
          "interval": t("Δ 的 95% CI", "95% CI for Δ"), "chartLabel": t("所选疾病与指标的模型比较", "Model comparison for the selected disease and metric")}
    template = Path(__file__).parent / "templates" / "prs_distillation_report.html"
    content = template.read_text()
    for key, value in {"LANG": "zh-CN" if lang == "zh" else "en", "TITLE": title, "BODY": body,
                       "DATA": json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"),
                       "UI": json.dumps(ui, ensure_ascii=False).replace("</", "<\\/")}.items():
        content = content.replace(f"@@{key}@@", value)
    assert "@@" not in content
    return content


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output
    for source in [RUN, LR_RUN]:
        if source.resolve() == out.resolve() or source.resolve() in out.resolve().parents:
            raise ValueError("Presentation output must not alter a sealed experiment directory")
    out.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence(RUN, LR_RUN)
    style()
    provenance = {
        "created_utc": datetime.now(UTC).isoformat(), "status": "presentation_only",
        "model_development_paused": True, "new_predictive_models_fitted": False,
        "new_jobs_submitted": False, "phase_b_executed": False, "primary_seed": 17,
        "cohort_policy": "original_full_sensitivity", "bootstrap_replicates": 1000,
        "bootstrap_seed": 20261027, "bootstrap_scope": "fixed_predictions_available_kinship_groups",
        "confidence_level": .95, "multiplicity_adjusted": False,
        "nagelkerke_null": "validation prevalence, recomputed for each bootstrap draw",
        "calibration_fit": "diagnostic only; predictions unchanged",
        "input_sha256": {str(p): sha256_file(p) for p in [RUN / "COMPLETE.json", LR_RUN / "VERIFIED.json",
                         evidence["label_path"], Path(__file__), Path(__file__).parent / "templates/prs_distillation_report.html",
                         Path("src/decipehr/metrics.py"), Path("src/decipehr/prs_distillation/reporting.py")]},
        "sealed_files_verified": evidence["verified_files"], "metric_references": SOURCES,
    }
    write_json(out / "PROVENANCE.json", provenance)
    for key, frame in evidence.items():
        if isinstance(frame, pd.DataFrame):
            assert not {"sampleid", "kinship_group", "sex", "y"} & set(frame.columns)
            frame.to_csv(out / f"{key}.csv", index=False)
    (out / "formulas").mkdir(exist_ok=True)
    for name, expression in FORMULAS.items():
        save_formula(expression, out / "formulas" / f"{name}.svg")
    for lang in ["zh", "en"]:
        directory = out / lang / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        for name, fig in figures(evidence, lang):
            fig.savefig(directory / f"{name}.svg", dpi=160)
            fig.savefig(directory / f"{name}.png", dpi=160)
            plt.close(fig)
        content = report_html(evidence, lang, out)
        (out / f"distillation_{lang}.html").write_text(content)
        print(f"HTML ready: {out / f'distillation_{lang}.html'}", flush=True)
    # Confirm that the presentation did not change any sealed experiment artifact.
    assert verify_seal(RUN, "COMPLETE.json") == evidence["verified_files"]["core"]
    assert verify_seal(LR_RUN, "VERIFIED.json") == evidence["verified_files"]["logistic"]
    readme = (
        "Open distillation_zh.html for Chinese or distillation_en.html for English.\n"
        "Both HTML reports are self-contained: no server, no external JS/fonts, and no patient-level rows.\n"
        "Figures are also available as SVG and PNG under zh/figures and en/figures.\n"
        "CSV files contain aggregate metrics, paired differences, bins, curves and settings.\n"
        "The model-development and monotonic-age experiments remain paused.\n"
        "No predictive model was retrained. Calibration regressions are diagnostic only.\n"
        "R2 definitions: Nagelkerke for binary outcomes; teacher-logit R2 for imitation.\n"
        "This is exploratory direct-prediction evaluation, not Phase B official Stage-2 evaluation.\n"
        "Reproduce from the repository root:\n"
        "UV_CACHE_DIR=outputs/uv-cache uv run --offline --no-sync python scripts/build_prs_distillation_presentation.py\n"
    )
    (out / "READ_ME.txt").write_text(readme)
    write_json(out / "VERIFIED.json", {
        "status": "rendered_numeric_checks_passed", "source_files_verified": evidence["verified_files"],
        "patient_alignment": True, "original_metrics_reproduced": True, "nagelkerke_independently_reproduced": True,
        "teacher_logit_r2_independently_reproduced": True, "calibration_bins_match_sklearn": True,
        "primary_training_logs_match_selection_lock": True, "bilingual_same_data": True,
        "patient_level_data_exported": False, "sealed_experiment_files_unchanged": True,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*")
                          if p.is_file() and p.name not in ["VERIFIED.json", "distillation_html_zh_en.zip"]},
    })
    with zipfile.ZipFile(out / "distillation_html_zh_en.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.suffix != ".zip":
                archive.write(p, p.relative_to(out))
    print(f"Complete: {out}", flush=True)


if __name__ == "__main__":
    main()


