"""Evaluate the locked centroid-initialization experiment and render offline HTML."""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from prs_report_math import check_latex, render_latex
from report_prs_distillation_bio import paired_bootstrap
from report_prs_plain_logistic import CheckHTML, image_uri
from run_prs_centroid_init import BIO, SCOPES, SEEDS, TRAITS
from sklearn.metrics import average_precision_score

from decipehr.external_pgs import sha256_file, write_json
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.reporting import basic_metrics, calibration

FILTERS = Path("outputs/prs_distillation_bio/validation_kinship_filters_20261027")
ASSESSMENTS = ["original_val", "exclude_shared_group", "exclude_direct_0177", "train_oof"]
METRICS = {"auroc": "AUROC ↑", "log_loss": "Log loss ↓", "nagelkerke_r2": "Nagelkerke R² ↑"}


def read(path):
    return json.loads(Path(path).read_text())


def evaluate(root):
    out = root / "report"
    out.mkdir(exist_ok=False)
    protocol = read(root / "PROTOCOL.json")
    verify_files(Path.cwd(), protocol["code_sha256"])
    verify_files(Path.cwd(), protocol["input_sha256"])
    assert sha256_file(root / "prs_ancestry_mapping.csv") == protocol["mapping_sha256"]
    frames, used, gates, checks = [], {}, [], []
    for scope in SCOPES:
        for trait in TRAITS:
            directory = root / "fits" / scope / trait
            receipt = read(directory / "COMPLETE.json")
            assert not receipt["evaluation_labels_read"] and receipt["new_models"] == 4
            assert receipt["protocol_sha256"] == sha256_file(root / "PROTOCOL.json")
            verify_files(directory, receipt["output_sha256"])
            frame = pd.read_parquet(directory / "predictions.parquet")
            assert frame.scope.eq(scope).all() and frame.trait.eq(trait).all()
            assert set(frame.seed) == set(SEEDS) and set(frame.arm) == {"M0_RANDOM", "M0_CENTROID"}
            frames.append(frame)
            gates.append(pd.read_csv(directory / "gate_by_ancestry.csv"))
            checks.extend({"scope": scope, "trait": trait, **c} for c in receipt["checks"])
            used[str(directory / "COMPLETE.json")] = sha256_file(directory / "COMPLETE.json")
            used[str(directory / "predictions.parquet")] = sha256_file(directory / "predictions.parquet")
    all_predictions = pd.concat(frames, ignore_index=True)
    assert not all_predictions.duplicated(["trait", "scope", "sampleid", "seed", "arm"]).any()
    filt_receipt = read(FILTERS / "VERIFIED.json")
    membership_path = FILTERS / "private/cohort_membership.parquet"
    assert sha256_file(membership_path) == filt_receipt["output_sha256"]["private/cohort_membership.parquet"]
    membership = pd.read_parquet(membership_path)
    used[str(membership_path)] = sha256_file(membership_path)
    write_json(root / "ASSESSMENT_LOCK.json", {"all_training_complete": True, "new_models": 72,
        "initialization_and_settings_reselected": False, "primary_seed": 17,
        "comparisons": "centroid minus random M0_KD", "assessment_labels_read_by_this_script": False,
        "assessments": ASSESSMENTS, "input_sha256": used})
    source = Path(read(BIO / "PROTOCOL.json")["identity"]["config"]["source_run"])
    audit = read(source / "audit_report.json")
    label_path = Path(audit["paths"]["evaluation"])
    assert sha256_file(label_path) == audit["source_sha256"][str(label_path)]
    labels = pd.read_csv(label_path, dtype={"sampleid": str}, usecols=["sampleid", *TRAITS]).set_index("sampleid")
    rows, differences, counts = [], [], []
    for trait in TRAITS:
        d, e, _ = load_inputs(source, trait)
        for assessment in ASSESSMENTS:
            nested = assessment == "train_oof"
            frame = all_predictions.loc[all_predictions.trait.eq(trait) & all_predictions.scope.eq("val").ne(nested)]
            target = d if nested else e
            truth = d.set_index("sampleid").y if nested else labels[trait]
            ids = target.sampleid.tolist()
            selected = np.ones(len(ids), bool)
            if not nested:
                member = membership.loc[membership.trait.eq(trait)].set_index("sampleid").loc[ids]
                np.testing.assert_array_equal(member.kinship_group, e.kinship_group)
                if assessment != "original_val":
                    selected = ~member["remove_rule1" if assessment == "exclude_shared_group" else "remove_rule2"].to_numpy()
            chosen_ids = np.asarray(ids)[selected]
            y = truth.loc[chosen_ids].to_numpy(int)
            groups = target.set_index("sampleid").loc[chosen_ids, "kinship_group"].to_numpy()
            counts.append({"trait": trait, "assessment": assessment, "n": len(y), "events": int(y.sum()),
                           "nonevents": int((1-y).sum()), "removed": int((~selected).sum())})
            for seed in SEEDS:
                arrays = {}
                for arm in ["M0_RANDOM", "M0_CENTROID"]:
                    part = frame.loc[frame.seed.eq(seed) & frame.arm.eq(arm)].set_index("sampleid")
                    assert part.index.is_unique and set(part.index) == set(ids)
                    np.testing.assert_array_equal(part.loc[chosen_ids, "kinship_group"], groups)
                    p = part.loc[chosen_ids, "probability"].to_numpy()
                    arrays[arm] = p
                    rows.append({"trait": trait, "assessment": assessment, "seed": seed, "arm": arm,
                        **basic_metrics(y, p), "nagelkerke_r2": nagelkerke_r2(y, p),
                        "average_precision": average_precision_score(y, p), **calibration(y, p)})
                if seed == 17:
                    result = paired_bootstrap(y, {"H_SELECTED": arrays["M0_CENTROID"], "M0_KD": arrays["M0_RANDOM"]},
                                              groups, "nested", replicates=1000)
                    differences.append(result.assign(arm="M0_CENTROID", reference="M0_RANDOM", trait=trait, assessment=assessment))
            print(f"Evaluated {trait}/{assessment}: N={len(y)}", flush=True)
    metrics, gaps, counts = pd.DataFrame(rows), pd.concat(differences, ignore_index=True), pd.DataFrame(counts)
    parent_receipt = read(BIO / "COMPLETE.json")
    for scope, assessment in [("val", "original_val"), ("nested", "train_oof")]:
        p = BIO / "report" / scope / "metrics.csv"
        assert sha256_file(p) == parent_receipt["aggregate_output_sha256"][str(p.relative_to(BIO))]
        old = pd.read_csv(p).set_index(["trait", "arm"])
        for row in metrics.loc[metrics.seed.eq(17) & metrics.arm.eq("M0_RANDOM") & metrics.assessment.eq(assessment)].itertuples():
            keys = ["auroc", "log_loss", "brier", "nagelkerke_r2"]
            np.testing.assert_allclose([getattr(row, k) for k in keys], old.loc[(row.trait, "M0_KD"), keys].to_numpy(float), atol=1e-12, rtol=0)
    stability = metrics.pivot(index=["trait", "assessment", "seed"], columns="arm", values=list(METRICS))
    deltas = []
    for (trait, assessment, seed), row in stability.iterrows():
        deltas.append({"trait": trait, "assessment": assessment, "seed": seed,
                       **{f"delta_{m}": row[m, "M0_CENTROID"]-row[m, "M0_RANDOM"] for m in METRICS}})
    for name, table in [("metrics", metrics), ("paired_differences", gaps), ("cohort_counts", counts),
                        ("seed_differences", pd.DataFrame(deltas)), ("gate_by_ancestry", pd.concat(gates, ignore_index=True))]:
        table.to_csv(out / f"{name}.csv", index=False)
    for name in ["prs_ancestry_mapping.csv", "scope_budgets.csv"]:
        (out / name).write_bytes((root / name).read_bytes())
    write_json(root / "VERIFIED.json", {"status": "completed_and_verified", "new_student_fits": 72,
        "new_teacher_fits": 0, "original_val_and_nested_metrics_reproduced": True,
        "reference_replays": 18, "primary_seed": 17, "bootstrap_replicates": 1000,
        "bootstrap_conditional_on_fixed_predictions": True, "multiplicity_adjusted": False,
        "checks": checks, "input_sha256": used, "label_source_sha256": sha256_file(label_path),
        "output_sha256": {str(p.relative_to(root)): sha256_file(p) for p in out.iterdir() if p.is_file()}})


def render(root):
    out = root / "report"
    verify_files(root, read(root / "VERIFIED.json")["output_sha256"])
    measured = pd.read_csv(out / "metrics.csv")
    primary = measured.loc[measured.seed.eq(17)]
    gaps = pd.read_csv(out / "paired_differences.csv")
    seeds = pd.read_csv(out / "seed_differences.csv")
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    titles = ["Original validation", "Validation: exclude shared training kinship groups",
              "Validation: exclude direct train–val KING ≥ 0.177", "Training: pooled five-fold OOF"]
    for assessment, title in zip(ASSESSMENTS, titles, strict=True):
        fig, axes = plt.subplots(3, 3, figsize=(12, 7.5), layout="constrained")
        for col, trait in enumerate(TRAITS):
            part = primary.loc[primary.trait.eq(trait) & primary.assessment.eq(assessment)].set_index("arm")
            for row, (metric, label) in enumerate(METRICS.items()):
                ax = axes[row, col]
                values = part.loc[["M0_RANDOM", "M0_CENTROID"], metric].to_numpy()
                span = max(float(np.ptp(values)), .003)
                ax.scatter(values, [0, 1], color=["#63798b", "#087f8c"], s=70)
                for i, value in enumerate(values):
                    ax.annotate(f"{value:.4f}", (value, i), xytext=(9, 0), textcoords="offset points", va="center")
                ax.set(xlim=(values.min()-.2*span, values.max()+.95*span), ylim=(1.65, -.65),
                       yticks=[0, 1], yticklabels=["Random init", "Centroid init"] if col == 0 else ["", ""], xlabel=label)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
                ax.tick_params(axis="y", length=0)
                ax.grid(axis="x", alpha=.13)
                if row == 0:
                    ax.set_title(f"{'Female BC' if trait == 'bc' else trait.upper()} · N={int(part.n.iloc[0]):,}")
        fig.suptitle(title + "\nM0_KD · only gate initialization changed · primary seed 17 · axes differ by panel", fontsize=13)
        for ext in ["svg", "png", "pdf"]:
            fig.savefig(out / f"performance_{assessment}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, trait in zip(axes, TRAITS, strict=True):
        part = measured.loc[measured.trait.eq(trait) & measured.assessment.eq("original_val")]
        values = part.pivot(index="seed", columns="arm", values="auroc").loc[list(SEEDS)]
        span = max(float(np.ptp(values.to_numpy())), .003)
        for row, seed in enumerate(SEEDS):
            pair = values.loc[seed, ["M0_RANDOM", "M0_CENTROID"]].to_numpy()
            ax.plot(pair, [row-.12, row+.12], color="#c3cdd5", zorder=1)
            for value, y, color, marker in zip(pair, [row-.12, row+.12], ["#63798b", "#087f8c"], ["o", "D"], strict=True):
                ax.scatter(value, y, color=color, marker=marker, s=42, zorder=2)
                ax.annotate(f"{value:.4f}", (value, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set(xlim=(values.to_numpy().min()-.15*span, values.to_numpy().max()+.55*span),
               ylim=(2.6, -.6), yticks=range(len(SEEDS)), yticklabels=[f"Seed {s}" for s in SEEDS],
               xlabel="AUROC ↑", title="Female BC" if trait == "bc" else trait.upper())
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(axis="x", alpha=.13)
    fig.suptitle("Original validation · all three paired seeds\nGray circles: random init · teal diamonds: centroid init · axes differ by panel", fontsize=12)
    for ext in ["svg", "png", "pdf"]:
        fig.savefig(out / f"seed_stability.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    equations = [
        r"$\pi(A)=\operatorname{softmax}(WA+c)$",
        r"$\begin{aligned}\mu_g&=\frac{1}{n_g}\sum_{\substack{i\in\mathrm{train}\\g_i=g}} A_i,\\[6pt]\tau^2&=\frac{1}{2}\operatorname*{median}_{g<h}\lVert\mu_g-\mu_h\rVert_2^2.\end{aligned}$",
        r"$\begin{aligned}W_k^{(0)}&=\frac{\mu_{g(k)}^\top}{\tau^2}+\epsilon_k,\\[5pt]c_k^{(0)}&=-\frac{\lVert\mu_{g(k)}\rVert_2^2}{2\tau^2},\\[5pt]\epsilon_{kj}&\sim\mathcal{N}(0,0.01^2).\end{aligned}$",
        r"$s_k(A)=-\frac{\lVert A-\mu_{g(k)}\rVert_2^2}{2\tau^2}\quad\Longrightarrow\quad \pi_k(A)\propto e^{s_k(A)}$",
    ]
    math_style, formulas, math_metadata = render_latex(equations)
    mapping = pd.read_csv(out / "prs_ancestry_mapping.csv")
    budgets = pd.read_csv(out / "scope_budgets.csv")
    gates = pd.read_csv(out / "gate_by_ancestry.csv")
    gates = gates.loc[gates.scope.eq("val") & gates.seed.eq(17)].merge(
        mapping[["trait", "prs", "mapped_group"]], on=["trait", "prs"], validate="many_to_one")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for col, trait in enumerate(TRAITS):
        for row, stage in enumerate(["initial", "trained"]):
            part = gates.loc[gates.trait.eq(trait) & gates.stage.eq(stage)]
            table = part.groupby(["patient_group", "mapped_group"]).mean_gate.sum().unstack(fill_value=0)
            np.testing.assert_allclose(table.sum(axis=1), 1, atol=1e-12)
            ax = axes[row, col]
            im = ax.imshow(table.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set(xticks=range(len(table.columns)), xticklabels=table.columns,
                   yticks=range(len(table)), yticklabels=table.index if col == 0 else [""]*len(table))
            ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"{'Female BC' if trait == 'bc' else trait.upper()} · {stage}")
            for i in range(len(table)):
                for j in range(len(table.columns)):
                    value = table.iloc[i, j]
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > .55 else "#17344c", fontsize=9)
    fig.colorbar(im, ax=axes, label="Mean gate mass summed by PRS source group", shrink=.65)
    fig.suptitle("Centroid arm · training patients · seed 17\nMixing weights, not ancestry fractions or causal effects", fontsize=13)
    for ext in ["svg", "png", "pdf"]:
        fig.savefig(out / f"gate_initialization.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    interpretation_path = root / "INTERPRETATION.json"
    interpretation = read(interpretation_path) if interpretation_path.exists() else {}
    for language in ["zh", "en"]:
        def t(zh, en, lang=language):
            return zh if lang == "zh" else en
        title = t("祖源中心初始化：M0_KD 配对对照", "Ancestry-centroid initialization: paired M0_KD comparison")
        html = [f'<!doctype html><html lang="{language}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>',
            '<style>body{margin:0;background:#f3f6f8;color:#17344c;font:16px/1.65 system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:25px}section{background:white;border:1px solid #dbe4e9;border-radius:12px;padding:24px;margin:18px 0}h1{font-size:29px;line-height:1.3}h2{font-size:21px;margin-top:0}.muted{color:#536779;font-size:14px}.note{background:#edf7f7;border-left:4px solid #087f8c;padding:14px}.plot{width:100%;height:auto}.formula{overflow:auto;margin:12px 0}a{color:#087f8c}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e2e9ee;white-space:nowrap}th{background:#f2f6f8}button{font:inherit;padding:8px 12px;margin:4px;border:1px solid #b5c5d0;border-radius:7px;background:white;cursor:pointer}button[aria-selected=true]{background:#087f8c;color:white}summary{cursor:pointer;font-weight:600}details{margin-top:15px}[hidden]{display:none!important}@media(max-width:650px){main{padding:12px}section{padding:15px}}@media print{button{display:none}.panel[hidden]{display:block!important}}</style><main>',
            f'<a href="centroid_{"en" if language == "zh" else "zh"}.html">{t("English version", "Chinese version")}</a><h1>{title}</h1>',
            '<p class="muted">' + t("同一 M0_KD 结构、Teacher、loss、正则化、步数及数据划分；只改变门控初始化。", "Same M0_KD architecture, teacher, loss, penalties, steps and data splits; only gate initialization changes.") + '</p>']
        if language in interpretation:
            html.append('<section><h2>' + t("当前结论", "Current findings") + '</h2>' + interpretation[language] + '</section>')
        html.append('<section><h2>' + t("三个种子是否一致", "Does the result repeat across seeds?") + '</h2><img class="plot" alt="Original validation AUROC across all three paired seeds" src="' + image_uri(out / "seed_stability.svg") + '"></section>')
        html += ['<section><h2>' + t("初始化如何构建", "How initialization is constructed") + '</h2>']
        html.extend(formulas)
        html += ['<p>' + t("中心由当前训练范围内、明确归入相应祖源组的患者计算，使用模型同一套标准化 PC1–PC10。它是训练队列中的祖源代表中心，不是原 GWAS 队列的实测中心。τ² 按训练中心距离固定；不搜索温度。",
            "Centers use ancestry-assigned patients within the current training context and the model’s standardized PC1–PC10 coordinates. They are training-cohort proxies, not measured centers of the original GWAS cohorts. The temperature is fixed from training-center distances, with no temperature search.") + '</p><p>' + t(
            "距离 softmax 的等价式适用于有明确中心的通道并忽略小随机扰动。每个通道保留原来相同种子的随机扰动；没有明确祖源的通道保持原始初始化。所有参数继续学习，祖源中心不固定约束最终权重。每个 PRS 的先验质量相同，不额外按祖源组的 PRS 数量重加权。",
            "The distance-softmax equivalence applies to mapped channels without the small random perturbation. Each channel retains the original perturbation from the paired seed; unmapped channels retain their original initialization. All parameters remain trainable. Per-PRS prior mass remains equal, without additional ancestry-count reweighting.") + '</p><div class="note">' + t(
            "明确映射：T2D 25/26、CAD 2/7、女性乳腺癌 4/5。D-PRISM 按 GWAS 权重来源映射，不用 trainedfor 标签覆盖；多祖源、NR 和仅注明 ASN 的通道不强行归入一个中心。AMBIGUOUS 患者不参与中心估计，但保留在模型训练和评估中。",
            "Mapped channels: T2D 25/26, CAD 2/7, female BC 4/5. D-PRISM uses GWAS weight-source ancestry; trainedfor labels do not override it. Multi-ancestry, NR and broadly labeled ASN channels receive no forced center. AMBIGUOUS patients are excluded from center estimation but retained for model training and evaluation.") + '</div></section>',
            '<section><h2>' + t("性能比较", "Performance comparison") + '</h2><div role="tablist">']
        labels = [t("原始 val", "Original val"), t("排除共享亲缘组", "Exclude shared groups"), t("排除直接近亲", "Exclude direct relatives"), t("训练五折 OOF", "Training five-fold OOF")]
        for i, label in enumerate(labels):
            html.append(f'<button role="tab" aria-selected="{str(i == 0).lower()}" onclick="selectCohort({i})">{label}</button>')
        html.append('</div>')
        for i, assessment in enumerate(ASSESSMENTS):
            html.append(f'<div class="panel" id="panel{i}" {"hidden" if i else ""}><img class="plot" alt="{labels[i]}" src="{image_uri(out / f"performance_{assessment}.svg")}">')
            table = []
            for trait in TRAITS:
                part = gaps.loc[gaps.trait.eq(trait) & gaps.assessment.eq(assessment)].set_index("metric")
                row = {t("疾病", "Endpoint"): "Female BC" if trait == "bc" else trait.upper()}
                for metric in METRICS:
                    p = part.loc[metric]
                    row["Δ " + metric] = f'{p.delta:+.5f} [{p.ci_low:+.5f}, {p.ci_high:+.5f}]'
                table.append(row)
            html.append('<p class="muted">' + t("中心初始化 − 随机初始化；名义 95% 配对区间。ΔAUROC/R² > 0、Δlog loss < 0 有利于中心初始化。", "Centroid − random initialization; nominal paired 95% intervals. Positive ΔAUROC/R² and negative Δlog loss favor centroid initialization.") + '</p><div class="table">' + pd.DataFrame(table).to_html(index=False, border=0) + '</div>')
            html.append('<details><summary>' + t("三个种子的差值", "Differences across all three seeds") + '</summary><div class="table">' + seeds.loc[seeds.assessment.eq(assessment)].drop(columns="assessment").to_html(index=False, border=0, float_format=lambda x: f"{x:+.5f}") + '</div></details></div>')
        html += ['</section><section><h2>' + t("解释范围", "Interpretation boundaries") + '</h2><p>' + t(
            "这是固定预算、固定超参数的初始化消融。沿用原随机初始化模型在 train 内选出的设置，没有为新起点重新调参。结果只能回答这一温和、按来源祖源映射的初始化是否改善现有训练流程；不能排除其他温度、中心定义或训练预算的效果。",
            "This is an initialization ablation with fixed budgets and hyperparameters originally selected within train for random initialization. The centroid arm is not retuned. Results concern this mild source-ancestry initialization within the existing training procedure, not every possible temperature, center definition or budget.") + '</p><p class="muted">' + t(
            "主种子固定为 17；29、43 用于稳定性。区间使用 1,000 次可用亲缘组配对 bootstrap，条件于固定预测，未校正多重比较。区间跨 0 不证明等效。Val 已参与开发观察；OOF 中每折重算中心并复用折内 Teacher 软标签。两种亲缘筛选只改变 val，亲缘未知者仍保留。年龄单位是 age_sd。",
            "Seed 17 is primary; 29 and 43 assess stability. Intervals use 1,000 paired available-kinship-group bootstrap resamples conditional on fixed predictions, without multiplicity adjustment. Crossing zero does not establish equivalence. Val has informed development. OOF refits centers within each outer training context and reuses fold-local teacher targets. Kinship filters affect val only and retain unresolved patients. Age is in age_sd units.") + '</p>',
            '<details><summary>' + t("初始化与训练后的门控变化", "Gate weights before and after training") + '</summary><p class="muted">' + t("每行按训练患者的祖源分组，列为 PRS 来源祖源，同来源 PRS 的门控权重相加。混合权重不等于祖源比例或因果贡献；正尺度 r 仍单独学习。", "Rows group training patients by ancestry; columns sum gate weights over PRSs sharing a source ancestry. Mixing weights are not ancestry fractions or causal contributions; positive scales r are learned separately.") + f'</p><img class="plot" alt="Initial and trained ancestry gate weights" src="{image_uri(out / "gate_initialization.svg")}"></details>',
            '<details><summary>' + t("PRS 祖源映射", "PRS ancestry mapping") + '</summary><div class="table">' + mapping[["trait", "prs", "display_name", "mapped_group", "mapping_reason"]].to_html(index=False, border=0) + '</div></details>',
            '<details><summary>' + t("每个训练范围的预算与中心支持", "Training scopes, budgets and center coverage") + '</summary><div class="table">' + budgets.drop(columns=["fit_ids_hash", "query_ids_hash"]).to_html(index=False, border=0, float_format=lambda x: f"{x:.5g}") + '</div></details>',
            '<p><a href="metrics.csv">All metrics CSV</a> · <a href="paired_differences.csv">Paired differences CSV</a> · <a href="gate_by_ancestry.csv">Gate weights CSV</a> · <a href="analysis_verification.json">Verification</a></p></section>',
            '<p class="muted">' + t("公式使用 KaTeX 排版 LaTeX，字体已内嵌，可离线显示；页面无患者标识。", "Equations use KaTeX to typeset LaTeX, with embedded fonts for offline display. No patient identifiers appear on this page.") + '</p></main>',
            '<script>function selectCohort(i){document.querySelectorAll(".panel").forEach((p,j)=>p.hidden=i!==j);document.querySelectorAll("[role=tab]").forEach((b,j)=>b.setAttribute("aria-selected",String(i===j)));}</script></html>']
        (out / f"centroid_{language}.html").write_text(''.join(html).replace('<main>', math_style+'<main>', 1))
    (out / "analysis_verification.json").write_bytes((root / "VERIFIED.json").read_bytes())
    for language in ["zh", "en"]:
        page = out / f"centroid_{language}.html"
        check = CheckHTML()
        check.feed(page.read_text())
        assert check.images == 6 and check.panels == 4
        check_latex(page.read_text(), len(equations))
        if language == "en":
            assert not re.search("[\u4e00-\u9fff]", page.read_text())
    with zipfile.ZipFile(out / "centroid_bilingual.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p.suffix != ".zip":
                if p.suffix == ".csv":
                    assert not {"sampleid", "kinship_group", "IID1", "IID2"} & set(pd.read_csv(p, nrows=0).columns)
                z.write(p, p.name)
    with zipfile.ZipFile(out / "centroid_bilingual.zip") as z:
        assert z.testzip() is None
    snapshot = root / "source_snapshot/scripts" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(root / "REPORT_VERIFIED.json", {"offline_resources": True, "formulas_per_language": 4,
        "panels_per_language": 4, "browser_render_tested": False, "analysis_sha256": sha256_file(root / "VERIFIED.json"),
        "script_sha256": sha256_file(__file__),
        "math_renderer": math_metadata,
        "interpretation_sha256": sha256_file(interpretation_path) if interpretation_path.exists() else None,
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}})
    print(f"Bilingual report: {out}", flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "render"])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.run_dir)
    else:
        render(args.run_dir)


if __name__ == "__main__":
    main()
