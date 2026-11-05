"""Evaluate sealed ancestry-transfer predictions and render a local research report."""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_ancestry_transfer import BIO, FILTERS, SOURCE, TRAITS, read
from sklearn.metrics import average_precision_score, roc_auc_score

from decipehr.external_pgs import sha256_file, write_json
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.reporting import basic_metrics, calibration

ARMS = ["LR_SINGLE_BEST", "LR_ALL", "M0_SUP", "M0_KD", "H_SELECTED", "TabPFN_v3"]
METRICS = ["log_loss", "auroc", "brier", "nagelkerke_r2"]


def metrics(y, p):
    return {**basic_metrics(y, p), "nagelkerke_r2": nagelkerke_r2(y, p),
            "average_precision": average_precision_score(y, p), **calibration(y, p)}


def contrasts():
    pairs = [(f"reference_pc/{a}", f"official_pc/{a}") for a in ARMS]
    for mode in ["official_pc", "reference_pc"]:
        pairs += [(f"{mode}/{a}", f"{mode}/{b}") for a, b in [
            ("M0_KD", "LR_SINGLE_BEST"), ("H_SELECTED", "LR_SINGLE_BEST"),
            ("TabPFN_v3", "LR_SINGLE_BEST"), ("M0_KD", "LR_ALL"),
            ("H_SELECTED", "LR_ALL"), ("M0_KD", "M0_SUP"),
            ("M0_KD", "TabPFN_v3"), ("H_SELECTED", "TabPFN_v3")]]
    return pairs


def paired_differences(y, arrays, groups, replicates=1000):
    """Shared group resampling; sklearn AUROC, conventional loss/Brier/R2 aggregation."""
    arms = list(arrays)
    p = np.column_stack([arrays[a] for a in arms])
    for column in p.T:
        basic_metrics(y, column)
    _, inverse = np.unique(groups, return_inverse=True)
    count_groups = inverse.max()+1
    n_group, y_group = np.bincount(inverse), np.bincount(inverse, weights=y)
    group_sums = []
    for matrix in [
        -(y[:, None]*np.log(np.clip(p, 1e-6, 1-1e-6))+(1-y[:, None])*np.log1p(-np.clip(p, 1e-6, 1-1e-6))),
        (p-y[:, None])**2,
        -(y[:, None]*np.log(np.clip(p, 1e-15, 1-1e-15))+(1-y[:, None])*np.log1p(-np.clip(p, 1e-15, 1-1e-15))),
    ]:
        sums = np.zeros((count_groups, len(arms)))
        np.add.at(sums, inverse, matrix)
        group_sums.append(sums)
    rng = np.random.default_rng(20261104)
    samples = []
    for _ in range(replicates):
        count = np.bincount(rng.integers(count_groups, size=count_groups), minlength=count_groups)
        n, events = count@n_group, count@y_group
        if not 0 < events < n:
            continue
        ll, brier, ll_r2 = [count@s/n for s in group_sums]
        rate = events/n
        null = rate*np.log(rate)+(1-rate)*np.log1p(-rate)
        r2 = -np.expm1(2*(null+ll_r2))/-np.expm1(2*null)
        auc = np.array([roc_auc_score(y, column, sample_weight=count[inverse]) for column in p.T])
        samples.append(np.column_stack([ll, auc, brier, r2]))
    samples = np.stack(samples)
    point = {a: {**basic_metrics(y, arrays[a]), "nagelkerke_r2": nagelkerke_r2(y, arrays[a])} for a in arms}
    rows = []
    for a, b in contrasts():
        delta = samples[:, arms.index(a)]-samples[:, arms.index(b)]
        for j, metric in enumerate(METRICS):
            lo, hi = np.quantile(delta[:, j], [.025, .975])
            rows.append({"arm": a, "reference": b, "metric": metric,
                         "delta": point[a][metric]-point[b][metric], "ci_low": lo, "ci_high": hi,
                         "replicates": len(samples), "resampling_unit": "available_kinship_group",
                         "models_refitted": False, "multiplicity_adjusted": False})
    return pd.DataFrame(rows)


def table(frame):
    return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.5f}")


def verify_original_metrics(measured, original, policy):
    """Match the frozen concrete model, not the legacy parametric-only alias."""
    selected = measured.loc[measured.cohort.eq("original_val") & measured["mode"].eq("official_pc")
                            & measured.arm.isin(["M0_KD", "M0_SUP", "H_SELECTED", "TabPFN_v3"])]
    for row in selected.itertuples():
        source_arm = policy["choices"][row.trait]["-1"] if row.arm == "H_SELECTED" else row.arm
        np.testing.assert_allclose([getattr(row, m) for m in METRICS],
                                   original.loc[(row.trait, source_arm), METRICS].to_numpy(float),
                                   atol=1e-12, rtol=0)


def render(root, measured, differences, selections, counts, subgroup):
    report = root / "report"
    full = measured.loc[measured.cohort.eq("original_val")]
    transfer = differences.loc[differences.cohort.eq("original_val") & differences.metric.eq("auroc")
                               & differences.arm.str.startswith("reference_pc/")
                               & differences.reference.str.startswith("official_pc/")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    shown = ["LR_SINGLE_BEST", "LR_ALL", "M0_KD", "H_SELECTED", "TabPFN_v3"]
    for ax, trait in zip(axes, TRAITS, strict=True):
        part = full.loc[full.trait.eq(trait)]
        for mode, color, offset in [("official_pc", "#555555", -.1), ("reference_pc", "#126c9c", .1)]:
            values = part.loc[part["mode"].eq(mode)].set_index("arm").loc[shown, "auroc"]
            ax.scatter(values, np.arange(len(shown))+offset, c=color, label=mode, s=35)
        ax.set_title({"T2D": "T2D", "cad": "CAD", "bc": "Female BC"}[trait])
        ax.set_yticks(np.arange(len(shown)), shown)
        ax.set_xlabel("Validation AUROC")
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=8)
    fig.suptitle("Frozen patients, score banks and student policies; refitted on reference PCs")
    fig.tight_layout()
    fig.savefig(report / "validation_auroc.png", dpi=180)
    fig.savefig(report / "validation_auroc.pdf")
    plt.close(fig)
    summary = full.pivot(index=["trait", "arm"], columns="mode", values="auroc").reset_index()
    summary["delta_reference_minus_official"] = summary.reference_pc-summary.official_pc
    summary.to_csv(report / "main_auc_comparison.csv", index=False)
    age_source = "现有模型的 age_sd 由主办方提供，训练包没有年龄年数到该尺度的可验证映射。将 AoU 年龄按它自身均值/SD 标准化，不能复现原模型年龄函数；应先恢复原变换，或在独立 AoU 训练子集中重训并单独评估。"
    sources = {
        "QC v9 (2026-10-30)": "https://support.researchallofus.org/hc/en-us/articles/50655639562900-All-of-Us-Genomics-Multi-omics-Quality-Report",
        "v9 update and known sample issue": "https://support.researchallofus.org/hc/en-us/articles/52422066751252-Incremental-Data-Change-for-Curated-Data-Repository-Version-9-Genomic-Data",
        "Genomic ancestry fields (archived v8; verify chosen release)": "https://support.researchallofus.org/hc/en-us/articles/29475228181908-How-the-All-of-Us-Genomic-data-are-organized-Archived-CDRv8",
        "Data dissemination policy": "https://support.researchallofus.org/hc/en-us/articles/22346276580372-Data-and-Statistics-Dissemination-Policy",
        "Current Workbench FAQ": "https://support.researchallofus.org/hc/en-us/articles/52861503885460-Frequently-Asked-Questions",
    }
    links = " ".join(f'<a href="{url}">{html.escape(name)}</a>.' for name, url in sources.items())
    text = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>统一祖源与 single-best LR</title>
<style>body{font:16px/1.65 system-ui;max-width:1200px;margin:35px auto;padding:0 24px;color:#202a36}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:6px 10px;border-bottom:1px solid #d9e0e7;text-align:right}th{background:#eef3f7}img{max-width:100%}p{max-width:1050px}code{background:#eef3f7}h1,h2{line-height:1.3}.note{padding:14px;background:#edf5fa}</style>
<h1>统一祖源坐标、蒸馏模型与 single-best logistic regression</h1>
<p class="note">本报告是 decipEHR 开发数据上的实测敏感性分析；没有读取 All of Us 个体数据，也不代表已完成 All of Us 外部验证。原 validation 之前已经用于开发，不能称为从未使用的测试集。</p>
<h2>本次实际比较</h2>
<p>既往模型输入使用主办方连续 PC1–PC10；用于分层展示的 hard_label 已是另行生成的 1000G 相似性标签。这两个角色不能混淆。本次对所有模型完整替换 PC 输入，再在对应训练集拟合标准化与权重。未在最终校准层加回官方 PCs。</p>
<p>统一坐标由 2,503 名 1000G 参考样本、GRCh38 上固定 53,571 个 SNP、固定参考等位基因频率及 PLINK2 PCA loadings 定义；只在外部参考人群拟合 PCA。覆盖全部 8,663 名 train/validation 患者。原始上游候选 marker panel 的覆盖率为 89.9914%，本次使用的是此前已另行拟合并冻结的 53,571 SNP 参考空间，不能称作原上游模型的完整复现。</p>
<p>PRS 输入保持 26/7/5 个已冻结的原始外部评分，未进行 PC 残差化。M0_KD 为原始简洁学生；H_SELECTED 为先前训练内部选择的疾病特定学生（最终 T2D=M0_KD、CAD=HGB_PAIR_KD、BC=G_A6_KD）。结构、超参数、迭代数及主种子 17 固定；新 PCs 下重算 Teacher OOF 软标签并重训 Student。这里检验固定策略的坐标敏感性，尚未在新空间重新搜索最优学生配置。</p>
<p>命名对应：本次 H_SELECTED 指旧报告的 BIO_POLICY，即跨参数化学生与 HGB 的最终冻结策略；旧表同名 H_SELECTED 仅在参数化模型中选择。复核时按具体模型名称逐一匹配，避免将 CAD 的 HGB_PAIR_KD 与 A6_KD 混在一起。</p>
<p>LR_SINGLE_BEST 每次只输入一个 PRS，加线性 age_sd、sex 和十个 PCs；在训练内部共享五折上按 pooled OOF log loss 选 PRS。LR_ALL 是全部 PRS 加相同协变量的普通无惩罚 LR。全训练选择用于 validation；训练 OOF 使用嵌套选择，严格排除外层患者与已知亲缘组。原官方-PC LR_ALL 已与旧封存预测复现至 1e-12。</p>
<p>两套 PC 下分别在训练内部选择 single-best，因此该 baseline 的跨 PC 差异可能同时包含最佳 PRS 的变化。LR_ALL、Teacher 和冻结 Student 保持同一组 PRS 输入，能更直接反映坐标替换的影响。single-best 的“最佳”仅限当前同一冻结 PRS 库，不表示遍历所有已发表评分后的全局最优。</p>
<p>所有训练 OOF 的 Student 软标签由外层训练集内再次交叉拟合生成；未直接使用含有外层患者影响的全训练 OOF 标签。外层 Student 继承相应旧外层训练内部选择；所有方法共享同一外层患者划分。校准斜率仅作诊断，未用验证标签修改预测。</p>
<h2>原 validation 的 AUROC</h2>"""
    text += table(summary)+ '<img src="validation_auroc.png" alt="Three-disease official versus reference PC validation AUROC">'
    text += "<p>下面差值统一为 reference PCs − official PCs。AUROC/R² 越高越好，log loss/Brier 越低越好。置信区间来自 1,000 次配对已知亲缘组 bootstrap，条件于已拟合模型；未作多重比较校正，适合描述敏感性，不用于宣称等效或普遍优越。</p>"
    text += table(transfer[["trait", "arm", "delta", "ci_low", "ci_high"]])
    text += "<h2>single-best LR 与学生</h2>"+table(selections.loc[selections.outer_fold.eq(-1)])
    delta = differences.loc[differences.cohort.eq("original_val") & differences.metric.eq("auroc")
                            & differences.reference.str.endswith("/LR_SINGLE_BEST")]
    text += table(delta[["trait", "arm", "reference", "delta", "ci_low", "ci_high"]])
    text += "<h2>All of Us 迁移建议</h2>"
    text += "<p><strong>不需要一概弃用官方祖源。</strong>建议保留 AoU 官方遗传相似性标签做核查与敏感性分析，并为跨队列模型冻结同一参考 panel、SNP 清单、等位基因、频率、loadings 和投影尺度。统一参考只是技术标准，不存在唯一的普适祖源真值。AoU 的既有官方文档提供参考 PCA loadings 和预测模型；也可将两个队列都投影到同一版 AoU 参考，但需要再次评估该版本，不能把官方 PC 数组按列名直接接到当前 Student。</p>"
    text += "<p>2026-10-30 的 CDRv9 QC 报告附录 G 说明，官方分类本身就基于 HGDP+1000G 参考、16 个 PCs 和随机森林，低置信度样本保留 OTH。它有可解释的技术来源，不应因是官方提供就弃用。我们的 1000G 单独参考对某些群体覆盖有限；跨群体建模可预先指定更广的参考，再统一重做投影与验证。保留连续 PCs 与不确定/分布外状态，不强制每位混合祖源参与者进入单一类别。遗传相似性、分类概率、祖源比例、自报 race/ethnicity 是不同变量。</p>"
    text += f"<p><strong>年龄输入：</strong>{age_source}</p>"
    text += "<p><strong>评分与基因型：</strong>锁定 CDR 和基因组版本；核对 effect allele、strand、重复位点、多等位位点、缺失率与 imputation/测序 QC。AoU 的小型 callset 不保证包含所有 PRS 位点，先统计每个 score 的位点覆盖和权重覆盖，再决定提取 VDS 或合适的较小 callset。未观察到的位点不能直接当成纯合参考基因型。移植既定预测器时沿用训练/reference 变换，避免在整份 AoU 数据上重估 PRS 均值、SD 或归一化参数。</p>"
    text += "<p><strong>表型与评估：</strong>先冻结 T2D/CAD/BC 的诊断定义、索引时间、观察窗口、病例/对照、既往病例排除和 BC 女性资格映射；无诊断记录不自动等于健康对照。当前二分类开发任务不能直接解释为未来发病风险。若做 incident disease，需要随访/删失定义和相应时间到事件评估。区分零适配外部验证、独立校准集校准和 AoU 内重训；测试集只做预测。参与者及亲缘组整体划分，同时审查 PRS discovery GWAS 与 AoU 的样本重叠。</p>"
    text += "<p><strong>迁移效果：</strong>同时报告 AUROC、AUPRC、log loss、Brier、校准斜率/截距和病例率，按遗传相似性、年龄、性别及站点/批次分层。Nagelkerke R² 会随病例构成和定义改变，不能把不同队列的值直接视作可比的解释率。总体 AUC 接近也不能证明各祖源组或校准均稳定。预先指定最差组和分布外样本的分析规则。</p>"
    text += "<p><strong>计算与部署：</strong>本地 Teacher 的训练 context 只有数千人，不能据此假定可将 AoU 全量训练集直接放入一次 TabPFN 拟合；应先验证 context 预算、GPU 内存和批量预测。移植 Teacher 还涉及训练 context 的数据使用许可，而 Student 推理只需保存的模型与预处理参数。若在 AoU 内重训蒸馏模型，Teacher 软标签必须按患者/亲缘组交叉拟合，再在训练部分选学生与 single-best LR。</p>"
    text += "<p><strong>Workbench 与当前版本：</strong>基因组分析需要 Controlled Tier，在 Researcher Workbench 内执行本地评分与模型推理，避免将个体数据发送到外部 API。现行 FAQ 对从参与者级数据训练的模型导出有限制；AoU 内重训的 Student/LR 也应按当时规则核查，不能假定权重可自由导出。正式传播时按政策抑制 1–20 人的计数及可反推该计数的结果。CDRv9 已发布样本质量问题与修订；应用所选 release 的最新排除清单，勿只复用旧版 bucket 路径或 QC 设置。</p>"
    text += "<p>具体例子：最新 v9 QC Known Issue #2 记录部分参与者在 chr4/chr19 的特定区域存在变异缺失，影响若干 VDS/MT/BGEN/BED 产品，而报告说明 cohort extraction、VCF 与 PGEN 不受该问题影响。PRS 若依赖这些区域，应按所选格式核查，而不能仅凭同为 CDRv9 就认为所有格式完全等价。</p>"
    text += "<p>外部依据（查阅于 2026-11-04）："+links+"</p>"
    text += "<h2>完整指标与可复核文件</h2><p>metrics.csv 为所有队列的指标；paired_differences.csv 为配对差值；nested_oof 为诚实的外层预测。selection_oof.parquet 只用于训练内部选 PRS，不能作为最终 single-best 的泛化性能。subgroup_metrics.csv 沿用同一旧 1000G 分层以保持比较人群一致；它不是 AoU 群体性能。</p>"
    text += table(counts)+"<details><summary>全部性能指标</summary>"+table(measured)+"</details>"
    text += "<details><summary>固定遗传相似性分层（小样本标记 unstable）</summary>"+table(subgroup)+"</details>"
    text += "<p>入口：scripts/run_ancestry_transfer.py；汇总：scripts/report_ancestry_transfer.py。PROTOCOL.json 记录分析约定，PREPARED.json 记录输入哈希，private/&lt;trait&gt;/COMPLETE.json 记录预测封存，VERIFIED.json 记录汇总验证。个体预测与模型都仅在 ignored outputs 内。</p></html>"
    (report / "report_zh.html").write_text(text)
    write_json(report / "sources.json", sources)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.out_dir
    verify_files(Path.cwd(), read(root / "PREPARED.json")["input_sha256"])
    predictions, selections = {}, []
    for trait in TRAITS:
        path = root / "private" / trait
        seal = read(path / "COMPLETE.json")
        if seal["validation_labels_read"]:
            raise ValueError("Training code read validation labels")
        verify_files(path, seal["output_sha256"])
        predictions[trait] = pd.read_parquet(path / "predictions.parquet")
        selections.append(pd.read_csv(path / "selections.csv"))
    # Open assessment outcomes only after every trait and arm is prediction-sealed.
    audit = read(SOURCE / "audit_report.json")
    label_path = Path(audit["paths"]["evaluation"])
    verify_files(Path.cwd(), {str(label_path): audit["source_sha256"][str(label_path)]})
    labels = pd.read_csv(label_path, dtype={"sampleid": str}, usecols=["sampleid", *TRAITS]).set_index("sampleid")
    member_path = "private/cohort_membership.parquet"
    verify_files(FILTERS, {member_path: read(FILTERS / "VERIFIED.json")["output_sha256"][member_path]})
    membership = pd.read_parquet(FILTERS / member_path)
    report = root / "report"
    report.mkdir(exist_ok=True)
    rows, differences, counts, subgroup = [], [], [], []
    for trait, frame in predictions.items():
        for scope in ["original_val", "nested_oof"]:
            selected = frame.loc[frame.fold.eq(-1) if scope == "original_val" else frame.fold.ge(0)].copy()
            selected["key"] = selected["mode"]+"/"+selected.arm
            wide = selected.pivot(index="sampleid", columns="key", values="probability")
            info = selected[["sampleid", "kinship_group", "hard_label"]].drop_duplicates().set_index("sampleid").loc[wide.index]
            if not info.index.is_unique or wide.isna().any().any():
                raise ValueError("Unmatched prediction coverage")
            if scope == "original_val":
                y = labels.loc[wide.index, trait].to_numpy(int)
                member = membership.loc[membership.trait.eq(trait)].set_index("sampleid")
                if set(member.index) != set(wide.index):
                    raise ValueError("Validation filter population differs")
                member = member.loc[wide.index]
                np.testing.assert_array_equal(member.kinship_group, info.kinship_group)
                masks = {"original_val": np.ones(len(wide), bool),
                         "exclude_shared_group": ~member.remove_rule1.to_numpy(),
                         "exclude_direct_0177": ~member.remove_rule2.to_numpy()}
            else:
                training = pd.read_parquet(SOURCE / trait / "training.parquet").set_index("sampleid")
                if set(training.index) != set(wide.index):
                    raise ValueError("OOF population differs")
                y = training.loc[wide.index, "y"].to_numpy(int)
                masks = {"nested_oof": np.ones(len(wide), bool)}
            for cohort, mask in masks.items():
                arrays = {k: wide[k].to_numpy()[mask] for k in wide}
                counts.append({"trait": trait, "cohort": cohort, "n": int(mask.sum()), "events": int(y[mask].sum()),
                               "n_kinship_groups": info.loc[mask, "kinship_group"].nunique()})
                for key, p in arrays.items():
                    mode, arm = key.split("/")
                    rows.append({"trait": trait, "cohort": cohort, "mode": mode, "arm": arm, **metrics(y[mask], p)})
                paired = paired_differences(y[mask], arrays, info.loc[mask, "kinship_group"].to_numpy()).assign(trait=trait, cohort=cohort)
                paired.to_csv(report / f"paired_{trait}_{cohort}.csv", index=False)
                differences.append(paired)
                print(f"Evaluated {trait}/{cohort}, n={mask.sum()}", flush=True)
            if scope == "original_val":
                for name in sorted(info.hard_label.unique()):
                    mask = info.hard_label.eq(name).to_numpy()
                    for key in wide:
                        mode, arm = key.split("/")
                        result = basic_metrics(y[mask], wide[key].to_numpy()[mask])
                        subgroup.append({"trait": trait, "group": name, "mode": mode, "arm": arm, **result,
                            "unstable": result["n"] < 100 or min(result["events"], result["nonevents"]) < 20,
                            "stratum_source": "preexisting_1000G_hard_label_held_fixed"})
    measured, delta, choices = pd.DataFrame(rows), pd.concat(differences, ignore_index=True), pd.concat(selections, ignore_index=True)
    cohort_counts, subgroup = pd.DataFrame(counts), pd.DataFrame(subgroup)
    # Every reported delta must match the matched point-estimate table.
    for row in delta.itertuples():
        part = measured.loc[measured.trait.eq(row.trait) & measured.cohort.eq(row.cohort)].copy()
        part.index = part["mode"]+"/"+part.arm
        np.testing.assert_allclose(row.delta, part.loc[row.arm, row.metric]-part.loc[row.reference, row.metric], atol=1e-13)
    old = pd.read_csv(BIO / "report/val/metrics.csv").set_index(["trait", "arm"])
    verify_original_metrics(measured, old, read(BIO / "report/val/POLICY_LOCK.json"))
    for name, frame in [("metrics", measured), ("paired_differences", delta), ("single_prs_selections", choices),
                        ("cohort_counts", cohort_counts), ("subgroup_metrics", subgroup)]:
        frame.to_csv(report / f"{name}.csv", index=False)
    render(root, measured, delta, choices, cohort_counts, subgroup)
    write_json(root / "VERIFIED.json", {"status": "completed_and_verified", "all_traits_complete": True,
        "primary_validation_rows": int(measured.cohort.eq("original_val").sum()), "total_metric_rows": len(measured),
        "all_pair_deltas_match_metrics": True, "old_teacher_student_metrics_reproduced": True,
        "label_source_sha256": sha256_file(label_path), "report_script_sha256": sha256_file(__file__),
        "prediction_receipts": {t: sha256_file(root / "private" / t / "COMPLETE.json") for t in TRAITS},
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "torch", "tabpfn"]},
        "output_sha256": {str(p.relative_to(root)): sha256_file(p) for p in report.iterdir() if p.is_file()}})


if __name__ == "__main__":
    main()
