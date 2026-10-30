"""Compare two validation-only relatedness filters using frozen full-train models."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_prs_distillation_bio import FM_ROOT, paired_bootstrap
from sklearn.metrics import average_precision_score, r2_score

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.internal_prs_benchmark import kinship_groups
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.reporting import basic_metrics, calibration, fidelity

TRAITS = ["T2D", "cad", "bc"]
COHORTS = ["original_val", "exclude_shared_group", "exclude_direct_0177"]
ARMS = ["M0_KD", "PRIMARY_SELECTED", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"]
COLORS = ["#0b877d", "#c57b17", "#263f57", "#7062bd", "#c54c70"]
NAMES = ["M0_KD", "Selected student", "TabPFN v3", "TabPFN 3.5", "Kumo Tabular"]
COHORT_LABELS = ["Original\nvalidation", "1: Drop shared\ngroup", "2: Drop direct\nKING ≥ 0.177"]


def read(path):
    return json.loads(Path(path).read_text())


def removal_masks(training, evaluation, edges):
    """Only validation IDs can be removed; transitive groups and direct edges differ."""
    assert training.sampleid.is_unique and evaluation.sampleid.is_unique
    assert not set(training.sampleid) & set(evaluation.sampleid)
    assert not training.kinship_group.isna().any() and not evaluation.kinship_group.isna().any()
    shared = evaluation.kinship_group.isin(set(training.kinship_group)).to_numpy()
    close = edges.loc[edges.KINSHIP.ge(.177)]
    linked = set(close.loc[close.IID1.isin(training.sampleid), "IID2"])
    linked |= set(close.loc[close.IID2.isin(training.sampleid), "IID1"])
    direct = evaluation.sampleid.isin(linked).to_numpy()
    unknown = evaluation.kinship_group.str.startswith("UNRESOLVED_QC::").to_numpy()
    assert not (direct & ~shared).any(), "Direct close edges must be inside the frozen groups"
    assert not ((shared | direct) & unknown).any(), "Unresolved patients must remain included"
    return shared, direct, unknown


def model_arrays(root, trait, ids, policy, used):
    pieces = []
    sealed_receipts = read(root / "COMPLETE.json")["upstream_receipt_sha256"]
    for directory, name, seal in [
        (root / "core" / trait, "direct_predictions.parquet", "PREDICTIONS_COMPLETE.json"),
        (root / "low_order" / trait, "predictions.parquet", "COMPLETE.json"),
    ]:
        receipt = read(directory / seal)
        assert not receipt["evaluation_labels_read"]
        assert sha256_file(directory / seal) == sealed_receipts[str(directory / seal)]
        verify_files(directory, {name: receipt["output_sha256"][name]})
        used[str(directory / seal)] = sha256_file(directory / seal)
        used[str(directory / name)] = sha256_file(directory / name)
        pieces.append(pd.read_parquet(directory / name, filters=[("seed", "=", 17)]))
    frame = pd.concat(pieces, ignore_index=True)
    arrays = {}
    for label, arm in [("M0_KD", "M0_KD"), ("PRIMARY_SELECTED", policy[trait]["-1"])]:
        selected = frame.loc[frame.arm.eq(arm)].set_index("sampleid")
        assert selected.index.is_unique and set(selected.index) == set(ids)
        arrays[label] = selected.loc[ids, "probability"].to_numpy()
    original = frame.loc[frame.arm.eq("M0_KD")].set_index("sampleid")
    arrays["TabPFN_v3"] = original.loc[ids, "p_teacher"].to_numpy()
    for backend, arm in [("tabpfn35", "TabPFN_v3.5"), ("kumo", "Kumo_Tabular")]:
        directory = FM_ROOT / backend / trait / "D_to_E"
        receipt = read(directory / "COMPLETE.json")
        verify_files(directory, receipt["output_sha256"])
        used[str(directory / "COMPLETE.json")] = sha256_file(directory / "COMPLETE.json")
        p = directory / "predictions.parquet"
        used[str(p)] = sha256_file(p)
        selected = pd.read_parquet(p).set_index("sampleid")
        assert selected.index.is_unique and set(selected.index) == set(ids)
        arrays[arm] = selected.loc[ids, "probability"].to_numpy()
    return arrays


def save_figure(fig, out, name):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figures(metrics, gaps, counts, out):
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "axes.labelcolor": "#18334b"})
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), layout="constrained")
    labels = {"auroc": "AUROC ↑", "log_loss": "Log loss ↓", "nagelkerke_r2": "Nagelkerke R² ↑"}
    for col, trait in enumerate(TRAITS):
        part = metrics.loc[metrics.trait.eq(trait)].set_index(["arm", "cohort"])
        n = counts.loc[counts.trait.eq(trait)].set_index("cohort").loc[COHORTS, "n"].tolist()
        title = "Female BC" if trait == "bc" else trait.upper()
        for row, (metric, label) in enumerate(labels.items()):
            ax = axes[row, col]
            for arm, color, name in zip(ARMS, COLORS, NAMES, strict=True):
                if trait == "T2D" and arm == "PRIMARY_SELECTED":
                    continue  # The selected T2D model is exactly M0_KD.
                values = part.loc[arm].loc[COHORTS, metric]
                ax.plot(range(3), values, "o-", color=color, label=name, lw=1.7, ms=5)
            ax.set(xticks=range(3), xticklabels=COHORT_LABELS, ylabel=label)
            ax.grid(axis="y", alpha=.16)
            ax.margins(x=.08, y=.18)
            if row == 0:
                ax.set_title(f"{title} · N = {n[0]:,} / {n[1]:,} / {n[2]:,}", pad=12)
    handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Validation only · frozen models fitted with all eligible training patients\n"
                 "Separate filters applied to original val; unresolved kinship retained; point estimates", fontsize=15)
    save_figure(fig, out, "validation_performance")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), layout="constrained")
    for col, trait in enumerate(TRAITS):
        for row, metric in enumerate(["auroc", "log_loss"]):
            ax = axes[row, col]
            for arm, color, offset in [("M0_KD", COLORS[0], -.07), ("PRIMARY_SELECTED", COLORS[1], .07)]:
                if trait == "T2D" and arm == "PRIMARY_SELECTED":
                    continue
                part = gaps.loc[gaps.trait.eq(trait) & gaps.arm.eq(arm) & gaps.reference.eq("TabPFN_v3") & gaps.metric.eq(metric)].set_index("cohort").loc[COHORTS]
                x = np.arange(3)+offset
                ax.vlines(x, part.ci_low, part.ci_high, color=color, alpha=.8, lw=2)
                ax.plot(x, part.delta, "o-", color=color, label="M0_KD" if arm == "M0_KD" else "Selected student")
            ax.axhline(0, color="#738293", ls="--", lw=1)
            ax.set(xticks=range(3), xticklabels=COHORT_LABELS,
                   ylabel="ΔAUROC: positive favors Student" if metric == "auroc" else "Δlog loss: negative favors Student")
            ax.grid(axis="y", alpha=.15)
            ax.margins(x=.1, y=.15)
            if row == 0:
                ax.set_title("Female BC" if trait == "bc" else trait.upper())
    handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Student − original TabPFN v3 · paired available-family bootstrap\n"
                 "1,000 draws; nominal 95% intervals conditional on frozen predictions", fontsize=14)
    save_figure(fig, out, "student_teacher_gap")


def html_reports(metrics, gaps, counts, policy, out):
    pictures = {name: base64.b64encode((out / f"{name}.svg").read_bytes()).decode()
                for name in ["validation_performance", "student_teacher_gap"]}
    for lang in ["zh", "en"]:
        def t(zh, en, lang=lang):
            return zh if lang == "zh" else en
        title = t("只筛选 validation：两种亲缘规则的性能比较", "Validation-only filtering: two relatedness rules")
        text = f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>'
        text += '<style>body{font:16px/1.65 system-ui;color:#18334b;background:#f4f7fa;margin:0}main{max-width:1300px;margin:auto;padding:28px}section,header{background:white;border-radius:12px;padding:24px;margin-bottom:20px}h1{font-size:30px;line-height:1.3}h2{font-size:23px}.note{background:#fff3db;padding:16px;border-radius:8px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%;white-space:nowrap}th,td{padding:9px;border-bottom:1px solid #e1e8ed;text-align:right}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}a{color:#087e73}details{margin:16px 0}summary{cursor:pointer;font-weight:650}.tag{color:#087e73;font-weight:700}</style></head><body><main>'
        text += '<header><p class="tag">'+t("全量 train 保持不变 · 只筛选 val · 冻结预测重算", "Full training context retained · filter val only · recompute frozen predictions")+f'</p><h1>{title}</h1>'
        text += '<p>'+t("Teacher 均使用完整训练集作为 context，直接预测官方 val；这里展示的不是 OOF 成绩。Student 使用原来训练内部锁定的模型。两种筛选各自应用于原始 val，不重新训练、校准或挑选模型。",
            "Teachers use the full eligible training context to directly predict official val; these are not OOF scores. Students retain the original train-internal selections. Both filters apply independently to original val, with no refitting, recalibration or model reselection.")+'</p></header>'
        text += '<section><h2>'+t("筛选规则与人数", "Rules and denominators")+'</h2><ol><li>'+t(
            "规则 1：排除与当前疾病 train 共享推断亲缘组的 val 患者。分组使用通过质控的 KING ≥ 0.0884 边及其连通分量，因此包括间接连接。",
            "Rule 1: exclude val patients sharing an inferred kinship group with the disease-specific train context. Groups are connected components of QC-valid KING ≥ 0.0884 edges, including indirect links.")+'</li><li>'+t(
            "规则 2：只排除与当前疾病 train 至少一人具有直接、通过质控的 KING ≥ 0.177 边的 val 患者。仅有间接连接或较低系数不会触发排除；val 内部的亲缘边本身也不会触发排除。",
            "Rule 2: exclude only val patients with at least one direct QC-valid KING ≥ 0.177 edge to the disease-specific train context. Indirect links, lower coefficients and val-to-val edges alone do not trigger removal.")+'</li></ol>'
        text += '<p class="note">'+t("两种规则均保留亲缘状态未确定者：T2D/CAD 为 42 人，女性 BC 为 26 人。这里的组是基因型推断结果，并非确认的家谱。规则 2 的保留集仍可能包含与 train 同组者；两个筛选结果都不能称为已确认完全无亲缘的外部测试集。",
            "Both rules retain unresolved-kinship patients: 42 for T2D/CAD and 26 for female BC. These are genotype-inferred groups, not verified pedigrees. Rule 2 may retain shared-group patients; neither subset is a certified fully unrelated external test set.")+'</p>'
        text += '<div class="scroll">'+counts.to_html(index=False, border=0)+'</div></section>'
        text += '<section><h2>'+t("验证集预测表现", "Validation predictive performance")+'</h2><p>'+t(
            "AUROC 越高越好，log loss 越低越好；Nagelkerke R² 是相对于该子集标签患病率基线的预测似然指标。各筛选改变了评估人群，线条只连接同一冻结模型的点估计，不表示因果改善。",
            "Higher AUROC and lower log loss are better. Nagelkerke R² uses each subset’s label-prevalence null model. Filtering changes the assessed population; lines connect point estimates for the same frozen model and do not indicate causal improvement.")+'</p>'
        text += f'<img alt="Validation predictive performance" src="data:image/svg+xml;base64,{pictures["validation_performance"]}">'
        text += '<p>'+t("当前选定 Student：", "Selected students: ")+", ".join(f"{k.upper()} = {v['-1']}" for k, v in policy.items())+'. '+t("T2D 的选定模型就是 M0_KD，图中合并显示。所有 Student 主结果使用 seed 17。", "The T2D selected model is exactly M0_KD and is drawn once. All student primary results use seed 17.")+'</p></section>'
        text += '<section><h2>'+t("Student 与原 Teacher 的差距", "Student versus original Teacher")+'</h2>'
        text += f'<img alt="Paired student teacher differences" src="data:image/svg+xml;base64,{pictures["student_teacher_gap"]}">'
        text += '<p class="note">'+t("区间来自 1,000 次可用亲缘组配对 bootstrap，条件于冻结预测，未重新训练或校正多重比较。未确定亲缘者使用患者级占位组，区间不覆盖未知亲缘依赖。区间跨 0 不代表已证明等效。该分析为开发 val 上的事后敏感性分析。",
            "Intervals use 1,000 paired available-family bootstrap draws conditional on frozen predictions, without refitting or multiplicity adjustment. Unresolved patients use patient-level placeholder groups, so unknown relatedness is not captured. Intervals crossing zero do not establish equivalence. This is a post-hoc sensitivity analysis on developmental val.")+'</p>'
        display = gaps.loc[gaps.arm.isin(["PRIMARY_SELECTED", "M0_KD"]) & gaps.reference.eq("TabPFN_v3") & gaps.metric.isin(["auroc", "log_loss"]), ["trait", "cohort", "arm", "metric", "delta", "ci_low", "ci_high"]]
        text += '<details><summary>'+t("展开配对差值和区间", "Show paired differences and intervals")+'</summary><div class="scroll">'+display.to_html(index=False, border=0, float_format=lambda x: f"{x:.5f}")+'</div></details></section>'
        text += '<section><h2>'+t("完整指标", "Complete metrics")+'</h2><p>'+t("Brier、average precision 和校准指标保存在 CSV。概率 R² 与 logit R² 衡量对原 TabPFN v3 的保真度，不是对真实疾病标签的 R²。", "Brier score, average precision and calibration diagnostics are included in CSV. Probability R² and logit R² measure fidelity to original TabPFN v3, not fit to true disease labels.")+'</p>'
        for cohort in COHORTS:
            table = metrics.loc[metrics.cohort.eq(cohort), ["trait", "arm", "n", "auroc", "log_loss", "brier", "nagelkerke_r2", "probability_r2_to_teacher", "logit_r2"]]
            text += f'<details><summary>{cohort}</summary><div class="scroll">'+table.to_html(index=False, border=0, float_format=lambda x: f"{x:.5f}")+'</div></details>'
        text += '<p><a href="metrics.csv">metrics.csv</a> · <a href="paired_differences.csv">paired_differences.csv</a> · <a href="cohort_counts.csv">cohort_counts.csv</a></p>'
        text += '<p><a href="validation_performance.png">PNG</a> · <a href="validation_performance.svg">SVG</a> · <a href="validation_performance.pdf">PDF</a></p></section></main></body></html>'
        (out / f"validation_filters_{lang}.html").write_text(text)


class OfflineQA(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = 0

    def handle_starttag(self, tag, attrs):
        assert tag != "script"
        if tag == "img":
            src = dict(attrs)["src"]
            assert src.startswith("data:image/svg+xml;base64,")
            ElementTree.fromstring(base64.b64decode(src.split(",", 1)[1], validate=True))
            self.images += 1


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eligibility-config", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_dir, args.out_dir
    assert root.resolve() not in out.resolve().parents and root.resolve() != out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report, private = out / "report", out / "private"
    report.mkdir()
    private.mkdir()
    complete = read(root / "COMPLETE.json")
    assert complete["status"] == "completed_and_verified"
    source = Path(read(root / "PROTOCOL.json")["identity"]["config"]["source_run"])
    audit = read(source / "audit_report.json")
    assert sha256_file(args.eligibility_config) == audit["source_sha256"][str(args.eligibility_config)]
    rules = load_eligibility_rules(args.eligibility_config)
    assert "bc" in rules
    policy_path = root / "report/val/POLICY_LOCK.json"
    assert sha256_file(policy_path) == complete["aggregate_output_sha256"]["report/val/POLICY_LOCK.json"]
    policy = read(policy_path)["choices"]
    qc_root = Path(audit["paths"]["qc_manifest"]).parent
    assert sha256_file(qc_root / "manifest.json") == audit["source_sha256"][str(qc_root / "manifest.json")]
    qc_manifest = read(qc_root / "manifest.json")
    assert qc_manifest["kinship_threshold"] == .0884
    assert sha256_file(qc_manifest["source_paths"]["kinship"]) == qc_manifest["source_sha256"]["kinship"]
    verify_files(qc_root, {n: qc_manifest["output_sha256"][n] for n in ["kinship_groups.csv", "retained_kinship_edges.csv", "sample_qc.csv"]})
    used = {str(p): sha256_file(p) for p in [root / "COMPLETE.json", root / "PROTOCOL.json", source / "audit_report.json", policy_path, args.eligibility_config, qc_root / "manifest.json", qc_root / "kinship_groups.csv", qc_root / "retained_kinship_edges.csv"]}
    groups = pd.read_csv(qc_root / "kinship_groups.csv", dtype=str)
    edges = pd.read_csv(qc_root / "retained_kinship_edges.csv", dtype={"IID1": str, "IID2": str})
    assert edges.KINSHIP.ge(.0884).all()
    rebuilt = kinship_groups(groups.sampleid.tolist(), edges, .0884)
    pd.testing.assert_frame_equal(groups.sort_values("sampleid").reset_index(drop=True), rebuilt)
    prepared, membership, counts = {}, [], []
    for trait in TRAITS:
        d, e, _ = load_inputs(source, trait)
        assert set(filter_eligible(d, trait, rules).sampleid) == set(d.sampleid)
        assert set(filter_eligible(e, trait, rules).sampleid) == set(e.sampleid)
        shared, direct, unknown = removal_masks(d, e, edges)
        masks = {"original_val": np.ones(len(e), dtype=bool), "exclude_shared_group": ~shared, "exclude_direct_0177": ~direct}
        membership.append(e[["sampleid", "kinship_group"]].assign(trait=trait, remove_rule1=shared, remove_rule2=direct, unresolved=unknown))
        arrays = model_arrays(root, trait, e.sampleid.tolist(), policy, used)
        prepared[trait] = (e, masks, arrays)
        for cohort, mask in masks.items():
            counts.append({"trait": trait, "cohort": cohort, "training_n_unchanged": len(d), "original_val_n": len(e), "removed": int((~mask).sum()), "n": int(mask.sum()), "unresolved_retained": int((mask & unknown).sum()), "shared_group_patients_remaining": int((mask & shared).sum())})
        print(f"Locked {trait}: train={len(d)}, val={len(e)}, drop group={shared.sum()}, drop direct={direct.sum()}", flush=True)
    pd.concat(membership, ignore_index=True).to_parquet(private / "cohort_membership.parquet", index=False)
    lock = {"filters": {"exclude_shared_group": "Remove val sharing a QC-valid KING>=0.0884 connected component with disease-specific train", "exclude_direct_0177": "Remove val with a direct QC-valid KING>=0.177 edge to disease-specific train"}, "unknown_kinship": "retained in both filters", "train_unchanged": True, "refitted": False, "reselected": False, "evaluation_labels_used_for_filtering": False, "primary_seed": 17, "selected_arms": policy, "membership_sha256": sha256_file(private / "cohort_membership.parquet"), "input_sha256": used, "script_sha256": sha256_file(__file__)}
    write_json(out / "ANALYSIS_LOCK.json", lock)
    # Patient membership and model choices are fixed before opening val labels.
    lp = Path(audit["paths"]["evaluation"])
    assert sha256_file(lp) == audit["source_sha256"][str(lp)]
    labels = pd.read_csv(lp, dtype={"sampleid": str}, usecols=["sampleid", *TRAITS]).set_index("sampleid")
    metrics, differences = [], []
    for trait, (e, masks, arrays) in prepared.items():
        y = labels.loc[e.sampleid, trait].to_numpy(int)
        for cohort, mask in masks.items():
            for arm, p in arrays.items():
                q, pred, truth = arrays["TabPFN_v3"][mask], p[mask], y[mask]
                metrics.append({"trait": trait, "cohort": cohort, "arm": arm, **basic_metrics(truth, pred), "nagelkerke_r2": nagelkerke_r2(truth, pred), "average_precision": average_precision_score(truth, pred), **calibration(truth, pred), **fidelity(pred, q), "probability_r2_to_teacher": r2_score(q, pred)})
            # Reuse the previously tested bootstrap with its expected contrast-slot names.
            slots = {"BIO_POLICY": "PRIMARY_SELECTED", "H_SELECTED": "M0_KD", "M0_KD": "M0_KD", "TabPFN_v3": "TabPFN_v3", "TabPFN_v3.5": "TabPFN_v3.5", "Kumo_Tabular": "Kumo_Tabular"}
            delta = paired_bootstrap(y[mask], {k: arrays[v][mask] for k, v in slots.items()}, e.loc[mask, "kinship_group"].to_numpy(), "nested", replicates=1000)
            delta["arm"], delta["reference"] = delta.arm.map(slots), delta.reference.map(slots)
            delta = delta.loc[delta.arm.ne(delta.reference)].drop_duplicates(["arm", "reference", "metric"])
            differences.append(delta.assign(trait=trait, cohort=cohort))
            print(f"Evaluated {trait}/{cohort}: n={mask.sum()}", flush=True)
    metrics, differences, counts = pd.DataFrame(metrics), pd.concat(differences, ignore_index=True), pd.DataFrame(counts)
    events = metrics.loc[metrics.arm.eq("M0_KD"), ["trait", "cohort", "events", "nonevents"]]
    counts = counts.merge(events, on=["trait", "cohort"], validate="one_to_one")
    prior_path = root / "report/val/metrics.csv"
    assert sha256_file(prior_path) == complete["aggregate_output_sha256"]["report/val/metrics.csv"]
    prior = pd.read_csv(prior_path).set_index(["trait", "arm"])
    for row in metrics.loc[metrics.cohort.eq("original_val")].itertuples():
        arm = "BIO_POLICY" if row.arm == "PRIMARY_SELECTED" else row.arm
        keys = ["auroc", "log_loss", "brier", "nagelkerke_r2"]
        np.testing.assert_allclose([getattr(row, k) for k in keys], prior.loc[(row.trait, arm), keys].to_numpy(float), atol=1e-12, rtol=0)
    for name, table in [("metrics", metrics), ("paired_differences", differences), ("cohort_counts", counts)]:
        table.to_csv(report / f"{name}.csv", index=False)
    figures(metrics, differences, counts, report)
    html_reports(metrics, differences, counts, policy, report)
    html_qa = []
    for lang in ["zh", "en"]:
        p = report / f"validation_filters_{lang}.html"
        checker = OfflineQA()
        checker.feed(p.read_text())
        assert checker.images == 2
        if lang == "en":
            assert not re.search("[\u4e00-\u9fff]", p.read_text())
        html_qa.append({"language": lang, "embedded_svg_images": checker.images, "sha256": sha256_file(p)})
    for p in report.glob("*.csv"):
        assert not set(pd.read_csv(p, nrows=0).columns) & {"sampleid", "kinship_group", "IID1", "IID2"}
    write_json(report / "METHODS.json", {"source_run": str(root), "analysis_lock_sha256": sha256_file(out / "ANALYSIS_LOCK.json"), "assessment": "full-train context to official val; not OOF", "filters_posthoc": True, "bootstrap_replicates": 1000, "bootstrap_unit": "available kinship group; unresolved patients use patient placeholders", "bootstrap_conditional_on_frozen_predictions": True, "multiplicity_adjusted": False, "original_val_metrics_reproduced": True, "html_qa": html_qa, "browser_render_tested": False})
    archive_path = report / "validation_filters_bilingual.zip"
    members = [p for p in report.iterdir() if p.is_file() and p.suffix != ".zip"]
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in members:
            z.write(p, p.name)
    with zipfile.ZipFile(archive_path) as z:
        assert z.testzip() is None
        assert all(z.read(p.name) == p.read_bytes() for p in members)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for p in [Path(__file__), Path("tests/test_prs_validation_kinship_filters.py")]:
        (snapshot / p.name).write_bytes(p.read_bytes())
    verify_files(Path.cwd(), used)
    write_json(out / "VERIFIED.json", {"status": "completed_and_verified", "original_val_metrics_exact": True, "group_reconstruction_exact": True, "train_unchanged": True, "refitted": False, "filters": 2, "cohorts": 3, "traits": 3, "model_arms": 5, "script_sha256": sha256_file(__file__), "label_source_sha256": sha256_file(lp), "upstream_input_sha256": used, "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*") if p.is_file() and p.name != "VERIFIED.json"}})
    print(f"Verified report: {report}", flush=True)


if __name__ == "__main__":
    main()
