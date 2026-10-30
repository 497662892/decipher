"""Secondary report separating age monotonicity from conditional PRS monotonicity."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from report_prs_distillation_bio import FM_ROOT, paired_bootstrap
from run_prs_distillation_bio import verify
from sklearn.metrics import average_precision_score

from decipehr.external_pgs import sha256_file, write_json
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.reporting import basic_metrics, calibration, fidelity


def read(path):
    return json.loads(Path(path).read_text())


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=["val", "nested"], required=True)
    args = parser.parse_args()
    root, scope = args.run_dir, args.scope
    _, cfg = verify(root)
    source = Path(cfg["source_run"])
    collected, selections, locks = {}, [], {}
    for trait in ["T2D", "cad", "bc"]:
        parts, locked = [], {}
        for fold in ([-1] if scope == "val" else range(5)):
            r = root if fold == -1 else root / "nested" / f"outer{fold}"
            for prior in ["STRICT", "AGE", "PRS", "FREE"]:
                d = r / "low_order" / trait if prior == "STRICT" else r / "constraint_ablation" / prior / trait
                receipt = read(d / "COMPLETE.json")
                assert receipt["models"] == 4 and not receipt["evaluation_labels_read"]
                verify_files(d, receipt["output_sha256"])
                selection = read(d / "SELECTION_LOCK.json")
                arm = selection["selected_arm"]
                locked[f"{fold}/{prior}"] = arm
                choice = selection["choices"][arm]
                selections.append({"trait": trait, "outer_fold": fold, "constraint": prior, "arm": arm,
                    "H_loss": choice["H_log_loss"], "lambda": choice["lambda"], "iterations": choice["iterations"],
                    "leaves": choice["leaves"], "l2": choice["l2"], "training_patients": selection["patient_n_D"]})
                p = pd.read_parquet(d / "predictions.parquet")
                parts.extend([p.assign(outer_fold=fold), p.loc[p.arm.eq(arm)].assign(arm=f"{prior}_SELECTED", outer_fold=fold)])
        collected[trait], locks[trait] = pd.concat(parts, ignore_index=True), locked
    for backend in ["tabpfn35", "kumo"]:
        verify_files(FM_ROOT / backend, read(FM_ROOT / backend / "COMPLETE.json")["output_sha256"])
    out = root / "report" / f"constraint_ablation_{scope}"
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "SELECTION_LOCK.json"
    if lock_path.exists():
        assert read(lock_path) == locks
    write_json(lock_path, locks)
    # All models in this scope are sealed before any assessment-label join.
    if scope == "val":
        audit = read(source / "audit_report.json")
        labels_path = Path(audit["paths"]["evaluation"])
        assert sha256_file(labels_path) == audit["source_sha256"][str(labels_path)]
        labels = pd.read_csv(labels_path, dtype={"sampleid": str}, usecols=["sampleid", "T2D", "cad", "bc"]).set_index("sampleid")
    metrics, effects = [], []
    for trait, frame in collected.items():
        base = frame.loc[frame.arm.eq("STRICT_SELECTED")].set_index("sampleid")
        assert base.index.is_unique
        ids = base.index.tolist()
        y = labels.loc[ids, trait].to_numpy(int) if scope == "val" else pd.read_parquet(
            source / trait / "training.parquet").set_index("sampleid").loc[ids, "y"].to_numpy(int)
        arrays = {}
        for arm, part in frame.groupby("arm"):
            indexed = part.set_index("sampleid")
            assert indexed.index.is_unique and set(indexed.index) == set(ids)
            arrays[arm] = indexed.loc[ids, "probability"].to_numpy()
        arrays["TabPFN_v3"] = base.p_teacher.to_numpy()
        for backend, name in [("tabpfn35", "TabPFN_v3.5"), ("kumo", "Kumo_Tabular")]:
            paths = [FM_ROOT / backend / trait / "D_to_E/predictions.parquet"] if scope == "val" else [
                FM_ROOT / backend / trait / f"D_oof_{fold}/predictions.parquet" for fold in range(5)]
            p = pd.concat([pd.read_parquet(path) for path in paths]).set_index("sampleid")
            assert p.index.is_unique and set(p.index) == set(ids)
            arrays[name] = p.loc[ids, "probability"].to_numpy()
        for arm, p in arrays.items():
            metrics.append({"scope": scope, "trait": trait, "arm": arm, **basic_metrics(y, p),
                "average_precision": average_precision_score(y, p), "nagelkerke_r2": nagelkerke_r2(y, p),
                **calibration(y, p), **fidelity(p, arrays["TabPFN_v3"])})
        # Reuse the independently tested family-bootstrap implementation. Slot
        # names are mapped back to this experiment's actual model names below.
        slots = {"H_SELECTED": "AGE_SELECTED", "M0_KD": "STRICT_SELECTED", "G_A6_KD": "PRS_SELECTED",
                 "BIO_POLICY": "FREE_SELECTED", "TabPFN_v3": "TabPFN_v3", "TabPFN_v3.5": "TabPFN_v3.5",
                 "Kumo_Tabular": "Kumo_Tabular"}
        delta = paired_bootstrap(y, {key: arrays[name] for key, name in slots.items()}, base.kinship_group.to_numpy(), "nested")
        delta["arm"], delta["reference"] = delta.arm.map(slots), delta.reference.map(slots)
        effects.append(delta.assign(trait=trait, scope=scope))
        print(f"Evaluated secondary {scope}/{trait}: {len(arrays)} models", flush=True)
    m, d, selected = pd.DataFrame(metrics), pd.concat(effects, ignore_index=True), pd.DataFrame(selections)
    m.to_csv(out / "metrics.csv", index=False)
    d.to_csv(out / "paired_differences.csv", index=False)
    selected.to_csv(out / "selection.csv", index=False)
    arms = ["STRICT_SELECTED", "AGE_SELECTED", "PRS_SELECTED", "FREE_SELECTED", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"]
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for j, trait in enumerate(["T2D", "cad", "bc"]):
        part = m.loc[m.trait.eq(trait)].set_index("arm")
        for i, metric in enumerate(["log_loss", "auroc"]):
            ax = axes[i, j]
            for k, arm in enumerate(arms):
                value = part.loc[arm, metric]
                ax.scatter(value, k, color="#0a8a77" if arm == "AGE_SELECTED" else "#4a617e")
                ax.annotate(f"{value:.4f}", (value, k), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
            ax.set(yticks=range(len(arms)), yticklabels=arms if j == 0 else [""]*len(arms),
                   ylim=(len(arms)-.5, -.5), xlabel="Log loss (lower)" if i == 0 else "AUROC (higher)")
            ax.margins(x=.65)
            if i == 0:
                ax.set_title(trait.upper())
    fig.suptitle(f"Secondary constraint ablation · {scope} · all choices made inside training")
    for ext in ["svg", "png"]:
        fig.savefig(out / f"performance.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    image = base64.b64encode((out / "performance.svg").read_bytes()).decode()
    for lang in ["zh", "en"]:
        def t(zh, en, lang=lang):
            return zh if lang == "zh" else en
        title = t("约束消融：保留哪些生物学先验？", "Constraint ablation: which biological priors should be retained?")
        text = f'<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
        text += '<style>body{font:16px/1.7 system-ui;color:#18304b;max-width:1220px;margin:auto;padding:25px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{background:#fff3d9;padding:18px}section{overflow:auto}</style>'
        text += f'<h1>{title}</h1><p class="note">'+t(f"范围：{scope}。这是主实验 val 结果和 train 内 Teacher 探针之后提出的二级探索。保留主实验的全部模型与选择；本轮每项超参数仍只由相应 U/H 决定，所有软标签来自同范围的交叉预测。它不是新的独立确认集。",
            f"Scope: {scope}. This secondary exploration follows the primary val results and train-internal teacher probes. Primary models and selections are preserved. Every hyperparameter still uses the corresponding U/H only, with scope-local cross-predicted soft targets. This is not a new independent confirmation set.")+'</p>'
        text += '<p>'+t("STRICT：年龄与每个 PRS 均单调；AGE：仅年龄单调；PRS：仅 PRS 单调；FREE：均不强制，作为诊断对照。四组共享 GAM/PAIR 交互图、预处理、soft BCE 和 40 个候选的选择预算；只改变单调约束坐标。每组分别在 H 选择结构、λ、叶子数、L2 和树数。",
            "STRICT constrains age and every PRS; AGE constrains age only; PRS constrains PRSs only; FREE imposes neither and serves as a diagnostic control. All share the GAM/PAIR graph, preprocessing, soft BCE and 40-candidate selection budget; only constrained coordinates change. Each group independently selects structure, λ, leaves, L2 and iterations on H.")+'</p>'
        text += '<div>η=η₀+Σfⱼ(xⱼ)+Σfⱼₖ(xⱼ,xₖ); p=sigmoid(η)</div><p>'+t("所有模型仍能精确拆成主效应与允许的二变量项。放宽 PRS 约束后，曲线表示控制其他相关 PRS 后的预测关系，不能解释为某个 PRS 的因果保护／危害作用。AGE 保留固定其他输入时年龄增加不降低总预测风险的保证。",
            "Every model retains exact main-effect and allowed pair-effect decomposition. Unconstrained PRS curves describe conditional prediction given other correlated scores, not causal protective or harmful effects. AGE preserves nondecreasing total risk with age at fixed other inputs.")+'</p>'
        text += f'<img alt="Matched constraint comparison" src="data:image/svg+xml;base64,{image}"><section>'
        text += m.loc[m.arm.isin(arms), ["trait", "arm", "n", "events", "log_loss", "auroc", "brier", "nagelkerke_r2", "logit_r2"]].to_html(index=False, float_format=lambda x: f"{x:.5f}", border=0)+'</section>'
        text += '<details><summary>'+t("配对 log loss 差值与 95% 区间", "Paired log-loss differences and 95% intervals")+'</summary>'+d.loc[d.metric.eq("log_loss"), ["trait", "arm", "reference", "delta", "ci_low", "ci_high"]].to_html(index=False, float_format=lambda x: f"{x:.5f}", border=0)+'</details>'
        text += '<details><summary>'+t("训练内部选择和实际预算", "Train-internal selections and actual budgets")+'</summary>'+selected.to_html(index=False, float_format=lambda x: f"{x:.5f}", border=0)+'</details>'
        text += '<p>'+t("区间使用 1,000 次已知亲缘组配对重采样，条件于冻结预测，未重训或校正多重比较。Nagelkerke R² 是预测似然指标；logit R² 是对原 Teacher 的模仿程度，均不是遗传解释度。",
            "Intervals use 1,000 paired available-family bootstrap draws conditional on frozen predictions, without refitting or multiplicity correction. Nagelkerke R² describes predictive likelihood; logit R² measures fidelity to the original teacher. Neither is genetic variance explained.")+'</p></html>'
        (out / f"constraint_ablation_{lang}.html").write_text(text)
    write_json(out / "VERIFIED.json", {"status": "secondary_exploration_reported", "scope": scope,
        "assessment_labels_used_for_selection": False, "script_sha256": sha256_file(__file__),
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*")
                          if p.is_file() and p.name != "VERIFIED.json"}})


if __name__ == "__main__":
    main()
