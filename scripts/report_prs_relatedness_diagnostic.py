"""Post-hoc attribution of the validation gap by known train/val family overlap."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report_prs_distillation_bio import FM_ROOT, paired_bootstrap
from scipy.special import logit
from sklearn.metrics import average_precision_score

from decipehr.external_pgs import sha256_file, write_json
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import AGE, PCS
from decipehr.prs_distillation.reporting import basic_metrics, calibration, fidelity


def read(path):
    return json.loads(Path(path).read_text())


def predictions(directory, filename):
    seal = "PREDICTIONS_COMPLETE.json" if filename == "direct_predictions.parquet" else "COMPLETE.json"
    receipt = read(directory / seal)
    assert sha256_file(directory / filename) == receipt["output_sha256"][filename]
    return pd.read_parquet(directory / filename)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    cfg = read(root / "PROTOCOL.json")["identity"]["config"]
    source = Path(cfg["source_run"])
    audit = read(source / "audit_report.json")
    label_path = Path(audit["paths"]["evaluation"])
    assert sha256_file(label_path) == audit["source_sha256"][str(label_path)]
    labels = pd.read_csv(label_path, dtype={"sampleid": str}, usecols=["sampleid", "T2D", "cad", "bc"]).set_index("sampleid")
    policy = read(root / "report/val/POLICY_LOCK.json")["choices"]
    replay = read(root / "teacher_replay/REPLAY.json")
    assert replay["status"] == "passed" and all(r["max_absolute_difference"] == 0 for r in replay["comparisons"])
    verify_files(root / "teacher_replay", replay["output_sha256"])
    metrics, deltas, decomposition, counts, members = [], [], [], [], []
    for trait in ["T2D", "cad", "bc"]:
        d, e, manifest = load_inputs(source, trait)
        ids = e.sampleid.tolist()
        y = labels.loc[ids, trait].to_numpy(int)
        shared = e.kinship_group.isin(set(d.kinship_group)).to_numpy()
        unresolved = e.kinship_group.str.startswith("UNRESOLVED_QC").to_numpy()
        assert not (shared & unresolved).any()
        strata = {"known_shared_family": shared, "no_known_shared_family": ~shared & ~unresolved,
                  "unresolved": unresolved}
        columns = [*manifest.column, AGE, *PCS, "sex"]
        duplicate = pd.util.hash_pandas_object(e[columns], index=False).isin(set(
            pd.util.hash_pandas_object(d[columns], index=False))).to_numpy()
        all_models = pd.concat([predictions(root / "core" / trait, "direct_predictions.parquet"),
            predictions(root / "low_order" / trait, "predictions.parquet")], ignore_index=True)
        arrays = {}
        for name, arm in [("PRIMARY_SELECTED", policy[trait]["-1"]), ("M0_KD", "M0_KD")]:
            p = all_models.loc[all_models.arm.eq(arm) & all_models.seed.eq(17)].set_index("sampleid")
            assert p.index.is_unique and set(p.index) == set(ids)
            arrays[name] = p.loc[ids, "probability"].to_numpy()
        directory = root / "constraint_ablation/AGE" / trait
        arm = read(directory / "SELECTION_LOCK.json")["selected_arm"]
        p = predictions(directory, "predictions.parquet")
        arrays["AGE_SELECTED"] = p.loc[p.arm.eq(arm)].set_index("sampleid").loc[ids, "probability"].to_numpy()
        arrays["TabPFN_v3"] = np.load(root / "inputs" / trait / "targets.npz")["q_E"]
        for backend, name in [("tabpfn35", "TabPFN_v3.5"), ("kumo", "Kumo_Tabular")]:
            directory = FM_ROOT / backend / trait / "D_to_E"
            verify_files(directory, read(directory / "COMPLETE.json")["output_sha256"])
            arrays[name] = pd.read_parquet(directory / "predictions.parquet").set_index("sampleid").loc[ids, "probability"].to_numpy()
        q = arrays["TabPFN_v3"]
        logit_error = (logit(np.clip(arrays["PRIMARY_SELECTED"], 1e-6, 1-1e-6))-logit(np.clip(q, 1e-6, 1-1e-6)))**2
        for group, mask in {"all_original_val": np.ones(len(e), dtype=bool), **strata}.items():
            qgroup = q[mask]
            counts.append({"trait": trait, "stratum": group, "n": int(mask.sum()), "events": int(y[mask].sum()),
                "available_groups": int(e.loc[mask, "kinship_group"].nunique()),
                "exact_duplicate_input_rows": int(duplicate[mask].sum()),
                "teacher_probability_below_0p001_or_above_0p999": int(((qgroup < .001) | (qgroup > .999)).sum()),
                "primary_logit_error_share": logit_error[mask].sum()/logit_error.sum()})
            for arm, p in arrays.items():
                values = basic_metrics(y[mask], p[mask])
                metrics.append({"trait": trait, "stratum": group, "arm": arm, **values,
                    "average_precision": average_precision_score(y[mask], p[mask]),
                    "nagelkerke_r2": nagelkerke_r2(y[mask], p[mask]), **calibration(y[mask], p[mask]),
                    **fidelity(p[mask], qgroup),
                    "probability_r2_to_teacher": 1-np.sum((p[mask]-qgroup)**2)/np.sum((qgroup-qgroup.mean())**2),
                    "unstable": mask.sum() < 100 or min(values["events"], values["nonevents"]) < 20})
            if group in strata:
                delta = basic_metrics(y[mask], arrays["PRIMARY_SELECTED"][mask])["log_loss"]-basic_metrics(y[mask], qgroup)["log_loss"]
                decomposition.append({"trait": trait, "stratum": group, "n": int(mask.sum()),
                    "n_fraction": mask.mean(), "primary_minus_teacher_log_loss": delta,
                    "contribution_to_full_val_gap": mask.mean()*delta})
                members.append(e.loc[mask, ["sampleid", "kinship_group"]].assign(trait=trait, stratum=group))
        full_delta = basic_metrics(y, arrays["PRIMARY_SELECTED"])["log_loss"]-basic_metrics(y, q)["log_loss"]
        np.testing.assert_allclose(sum(x["contribution_to_full_val_gap"] for x in decomposition if x["trait"] == trait), full_delta, atol=1e-12)
        mask = strata["no_known_shared_family"]
        slots = {"H_SELECTED": "PRIMARY_SELECTED", "M0_KD": "M0_KD", "G_A6_KD": "AGE_SELECTED",
                 "TabPFN_v3": "TabPFN_v3", "TabPFN_v3.5": "TabPFN_v3.5", "Kumo_Tabular": "Kumo_Tabular"}
        interval = paired_bootstrap(y[mask], {k: arrays[v][mask] for k, v in slots.items()},
            e.loc[mask, "kinship_group"].to_numpy(), "nested")
        interval["arm"], interval["reference"] = interval.arm.map(slots), interval.reference.map(slots)
        deltas.append(interval.assign(trait=trait, stratum="no_known_shared_family"))
        print(f"{trait}: shared={shared.sum()}, no known shared={mask.sum()}, unresolved={unresolved.sum()}", flush=True)
    out = root / "report/relatedness_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    private = root / "relatedness_diagnostic"
    private.mkdir(exist_ok=True)
    pd.concat(members, ignore_index=True).to_parquet(private / "private_membership.parquet", index=False)
    tables = {"metrics": pd.DataFrame(metrics), "paired_differences": pd.concat(deltas, ignore_index=True),
              "gap_decomposition": pd.DataFrame(decomposition), "counts": pd.DataFrame(counts)}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, trait in zip(axes, ["T2D", "cad", "bc"], strict=True):
        part = tables["gap_decomposition"].loc[tables["gap_decomposition"].trait.eq(trait)]
        values = 1000*part.contribution_to_full_val_gap
        ax.bar(["Shared known\nfamily", "No known shared\nfamily", "Unresolved"], values, color=["#b36b27", "#148d80", "#8491a5"])
        for i, value in enumerate(values):
            ax.annotate(f"{value:+.2f}", (i, value), xytext=(0, 5 if value >= 0 else -15), textcoords="offset points", ha="center")
        ax.axhline(0, color="gray", lw=1)
        ax.set(title=trait.upper(), ylabel="Contribution to full-val Δ log loss × 1,000")
        ax.margins(y=.25)
    fig.suptitle("Arithmetic decomposition of the original validation gap · Student − TabPFN v3")
    for ext in ["svg", "png"]:
        fig.savefig(out / f"gap_decomposition.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    payload = base64.b64encode((out / "gap_decomposition.svg").read_bytes()).decode()
    for lang in ["zh", "en"]:
        def t(zh, en, lang=lang):
            return zh if lang == "zh" else en
        title = t("验证差距与训练／验证亲缘重叠", "Validation gaps and train/validation family overlap")
        text = f'<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
        text += '<style>body{font:16px/1.7 system-ui;color:#18304b;max-width:1220px;margin:auto;padding:25px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{background:#fff3d9;padding:18px}section{overflow:auto}</style>'
        text += f'<h1>{title}</h1><p class="note">'+t("这是事后分层诊断。所有模型、超参数和主实验选择保持原来训练内部锁定的状态；训练患者仍为全量。仅按评估患者是否与当前疾病的训练患者共享已知亲缘组来划分原始 val。未依据这些分层结果重新挑选模型。",
            "This is a post-hoc stratified diagnostic. All models, hyperparameters and primary selections retain their original train-internal locks, using all training patients. Original val is partitioned by known family-group overlap with the disease-specific training context. No models are reselected using these strata.")+'</p>'
        text += '<p>'+t("T2D/CAD：159 人处于跨界已知亲缘组，2,651 人没有已知同组训练患者，42 人亲缘状态未确定。女性 BC 对应为 75、1,527、26 人。原 Teacher 从同一 checkpoint 重新推理，三项任务的概率与缓存逐项完全一致。没有发现完全相同的输入特征行。",
            "For T2D/CAD, 159 validation patients share a known family group with training, 2,651 have no known shared group, and 42 have unresolved kinship. Female BC counts are 75, 1,527 and 26. Fresh inference from the original teacher checkpoint exactly reproduces every cached probability in all three tasks. No exactly duplicated input-feature rows were found.")+'</p>'
        text += f'<img alt="Family-overlap gap decomposition" src="data:image/svg+xml;base64,{payload}">'
        text += '<p>ΔLL<sub>all</sub>=Σ<sub>g</sub>(n<sub>g</sub>/N)·[LL<sub>Student,g</sub>−LL<sub>Teacher,g</sub>]</p>'
        text += '<p>'+t("这是对已训练模型误差差值的算术分解，不证明亲缘关系导致差距；组间年龄、祖源等构成也可能不同。无已知同组者仍不是完全确认无亲缘关系的新外部测试集。",
            "This is an arithmetic decomposition of error differences for frozen models, not proof that relatedness causes the gap; strata may also differ in age, ancestry and other composition. The no-known-shared-family subset is not a new external test set with fully established unrelatedness.")+'</p>'
        text += '<h2>'+t("没有已知同组训练患者的验证子集", "Validation subset without a known shared training family")+'</h2><section>'
        display = tables["metrics"].loc[lambda x: x.stratum.eq("no_known_shared_family"),
            ["trait", "arm", "n", "events", "log_loss", "auroc", "brier", "nagelkerke_r2", "probability_r2_to_teacher", "logit_r2"]]
        text += display.to_html(index=False, float_format=lambda x: f"{x:.5f}", border=0)+'</section>'
        text += '<details><summary>'+t("配对差值与名义 95% 区间", "Paired differences and nominal 95% intervals")+'</summary>'+tables["paired_differences"].query("metric == 'log_loss'")[["trait", "arm", "reference", "delta", "ci_low", "ci_high"]].to_html(index=False, float_format=lambda x: f"{x:.5f}", border=0)+'</details>'
        text += '<details><summary>'+t("各层人数、极端概率与保真误差分布", "Stratum counts, extreme probabilities and fidelity-error distribution")+'</summary>'+tables["counts"].to_html(index=False, float_format=lambda x: f"{x:.5f}", border=0)+'</details>'
        text += '<p>'+t("区间为 1,000 次可用亲缘组配对 bootstrap，条件于冻结预测，未校正多重比较。probability R² 与 logit R² 均衡量对 Teacher 的保真度；Nagelkerke R² 是对真实标签的预测似然指标。极端概率会显著影响 logit R²，因此同时查看概率误差和概率尺度 R²。下一步应在训练、验证之间实施统一亲缘隔离后，再确认 Student 与 Teacher 的差距。",
            "Intervals use 1,000 paired available-family bootstrap draws conditional on frozen predictions, without multiplicity correction. Probability R² and logit R² measure teacher fidelity; Nagelkerke R² measures predictive likelihood against true labels. Extreme probabilities can dominate logit R², so probability errors and probability-scale R² should be considered alongside it. A consistently family-isolated train/validation comparison is the next confirmation step.")+'</p></html>'
        (out / f"relatedness_{lang}.html").write_text(text)
    write_json(out / "VERIFIED.json", {"status": "posthoc_stratified_diagnostic_complete", "script_sha256": sha256_file(__file__),
        "no_model_refitting_or_reselection": True, "teacher_replay_exact": True,
        "private_membership_sha256": sha256_file(private / "private_membership.parquet"),
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*")
                          if p.is_file() and p.name != "VERIFIED.json"}})


if __name__ == "__main__":
    main()
