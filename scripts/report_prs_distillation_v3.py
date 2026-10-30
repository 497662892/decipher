"""Evaluate locked v3 predictions; emit aggregate interpretation and bilingual HTML."""

from __future__ import annotations

import argparse
import base64
import html
import os
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from run_prs_distillation_v3 import read_json, verify_run
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score

from decipehr.external_pgs import sha256_file, write_json
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.reporting import basic_metrics, calibration, fidelity
from decipehr.prs_distillation_v3 import load_bundle

VARIANTS = ["M0", "M1", "M2"]
TRAITS = ["T2D", "cad", "bc"]
NEW_ARMS = [f"{v}_{m}" for v in VARIANTS for m in ["SUP", "KD"]]
COLORS = {"M0_KD": "#C98929", "M1_KD": "#4489BA", "M2_KD": "#087F70",
          "V1_KD_MIX": "#9072B0", "T_MATCHED": "#263D6B", "LR": "#929EAC"}
METRICS = ["log_loss", "auroc", "brier", "nagelkerke_r2"]


def contrasts():
    pairs = [(f"M2_{m}", f"{v}_{m}") for m in ["SUP", "KD"] for v in ["M0", "M1"]]
    pairs += [(f"M1_{m}", f"M0_{m}") for m in ["SUP", "KD"]]
    pairs += [(f"{v}_KD", f"{v}_SUP") for v in VARIANTS]
    pairs += [(f"{v}_KD", "V1_KD_MIX") for v in VARIANTS]
    return pairs + [("M2_KD", a) for a in ["T_MATCHED", "LR", "META_V1"]]


def paired_intervals(y, arrays, groups, replicates, seed):
    arms = sorted({a for pair in contrasts() for a in pair})
    p = np.column_stack([arrays[a] for a in arms])
    loss = -(y[:, None]*np.log(np.clip(p, 1e-6, 1-1e-6)) +
             (1-y[:, None])*np.log1p(-np.clip(p, 1e-6, 1-1e-6)))
    loss_r2 = -(y[:, None]*np.log(np.clip(p, 1e-15, 1-1e-15)) +
                (1-y[:, None])*np.log1p(-np.clip(p, 1e-15, 1-1e-15)))
    brier = (p-y[:, None])**2
    _, inverse = np.unique(groups, return_inverse=True)
    g = inverse.max()+1
    group_n, group_y = np.bincount(inverse), np.bincount(inverse, weights=y)
    grouped = []
    for matrix in [loss, loss_r2, brier]:
        sums = np.zeros((g, len(arms)))
        np.add.at(sums, inverse, matrix)
        grouped.append(sums)
    collected = []
    rng = np.random.default_rng(seed)
    for b in range(replicates):
        counts = np.bincount(rng.integers(g, size=g), minlength=g)
        n, cases = counts @ group_n, counts @ group_y
        if not 0 < cases < n:
            continue
        mean_loss, mean_r2_loss, mean_brier = [counts @ sums / n for sums in grouped]
        rate = cases/n
        null_ll = rate*np.log(rate)+(1-rate)*np.log1p(-rate)
        r2 = -np.expm1(2*(null_ll+mean_r2_loss)) / -np.expm1(2*null_ll)
        weight = counts[inverse]
        auc = np.array([roc_auc_score(y, p[:, i], sample_weight=weight) for i in range(len(arms))])
        collected.append(np.column_stack([mean_loss, auc, mean_brier, r2]))
        if b % 250 == 249:
            print(f"  paired bootstrap {b+1}/{replicates}", flush=True)
    samples = np.stack(collected)
    measured = {a: {**basic_metrics(y, arrays[a]), "nagelkerke_r2": nagelkerke_r2(y, arrays[a])} for a in arms}
    rows = []
    for arm, reference in contrasts():
        difference = samples[:, arms.index(arm)]-samples[:, arms.index(reference)]
        for i, metric in enumerate(METRICS):
            lo, hi = np.quantile(difference[:, i], [.025, .975])
            rows.append({"arm": arm, "reference": reference, "metric": metric,
                         "delta": measured[arm][metric]-measured[reference][metric], "ci_low": lo, "ci_high": hi,
                         "valid_replicates": len(samples), "requested_replicates": replicates,
                         "resampling_unit": "available_kinship_group", "refitted": False,
                         "multiplicity_adjusted": False})
    return pd.DataFrame(rows)


def evaluate(root):
    protocol, cfg, source = verify_run(root)
    lr_root = Path(cfg["matched_logistic_run"])
    verify_files(lr_root, read_json(lr_root / "VERIFIED.json")["output_sha256"])
    # All disease predictions/configurations must exist before opening assessment outcomes.
    for trait in TRAITS:
        record = read_json(root / "core" / trait / "PREDICTIONS_COMPLETE.json")
        if record["smoke"] or record["models"] != 18 or record["evaluation_labels_read"]:
            raise ValueError("Full locked predictions are required for all diseases")
        verify_files(root / "core" / trait, record["output_sha256"])
    audit = read_json(source / "audit_report.json")
    labels_path = Path(audit["paths"]["evaluation"])
    assert sha256_file(labels_path) == audit["source_sha256"][str(labels_path)]
    labels = pd.read_csv(labels_path, dtype={"sampleid": str}, usecols=["sampleid", *TRAITS])
    out = root / "report"
    out.mkdir(exist_ok=True)
    measurements, effects, subgroup_rows, stability, gate_rows, contribution_rows, support_rows = [], [], [], [], [], [], []
    plot_data, decisions = {}, {}
    for trait in TRAITS:
        core = root / "core" / trait
        new = pd.read_parquet(core / "direct_predictions.parquet")
        old = pd.read_parquet(source / "core" / trait / "direct_predictions.parquet")
        primary = new.loc[new.seed.eq(17)].copy()
        ids = primary.loc[primary.arm.eq("M0_SUP"), "sampleid"].tolist()
        label = labels.set_index("sampleid").loc[ids, trait]
        assert label.isin([0, 1]).all()
        y = label.to_numpy(dtype=int)
        arrays = {}
        for arm, part in primary.groupby("arm"):
            assert part.sampleid.tolist() == ids and part.sampleid.is_unique
            arrays[arm] = part.probability.to_numpy()
        for old_arm, name in [("S_KD_MIX", "V1_KD_MIX"), ("S_SUP", "V1_SUP"), ("S_KD_PURE", "V1_KD_PURE"),
                              ("META", "META_V1"), ("T_MATCHED", "T_MATCHED")]:
            p = old.loc[old.arm.eq(old_arm) & old.seed.eq(17)].set_index("sampleid")
            assert p.index.is_unique and set(p.index) == set(ids)
            arrays[name] = p.loc[ids, "probability"].to_numpy()
        lr = pd.read_parquet(lr_root / trait / "validation_predictions.parquet").set_index("sampleid")
        assert set(lr.index) == set(ids)
        arrays["LR"] = lr.loc[ids, "probability"].to_numpy()
        groups = primary.loc[primary.arm.eq("M0_SUP"), "kinship_group"].to_numpy()
        np.testing.assert_array_equal(groups, lr.loc[ids, "kinship_group"])
        for arm, p in arrays.items():
            measurements.append({"trait": trait, "arm": arm, **basic_metrics(y, p),
                "average_precision": average_precision_score(y, p), "nagelkerke_r2": nagelkerke_r2(y, p),
                **calibration(y, p), **fidelity(p, arrays["T_MATCHED"])})
        print(f"Evaluating {trait}: {len(y)} paired patients / {len(arrays)} arms", flush=True)
        intervals = paired_intervals(y, arrays, groups, cfg["bootstrap_replicates"], cfg["bootstrap_seed"]).assign(trait=trait)
        effects.append(intervals)
        # Matched-seed comparisons use the same patient's frozen prediction in each fit.
        params = pd.read_csv(core / "parameters.csv")
        for arm in NEW_ARMS:
            base = new.loc[new.arm.eq(arm) & new.seed.eq(17)].set_index("sampleid")
            for seed in [29, 43]:
                other = new.loc[new.arm.eq(arm) & new.seed.eq(seed)].set_index("sampleid").loc[ids]
                delta = other.probability.to_numpy()-base.loc[ids].probability.to_numpy()
                row = {"trait": trait, "arm": arm, "seed": seed, "reference_seed": 17,
                       "probability_mae": np.mean(np.abs(delta)), "probability_max_abs": np.max(np.abs(delta))}
                for parameter in ["r", "d"]:
                    first = params.loc[params.arm.eq(arm) & params.seed.eq(17)].set_index("prs")[parameter]
                    second = params.loc[params.arm.eq(arm) & params.seed.eq(seed)].set_index("prs").loc[first.index, parameter]
                    row[f"{parameter}_mae"] = float(np.abs(first-second).mean())
                    row[f"{parameter}_max_abs"] = float(np.abs(first-second).max())
                stability.append(row)
        primary = primary.merge(labels[["sampleid", trait]].rename(columns={trait: "y"}), on="sampleid", validate="many_to_one")
        prep = read_json(core / "PREPROCESSING.json")["fixed_preprocessing"]
        edges = np.unique([-np.inf, *prep["age_quartiles"], np.inf])
        primary["age_bin"] = pd.cut(primary.age_sd, edges, include_lowest=True).astype(str)
        primary["joint_group"] = primary.hard_label+" | "+primary.age_bin
        for arm, part in primary.groupby("arm"):
            for dimension in ["hard_label", "age_bin", "joint_group"]:
                for group, sub in part.groupby(dimension, observed=True):
                    metrics = basic_metrics(sub.y, sub.probability)
                    subgroup_rows.append({"trait": trait, "arm": arm, "dimension": dimension, "group": group,
                        **metrics, "age_out_of_support": int(sub.age_out_of_support.sum()),
                        "unstable_exploratory": len(sub) < 100 or min(metrics["events"], metrics["nonevents"]) < 20})
            for group, sub in part.groupby("hard_label"):
                for column in [c for c in sub if c.startswith("pi__")]:
                    gate_rows.append({"trait": trait, "arm": arm, "group": group, "prs": column[4:],
                                      "n": len(sub), "mean_pi": sub[column].mean()})
            for column in [c for c in part if c.startswith("contributions__")]:
                contribution_rows.append({"trait": trait, "arm": arm, "prs": column.split("__", 1)[1],
                                          "mean_logit_contribution": part[column].mean(),
                                          "mean_abs_logit_contribution": part[column].abs().mean()})
        train = pd.read_parquet(source / trait / "training.parquet", columns=["age_sd", "hard_label", "y"])
        val = primary.loc[primary.arm.eq("M2_KD"), ["age_sd", "hard_label", "y"]]
        for scope, frame in [("train", train), ("val", val)]:
            frame = frame.copy()
            frame["age_bin"] = pd.cut(frame.age_sd, edges, include_lowest=True).astype(str)
            for (group, age_bin), sub in frame.groupby(["hard_label", "age_bin"], observed=True):
                support_rows.append({"trait": trait, "scope": scope, "ancestry": group, "age_bin": age_bin,
                                     "n": len(sub), "events": int(sub.y.sum()), "age_min": sub.age_sd.min(), "age_max": sub.age_sd.max()})
        lock = read_json(core / "SELECTION_LOCK.json")
        primary_intervals = intervals.loc[intervals.metric.eq("log_loss")]
        def contrast_row(arm, reference, primary_intervals=primary_intervals):
            return primary_intervals.loc[primary_intervals.arm.eq(arm) & primary_intervals.reference.eq(reference),
                                          ["delta", "ci_low", "ci_high"]].iloc[0].to_dict()
        decisions[trait] = {"H_selected_structure": lock["selected_structure_by_H"],
            "M2_KD_minus_M0_KD": contrast_row("M2_KD", "M0_KD"),
            "M2_KD_minus_M1_KD": contrast_row("M2_KD", "M1_KD"),
            "KD_minus_SUP": {v: contrast_row(f"{v}_KD", f"{v}_SUP") for v in VARIANTS}}
        plot_data[trait] = {"y": y, "arrays": arrays, "train_age": train.age_sd.to_numpy(),
                            "val_age": val.age_sd.to_numpy(), "parameters": params, "lock": lock}
    frames = {"metrics": pd.DataFrame(measurements), "paired_differences": pd.concat(effects, ignore_index=True),
              "subgroups": pd.DataFrame(subgroup_rows), "initialization_stability": pd.DataFrame(stability),
              "ancestry_gate": pd.DataFrame(gate_rows), "effective_contribution_summary": pd.DataFrame(contribution_rows),
              "age_ancestry_joint_support": pd.DataFrame(support_rows)}
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    write_json(out / "DECISION.json", decisions)
    return frames, plot_data, decisions, protocol


def plot_all(root, frames, data):
    out = root / "report/figures"
    out.mkdir(exist_ok=True)
    font_manager.fontManager.addfont("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Droid Sans Fallback"], "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "path"})
    def save(fig, name):
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    shown = ["M0_KD", "M1_KD", "M2_KD", "V1_KD_MIX", "T_MATCHED"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout="constrained")
    for j, trait in enumerate(TRAITS):
        m = frames["metrics"].query("trait == @trait").set_index("arm")
        for i, metric in enumerate(["auroc", "log_loss", "nagelkerke_r2"]):
            ax = axes[i, j]
            for y, arm in enumerate(shown):
                value = m.loc[arm, metric]
                ax.scatter(value, y, c=COLORS[arm], s=45)
                ax.annotate(f"{value:.4f}", (value, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
            ax.set(yticks=range(len(shown)), yticklabels=shown if j == 0 else [""]*len(shown), ylim=(4.6, -.6),
                   xlabel={"auroc": "AUROC ↑", "log_loss": "Log loss ↓", "nagelkerke_r2": "Nagelkerke R² ↑"}[metric])
            ax.margins(x=.65)
            ax.grid(axis="x", alpha=.2)
            if i == 0:
                ax.set_title({"T2D": "T2D", "cad": "CAD", "bc": "Female BC"}[trait], weight="bold")
    fig.suptitle("Direct validation performance · primary seed 17", fontsize=18)
    save(fig, "performance")
    plot_paired(root, frames)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for j, trait in enumerate(TRAITS):
        params = data[trait]["parameters"].query("arm == 'M2_KD'")
        prs = params.prs.unique().tolist()
        for i, parameter in enumerate(["r", "d"]):
            ax = axes[i, j]
            for seed, marker in [(17, "o"), (29, "x"), (43, "+")]:
                values = params.loc[params.seed.eq(seed)].set_index("prs").loc[prs, parameter]
                ax.plot(np.arange(len(prs)), values, marker=marker, ls="none", label=f"seed {seed}", alpha=.8)
            ax.set(xticks=range(len(prs)), xticklabels=prs, ylabel=parameter, title=f"{trait.upper()} · M2 KD")
            ax.tick_params(axis="x", rotation=90, labelsize=6.5)
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Per-PRS magnitude r and sign-aware strength d · seed spread is not a confidence interval", fontsize=15)
    save(fig, "r_d_parameters")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for j, trait in enumerate(TRAITS):
        model, prep, _ = load_bundle(root / "core" / trait / "models/M2_KD/seed17")
        age = np.linspace(prep.age_boundaries[0], prep.age_boundaries[-1], 150)
        t = (age-prep.reference_age)/prep.age_unit
        d = model.effective_d().detach().numpy()
        for z, color in [(-1, "#C98929"), (0, "gray"), (1, "#087F70")]:
            phi = z*np.exp(t[:, None]*d*np.sign(z))
            axes[0, j].plot(age, np.median(phi, axis=1), c=color, label=f"z = {z}")
            axes[0, j].fill_between(age, phi.min(1), phi.max(1), color=color, alpha=.12)
        axes[0, j].axvline(prep.reference_age, color="gray", ls="--", lw=1)
        axes[0, j].set(title=trait.upper(), xlabel="age_sd", ylabel="Transformed fixed score φ(z, age)")
        axes[1, j].hist(data[trait]["train_age"], bins=20, alpha=.5, label="train")
        axes[1, j].hist(data[trait]["val_age"], bins=20, alpha=.5, label="val")
        axes[1, j].set(xlabel="age_sd", ylabel="Patients")
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.suptitle("Sign-aware age response and observed support · bands span PRSs, not uncertainty", fontsize=15)
    save(fig, "sign_aware_age")
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), layout="constrained")
    for ax, trait in zip(axes, TRAITS, strict=True):
        d = frames["ancestry_gate"].query("trait == @trait and arm == 'M2_KD'")
        grid = d.pivot(index="prs", columns="group", values="mean_pi")
        image = ax.imshow(grid, aspect="auto", vmin=0, cmap="Blues")
        counts = d.groupby("group").n.first()
        ax.set(title=trait.upper(), yticks=range(len(grid)), yticklabels=grid.index,
               xticks=range(len(grid.columns)), xticklabels=[f"{g}\nn={counts[g]}" for g in grid.columns])
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.tick_params(axis="y", labelsize=7)
        fig.colorbar(image, ax=ax, shrink=.6, label="Mean π (model weight)")
    fig.suptitle("Observed ancestry groups · M2 KD · mixing weights are not ancestry fractions", fontsize=16)
    save(fig, "ancestry_gate")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for j, trait in enumerate(TRAITS):
        d = data[trait]
        for arm in ["M0_KD", "M1_KD", "M2_KD", "T_MATCHED"]:
            observed, predicted = calibration_curve(d["y"], d["arrays"][arm], n_bins=10, strategy="quantile")
            axes[0, j].plot(predicted, observed, "o-", ms=3, c=COLORS[arm], label=arm)
        axes[0, j].plot([0, 1], [0, 1], "--", c="gray")
        axes[0, j].set(title=trait.upper(), xlim=(0, 1), ylim=(0, 1), xlabel="Mean probability", ylabel="Observed event fraction")
        axes[1, j].hist2d(d["arrays"]["T_MATCHED"], d["arrays"]["M2_KD"], bins=25, range=[[0, 1], [0, 1]], cmap="Blues", cmin=1)
        axes[1, j].plot([0, 1], [0, 1], "--", c="#087F70")
        axes[1, j].set(xlabel="Teacher probability", ylabel="M2 KD probability")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Direct calibration and teacher fidelity · no recalibration on val", fontsize=17)
    save(fig, "calibration_fidelity")


def plot_paired(root, frames):
    pairs = [("M2_KD", "M0_KD"), ("M2_KD", "M1_KD"), ("M1_KD", "M0_KD")]
    pairs += [(f"{v}_KD", f"{v}_SUP") for v in VARIANTS]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "path"}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
        for ax, trait in zip(axes, TRAITS, strict=True):
            d = frames["paired_differences"].query("trait == @trait and metric == 'log_loss'")
            ax.axvline(0, ls="--", c="gray", lw=1)
            for i, (arm, reference) in enumerate(pairs):
                row = d.loc[d.arm.eq(arm) & d.reference.eq(reference)].iloc[0]
                ax.errorbar(row.delta, i, xerr=[[row.delta-row.ci_low], [row.ci_high-row.delta]], fmt="o",
                            c="#087F70" if i >= 3 else "#4489BA", capsize=3)
            ax.set(yticks=range(len(pairs)), yticklabels=[f"{a} − {b}" for a, b in pairs], ylim=(5.6, -.6),
                   title=trait.upper() if trait != "bc" else "Female BC", xlabel="Δ log loss (left favors first model)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis="x", alpha=.15)
        fig.suptitle("Paired comparisons · nominal 95% group-bootstrap intervals", fontsize=17)
        for suffix in ["svg", "png"]:
            fig.savefig(root / "report/figures" / f"paired_comparisons.{suffix}", dpi=160, bbox_inches="tight")
        plt.close(fig)


def training_details(root):
    rows = []
    for trait in TRAITS:
        core = root / "core" / trait
        for arm in NEW_ARMS:
            meta = read_json(core / "models" / arm / "seed17" / "metadata.json")
            hp = meta["hyperparameters"]
            warm, stage = meta["warm_steps"], meta["additional_steps"]
            if arm.startswith("M0_"):
                warm, stage = 0, warm+stage
            rows.append({"trait": trait.upper(), "arm": arm,
                         "full_batch_n": meta["preprocessing"]["fixed_preprocessing"]["fit_n"],
                         "lambda_KD": hp["lambda"], "kappa": hp["kappa"], "gate_ridge": hp["rho"],
                         "M0_warm_steps": warm, "stage_steps": stage, "total_steps": warm+stage,
                         "H_log_loss_mean_3_seeds": hp["H_log_loss"]})
    return pd.DataFrame(rows)


def make_html(root, frames, data, decisions, protocol, lang):
    def t(zh, en):
        return zh if lang == "zh" else en
    def table(frame):
        return frame.to_html(index=False, float_format=lambda x: f"{x:.5f}", border=0, escape=True)
    def figure(name):
        encoded = base64.b64encode((root / "report/figures" / f"{name}.svg").read_bytes()).decode()
        return f'<img alt="{name}" src="data:image/svg+xml;base64,{encoded}">'
    title = t("PRS 蒸馏 v3：Sign-aware 结构试验", "PRS distillation v3: sign-aware structural experiment")
    text = f'''<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
    <style>body{{font:16px/1.7 system-ui,sans-serif;color:#172b44;background:#f3f6fa;margin:0}}main{{max-width:1200px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dbe4ed;border-radius:10px;padding:28px;margin:20px 0;overflow:auto}}h1{{font-size:34px}}h2{{color:#087f70}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px;width:100%}}td,th{{padding:9px;border-bottom:1px solid #dbe4ed;text-align:left}}th{{background:#edf3f8}}.formula{{background:#eef8f5;padding:20px;font-size:19px;overflow:auto}}.note{{border-left:4px solid #c78929;background:#fff8eb;padding:16px}}a{{color:#087f70}}small{{color:#607187}}</style><main><h1>{title}</h1>'''
    text += '<p>'+t("原始 train 内开发，冻结后在原始 val 评估。Teacher、固定 PRS、患者及拆分沿用已审计的全量敏感性实验。",
                    "Developed within original train and evaluated on original val after locking. Teacher, fixed PRS, patients and folds reuse the audited full-cohort sensitivity experiment.")+'</p>'
    text += '<p class="note">'+t(
        "主要发现：三个疾病的 v3 KD 验证集点估计均优于旧版混合蒸馏，但 M2 相对 M0 的 log loss 差异区间均跨过零。H 内按预设容差优先简单结构，三个疾病均选中 M0。KD 对同结构 SUP 的改善在 T2D 较明确，CAD / BC 尚不明确；Teacher 仍领先。",
        "Main finding: v3 KD improves the validation point estimates over historical mixed distillation for all three diseases, but every M2-versus-M0 log-loss interval crosses zero. The predeclared H selection rule favors simpler structures within tolerance and selects M0 for all three diseases. KD has clearer benefit over matched SUP in T2D; CAD / BC remain uncertain. The teacher still leads.")+'</p>'
    text += '<section><h2>'+t("模型与训练", "Model and training")+'</h2><div class="formula">'
    text += 'π(A) = softmax(WA + c), &nbsp; r<sub>k</sub> = exp(log r<sub>k</sub>) &gt; 0, &nbsp; d<sub>k</sub> ≥ 0<br>'
    text += 't(a) = (clip(a, a<sub>min</sub>, a<sub>max</sub>) − a<sub>ref</sub>) / age_unit<br>'
    text += 'φ<sub>k</sub>(Z<sub>k</sub>,a) = Z<sub>k</sub> exp[d<sub>k</sub> t(a) sign(Z<sub>k</sub>)]<br>'
    text += 'η = α + γ<sup>T</sup>A + δ<sup>T</sup>C + f<sub>↑</sub>(a) + Σ<sub>k</sub> π<sub>k</sub>(A) r<sub>k</sub> φ<sub>k</sub>(Z<sub>k</sub>,a); &nbsp; p = sigmoid(η)</div>'
    text += '<div class="formula">B<sub>j</sub>(t) = clip(t−b<sub>j</sub>, 0, w<sub>j</sub>) − clip(−b<sub>j</sub>, 0, w<sub>j</sub>)<br>'
    text += 'f<sub>↑</sub>(a) = Σ<sub>j</sub> m<sub>j</sub>B<sub>j</sub>(t(a)), &nbsp; m<sub>j</sub> ≥ 0<br>'
    text += '∂φ<sub>k</sub>/∂a = (d<sub>k</sub>/age_unit)|Z<sub>k</sub>| exp[d<sub>k</sub>t sign(Z<sub>k</sub>)] ≥ 0</div>'
    text += '<p>'+t(
        "年龄区间由当前训练范围的 0、1/3、2/3、1 分位点确定，b 为中心化边界，w 为区间宽度；参考年龄取训练中位数。上面的导数适用于支持范围内部，范围外两条年龄分支均为常数延伸。正 PRS 在高龄时贡献增大；负 PRS 在高龄时负贡献减弱，因此两种符号的年龄方向一致。",
        "Age boundaries use the current fit scope's 0, 1/3, 2/3 and 1 quantiles; b denotes centered boundaries and w interval widths. The reference age is the training median. The derivative holds within age support; both branches extrapolate constantly outside it. At older ages positive PRS contributions increase and negative contributions become less negative, giving the same age direction for both signs.")+'</p>'
    text += '<p>'+t("M0：d=0；M1：共享一个非负 d；M2：每个 PRS 一个非负 d。三者均有自由正值 r 和最多三段的单调年龄基础项。M1/M2 从同作用域、λ、κ、seed 的 M0 精确初始化，再共同优化全部参数。",
        "M0 fixes d=0; M1 shares one nonnegative d; M2 learns a nonnegative d for each PRS. All have free positive r and a monotone baseline with at most three age segments. M1/M2 start exactly from the M0 fitted in the same scope with the same λ, κ and seed, then jointly optimize all components.")+'</p>'
    text += '<div class="formula">L = mean BCEWithLogits(η, (1−λ)y + λq<sub>OOF</sub>) + Ω<sub>baseline,gate</sub> + κ mean[(log r)²] + κ mean[d²]</div>'
    text += '<p>'+t("复用完整核验的 U 内 OOF 与 D 内 OOF Teacher 软标签；SUP 使用 λ=0，KD 固定沿用 T2D=0.75、CAD=0.5、BC=0.75。同一疾病六臂使用同一门控 ridge；κ∈{10⁻⁴,10⁻³,10⁻²} 只在 H 上按真实标签 log loss 选择。Adam/full batch/lr=0.01；每阶段最多 2,000 步；三个 seeds=17/29/43，主结果固定 17。d 和年龄斜率每步投影为非负。",
        "Reuse verified U-OOF and D-OOF teacher targets. SUP uses λ=0; KD keeps T2D=0.75, CAD=0.5 and BC=0.75. All six arms within a disease share the gate ridge; κ∈{10⁻⁴,10⁻³,10⁻²} is selected only on H true-label log loss. Adam/full batch/lr=0.01, at most 2,000 steps per stage, seeds 17/29/43 with primary seed 17. d and age slopes are projected nonnegative after each update.")+'</p>'
    text += '<p class="note">'+t("年龄仍是 age_sd，age_unit=1；不是岁或十年。保证的是固定 Z/A/C 时直接 Student 的年龄单调性。该 sign-aware 形式是待检验的结构先验，不是已证实的遗传规律。M1/M2 的计算量包含 M0 预训练阶段，不能宣称与 M0 总步数相同。",
        "Age remains age_sd with age_unit=1, not years or decades. The guarantee concerns the direct student's age response at fixed Z/A/C. The sign-aware form is a candidate structural prior, not an established genetic law. M1/M2 include the M0 warm-fit stage and do not have the same total step count as M0.")+'</p>'
    text += '<details><summary>'+t("实际全量 refit 的 batch、步数与锁定参数", "Actual full-cohort refit batches, steps and locked parameters")+'</summary>'
    text += '<p>'+t(
        "D 为全量 train，U/H 是 train 内的拟合/调参划分；val 不参与选择。下表为主 seed=17；三个 seed 复用同一锁定步数。stage_steps 是本阶段步数，M1/M2 另外包含 M0_warm_steps。κ 在 H 上按三 seed 平均 log loss 选择，最终步数取三 seed 最佳步数的中位数。",
        "D denotes full train; U/H are the fitting/tuning partitions within train. Val does not enter selection. The table shows primary seed 17; all three seeds reuse the same locked step budget. stage_steps is the current stage; M1/M2 additionally include M0_warm_steps. κ is selected by mean H log loss across three seeds; final steps use the median of their best steps.")+'</p>'
    text += table(frames["training_details"])+ '</details></section>'
    text += '<section><h2>'+t("主要结果与结构决策", "Main results and structural decisions")+'</h2>'
    for trait in TRAITS:
        decision = decisions[trait]
        a, b = decision["M2_KD_minus_M0_KD"], decision["M2_KD_minus_M1_KD"]
        supported = a["ci_high"] < 0 and b["ci_high"] < 0
        text += f'<h3>{trait.upper()}</h3><p>'+t(
            f'H 内选中：{decision["H_selected_structure"]}。M2 KD 相对 M0 和 M1 的 log loss 均有明确改善。' if supported else
            f'H 内选中：{decision["H_selected_structure"]}。目前不能确认 M2 KD 同时优于 M0 与 M1。',
            f'H-selected structures: {decision["H_selected_structure"]}. M2 KD improves log loss over both M0 and M1.' if supported else
            f'H-selected structures: {decision["H_selected_structure"]}. M2 KD is not established as better than both M0 and M1.')+'</p>'
        rows = []
        for reference in ["M0_KD", "M1_KD", "V1_KD_MIX"]:
            part = frames["paired_differences"].query("trait == @trait and arm == 'M2_KD' and reference == @reference and metric == 'log_loss'").iloc[0]
            rows.append({"comparison": f"M2_KD − {reference}", "Δ log loss": part.delta,
                         "95% CI": f'[{part.ci_low:+.5f}, {part.ci_high:+.5f}]'})
        for v, row in decision["KD_minus_SUP"].items():
            rows.append({"comparison": f"{v}_KD − {v}_SUP", "Δ log loss": row["delta"],
                         "95% CI": f'[{row["ci_low"]:+.5f}, {row["ci_high"]:+.5f}]'})
        text += table(pd.DataFrame(rows))
    text += figure("performance")+figure("paired_comparisons")
    text += '<p>'+t("主比较指标是 log loss。区间为 1,000 次可用亲缘组配对 bootstrap 的名义 95% CI，条件于已拟合模型与预测，未校正多重比较。M0 允许在 train 内胜出；没有按 val 改选部署模型。",
        "The primary comparison metric is log loss. Intervals are nominal 95% CIs from 1,000 paired available-family-group bootstrap draws, conditional on fitted models/predictions and without multiplicity correction. M0 is allowed to win within train; no deployment model is reselected on val.")+'</p></section>'
    text += '<section><h2>'+t("完整指标与两种 R²", "All metrics and the two R² definitions")+'</h2>'
    text += '<p>'+t("Nagelkerke R²：固定概率的 val 似然相对 val 事件率常数参考，越高越好；它不是遗传解释度，也不是 Phase B 正式 wrapper 分数。Teacher-logit R²：1−Σ(logit p−logit q)²/Σ(logit q−mean logit q)²，衡量模仿教师的程度。AP 是 average precision。",
        "Nagelkerke R² compares frozen-probability val likelihood against a constant val-prevalence reference; higher is better. It is not genetic variance explained or a Phase B wrapper score. Teacher-logit R² = 1−Σ(logit p−logit q)²/Σ(logit q−mean logit q)² measures teacher imitation. AP denotes average precision.")+'</p>'
    for trait in TRAITS:
        text += f'<h3>{trait.upper()}</h3>'+table(frames["metrics"].query("trait == @trait")[["arm", "n", "events", "auroc", "average_precision", "log_loss", "brier", "nagelkerke_r2", "probability_mae", "logit_r2"]])
    text += figure("calibration_fidelity")+'</section>'
    text += '<section><h2>'+t("解释、支持范围与初始化稳定性", "Interpretation, support and initialization stability")+'</h2>'
    text += '<div class="formula">η<sub>ref</sub> = α + γ<sup>T</sup>A + δ<sup>T</sup>C + Σ π<sub>k</sub>r<sub>k</sub>Z<sub>k</sub><br>Δ<sub>age</sub> = f<sub>↑</sub>(a) + Σ(C<sub>k</sub>−C<sub>ref,k</sub>); &nbsp; η = η<sub>ref</sub> + Δ<sub>age</sub></div>'
    text += '<p>'+t("高于参考年龄时 Δage≥0，低于参考年龄时 Δage≤0。贡献位于 logit 尺度，不是相加的概率。β 随个体 PRS 的符号而变，不能当作跨越零点的全局每 SD OR。相关 PRS 的 π/r/d 仍可能不稳定，不能解释成唯一遗传机制。",
        "Δage≥0 above the reference age and ≤0 below it. Contributions add on the logit scale, not the probability scale. β depends on the individual's PRS sign and is not a global per-SD OR across zero. Correlated PRS can still produce unstable π/r/d, so they do not identify a unique genetic mechanism.")+'</p>'
    for name in ["r_d_parameters", "sign_aware_age", "ancestry_gate"]:
        text += figure(name)
    text += '<details><summary>'+t("三个初始化的稳定性（不是置信区间）", "Three-initialization stability (not confidence intervals)")+'</summary>'+table(frames["initialization_stability"])+ '</details>'
    text += '<details><summary>'+t("年龄 × 祖源联合支持计数", "Age × ancestry joint support counts")+'</summary>'+table(frames["age_ancestry_joint_support"])+ '</details></section>'
    text += '<section><h2>'+t("范围与审计", "Scope and audit")+'</h2><ul>'
    notes = [t("三病分别使用原始全量 train/val；BC 保持女性资格。Teacher 与 v1 预测全部保留，未重新训练 Teacher。",
               "Original full train/val cohorts are retained by disease; BC remains female-eligible. Teacher and v1 predictions are preserved, with no teacher retraining."),
             t("全量队列仍保留 115 个跨 train/val 亲缘组，以及 train 112 / val 42 人亲缘未确定；区间未覆盖未知亲缘依赖。",
               "The cohort retains 115 family groups crossing train/val and unresolved kinship for 112 train / 42 val patients; intervals do not capture unknown family dependence."),
             t("这是反复使用过的开发 val 上的 post-hoc 证据；未读取 blind holdout，未运行 Phase B。单调性保证仅针对直接 Student。",
               "This is post-hoc evidence on a previously used development val cohort. Blind holdout was not opened and Phase B was not run. Monotonicity applies only to the direct student."),
             t("META_V1 作为既有 metadata-only 参照保留，其年龄结构不同；差异不能解释成纯粹的遗传增益。患者级完整分解仅保存在本地 core 目录。",
               "META_V1 retains the existing metadata-only reference with a different age structure; differences do not isolate a pure genetic increment. Full patient decompositions remain local under core."),
             t("结构检查、独立 NumPy 重建、保存加载、参考年龄恒等式和固定协变量年龄扫描均已逐模型记录。",
               "Structural checks, independent NumPy reconstruction, save/load parity, reference-age identities and fixed-covariate age sweeps are recorded for every model.")]
    text += ''.join(f'<li>{s}</li>' for s in notes)+'</ul><small>'+html.escape(str(root))+'</small></section></main></html>'
    return text


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true", help="Refresh HTML from sealed results without refitting or re-evaluating")
    args = parser.parse_args()
    root = args.run_dir
    out = root / "report"
    if args.render_only:
        verify_files(root, read_json(root / "COMPLETE.json")["output_sha256"])
        protocol, cfg, _ = verify_run(root)
        lr_root = Path(cfg["matched_logistic_run"])
        verify_files(lr_root, read_json(lr_root / "VERIFIED.json")["output_sha256"])
        frames = {p.stem: pd.read_csv(p) for p in out.glob("*.csv")}
        data, decisions = {}, read_json(out / "DECISION.json")
        plot_paired(root, frames)
    else:
        frames, data, decisions, protocol = evaluate(root)
        plot_all(root, frames, data)
    frames["training_details"] = training_details(root)
    frames["training_details"].to_csv(out / "training_details.csv", index=False)
    for lang in ["zh", "en"]:
        (out / f"signaware_v3_{lang}.html").write_text(make_html(root, frames, data, decisions, protocol, lang))
    source = Path(protocol["source_run"])
    verify_files(source, read_json(source / "COMPLETE.json")["output_sha256"])
    write_json(out / "VERIFIED.json", {"status": "evaluation_and_aggregate_report_complete",
        "model_count": 54, "patient_sets_matched": True, "source_models_unchanged": True,
        "matched_logistic_source_verified": True,
        "assessment_used_for_selection": False, "phase_b": False,
        "report_script_sha256": sha256_file(Path(__file__)),
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*")
                          if p.is_file() and p.name != "VERIFIED.json" and p.suffix != ".zip"}})
    with zipfile.ZipFile(out / "signaware_v3_report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.suffix != ".zip":
                archive.write(p, p.relative_to(out))
    write_json(root / "COMPLETE.json", {"status": "v3_signaware_complete_verified", "models": 54,
        "phase_b": False, "new_teacher_fits": False,
        "output_sha256": {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*")
                          if p.is_file() and p.name != "COMPLETE.json"}})
    print(f"Completed v3 evaluation: {root}", flush=True)


if __name__ == "__main__":
    main()
