"""Sealed biological-student / foundation-model evaluation and offline HTML."""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from run_prs_distillation_bio import verify
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score

from decipehr.external_pgs import sha256_file, write_json
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.reporting import basic_metrics, calibration, fidelity
from decipehr.prs_distillation_bio import load, predict
from decipehr.prs_monotone_boosting import raw_predict as boosting_logit

TRAITS = ["T2D", "cad", "bc"]
FM_ROOT = Path("outputs/foundation_model_probe_20261026/benchmark")
METRICS = ["log_loss", "auroc", "brier", "nagelkerke_r2"]
PRED_COLUMNS = ["sampleid", "kinship_group", "age_sd", "hard_label", "arm", "seed", "probability", "p_teacher"]


def read(path):
    return json.loads(Path(path).read_text())


def read_locked(core):
    receipt = read(core / "PREDICTIONS_COMPLETE.json")
    if receipt["smoke"] or receipt["evaluation_labels_read"] or receipt["models"] != 72:
        raise ValueError("Expected all 72 locked, unevaluated final models")
    verify_files(core, receipt["output_sha256"])
    return pd.read_parquet(core / "direct_predictions.parquet", columns=PRED_COLUMNS, filters=[("seed", "=", 17)])


def pairs(arrays, scope):
    references = ["M0_KD", "M0_SUP", "G_A6_KD", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"]
    contrasts = [("H_SELECTED", x) for x in references]
    contrasts += [("G_A6_KD", "M0_KD"), ("G_KD", "M0_KD"), ("A6_KD", "M0_KD")]
    contrasts += [(f"{family}_KD", f"{family}_SUP") for family in ["M0", "A6", "G", "G_A6"]]
    contrasts += [("BIO_POLICY", a) for a in ["H_SELECTED", "M0_KD", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"]]
    contrasts += [(f"HGB_{family}_KD", f"HGB_{family}_SUP") for family in ["GAM", "PAIR"]]
    contrasts += [("HGB_PAIR_KD", "HGB_GAM_KD"), ("HGB_GAM_KD", "M0_KD")]
    contrasts += [("TabPFN_v3.5", "TabPFN_v3"), ("Kumo_Tabular", "TabPFN_v3"),
                  ("Kumo_Tabular", "TabPFN_v3.5")]
    for family in ["M0", "G_A6"]:
        for interaction in ["CONTINUE", "D1", "D2", "D2_ONLY"]:
            contrasts.append((f"{family}_KD_{interaction}", f"{family}_KD"))
    if scope == "val":
        contrasts += [("H_SELECTED", "V3_M0_KD"), ("H_SELECTED", "V1_KD_MIX"), ("H_SELECTED", "LR")]
    return [(a, b) for a, b in contrasts if a in arrays and b in arrays]


def paired_bootstrap(y, arrays, groups, scope, replicates=1000):
    comparisons = pairs(arrays, scope)
    arms = sorted({a for pair in comparisons for a in pair})
    p = np.column_stack([arrays[a] for a in arms])
    loss = -(y[:, None]*np.log(np.clip(p, 1e-6, 1-1e-6))+(1-y[:, None])*np.log1p(-np.clip(p, 1e-6, 1-1e-6)))
    r2_loss = -(y[:, None]*np.log(np.clip(p, 1e-15, 1-1e-15))+(1-y[:, None])*np.log1p(-np.clip(p, 1e-15, 1-1e-15)))
    _, inverse = np.unique(groups, return_inverse=True)
    g = inverse.max()+1
    n_group, y_group = np.bincount(inverse), np.bincount(inverse, weights=y)
    sums = []
    for matrix in [loss, (p-y[:, None])**2, r2_loss]:
        result = np.zeros((g, len(arms)))
        np.add.at(result, inverse, matrix)
        sums.append(result)
    rng, samples = np.random.default_rng(20261028), []
    for _ in range(replicates):
        count = np.bincount(rng.integers(g, size=g), minlength=g)
        n, positives = count@n_group, count@y_group
        if not 0 < positives < n:
            continue
        ll, brier, ll_r2 = [count@s/n for s in sums]
        rate = positives/n
        null = rate*np.log(rate)+(1-rate)*np.log1p(-rate)
        nr2 = -np.expm1(2*(null+ll_r2)) / -np.expm1(2*null)
        auc = np.array([roc_auc_score(y, p[:, j], sample_weight=count[inverse]) for j in range(len(arms))])
        samples.append(np.column_stack([ll, auc, brier, nr2]))
    samples = np.stack(samples)
    point = {a: {**basic_metrics(y, arrays[a]), "nagelkerke_r2": nagelkerke_r2(y, arrays[a])} for a in arms}
    rows = []
    for arm, reference in comparisons:
        delta = samples[:, arms.index(arm)]-samples[:, arms.index(reference)]
        for j, metric in enumerate(METRICS):
            lo, hi = np.quantile(delta[:, j], [.025, .975])
            rows.append({"arm": arm, "reference": reference, "metric": metric,
                         "delta": point[arm][metric]-point[reference][metric], "ci_low": lo, "ci_high": hi,
                         "replicates": len(samples), "unit": "available_kinship_group", "refitted": False,
                         "multiplicity_adjusted": False})
    return pd.DataFrame(rows)


def age_groups(root, scope, trait, base):
    """Assign assessment ages using training-only quartiles, separately per fold."""
    result = pd.Series(index=base.index, dtype="object")
    for fold in ([-1] if scope == "val" else range(5)):
        scope_root = root if fold == -1 else root / "nested" / f"outer{fold}"
        meta = read(scope_root / "core" / trait / "models/M0_SUP/seed17/metadata.json")
        edges = meta["preprocessing"]["fixed_preprocessing"]["age_quartiles"]
        selected = base if fold == -1 else base.loc[base.outer_fold.eq(fold)]
        result.loc[selected.index] = [f"Q{i+1}" for i in np.searchsorted(edges, selected.age_sd)]
    assert result.notna().all()
    return result


def seed_metrics(root, scope, trait, ids, y, locks):
    parts = []
    for fold in ([-1] if scope == "val" else range(5)):
        scope_root = root if fold == -1 else root / "nested" / f"outer{fold}"
        lock = locks if fold == -1 else locks[str(fold)]
        frame = pd.read_parquet(scope_root / "core" / trait / "direct_predictions.parquet",
                                columns=["sampleid", "arm", "seed", "probability"])
        parts.extend([frame, frame.loc[frame.arm.eq(lock["selected_arm"])].assign(arm="H_SELECTED")])
    frame = pd.concat(parts, ignore_index=True)
    rows = []
    for arm, sub in frame.groupby("arm"):
        reference = sub.loc[sub.seed.eq(17)].set_index("sampleid").loc[ids, "probability"].to_numpy()
        for seed, current in sub.groupby("seed"):
            indexed = current.set_index("sampleid")
            assert indexed.index.is_unique and set(indexed.index) == set(ids)
            p = indexed.loc[ids, "probability"].to_numpy()
            rows.append({"scope": scope, "trait": trait, "arm": arm, "seed": seed,
                         **basic_metrics(y, p), "nagelkerke_r2": nagelkerke_r2(y, p),
                         "probability_mae_to_seed17": float(np.abs(p-reference).mean()),
                         "probability_max_difference_to_seed17": float(np.abs(p-reference).max())})
    return rows


def evaluate(root, scope):
    protocol, cfg = verify(root)
    source = Path(cfg["source_run"])
    for backend in ["tabpfn35", "kumo"]:
        verify_files(FM_ROOT / backend, read(FM_ROOT / backend / "COMPLETE.json")["output_sha256"])
    loaded, locks, low_locks, low_training = {}, {}, {}, []
    def low_order(scope_root, trait):
        directory = scope_root / "low_order" / trait
        record = read(directory / "COMPLETE.json")
        if record["models"] != 4 or record["evaluation_labels_read"]:
            raise ValueError("Low-order models are not fully locked")
        verify_files(directory, record["output_sha256"])
        lock = read(directory / "SELECTION_LOCK.json")
        for arm, choice in lock["choices"].items():
            low_training.append({"trait": trait, "scope_root": str(scope_root), "arm": arm,
                "lambda": choice["lambda"], "leaves": choice["leaves"], "l2": choice["l2"],
                "iterations": choice["iterations"], "H_loss": choice["H_log_loss"],
                "training_patients": lock["patient_n_D"], "binary_rows": lock["binary_rows_D"]})
        return pd.read_parquet(directory / "predictions.parquet", columns=PRED_COLUMNS), lock
    for trait in TRAITS:
        if scope == "val":
            loaded[trait] = read_locked(root / "core" / trait)
            locks[trait] = read(root / "core" / trait / "SELECTION_LOCK.json")
            extra, low_locks[trait] = low_order(root, trait)
            loaded[trait] = pd.concat([loaded[trait], extra], ignore_index=True)
        else:
            pieces, choices, extra_choices = [], {}, {}
            for fold in range(5):
                core = root / "nested" / f"outer{fold}" / "core" / trait
                pieces.append(read_locked(core).assign(outer_fold=fold))
                choices[str(fold)] = read(core / "SELECTION_LOCK.json")
                extra, extra_choices[str(fold)] = low_order(root / "nested" / f"outer{fold}", trait)
                pieces.append(extra.assign(outer_fold=fold))
            loaded[trait], locks[trait] = pd.concat(pieces, ignore_index=True), choices
            low_locks[trait] = extra_choices
    policy = {}
    for trait in TRAITS:
        choices = {}
        for fold in ([-1] if scope == "val" else range(5)):
            par = locks[trait] if fold == -1 else locks[trait][str(fold)]
            low = low_locks[trait] if fold == -1 else low_locks[trait][str(fold)]
            a, b = par["selected_arm"], low["selected_arm"]
            chosen = b if low["choices"][b]["H_log_loss"] < par["choices"][a]["H_loss"]-1e-4 else a
            choices[str(fold)] = chosen
        policy[trait] = choices
    out = root / "report" / scope
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "POLICY_LOCK.json"
    if lock_path.exists() and read(lock_path)["choices"] != policy:
        raise ValueError("Frozen biological selection policy changed")
    write_json(lock_path, {"choices": policy, "rule": "H loss; prefer parametric within 0.0001",
                          "assessment_labels_used_for_selection": False})
    # No assessment labels are opened until every model in this scope is sealed.
    if scope == "val":
        audit = read(source / "audit_report.json")
        lp = Path(audit["paths"]["evaluation"])
        assert sha256_file(lp) == audit["source_sha256"][str(lp)]
        labels = pd.read_csv(lp, dtype={"sampleid": str}, usecols=["sampleid", *TRAITS]).set_index("sampleid")
    all_metrics, all_effects, subgroup, selection, plot_data, training, stability = [], [], [], [], {}, [], []
    for trait in TRAITS:
        frame = loaded[trait]
        base = frame.loc[frame.arm.eq("M0_SUP")].set_index("sampleid")
        assert base.index.is_unique
        ids = base.index.tolist()
        if scope == "val":
            y = labels.loc[ids, trait].to_numpy(dtype=int)
        else:
            training_labels = pd.read_parquet(source / trait / "training.parquet").set_index("sampleid")
            assert set(ids) == set(training_labels.index)
            y = training_labels.loc[ids, "y"].to_numpy(dtype=int)
        arrays = {}
        for arm, part in frame.groupby("arm"):
            p = part.set_index("sampleid")
            assert p.index.is_unique and set(p.index) == set(ids)
            arrays[arm] = p.loc[ids, "probability"].to_numpy()
        if scope == "val":
            selected_arm = locks[trait]["selected_arm"]
            arrays["H_SELECTED"] = arrays[selected_arm]
            selection.append({"trait": trait, "scope": scope, "arm": selected_arm, "outer_fold": -1,
                              "policy": "parametric", "H_loss": locks[trait]["choices"][selected_arm]["H_loss"]})
            low_arm = low_locks[trait]["selected_arm"]
            arrays["HGB_SELECTED"] = arrays[low_arm]
            low_loss = low_locks[trait]["choices"][low_arm]["H_log_loss"]
            total_arm = low_arm if low_loss < locks[trait]["choices"][selected_arm]["H_loss"]-1e-4 else selected_arm
            assert total_arm == policy[trait]["-1"]
            arrays["BIO_POLICY"] = arrays[total_arm]
            selection.append({"trait": trait, "scope": scope, "arm": total_arm, "outer_fold": -1,
                              "policy": "all_biological", "H_loss": low_loss if total_arm == low_arm else locks[trait]["choices"][selected_arm]["H_loss"]})
            v3 = pd.read_parquet(Path(cfg["v3_run"]) / "core" / trait / "direct_predictions.parquet",
                                 columns=["sampleid", "arm", "seed", "probability"], filters=[("seed", "=", 17)])
            for arm in ["M0_KD", "M2_KD"]:
                arrays[f"V3_{arm}"] = v3.loc[v3.arm.eq(arm)].set_index("sampleid").loc[ids, "probability"].to_numpy()
            v1 = pd.read_parquet(source / "core" / trait / "direct_predictions.parquet",
                                 columns=["sampleid", "arm", "seed", "probability"], filters=[("seed", "=", 17)])
            for arm, name in [("S_KD_MIX", "V1_KD_MIX"), ("META", "META_V1")]:
                arrays[name] = v1.loc[v1.arm.eq(arm)].set_index("sampleid").loc[ids, "probability"].to_numpy()
            lr_root = Path("outputs/prs_distillation_v1/joint_logistic_comparison_fulltrain_20261026")
            verify_files(lr_root, read(lr_root / "VERIFIED.json")["output_sha256"])
            arrays["LR"] = pd.read_parquet(lr_root / trait / "validation_predictions.parquet").set_index("sampleid").loc[ids, "probability"].to_numpy()
        else:
            selected_parts, total_parts, low_parts = [], [], []
            for fold, lock in locks[trait].items():
                selected_parts.append(frame.loc[frame.outer_fold.eq(int(fold)) & frame.arm.eq(lock["selected_arm"])])
                selection.append({"trait": trait, "scope": scope, "outer_fold": int(fold), "arm": lock["selected_arm"],
                                  "policy": "parametric", "H_loss": lock["choices"][lock["selected_arm"]]["H_loss"]})
                low = low_locks[trait][fold]
                low_arm = low["selected_arm"]
                low_loss = low["choices"][low_arm]["H_log_loss"]
                bio_loss = lock["choices"][lock["selected_arm"]]["H_loss"]
                total_arm = low_arm if low_loss < bio_loss-1e-4 else lock["selected_arm"]
                assert total_arm == policy[trait][fold]
                total_parts.append(frame.loc[frame.outer_fold.eq(int(fold)) & frame.arm.eq(total_arm)])
                low_parts.append(frame.loc[frame.outer_fold.eq(int(fold)) & frame.arm.eq(low_arm)])
                selection.append({"trait": trait, "scope": scope, "outer_fold": int(fold), "arm": total_arm,
                                  "policy": "all_biological", "H_loss": low_loss if total_arm == low_arm else bio_loss})
            chosen = pd.concat(selected_parts).set_index("sampleid")
            assert chosen.index.is_unique and set(chosen.index) == set(ids)
            arrays["H_SELECTED"] = chosen.loc[ids, "probability"].to_numpy()
            for name, pieces in [("BIO_POLICY", total_parts), ("HGB_SELECTED", low_parts)]:
                chosen = pd.concat(pieces).set_index("sampleid")
                assert chosen.index.is_unique and set(chosen.index) == set(ids)
                arrays[name] = chosen.loc[ids, "probability"].to_numpy()
        arrays["TabPFN_v3"] = base.loc[ids, "p_teacher"].to_numpy()
        for backend, name in [("tabpfn35", "TabPFN_v3.5"), ("kumo", "Kumo_Tabular")]:
            paths = [FM_ROOT / backend / trait / "D_to_E/predictions.parquet"] if scope == "val" else [
                FM_ROOT / backend / trait / f"D_oof_{f}/predictions.parquet" for f in range(5)]
            pred = pd.concat([pd.read_parquet(p) for p in paths]).set_index("sampleid")
            assert pred.index.is_unique and set(pred.index) == set(ids)
            arrays[name] = pred.loc[ids, "probability"].to_numpy()
        for arm, p in arrays.items():
            all_metrics.append({"scope": scope, "trait": trait, "arm": arm, **basic_metrics(y, p),
                "average_precision": average_precision_score(y, p), "nagelkerke_r2": nagelkerke_r2(y, p),
                **calibration(y, p), **fidelity(p, arrays["TabPFN_v3"])})
        stability.extend(seed_metrics(root, scope, trait, ids, y, locks[trait]))
        effects = paired_bootstrap(y, arrays, base.loc[ids, "kinship_group"].to_numpy(), scope,
                                   cfg["bootstrap_replicates"])
        all_effects.append(effects.assign(trait=trait, scope=scope))
        bins = age_groups(root, scope, trait, base)
        for arm in ["BIO_POLICY", "H_SELECTED", "M0_KD", "G_A6_KD", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"]:
            values = base.loc[ids].assign(y=y, p=arrays[arm], age_quartile=bins)
            values["ancestry_age"] = values.hard_label.astype(str)+" / "+values.age_quartile
            for dimension in ["hard_label", "age_quartile", "ancestry_age"]:
                for group, sub in values.groupby(dimension):
                    m = basic_metrics(sub.y, sub.p)
                    subgroup.append({"scope": scope, "trait": trait, "arm": arm, "dimension": dimension,
                        "group": group, **m, "unstable": len(sub) < 100 or min(m["events"], m["nonevents"]) < 20})
        if scope == "val":
            for arm, choice in locks[trait]["choices"].items():
                meta = read(root / "core" / trait / "models" / arm / "seed17/metadata.json")
                model, _, _ = load(root / "core" / trait / "models" / arm / "seed17")
                parent_steps = locks[trait]["choices"][choice["parent"]]["steps"] if choice["parent"] else 0
                training.append({"trait": trait, "arm": arm, "lambda": choice["settings"]["lam"],
                    "kappa_r": choice["settings"]["kappa_r"], "kappa_d": choice["settings"]["kappa_d"],
                    "stage_steps": choice["steps"], "parent_steps": parent_steps,
                    "full_batch_n": meta["preprocessing"]["fixed_preprocessing"]["fit_n"],
                    "raw_parameter_count": sum(p.numel() for p in model.parameters()),
                    "age_reference": meta["preprocessing"]["reference_age"], "H_loss": choice["H_loss"]})
        plot_data[trait] = {"y": y, "arrays": arrays, "base": base, "locks": locks[trait], "low_locks": low_locks[trait]}
        print(f"Evaluated {scope}/{trait}: {len(arrays)} arms on {len(y)} patients", flush=True)
    out = root / "report" / scope
    out.mkdir(parents=True, exist_ok=True)
    frames = {"metrics": pd.DataFrame(all_metrics), "paired_differences": pd.concat(all_effects, ignore_index=True),
              "subgroups": pd.DataFrame(subgroup), "selection": pd.DataFrame(selection),
              "seed_stability": pd.DataFrame(stability),
              "training_details": pd.DataFrame(training), "low_order_training_details": pd.DataFrame(low_training)}
    for name, frame in frames.items():
        if len(frame):
            frame.to_csv(out / f"{name}.csv", index=False)
    write_json(out / "EVALUATION.json", {"scope": scope, "assessment_used_for_selection": False,
        "nested_gate_ridge": .001 if scope == "nested" else "historical mixed-KD ridge",
        "bootstrap": "available-family paired; fixed predictions; nominal 95%; no multiplicity correction",
        "unresolved_kinship": protocol["unresolved_kinship_retained"], "report_script_sha256": sha256_file(__file__)})
    return frames, plot_data, protocol


def plots(root, scope, frames, data):
    dest = root / "report" / scope / "figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False})
    def savefig(fig, name):
        for ext in ["svg", "png"]:
            fig.savefig(dest / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)
    arms = ["M0_KD", "G_A6_KD", "HGB_GAM_KD", "HGB_PAIR_KD", "BIO_POLICY", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")
    for j, trait in enumerate(TRAITS):
        m = frames["metrics"].query("trait == @trait").set_index("arm")
        for i, metric in enumerate(["log_loss", "auroc", "nagelkerke_r2"]):
            ax = axes[i, j]
            for y, arm in enumerate(arms):
                color = "#0b8576" if arm == "BIO_POLICY" else ("#3c527a" if "Tab" in arm else "#9b6d24")
                value = m.loc[arm, metric]
                ax.scatter(value, y, color=color, s=40)
                ax.annotate(f"{value:.4f}", (value, y), xytext=(5, 0), textcoords="offset points", fontsize=8, va="center")
            ax.set(yticks=range(len(arms)), yticklabels=arms if j == 0 else [""]*len(arms), ylim=(len(arms)-.4, -.6),
                   xlabel={"log_loss": "Log loss (lower)", "auroc": "AUROC (higher)", "nagelkerke_r2": "Nagelkerke R² (higher)"}[metric])
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.margins(x=.65)
            ax.grid(axis="x", alpha=.2)
            if i == 0:
                ax.set_title(trait.upper() if trait != "bc" else "Female BC")
    fig.suptitle(f"{'Original validation' if scope == 'val' else 'Nested train-only pooled OOF'} · primary seed 17", fontsize=18)
    savefig(fig, "performance")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, trait in zip(axes, TRAITS, strict=True):
        for arm in ["BIO_POLICY", "M0_KD", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"]:
            observed, probability = calibration_curve(data[trait]["y"], data[trait]["arrays"][arm], n_bins=10, strategy="quantile")
            ax.plot(probability, observed, "o-", ms=3, label=arm)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(title=trait.upper(), xlabel="Mean probability", ylabel="Observed event fraction", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen direct predictions · calibration curves are diagnostic only")
    savefig(fig, "calibration")
    if scope == "val":
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for j, trait in enumerate(TRAITS):
            arm = data[trait]["locks"]["selected_arm"]
            model, prep, _ = load(root / "core" / trait / "models" / arm / "seed17")
            z = np.linspace(-3, 3, 151)
            slopes = model.shape_slopes().detach().numpy()
            for k, name in enumerate(prep.age.fixed.kept_prs):
                hinges = np.maximum(z[:, None]-prep.prs_knots[k], 0)-np.maximum(-prep.prs_knots[k], 0)
                g = slopes[k, 0]*z + (hinges*np.diff(slopes[k])).sum(1)
                axes[0, j].plot(z, g, alpha=.6, label=name)
            axes[0, j].plot(z, z, "--", color="black", lw=1)
            axes[0, j].set(title=f"{trait.upper()} · {arm}", xlabel="Fixed standardized PRS", ylabel="Monotone shaped PRS g(z)")
            train = pd.read_parquet(Path(read(root / "PROTOCOL.json")["identity"]["config"]["source_run"]) / trait / "training.parquet")
            age = np.linspace(*prep.age.age_boundaries[[0, -1]], 101)
            fixed = prep.age.fixed
            for prs_level in [-1, 0, 1]:
                profile = pd.DataFrame({name: np.full(len(age), train[name].mean())
                    for name in fixed.source_columns if name != "sex"})
                profile["sex"] = train.sex.mode().iloc[0]
                for k, name in enumerate(fixed.kept_prs):
                    profile[name] = fixed.scaler.mean_[k]+prs_level*fixed.scaler.scale_[k]
                profile = profile[fixed.source_columns]
                profile.age_sd = age
                curve = predict(model, prep.transform(profile))["probability"]
                axes[1, j].plot(age, curve, alpha=.8, label=f"All PRS z={prs_level:+d}")
            axes[1, j].axvline(prep.age.reference_age, ls="--", color="gray")
            axes[1, j].set(xlabel="age_sd (not years)", ylabel="Direct probability at fixed other inputs")
            axes[1, j].legend(fontsize=8)
        fig.suptitle("Exact response curves · model priors, not identified causal effects", fontsize=16)
        savefig(fig, "interpretation")
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        for j, trait in enumerate(TRAITS):
            arm = data[trait]["low_locks"]["selected_arm"]
            bundle = joblib.load(root / "low_order" / trait / "models" / arm / "model.joblib")
            features = bundle["features"]
            reference = features.reference
            k_age = features.k
            age = np.linspace(*features.prep.age.age_boundaries[[0, -1]], 81)
            t_age = age-features.prep.age.reference_age
            z = np.linspace(-2, 2, 65)
            zz, tt = np.meshgrid(z, t_age)
            def interaction(k_prs, reference=reference, zz=zz, tt=tt, k_age=k_age, bundle=bundle):
                x = np.tile(reference, (zz.size, 1))
                x[:, k_prs], x[:, k_age] = zz.ravel(), tt.ravel()
                eta = boosting_logit(bundle, x)
                x[:, k_age] = 0
                main_prs = boosting_logit(bundle, x)
                x[:, k_prs], x[:, k_age] = 0, tt.ravel()
                main_age = boosting_logit(bundle, x)
                return (eta-main_prs-main_age+boosting_logit(bundle, reference[None])[0]).reshape(zz.shape)
            candidates = [i for i, k in bundle["active_pairs"] if k == k_age]
            candidates = candidates or [0]
            effects = {k: interaction(k) for k in candidates}
            chosen = max(effects, key=lambda k: np.max(np.abs(effects[k])))
            for level in [-1, 0, 1]:
                x = np.tile(reference, (len(age), 1))
                x[:, chosen], x[:, k_age] = level, t_age
                eta = boosting_logit(bundle, x)
                axes[0, j].plot(age, 1/(1+np.exp(-eta)), label=f"PRS z={level:+d}")
            axes[0, j].set(title=f"{trait.upper()} · {arm}", xlabel="age_sd (not years)", ylabel="Direct probability")
            axes[0, j].legend(fontsize=8)
            bound = max(float(np.abs(effects[chosen]).max()), .01)
            img = axes[1, j].imshow(effects[chosen], origin="lower", aspect="auto", cmap="RdBu_r",
                vmin=-bound, vmax=bound, extent=[z[0], z[-1], age[0], age[-1]])
            axes[1, j].set(xlabel=features.names[chosen]+" (standardized)", ylabel="age_sd",
                          title="Exact age × PRS logit interaction")
            fig.colorbar(img, ax=axes[1, j], label="Anchored logit contribution")
        fig.suptitle("Low-order model responses at the reference profile · strongest fitted age × PRS surface", fontsize=15)
        savefig(fig, "low_order_interpretation")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
        for ax, trait in zip(axes, TRAITS, strict=True):
            lock = data[trait]["locks"]
            arm = lock["selected_arm"]
            candidate = lock["choices"][arm]["candidate"]
            info = read(root / "core" / trait / "tuning" / f"{candidate}_seed17/metadata.json")["training"]
            curve = pd.DataFrame(info["logs"])
            ax.plot(curve.step, curve.training_soft_bce, label="U soft-target BCE")
            ax.plot(curve.step, curve.training_soft_bce+curve.training_penalty, label="U BCE + penalty")
            ax.plot(curve.step, curve.selection_loss, label="H true-label log loss")
            ax.axvline(info["best_step"], ls="--", color="gray", label="Selected step (seed 17)")
            ax.set(title=f"{trait.upper()} · {arm}", xlabel="Optimization step in selected stage", ylabel="Loss")
        axes[0].legend(fontsize=8)
        fig.suptitle("Train-internal optimization traces · soft-target and true-label losses have different targets")
        savefig(fig, "training_loss")


def make_html(root, scope, frames, data, lang):
    def t(zh, en):
        return zh if lang == "zh" else en
    def table(frame):
        return frame.to_html(index=False, float_format=lambda v: f"{v:.5f}", border=0, na_rep="—")
    def image(name):
        payload = base64.b64encode((root / "report" / scope / "figures" / f"{name}.svg").read_bytes()).decode()
        return f'<img alt="{name}" src="data:image/svg+xml;base64,{payload}">'
    title = t("生物学先验蒸馏：可解释模型与新基础模型", "Biologically inspired distillation: interpretable students and new foundation models")
    text = f'''<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
    <style>body{{font:16px/1.7 system-ui,sans-serif;color:#18304b;background:#f1f5f9;margin:0}}main{{max-width:1220px;margin:auto;padding:30px}}section{{padding:26px;background:white;margin:20px 0;border-radius:10px;overflow:auto}}h1{{font-size:32px}}h2{{color:#087f70}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dce4ed}}th{{background:#eef4f8}}img{{width:100%;height:auto}}.formula{{background:#edf8f4;padding:20px;font-size:19px}}.note{{background:#fff6e6;padding:18px;border-left:4px solid #be8b32}}a{{color:#087f70}}</style><main><h1>{title}</h1>'''
    text += '<p>'+t(f"评估范围：{scope}。固定 PRS、原始 train/val 和女性 BC 资格保持一致。", f"Evaluation scope: {scope}. Fixed PRS panels, original train/val membership and female BC eligibility are retained.")+'</p>'
    if scope == "nested":
        text += '<p class="note">'+t("五个外折共享原有患者／可用亲缘组划分。每折内部重新选择结构、λ、正则化和步数；所有 Teacher 软标签均在该外折训练范围内重新交叉预测。门控 ridge 固定为共同的 0.001，避免沿用旧全量 H 调出的值；因此这是单独的嵌套敏感性流程。",
            "Five outer folds reuse the shared patient/available-family assignment. Structure, λ, penalties and steps are reselected inside each outer training fold; teacher soft targets are cross-predicted within that scope. A common gate ridge of 0.001 avoids inheriting the historical full-train H-selected value, making this a separate nested sensitivity pipeline.")+'</p>'
    else:
        text += '<p class="note">'+t("全部模型在 train 内 H 选择并锁定后，才读取本次 val 指标。原始 val 已参与过此前方案讨论，且全量队列保留已知亲缘重叠；这些结果属于开发证据。",
            "All choices were locked on H within train before this evaluation. Original val has informed earlier design discussions, and the full cohort retains known family overlap; these are development results.")+'</p>'
    text += '<section><h2>'+t("先看训练内部选出的模型", "Start with the models selected inside training")+'</h2><ul>'
    for trait in TRAITS:
        m = frames["metrics"].query("trait == @trait").set_index("arm")
        selected = frames["selection"].query("trait == @trait and policy == 'all_biological'")
        names = ", ".join(f"{a} ({n}/5)" for a, n in selected.arm.value_counts().items()) if scope == "nested" else selected.arm.iloc[0]
        delta = frames["paired_differences"].query("trait == @trait and arm == 'BIO_POLICY' and reference == 'TabPFN_v3' and metric == 'log_loss'").iloc[0]
        text += '<li>'+html.escape(f"{trait.upper()}: {names}")+". "+t(
            f"Log loss {m.loc['BIO_POLICY', 'log_loss']:.4f}，AUROC {m.loc['BIO_POLICY', 'auroc']:.4f}，Nagelkerke R² {m.loc['BIO_POLICY', 'nagelkerke_r2']:.4f}。相对原 Teacher 的 Δlog loss={delta.delta:+.4f}，95% CI [{delta.ci_low:+.4f}, {delta.ci_high:+.4f}]。",
            f"Log loss {m.loc['BIO_POLICY', 'log_loss']:.4f}, AUROC {m.loc['BIO_POLICY', 'auroc']:.4f}, Nagelkerke R² {m.loc['BIO_POLICY', 'nagelkerke_r2']:.4f}. Δlog loss vs original teacher={delta.delta:+.4f}, 95% CI [{delta.ci_low:+.4f}, {delta.ci_high:+.4f}].")+'</li>'
    text += '</ul><p>'+t("Δlog loss 越小越好。模型由 H 选择，下面所有候选的 val/外折指标仅用于诊断；区间跨零不能证明等效。不同疾病允许选择不同结构。",
        "Lower Δlog loss is better. Models are selected on H; assessment metrics for all other candidates below are diagnostic. A CI spanning zero does not establish equivalence. Different diseases may select different structures.")+'</p>'+image("performance")+'</section>'
    text += '<section><h2>'+t("模型设计与训练", "Model design and training")+'</h2>'
    text += '<div class="formula">π(A)=softmax(WA+c), r<sub>k</sub>&gt;0, d<sub>k</sub>≥0<br>'
    text += 'g<sub>k</sub>(0)=0, &nbsp; g′<sub>k</sub>(z)&gt;0<br>'
    text += 'η=b(A,C)+f<sub>↑</sub>(a)+Σ<sub>k</sub>π<sub>k</sub>(A)r<sub>k</sub>g<sub>k</sub>(Z<sub>k</sub>)exp[d<sub>k</sub>t(a)sign(g<sub>k</sub>(Z<sub>k</sub>))]<br>p=sigmoid(η)</div>'
    text += '<p>'+t("M0：线性 PRS + 三段单调年龄；A6：六段年龄；G：五段单调 PRS 曲线；G_A6：组合两者。PRS 曲线在训练分位点设结点、在零点锚定，正斜率保证每个 PRS 的方向。它们是可检验的结构先验，不是遗传因果规律。",
        "M0 uses linear PRS and three monotone age segments; A6 uses six age segments; G learns five-segment monotone PRS curves; G_A6 combines both. PRS knots use training quantiles, curves are anchored at zero, and positive slopes preserve each score's direction. These are testable structural priors, not genetic causal laws.")+'</p>'
    text += '<div class="formula">L=BCEWithLogits(η,(1−λ)y+λq<sub>OOF</sub>)+Ω<sub>base,gate</sub>+κ<sub>r</sub>mean(log r)²+κ<sub>d</sub>mean(d²)+κ<sub>g</sub>mean(centered log slopes)²</div>'
    text += '<p>'+t("λ∈{0,0.25,0.5,0.75,1}；SUP 固定 λ=0，KD 在正值中选择。Adam/full batch，初始 lr=0.01，交互微调 lr=0.001，每阶段最多 2,000 步。d 和年龄斜率投影非负；第 0 步可以胜出。CONTINUE 是同优化器重启的无交互对照；D1 共享 d，D2 每 PRS 一个 d，D2_ONLY 只拟合 d。三个 seeds=17/29/43，主结果固定 17。",
        "λ∈{0,0.25,0.5,0.75,1}; SUP fixes λ=0 and KD selects among positive values. Adam/full batch uses initial lr=0.01 and interaction lr=0.001, with at most 2,000 steps per stage. d and age slopes are projected nonnegative; step zero can win. CONTINUE controls for optimizer restart without interaction; D1 shares d, D2 uses per-PRS d, and D2_ONLY fits d alone. Seeds are 17/29/43 with primary seed 17.")+'</p>'
    text += '<p>'+t("Teacher 始终是原 TabPFN v3。TabPFN 3.5（tabpfn 9.2.0）与 NVIDIA Kumo Tabular large（SDM 0.2.1）作为独立对照，各使用同一输入信息、16 个 estimators、seed=42，在本地推理。",
        "The distillation teacher remains the original TabPFN v3. TabPFN 3.5 (tabpfn 9.2.0) and NVIDIA Kumo Tabular large (SDM 0.2.1) are separate local comparators with the same information set, 16 estimators and seed 42.")+'</p></section>'
    text += '<section><h2>'+t("较弱的生物学约束：单调低阶加性模型", "A weaker biological prior: monotone low-order additive models")+'</h2>'
    text += '<div class="formula">η=η<sub>0</sub>+Σ<sub>j</sub>f<sub>j</sub>(x<sub>j</sub>)+Σ<sub>(j,k)∈E</sub>f<sub>jk</sub>(x<sub>j</sub>,x<sub>k</sub>)<br>∂p/∂age ≥ 0, &nbsp; ∂p/∂PRS<sub>k</sub> ≥ 0</div>'
    text += '<p>'+t("HGB_GAM 只允许单变量项；HGB_PAIR 只允许 PRS×年龄、PRS×PC1–PC10、年龄×性别，以及单变量项。使用 sklearn HistGradientBoostingClassifier 的单调和交互约束；逐树分支核验最多两个特征，并通过参考点分解精确重建 logit。总年龄风险单调，但遗传效应随年龄的变化可增强或减弱。",
        "HGB_GAM allows main effects only. HGB_PAIR permits PRS×age, PRS×PC1–PC10, age×sex, and main effects. sklearn HistGradientBoostingClassifier enforces monotonicity and interaction constraints; every branch is checked to contain at most two features, and anchored components exactly reconstruct logits. Total risk is age-monotone while genetic contrasts may strengthen or attenuate across age.")+'</p>'
    text += '<p>'+t("软标签通过每名患者两个二元训练行、权重 (1−q,q) 实现，精确等价于软标签 BCE；每名患者总权重仍为 1。固定 lr=0.05，最多 400 棵树；叶子数 {4,8}、L2 {1,10} 和同一 λ 网格均只在 H 选择，包含零棵树候选。HGB 固定 seed=17，无随机子采样。BIO_POLICY 在 H 比较参数模型与低阶模型，容差内优先保留参数模型；val/外折标签不参与。",
        "Soft labels use two binary rows per patient with weights (1−q,q), exactly matching soft-label BCE with total weight one per patient. lr=0.05 and at most 400 trees; leaves {4,8}, L2 {1,10}, and the same λ grid are selected on H, including zero trees. HGB uses seed 17 without stochastic subsampling. BIO_POLICY compares parametric and low-order choices on H, favoring the parametric model within tolerance; val/outer-fold labels are excluded.")+'</p>'
    text += '<details><summary>'+t("低阶模型实际训练预算", "Actual low-order training budgets")+'</summary>'+table(frames["low_order_training_details"])+ '</details></section>'
    text += '<section><h2>'+t("训练内部选择与性能", "Train-internal selection and performance")+'</h2>'+table(frames["selection"])
    for trait in TRAITS:
        m = frames["metrics"].query("trait == @trait")
        text += f'<details><summary>{trait.upper()} — '+t("所有候选与对照", "all candidates and comparators")+'</summary>'+table(m[["arm", "n", "events", "log_loss", "auroc", "average_precision", "brier", "nagelkerke_r2", "probability_mae", "logit_r2"]])+ '</details>'
    text += '<p>'+t("Nagelkerke R² 使用冻结概率相对本评估队列事件率常数模型的似然；logit R² 衡量对原 Teacher 的模仿。二者都不是遗传解释度。本轮全部是直接概率，未运行 Phase B / Stage-2 wrapper。",
        "Nagelkerke R² compares frozen probabilities with a constant prevalence reference in this assessment cohort; logit R² measures fidelity to the original teacher. Neither is genetic variance explained. All results use direct probabilities; Phase B / Stage-2 wrappers were not run.")+'</p></section>'
    text += '<section><h2>'+t("配对不确定性与校准", "Paired uncertainty and calibration")+'</h2>'
    contrasts = frames["paired_differences"].query("metric == 'log_loss'")
    text += table(contrasts[["trait", "arm", "reference", "delta", "ci_low", "ci_high"]])+image("calibration")
    text += '<p>'+t("区间为 1,000 次可用亲缘组配对 bootstrap 的名义 95% CI，条件于固定预测；没有重新训练模型或校正多重比较，也没有覆盖未知亲缘依赖。nested 区间同样不覆盖重新划分与重新训练的不确定性。",
        "Intervals are nominal 95% CIs from 1,000 paired available-family bootstrap draws conditional on frozen predictions. They do not refit models, adjust multiplicity or capture unknown family dependence. Nested intervals likewise exclude new-split and refitting uncertainty.")+'</p></section>'
    if scope == "val":
        text += '<section><h2>'+t("可解释曲线与实际训练预算", "Interpretable curves and actual training budgets")+'</h2>'+image("interpretation")
        text += '<p>'+t("年龄单位仍为 age_sd。曲线保持其余输入固定；PRS / 年龄变换属于模型特征，不改变患者基因型。pi 是模型混合权重，r/d/曲线形状不是唯一生物学贡献；相关 PRS 仍会造成参数不唯一。",
            "Age remains age_sd. Curves hold all other inputs fixed; PRS/age transforms are model features and do not alter genotype. pi denotes model mixing weights; r/d/shape are not unique biological contributions, and correlated PRS still permit parameter ambiguity.")+'</p>'+image("low_order_interpretation")
        text += '<p>'+t("上图使用合成参考特征组合，不代表单个患者。低阶模型热图是精确的年龄×PRS logit 交互；零曲面表示该模型未学习此交互。它与总风险随年龄单调上升是不同性质。",
            "Curves use synthetic reference profiles, not individual patients. Low-order heatmaps show exact age×PRS logit interactions; a zero surface indicates no learned interaction. Interaction shape is distinct from monotonicity of total risk across age.")+'</p>'+image("training_loss")
        text += '<details><summary>'+t("参数模型实际训练预算", "Actual parametric training budgets")+'</summary>'+table(frames["training_details"])+ '</details></section>'
    text += '<section><h2>'+t("随机初始化稳定性", "Initialization stability")+'</h2><p>'+t(
        "三个 seed 使用同一训练内部选择结果；主分析仍固定 seed=17，没有根据评估表现挑 seed。以下变化衡量初始化敏感性，不是总体不确定性的置信区间。",
        "Three seeds share the same train-internal selection; primary results remain seed 17, without assessment-based seed selection. These differences measure initialization sensitivity, not population confidence intervals.")+'</p>'+table(frames["seed_stability"].query("arm == 'H_SELECTED'"))+ '</section>'
    text += '<section><h2>'+t("年龄、祖源分组与证据", "Age, ancestry subgroups and evidence")+'</h2><p>'+t(
        "年龄四分位点仅由相应训练范围确定。人数少于 100 或任一类别少于 20 的分组标记 unstable，不据此作强结论。",
        "Age quartiles are fitted in the corresponding training scope only. Groups with fewer than 100 patients or fewer than 20 in either outcome class are flagged unstable.")+'</p><details><summary>'+t("查看分组人数、事件数和指标", "View subgroup counts, events and metrics")+'</summary>'+table(frames["subgroups"])+ '</details>'
    text += '<p>'+t("已知 train/val 跨界亲缘组 115 个；全量 train 112 人、val 42 人亲缘状态未确定。嵌套外折排除了可用亲缘组交叉，但未知亲缘仍是限制。患者级预测和完整模型仅保存在本地。",
        "Original train/val retain 115 crossing family groups, with unresolved kinship for 112 train and 42 val patients. Nested outer folds exclude available-family overlap, while unresolved kinship remains a limitation. Patient predictions and complete models stay local.")+'</p>'
    text += '<p><a href="https://huggingface.co/Prior-Labs/tabpfn_3_5">TabPFN 3.5 model card</a> · <a href="https://huggingface.co/nvidia/Kumo-Tabular">Kumo Tabular model card</a></p>'
    text += '<small>'+html.escape(str(root))+'</small></section></main></html>'
    return text


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--scope", choices=["val", "nested"], required=True)
    a = p.parse_args()
    frames, data, _ = evaluate(a.run_dir, a.scope)
    plots(a.run_dir, a.scope, frames, data)
    out = a.run_dir / "report" / a.scope
    for lang in ["zh", "en"]:
        (out / f"bio_distillation_{lang}.html").write_text(make_html(a.run_dir, a.scope, frames, data, lang))
    write_json(out / "VERIFIED.json", {"status": "evaluated_and_reported", "scope": a.scope,
        "output_sha256": {str(f.relative_to(out)): sha256_file(f) for f in out.rglob("*")
                          if f.is_file() and f.name != "VERIFIED.json" and f.suffix != ".zip"}})
    with zipfile.ZipFile(out / "bio_distillation_report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for f in out.rglob("*"):
            if f.is_file() and f.suffix != ".zip":
                archive.write(f, f.relative_to(out))
    print(f"Report ready: {out}", flush=True)


if __name__ == "__main__":
    main()
