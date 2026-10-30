"""Read evaluation labels only after prediction completion; never refit prediction models."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import read_csv, verify_files
from decipehr.prs_distillation.student import load_student_bundle

EPS = 1e-6


def basic_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or \
            not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary evaluation data")
    clipped = np.clip(p, EPS, 1 - EPS)
    return {"n": len(y), "events": int(y.sum()), "nonevents": int((1 - y).sum()),
            "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
            "brier": float(np.mean((p - y) ** 2))}


def fidelity(p, q):
    p, q = np.asarray(p), np.asarray(q)
    pc, qc = np.clip(p, EPS, 1 - EPS), np.clip(q, EPS, 1 - EPS)
    pl, ql = logit(pc), logit(qc)
    return {"probability_mae": float(np.abs(p - q).mean()),
            "probability_rmse": float(np.sqrt(np.mean((p - q) ** 2))),
            "bernoulli_kl": float(np.mean(qc * np.log(qc / pc) +
                                           (1 - qc) * np.log((1 - qc) / (1 - pc)))),
            "logit_r2": float(1 - np.sum((pl - ql) ** 2) / np.sum((ql - ql.mean()) ** 2))
            if np.var(ql) > 1e-12 else float("nan")}


def calibration(y, p):
    result = {"calibration_intercept": float("nan"), "calibration_slope": float("nan"),
              "calibration_status": "not_estimable"}
    x = logit(np.clip(p, EPS, 1 - EPS))
    if len(np.unique(y)) != 2 or np.var(x) <= 1e-12:
        return result
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
        if fit.converged and np.isfinite(fit.params).all():
            result.update(calibration_intercept=float(fit.params[0]),
                          calibration_slope=float(fit.params[1]), calibration_status="diagnostic_only")
    except (ValueError, np.linalg.LinAlgError, Warning) as exc:
        result["calibration_status"] = f"not_estimable:{type(exc).__name__}"
    return result


def paired_bootstrap(y, predictions, groups, replicates, seed):
    """All arms share the same cluster resample; intervals condition on fixed predictions."""
    y, groups = np.asarray(y), np.asarray(groups)
    if pd.isna(groups).any() or len(groups) != len(y):
        raise ValueError("Bootstrap group coverage mismatch")
    arms = list(predictions)
    for p in predictions.values():
        basic_metrics(y, p)
    contrasts = [("S_KD_MIX", "S_SUP"), ("S_KD_PURE", "S_SUP")]
    contrasts += [(arm, "T_MATCHED") for arm in ["S_SUP", "S_KD_PURE", "S_KD_MIX"]]
    contrasts += [(arm, "S_KD_MIX") for arm in ("S_STATIC_GATE", "S_CONSTANT_SCALE") if arm in arms]
    point = {arm: basic_metrics(y, predictions[arm]) for arm in arms}
    metrics = ("log_loss", "auroc", "brier")
    samples = {(a, b, m): [] for a, b in contrasts for m in metrics}
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        selected = rng.integers(len(unique), size=len(unique))
        indices = np.concatenate([members[i] for i in selected])
        measured = {a: basic_metrics(y[indices], predictions[a][indices]) for a in arms}
        for a, b, m in samples:
            samples[a, b, m].append(measured[a][m] - measured[b][m])
    rows = []
    for (a, b, metric), values in samples.items():
        valid = np.asarray(values)[np.isfinite(values)]
        lo, hi = np.quantile(valid, [0.025, 0.975]) if len(valid) else [np.nan, np.nan]
        rows.append({"arm": a, "reference": b, "metric": metric,
                     "delta": point[a][metric] - point[b][metric], "ci_low": lo, "ci_high": hi,
                     "n_valid_replicates": len(valid), "n_requested_replicates": replicates,
                     "resampling_unit": "kinship_group", "refit_models": False})
    return pd.DataFrame(rows)


def subgroup_metrics(primary, age_quartiles):
    rows = []
    edges = np.unique([-np.inf, *age_quartiles, np.inf])
    for arm, part in primary.groupby("arm", sort=False):
        part = part.copy()
        part["age_bin"] = pd.cut(part.age_sd, edges, include_lowest=True).astype(str)
        for dimension in ("hard_label", "age_bin"):
            for name, group in part.groupby(dimension, observed=True):
                values = basic_metrics(group.y.to_numpy(), group.probability.to_numpy())
                unstable = values["n"] < 100 or min(values["events"], values["nonevents"]) < 20
                rows.append({"arm": arm, "dimension": dimension, "group": name, **values,
                             "unstable_exploratory": unstable})
    return pd.DataFrame(rows)


def stability(predictions):
    rows = []
    for arm, group in predictions.groupby("arm", sort=False):
        if arm == "T_MATCHED":
            continue
        base = group.loc[group.seed == 17].sort_values("sampleid")
        for seed in (29, 43):
            current = group.loc[group.seed == seed].sort_values("sampleid")
            if base.sampleid.tolist() != current.sampleid.tolist():
                raise ValueError("Stability comparison has different patient IDs")
            columns = ["probability"]
            if arm != "META":
                columns += ["age_scale", *[c for c in group if c.startswith("pi__")]]
            for column in columns:
                delta = current[column].to_numpy() - base[column].to_numpy()
                rows.append({"arm": arm, "seed": seed, "reference_seed": 17, "component": column,
                             "mean_absolute_difference": np.mean(np.abs(delta)),
                             "maximum_absolute_difference": np.max(np.abs(delta)),
                             "interpretation": "initialization stability, not a confidence interval"})
    return pd.DataFrame(rows)


def verify_decompositions(predictions, out, evaluation):
    for (arm, seed), part in predictions.groupby(["arm", "seed"], sort=False):
        if arm == "T_MATCHED":
            continue
        bundle = load_student_bundle(out / "models" / arm / f"seed{seed}")
        ordered = evaluation.set_index("sampleid", drop=False).loc[part.sampleid]
        replay = bundle.predict(ordered)
        np.testing.assert_allclose(replay["probability"], part.probability, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(replay["logit"], part.logit, atol=1e-12, rtol=1e-12)
        if arm != "META":
            for key, prefix in (("weights", "pi"), ("effective_coefficients", "beta"),
                                ("contributions", "contribution")):
                columns = [f"{prefix}__{name}" for name in bundle.preprocessing.kept_prs]
                np.testing.assert_allclose(replay[key], part[columns], atol=1e-12, rtol=1e-12)


def figures(out, primary, training, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = out / "figures"
    target.mkdir(exist_ok=True)
    x = np.linspace(training.age_sd.min(), training.age_sd.max(), 100)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for arm in ("S_SUP", "S_KD_PURE", "S_KD_MIX"):
        bundle = load_student_bundle(out / "models" / arm / "seed17")
        params = bundle.model.state_dict()
        scale = np.exp(float(params["u0"]) + bundle.preprocessing.age_basis(x) @ params["theta_s"].numpy())
        ref = float(np.exp(float(params["u0"]) +
                           (bundle.preprocessing.age_basis([bundle.preprocessing.reference_age]) @
                            params["theta_s"].numpy())[0]))
        axes[0].plot(x, scale, label=arm)
        axes[1].plot(x, scale / ref, label=arm)
    axes[2].hist(training.age_sd, bins=30, color="gray")
    for ax, label in zip(axes, ["Absolute PRS scale", "Scale / training median-age scale", "Training count"], strict=True):
        ax.set(xlabel="Provided standardized age (age_sd), not years", ylabel=label)
    axes[0].legend(fontsize=8)
    fig.savefig(target / "age_scaling.png", dpi=160)
    plt.close(fig)
    mixed = primary.loc[primary.arm == "S_KD_MIX"]
    weights = [c for c in mixed if c.startswith("pi__") and mixed[c].notna().all()]
    means = mixed.groupby("hard_label", observed=True)[weights].mean()
    means.to_csv(out / "gate_by_ancestry.csv")
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.26 * len(weights))), layout="constrained")
    img = ax.imshow(means.T.to_numpy(), aspect="auto", vmin=0, vmax=max(0.1, means.max().max()))
    ax.set_xticks(range(len(means)), means.index, rotation=30, ha="right")
    ax.set_yticks(range(len(weights)), [w.removeprefix("pi__") for w in weights], fontsize=8)
    ax.set_title("Mean gate weight in observed genetic-similarity groups")
    fig.colorbar(img, ax=ax, label="Model weight, not ancestry proportion")
    fig.savefig(target / "gate_by_ancestry.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    axes[0].scatter(mixed.p_teacher, mixed.probability, s=5, alpha=0.25)
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[0].set(xlabel="Matched teacher probability", ylabel="Mixed student probability")
    axes[1].hist(mixed.probability - mixed.p_teacher, bins=40)
    axes[1].set(xlabel="Student minus teacher probability", ylabel="Patients")
    fig.savefig(target / "teacher_student_fidelity.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    for arm in ("T_MATCHED", "S_SUP", "S_KD_MIX"):
        p = primary.loc[primary.arm == arm]
        observed, predicted = calibration_curve(p.y, p.probability, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, "o-", label=arm)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction", title="Direct calibration diagnostic")
    ax.legend(fontsize=8)
    fig.savefig(target / "calibration.png", dpi=160)
    plt.close(fig)
    # Save all seed curves as tabular data; no confidence-band interpretation.
    curves = []
    for arm in predictions.arm.unique():
        if arm in {"T_MATCHED", "META"}:
            continue
        for seed in (17, 29, 43):
            bundle = load_student_bundle(out / "models" / arm / f"seed{seed}")
            state = bundle.model.state_dict()
            scale = np.full(len(x), float(state["u0"]))
            if "theta_s" in state:
                scale += bundle.preprocessing.age_basis(x) @ state["theta_s"].numpy()
            curves.extend({"arm": arm, "seed": seed, "age_sd": a, "age_scale": s}
                          for a, s in zip(x, np.exp(scale), strict=True))
    pd.DataFrame(curves).to_csv(out / "age_scaling_by_seed.csv", index=False)


def report(root, cfg, trait, *, smoke=False):
    """First semantic consumer of E outcomes. Read-only with respect to all fitted models."""
    from decipehr.prs_distillation.experiment import load_inputs, verify_run

    verify_run(root, cfg)
    out = Path(root) / ("smoke" if smoke else "core") / trait
    complete = json.loads((out / "PREDICTIONS_COMPLETE.json").read_text())
    verify_files(out, complete["output_sha256"])
    predictions = pd.read_parquet(out / "direct_predictions.parquet")
    if (out / "ABLATIONS_COMPLETE.json").exists():
        verify_files(out, json.loads((out / "ABLATIONS_COMPLETE.json").read_text())["output_sha256"])
        predictions = pd.concat([predictions, pd.read_parquet(out / "ablation_predictions.parquet")],
                                ignore_index=True)
    training, evaluation, _ = load_inputs(root, trait)
    if smoke:
        from decipehr.prs_distillation.experiment import group_subset
        training, evaluation = group_subset(training, 256, 1717), group_subset(evaluation, 64, 2929)
    ids = evaluation.sampleid.tolist()
    for _, frame in predictions.groupby(["arm", "seed"]):
        if frame.sampleid.tolist() != ids or not frame.sampleid.is_unique:
            raise ValueError("Models do not share the exact evaluation cohort and order")
    verify_decompositions(predictions, out, evaluation)
    # Check all teacher context/query scopes, hashes and group exclusions independently.
    cache_manifest = json.loads((out / "teacher_cache_manifest.json").read_text())
    verify_files(out, cache_manifest)
    split_path = out / "split_manifest.parquet" if smoke else Path(root) / "split_manifest.parquet"
    assignments = pd.read_parquet(split_path).set_index("sampleid").loc[training.sampleid]
    u_ids = set(assignments.index[assignments.internal_role == "U"])
    h_ids = set(assignments.index[assignments.internal_role == "H"])
    d_ids, e_ids = set(training.sampleid), set(evaluation.sampleid)
    for relative in cache_manifest:
        cache_dir = (out / relative).parent
        record = json.loads((out / relative).read_text())
        verify_files(cache_dir, record["output_sha256"])
        scope = pd.read_parquet(cache_dir / "scope.parquet")
        from decipehr.prs_distillation.preprocessing import assert_boundary
        parts = Path(relative).parts
        allow_overlap = cfg.cohort_policy == "original_full_sensitivity" and parts[1] == "D_to_E"
        assert_boundary(scope.loc[scope.role == "context"], scope.loc[scope.role == "query"],
                        require_group_disjoint=not allow_overlap)
        if record["fingerprint"]["require_group_disjoint"] == allow_overlap:
            raise ValueError("Teacher boundary policy differs from the declared experiment")
        if parts[1] == "U_to_H":
            expected_context, expected_query = u_ids, h_ids
        elif parts[1] == "D_to_E":
            expected_context, expected_query = d_ids, e_ids
        else:
            fold = int(parts[2])
            is_u = parts[1] == "U_oof"
            pool = u_ids if is_u else d_ids
            column = "u_teacher_fold" if is_u else "final_teacher_fold"
            expected_query = set(assignments.index[assignments[column] == fold]) & pool
            expected_context = pool - expected_query
        if set(scope.loc[scope.role == "context", "sampleid"]) != expected_context or \
                set(scope.loc[scope.role == "query", "sampleid"]) != expected_query:
            raise ValueError("Teacher cache does not match its prescribed U/H/D/E scope")
        from decipehr.prs_distillation.preprocessing import id_hash
        context = scope.loc[scope.role == "context"]
        if record["preprocessing"]["fit_n"] != len(context) or \
                record["preprocessing"]["fit_ids_hash"] != id_hash(context):
            raise ValueError("Teacher preprocessing fit scope differs from its context")
    labels = read_csv(cfg.evaluation_path, usecols=["sampleid", trait]).rename(columns={trait: "y"})
    predictions = predictions.merge(labels, on="sampleid", how="left", validate="many_to_one")
    if predictions.y.isna().any() or not predictions.y.isin([0, 1]).all():
        raise ValueError("Evaluation labels are missing or not binary")
    primary = predictions.loc[predictions.seed == 17].copy()
    rows = []
    for arm, part in primary.groupby("arm", sort=False):
        y, p, q = part.y.to_numpy(), part.probability.to_numpy(), part.p_teacher.to_numpy()
        rows.append({"arm": arm, **basic_metrics(y, p), **calibration(y, p), **fidelity(p, q)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "direct_metrics.csv", index=False)
    matrix = {a: p.probability.to_numpy() for a, p in primary.groupby("arm", sort=False)}
    reference = primary.loc[primary.arm == "T_MATCHED"]
    differences = paired_bootstrap(reference.y.to_numpy(), matrix, reference.kinship_group.to_numpy(),
                                   30 if smoke else cfg.bootstrap_replicates, cfg.bootstrap_seed)
    differences.to_csv(out / "paired_differences.csv", index=False)
    preprocessing = json.loads((out / "preprocessing.json").read_text())
    subgroup_metrics(primary, preprocessing["age_quartiles"]).to_csv(out / "subgroup_metrics.csv", index=False)
    stable = stability(predictions)
    stable.to_csv(out / "gate_stability.csv", index=False)
    figures(out, primary, training, predictions)
    lock = json.loads((out / "selection_lock.json").read_text())
    lines = ["# Direct distillation diagnostics", "",
             "SMOKE: interface verification only; not a model-performance estimate." if smoke else
             f"Phase A: post-hoc direct validation diagnostics; cohort policy: {cfg.cohort_policy}.", "",
             f"Trait: {trait}. Train: {len(training)}. Validation: {len(evaluation)}; "
             f"events: {int(reference.y.sum())}. PRS: {len(preprocessing['prs_columns'])}.",
             "All development, early stopping, soft targets and selection use official train only.",
             "Validation labels were loaded after locked refits and prediction completion.",
             f"Internal SELECTED_POLICY: {lock['SELECTED_POLICY']}. Primary model seed: 17.",
             "", "| Model | AUROC | Log loss | Brier | Calibration intercept | Calibration slope |",
             "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.arm} | {r.auroc:.5f} | {r.log_loss:.5f} | {r.brier:.5f} | "
              f"{r.calibration_intercept:.3f} | {r.calibration_slope:.3f} |" for r in metrics.itertuples()]
    lines += ["", "Mixed minus supervised (negative log loss/Brier or positive AUROC favors mixed):", ""]
    for r in differences.loc[(differences.arm == "S_KD_MIX") &
                             (differences.reference == "S_SUP")].itertuples():
        lines.append(f"- {r.metric}: {r.delta:+.6f}; paired 95% interval [{r.ci_low:+.6f}, {r.ci_high:+.6f}].")
    ll = differences.loc[(differences.arm == "S_KD_MIX") & (differences.reference == "S_SUP") &
                         (differences.metric == "log_loss")].iloc[0]
    if ll.ci_high < 0:
        conclusion = "The paired log-loss interval supports improvement from mixed distillation on this cohort."
    elif ll.ci_low > 0:
        conclusion = "Mixed distillation worsened log loss on this cohort."
    else:
        conclusion = "The paired log-loss interval crosses zero; no clear benefit is established, and equivalence is not shown."
    lines += ["", conclusion]
    if cfg.cohort_policy == "original_full_sensitivity":
        lines += ["", "FULL ORIGINAL COHORT SENSITIVITY: train/validation relatives are retained. "
                  "Known QC-valid families stay in one training fold. QC-excluded patients have "
                  "unresolved kinship and patient-level placeholders, not certified singleton families. "
                  "Intervals using these available groups do not cover unresolved relatedness."]
    if lock["SELECTED_POLICY"] == "S_SUP":
        lines.append("Internal selection chose direct supervision (lambda=0); distillation was not selected.")
    means = metrics.set_index("arm").log_loss
    denom = means.META - means.T_MATCHED
    gain = (means.META - means.S_KD_MIX) / denom if denom > 1e-4 else np.nan
    lines += ["", f"Operational gain retention relative to shared simple META: {gain:.4f}.",
             "This ratio is not a biological information fraction. T_METADATA was not run; teacher improvements cannot be assigned solely to PRS interactions.",
             "", "Weight/scale/prediction differences across seeds 17/29/43 are in gate_stability.csv.",
             "These measure optimization stability, not confidence intervals or unique causal contributions.",
             "Training PRS correlations are in prs_correlation.csv; correlated scores may have nonunique weights.",
             "Age plots use standardized age_sd, not years. Ancestry groups describe genetic similarity, not race or proportions.",
             "", "Intervals resample kinship groups with fixed predictions, without refitting teachers or students.",
             "Small subgroups are flagged when N<100 or events/non-events<20; their estimates are exploratory.",
             "Previously used development data, historically selected panels and frozen genotype preprocessing limit confirmatory interpretation.",
             "Clinical PC construction population and original age conversion were not recoverable.",
             "Probabilities describe this sampled evaluation distribution, not population absolute risk.",
             "Phase B / official Stage-2 comparison was NOT run. No blind labels or files were read."]
    (out / "direct_distillation_diagnostics.md").write_text("\n".join(lines) + "\n")
    write_json(out / "VERIFIED.json", {
        "status": "smoke_verified" if smoke else "phase_a_verified",
        "group_isolation": cfg.cohort_policy != "original_full_sensitivity",
        "patient_isolation": True, "known_training_groups_isolated": True,
        "cohort_policy": cfg.cohort_policy,
        "matched_evaluation_ids": True, "bundle_replay": True,
        "component_replay": True, "teacher_cache_hashes": True, "teacher_prediction_scopes": len(cache_manifest),
        "selection_policy": lock["SELECTED_POLICY"], "n_evaluation": len(evaluation),
        "phase_b_executed": False, "blind_data_opened": False,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*")
                          if p.is_file() and p.name != "VERIFIED.json"},
    })
    return metrics, differences
