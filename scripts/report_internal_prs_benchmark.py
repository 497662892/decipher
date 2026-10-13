"""Verify all partitions and report conditional, cluster-paired internal evidence."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from statsmodels.stats.multitest import multipletests

from decipehr.ancestry_performance import prediction_metrics
from decipehr.external_pgs import sha256_file, write_json
from decipehr.internal_prs_benchmark import (
    DISEASES,
    METHODS,
    assert_boundary,
    ids_hash,
    inner_splits,
)

COMPARISONS = (
    ("full_vs_selected_baseline", "tabpfn_full", "selected_baseline", True),
    ("matched_vs_selected_disco", "tabpfn_matched", "selected_disco", True),
    ("full_vs_single_tabpfn", "tabpfn_full", "tabpfn_single", False),
    ("full_vs_covariates", "tabpfn_full", "tabpfn_covariates", False),
    ("full_vs_lr", "tabpfn_full", "full_lr", False),
    ("full_vs_catboost", "tabpfn_full", "catboost", False),
    ("fixed_genetic_increment", "flex_fixed", "flex_covariates", False),
    ("varying_vs_fixed", "flex_varying", "flex_fixed", False),
    ("continuous_vs_no_pc", "tabpfn_full", "tabpfn_no_pc", False),
    ("continuous_vs_hard", "tabpfn_full", "tabpfn_hard", False),
    ("matched_vs_fixed", "tabpfn_matched", "matched_fixed", False),
)


def verified_predictions(root: Path):
    prepared = root / "prepared"
    manifest_hash = sha256_file(prepared / "manifest.json")
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name, digest in manifest["code_sha256"].items():
        if sha256_file(name) != digest:
            raise ValueError(f"Frozen code changed: {name}")
    for name, digest in manifest["source_sha256"].items():
        if sha256_file(manifest["source_paths"][name]) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    predictions, selection, fits = [], [], []
    for disease in DISEASES:
        training = pd.read_parquet(prepared / f"{disease}.training.parquet")
        dev = pd.read_parquet(prepared / f"{disease}.development.parquet")
        for partition in ["0", "1", "2", "3", "4", "development"]:
            out = root / "partitions" / disease / partition
            receipt = json.loads((out / "COMPLETE.json").read_text())
            if receipt["status"] != "complete" or receipt["prepared_manifest_sha256"] != manifest_hash:
                raise ValueError("Incomplete or incompatible partition")
            for name, digest in receipt["output_sha256"].items():
                if sha256_file(out / name) != digest:
                    raise ValueError(f"Output changed: {out / name}")
            frame = pd.read_csv(out / "predictions.csv", dtype={"sampleid": str, "kinship_group": str,
                                                               "partition": str}, float_precision="round_trip")
            expected = dev if partition == "development" else training.loc[training.fold == int(partition)]
            context = (training.loc[training.development_context] if partition == "development"
                       else training.loc[training.fold != int(partition)]).reset_index(drop=True)
            assert_boundary(context, expected)
            boundaries = {"final": (ids_hash(context), ids_hash(expected), len(context), len(expected))}
            for inner_fold, (a, b) in enumerate(inner_splits(context, disease, manifest["seed"])):
                boundaries[inner_fold] = (ids_hash(context.iloc[a]), ids_hash(context.iloc[b]), len(a), len(b))
            fitted = pd.read_csv(out / "fits.csv")
            for row in fitted.itertuples():
                key = "final" if row.stage == "final" else int(row.inner_fold)
                observed = (row.context_ids_sha256, row.query_ids_sha256, row.n_context, row.n_query)
                if observed != boundaries[key]:
                    raise ValueError(f"Fit boundary changed: {disease}/{partition}/{row.method}")
            if set(frame.method) != set(METHODS) or frame.duplicated(["sampleid", "method"]).any():
                raise ValueError("Missing methods or duplicate predictions")
            for method, current in frame.groupby("method", observed=True):
                checked = current.set_index("sampleid").sort_index()
                truth = expected.set_index("sampleid").sort_index()
                if list(checked.index) != list(truth.index):
                    raise ValueError(f"Coverage mismatch: {disease}/{partition}/{method}")
                np.testing.assert_array_equal(checked.label.to_numpy(), truth[disease].to_numpy())
                np.testing.assert_array_equal(checked.kinship_group.to_numpy(), truth.kinship_group.astype(str).to_numpy())
                if not np.isfinite(checked.probability).all() or not checked.probability.between(0, 1).all():
                    raise ValueError("Invalid saved probabilities")
            frame["evaluation"] = "development" if partition == "development" else "training_oof"
            predictions.append(frame)
            for destination, name in ((selection, "selection.csv"), (fits, "fits.csv")):
                table = pd.read_csv(out / name)
                table["disease"], table["partition"] = disease, partition
                destination.append(table)
    return pd.concat(predictions), pd.concat(selection), pd.concat(fits)


def cluster_comparisons(frame, n_bootstraps, seed):
    meta = frame[["sampleid", "label", "kinship_group"]].drop_duplicates().set_index("sampleid")
    wide = frame.pivot(index="sampleid", columns="method", values="probability").loc[meta.index]
    y = meta.label.to_numpy(dtype=int)
    group_index, labels = pd.factorize(meta.kinship_group, sort=True)
    methods = sorted({m for _, a, b, _ in COMPARISONS for m in (a, b)})
    prob = {m: wide[m].to_numpy() for m in methods}
    metrics = ("auc", "log_loss", "brier")
    functions = (roc_auc_score, log_loss, brier_score_loss)
    point = {m: np.array([fn(y, prob[m]) for fn in functions]) for m in methods}
    # Resample whole groups. Every method receives the same patient multiplicities.
    boot = {m: np.full((n_bootstraps, 3), np.nan) for m in methods}
    rng = np.random.default_rng(seed)
    for i in range(n_bootstraps):
        counts = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))
        weights = counts[group_index]
        if np.dot(weights, y) == 0 or np.dot(weights, 1 - y) == 0:
            continue
        for m in methods:
            boot[m][i] = [fn(y, prob[m], sample_weight=weights) for fn in functions]
    rows = []
    for name, candidate, reference, primary in COMPARISONS:
        for j, metric in enumerate(metrics):
            delta = float(point[candidate][j] - point[reference][j])
            samples = boot[candidate][:, j] - boot[reference][:, j]
            samples = samples[np.isfinite(samples)]
            family = 6 if primary and metric == "auc" else 1
            lo, hi = np.quantile(samples, [.025, .975])
            flo, fhi = np.quantile(samples, [.025 / family, 1 - .025 / family])
            # Approximate centered-bootstrap null test; conditional on stored predictions.
            p = (1 + np.sum(np.abs(samples - delta) >= abs(delta))) / (1 + len(samples))
            rows.append({"comparison": name, "candidate": candidate, "reference": reference,
                         "primary": primary and metric == "auc", "metric": metric,
                         "candidate_value": point[candidate][j], "reference_value": point[reference][j],
                         "delta": delta, "ci_low": lo, "ci_high": hi,
                         "familywise_ci_low": flo, "familywise_ci_high": fhi,
                         "p_centered_bootstrap": p, "n": len(y), "n_clusters": len(labels),
                         "n_valid_bootstraps": len(samples), "seed": seed})
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/internal_prs/nyu_20261011")
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.run_dir)
    prediction, selection, fits = verified_predictions(root)
    out = root / "report"
    out.mkdir(exist_ok=False)
    prediction.to_parquet(out / "predictions.parquet", index=False)
    selection.to_csv(out / "selection.csv", index=False)
    fits.to_csv(out / "fit_audit.csv", index=False)
    rows, fold_rows, calibration = [], [], []
    for (disease, evaluation, method), frame in prediction.groupby(["disease", "evaluation", "method"], observed=True):
        groups = [("ALL", frame), *list(frame.groupby("hard_label", observed=True))]
        for label, g in groups:
            row = {"disease": disease, "evaluation": evaluation, "method": method, "ancestry": label}
            if g.label.nunique() == 2:
                row.update(prediction_metrics(g.label.to_numpy(), g.probability.to_numpy()))
                # Preserve rank from the un-clipped saved probabilities.
                row["auc"] = roc_auc_score(g.label, g.probability)
                row["log_loss"] = log_loss(g.label, g.probability)
                row["brier"] = brier_score_loss(g.label, g.probability)
            else:
                row.update({"n": len(g), "cases": int(g.label.sum()), "controls": int((1-g.label).sum()),
                            "auc": np.nan, "log_loss": log_loss(g.label, g.probability, labels=[0, 1]),
                            "brier": brier_score_loss(g.label, g.probability)})
            row["small_subgroup"] = min(row["cases"], row["controls"]) < 20
            rows.append(row)
        for partition, g in frame.groupby("partition", observed=True):
            fold_rows.append({"disease": disease, "evaluation": evaluation, "method": method,
                              "partition": partition, **prediction_metrics(g.label.to_numpy(), g.probability.to_numpy()),
                              "auc": roc_auc_score(g.label, g.probability),
                              "log_loss": log_loss(g.label, g.probability),
                              "brier": brier_score_loss(g.label, g.probability)})
        bins = pd.qcut(frame.probability, 10, duplicates="drop")
        for _, g in frame.groupby(bins, observed=True):
            calibration.append({"disease": disease, "evaluation": evaluation, "method": method,
                                "n": len(g), "mean_prediction": g.probability.mean(), "observed": g.label.mean()})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics_by_ancestry.csv", index=False)
    overall = metrics.loc[metrics.ancestry == "ALL"]
    overall.to_csv(out / "overall_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(out / "calibration_bins.csv", index=False)
    comparisons = []
    groups = list(prediction.groupby(["disease", "evaluation"], observed=True))
    # Each group has its own fixed seed; scheduling cannot change resamples or output order.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(cluster_comparisons, frame, args.n_bootstraps, 20261011 + i)
                   for i, (_, frame) in enumerate(groups)]
        for ((disease, evaluation), _), future in zip(groups, futures, strict=True):
            table = future.result()
            table["disease"], table["evaluation"] = disease, evaluation
            comparisons.append(table)
            print(f"Cluster bootstrap complete: {disease}/{evaluation}", flush=True)
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison["p_holm_primary"] = np.nan
    for evaluation in ("training_oof", "development"):
        mask = comparison.primary & (comparison.evaluation == evaluation)
        if mask.sum() != 6:
            raise ValueError("Primary family is incomplete")
        comparison.loc[mask, "p_holm_primary"] = multipletests(
            comparison.loc[mask, "p_centered_bootstrap"], method="holm")[1]
    comparison.to_csv(out / "paired_cluster_comparisons.csv", index=False)
    selected = selection.loc[selection.selected.fillna(False).astype(bool) | selection.chosen_method.notna()]
    selected.to_csv(out / "selected_configurations.csv", index=False)
    display = ["tabpfn_covariates", "tabpfn_single", "tabpfn_full", "full_lr", "flex_fixed", "flex_varying", "catboost"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, disease in zip(axes, DISEASES, strict=True):
        for evaluation, marker in (("training_oof", "o"), ("development", "s")):
            values = overall.loc[(overall.disease == disease) & (overall.evaluation == evaluation)].set_index("method").loc[display]
            ax.plot(values.auc, np.arange(len(display)), marker, label=evaluation, alpha=.8)
        ax.set_yticks(np.arange(len(display)), display)
        ax.set_xlabel("AUC")
        ax.set_title(disease)
        ax.grid(axis="x", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(out / "internal_auc.svg")
    fig.savefig(out / "internal_auc.pdf")
    plt.close(fig)
    primary = comparison.loc[comparison.primary & (comparison.evaluation == "training_oof")]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    labels = [f"{r.disease}: {r.comparison}" for r in primary.itertuples()]
    ax.hlines(np.arange(len(primary)), primary.ci_low, primary.ci_high)
    ax.scatter(primary.delta, np.arange(len(primary)))
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(np.arange(len(primary)), labels)
    ax.set_xlabel("Paired delta AUC (95% cluster bootstrap CI; conditional on predictions)")
    fig.savefig(out / "primary_oof_deltas.svg")
    fig.savefig(out / "primary_oof_deltas.pdf")
    plt.close(fig)
    write_json(out / "VERIFIED.json", {
        "status": "complete", "partitions": 18, "methods_per_partition": len(METHODS),
        "n_bootstraps": args.n_bootstraps, "bootstrap_unit": "kinship connected component",
        "workers": args.workers,
        "interpretation": "internal development; conditional uncertainty, not external confirmation",
        "script_sha256": sha256_file(__file__),
        "prepared_manifest_sha256": sha256_file(root / "prepared/manifest.json"),
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
    })
    print(f"Verified complete report: {out}")


if __name__ == "__main__":
    main()
