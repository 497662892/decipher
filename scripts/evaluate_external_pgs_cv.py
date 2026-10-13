from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import pandas as pd

from decipehr.eligibility import load_eligibility_rules
from decipehr.external_pgs import load_weight_lock, sha256_file, write_json
from decipehr.external_pgs_benchmark import (
    DEFAULT_METHODS,
    HARD_ANCESTRY_METHOD,
    METHODS,
    PC_ONLY_LR_METHOD,
    PRIMARY_METHOD,
    SOFT,
    evaluate_disease,
    evaluate_official_validation,
    method_features,
    prepare_disease_data,
    prepare_official_validation_data,
    summarize_benchmark,
)
from decipehr.io import read_table, write_table


def _read(path: str) -> pd.DataFrame:
    if Path(path).suffix == ".parquet":
        return read_table(path)
    return read_table(path, dtype={"sampleid": str, "IID": str})


def _tabpfn_factory(device: str, model_path: str | None):
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    def create(seed: int):
        kwargs = {"n_estimators": 16, "device": device, "random_state": seed}
        if model_path:
            kwargs["model_path"] = model_path
        return TabPFNClassifier.create_default_for_version(ModelVersion.V3, **kwargs)

    return create


def main(protocol: str = "shared_5fold_cv") -> None:
    if protocol not in {"official_train_val", "shared_5fold_cv"}:
        raise ValueError("Unknown external PGS evaluation protocol")
    official = protocol == "official_train_val"
    label = "official train-context / validation-query" if official else "optional shared five-fold CV"
    parser = argparse.ArgumentParser(description=f"Frozen external PGS benchmark: {label}.")
    parser.add_argument("--weight-lock", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--scoring-run", required=True)
    if official:
        parser.add_argument("--train-phenotype", default="data/processed/phenotype.csv")
        parser.add_argument("--validation-phenotype", default="data/processed/official_validation_phenotype.csv")
    else:
        parser.add_argument("--phenotype", default="data/processed/phenotype.csv")
        parser.add_argument(
            "--splits", required=True,
            help="Explicit shared five-fold assignments restricted to the supplied training cohort.",
        )
    parser.add_argument("--ancestry-weights", help="Required only for soft-similarity methods or soft_z normalization.")
    parser.add_argument("--ancestry-labels", required=True)
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--diseases", default="T2D,cad,bc")
    parser.add_argument("--score-set", default="all", help="all, or a score-set name from the resource manifest.")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help="Comma-separated methods; default is PGS + age/sex + PCs. Use all for the original eight-arm benchmark, or tabpfn_prs_hard_ancestry for the discrete-ancestry ablation.")
    parser.add_argument("--normalization", choices=["none", "soft_z"], default="none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.n_bootstraps < 1:
        raise ValueError("--n-bootstraps must be positive")
    methods = METHODS if args.methods == "all" else tuple(args.methods.split(","))
    if len(set(methods)) != len(methods) or set(methods) - {*METHODS, HARD_ANCESTRY_METHOD, PC_ONLY_LR_METHOD}:
        raise ValueError("Unknown or duplicated benchmark methods")
    args.methods = ",".join(methods)
    needs_soft = args.normalization == "soft_z" or any(
        set(method_features(method, [])[1]).intersection(SOFT) for method in methods
    )
    if needs_soft and not args.ancestry_weights:
        parser.error("--ancestry-weights is required for the selected methods or soft_z normalization")
    diseases = args.diseases.split(",")
    if len(set(diseases)) != len(diseases) or set(diseases) - {"T2D", "cad", "bc"}:
        raise ValueError("Unknown or duplicated diseases")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory for each benchmark")
    lock = load_weight_lock(args.weight_lock)
    scoring_run = json.loads(Path(args.scoring_run).read_text(encoding="utf-8"))
    if scoring_run["weight_lock_sha256"] != sha256_file(args.weight_lock):
        raise ValueError("Scoring run was produced with a different weight lock")
    if scoring_run["scores_sha256"] != sha256_file(args.scores):
        raise ValueError("Score file does not match the audited scoring run")
    scores = _read(args.scores)
    if official:
        phenotype, validation = _read(args.train_phenotype), _read(args.validation_phenotype)
    else:
        phenotype, splits = _read(args.phenotype), _read(args.splits)
        if set(splits["sampleid"]) - set(phenotype["sampleid"]):
            raise ValueError("Optional CV splits must contain training-cohort patients only")
    weights = _read(args.ancestry_weights) if args.ancestry_weights else None
    labels = _read(args.ancestry_labels)
    rules = load_eligibility_rules(args.eligibility_config)
    prepared, counts = {}, []
    for disease in diseases:
        model_ids = [
            item["id"] for item in lock["resources"] if item["disease"] == disease
            and (args.score_set == "all" or item["name"] == args.score_set)
        ]
        shared = dict(
            scores=scores, soft_weights=weights,
            hard_labels=labels, disease=disease, model_ids=model_ids, eligibility_rules=rules,
        )
        if official:
            data, query = prepare_official_validation_data(
                train_phenotype=phenotype, validation_phenotype=validation, **shared,
            )
            groups = [("train_context", data), ("validation_query", query)]
        else:
            data = prepare_disease_data(phenotype=phenotype, splits=splits, **shared)
            query = None
            groups = list(data.groupby("fold"))
        for partition, frame in groups:
            context = data if official else data.loc[data["fold"] != partition]
            context_counts = context[disease].value_counts()
            if len(context_counts) != 2 or context_counts.min() < 5:
                raise ValueError("Insufficient context class counts for nested inner three-fold selection")
            counts.append({
                "disease": disease,
                **({"split": partition} if official else {"fold": int(partition)}), "n": len(frame),
                "cases": int(frame[disease].sum()), "controls": int((1 - frame[disease]).sum()),
                "n_scores": len(model_ids),
            })
        prepared[disease] = (data, query, model_ids)
    out.mkdir(parents=True, exist_ok=True)
    write_table(pd.DataFrame(counts), out / "cohort_counts.csv")
    phenotype_inputs = ("train_phenotype", "validation_phenotype") if official else ("phenotype", "splits")
    provenance = {
        "arguments": vars(args), "tabpfn_model_version": "v3", "n_estimators": 16,
        "inner_folds": 3, "ridge_C_grid": [0.1, 1.0, 10.0], "wrapper_clip_eps": 0.001,
        "evaluation_protocol": protocol,
        "primary_metric": "raw_validation_auroc" if official else "raw_oof_auroc",
        "primary_method": PRIMARY_METHOD if PRIMARY_METHOD in methods else None,
        "primary_method_decision": "PC-only default adopted after the initial validation comparison; soft similarities are optional sensitivity inputs.",
        "multiplicity_family_size": 6 if "tabpfn_prs_pc_soft" in methods else None,
        "score_banks": {disease: ids for disease, (_, _, ids) in prepared.items()},
        "input_sha256": {key: sha256_file(getattr(args, key)) for key in (
            "weight_lock", "scores", "scoring_run",
            *phenotype_inputs,
            "ancestry_weights", "ancestry_labels", "eligibility_config",
        ) if getattr(args, key) is not None},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "scikit-learn", "scipy", "statsmodels")},
    }
    write_json(out / "preflight.json", provenance)
    if args.preflight_only:
        print(f"Input audit completed for {len(diseases)} disease(s), {label}; no models were fitted.")
        return
    factory = None
    if any(not method.endswith("_lr") for method in methods):
        factory = _tabpfn_factory(args.device, args.model_path)
        provenance["versions"]["tabpfn"] = importlib.metadata.version("tabpfn")
        if args.model_path:
            provenance["tabpfn_checkpoint_sha256"] = sha256_file(args.model_path)
    prediction_frames = []
    predictions_name = "validation_predictions.csv" if official else "oof_predictions.csv"
    for disease, (data, query, model_ids) in prepared.items():
        print(f"Evaluating {disease}: {len(data)} eligible patients, {len(model_ids)} frozen scores.", flush=True)
        options = dict(
            disease=disease, model_ids=model_ids, methods=methods, seed=args.seed,
            tabpfn_factory=factory, normalization=args.normalization,
        )
        result = evaluate_official_validation(data, query, **options) if official else evaluate_disease(data, **options)
        write_table(result.predictions, out / disease / predictions_name)
        write_table(result.fits, out / disease / "fit_audit.csv")
        write_table(result.preprocessing, out / disease / "preprocessing.csv")
        prediction_frames.append(result.predictions)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics, comparisons = summarize_benchmark(
        predictions, n_bootstraps=args.n_bootstraps, seed=args.seed,
    )
    if args.score_set != "all" and not comparisons.empty:
        comparisons["exploratory"] = True
    write_table(predictions, out / predictions_name)
    write_table(metrics, out / "summary_metrics.csv")
    if not comparisons.empty:
        write_table(comparisons, out / "paired_auc_comparisons.csv")
    provenance["status"] = "complete"
    provenance["interpretation"] = (
        ("Official train context and fixed validation query; all selection and fitting use train only. "
         if official else "Development OOF comparison. ")
        + "Bootstrap intervals condition on frozen predictions. "
        "Genetic similarities are not ancestry proportions. Wrapper results are secondary."
    )
    write_json(out / "run.json", provenance)
    report = [
        "# Frozen external PGS benchmark", "",
        f"Primary: raw {'validation' if official else 'pooled OOF'} AUROC. Secondary: covariate-plus-score wrapper.",
        f"Primary model: {provenance['primary_method'] or 'custom method subset'}.",
        "The PC-only default was adopted after reviewing the initial validation comparison; it is a subsequent development choice.",
        ("Official train is the full context; official val is prediction-only. All methods share this split."
         if official else "All methods use the same eligible patients and five patient-level folds."),
        "SNP weights are frozen. Internal labels only fit the downstream integrator.",
        "1000G similarities are continuous genetic similarities, not ancestry proportions.",
        f"PRS normalization: {args.normalization}; score set: {args.score_set}.", "",
        f"| Disease | Method | N | Cases | Raw {'validation' if official else 'OOF'} AUROC | OR90 | R² | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.loc[
        metrics["ancestry_group"].eq("ALL") & metrics["prediction_scale"].eq("raw")
    ].to_dict("records"):
        report.append(
            f"| {row['disease']} | {row['method']} | {row['n']} | {row['cases']} | "
            f"{row['auc']:.4f} | {row['top10_vs_lower90_or']:.4f} | "
            f"{row['nagelkerke_r2']:.4f} | {row['brier']:.4f} |"
        )
    report.extend([
        "", "Paired comparisons and subgroup estimates are in the adjacent CSV files.",
        "When the original soft-similarity comparison arms are requested, their two contrasts across three diseases retain a six-comparison family.",
        "familywise_ci_low/high use Bonferroni percentile intervals (individual coverage 99.1667%).",
        "Subgroup, within-score-set, and normalization sensitivity comparisons are exploratory.",
        "Bootstrap uncertainty conditions on frozen predictions and does not include refitting uncertainty.",
        "Previously used validation patients remain development data, not an independent external test.", "",
    ])
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote external PGS {label} benchmark to {out}")


if __name__ == "__main__":
    main()
