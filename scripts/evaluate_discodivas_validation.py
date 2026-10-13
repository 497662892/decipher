"""Compare official DiscoDivas, linear stacking and TabPFN on identical frozen PRSs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_external_pgs_cv import _tabpfn_factory

from decipehr.discodivas import (
    COMPARISON_METHODS,
    REFERENCE_PCS,
    TrainScoreHarmonizer,
    comparison_bank,
    official_interpolation,
)
from decipehr.eligibility import load_eligibility_rules
from decipehr.external_pgs import load_weight_lock, sha256_file, write_json
from decipehr.external_pgs_benchmark import (
    evaluate_official_validation,
    prepare_official_validation_data,
    summarize_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", required=True, choices=["T2D", "cad", "bc"])
    parser.add_argument("--scoring-dir", required=True)
    parser.add_argument("--projection-dir", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--train-phenotype", default="data/processed/phenotype.csv")
    parser.add_argument("--validation-phenotype", default="data/processed/official_validation_phenotype.csv")
    parser.add_argument("--ancestry-labels", required=True)
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out, scoring, projection, resource = map(Path, (
        args.out_dir, args.scoring_dir, args.projection_dir, args.resource_dir,
    ))
    out.mkdir(parents=True, exist_ok=False)
    paths = {
        "weight_lock": scoring / "weights.scored.lock.json",
        "scores": scoring / "scores.long.csv", "scoring_run": scoring / "scoring_run.json",
        "projection_run": projection / "projection.json", "pcs": projection / "target_pcs.tsv",
        "centers": projection / "reference_medians.tsv",
        "source_lock": resource / "source.lock.json",
        "coefficient_mapping": resource / "dprism_component_mapping.csv",
        "resource_audit": resource / "AUDIT.json",
        "dprism_source_lock": resource / "dprism/source.lock.json",
        "train_phenotype": Path(args.train_phenotype),
        "validation_phenotype": Path(args.validation_phenotype),
        "ancestry_labels": Path(args.ancestry_labels),
        "eligibility_config": Path(args.eligibility_config), "model_path": Path(args.model_path),
    }
    hashes = {key: sha256_file(value) for key, value in paths.items()}
    lock = load_weight_lock(paths["weight_lock"])
    run = json.loads(paths["scoring_run"].read_text())
    if run["scores_sha256"] != hashes["scores"]:
        raise ValueError("Changed frozen PRS scores")
    if run["weight_lock_sha256"] != hashes["weight_lock"]:
        raise ValueError("Changed frozen SNP weights")
    projection_run = json.loads(paths["projection_run"].read_text())
    if projection_run["status"] != "complete":
        raise ValueError("Incomplete reference PC projection")
    for name, digest in projection_run["output_sha256"].items():
        if sha256_file(projection / name) != digest:
            raise ValueError(f"Changed projection output: {name}")
    audit = json.loads(paths["resource_audit"].read_text())
    expected = audit["outputs"]["dprism_component_mapping.csv"]["sha256"]
    if hashes["coefficient_mapping"] != expected:
        raise ValueError("Changed published coefficient mapping")
    for source_lock in (paths["source_lock"], paths["dprism_source_lock"]):
        for item in json.loads(source_lock.read_text())["files"]:
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError(f"Changed public source: {item['path']}")
    bank = comparison_bank(args.disease, pd.read_csv(paths["coefficient_mapping"]))
    resources = {r["id"]: r for r in lock["resources"]}
    if any(resources[key]["disease"] != args.disease for key in bank.source_ids):
        raise ValueError("Score bank contains a different trait")
    rules = load_eligibility_rules(args.eligibility_config)
    context, query = prepare_official_validation_data(
        train_phenotype=pd.read_csv(args.train_phenotype, dtype={"sampleid": str}),
        validation_phenotype=pd.read_csv(args.validation_phenotype, dtype={"sampleid": str}),
        scores=pd.read_csv(paths["scores"], dtype={"sampleid": str}), soft_weights=None,
        hard_labels=pd.read_csv(args.ancestry_labels, sep="\t", dtype={"IID": str}),
        disease=args.disease, model_ids=list(bank.source_ids), eligibility_rules=rules,
    )
    pcs = pd.read_csv(paths["pcs"], sep="\t", dtype={"sampleid": str})
    centers = pd.read_csv(paths["centers"], sep="\t")
    for frame in (context, query):
        if not set(frame.sampleid).issubset(pcs.sampleid):
            raise ValueError("Missing patients in reference PC projection")
    context = context.merge(pcs, on="sampleid", validate="one_to_one", how="left")
    query = query.merge(pcs, on="sampleid", validate="one_to_one", how="left")
    sources = json.loads(paths["source_lock"].read_text())
    r_script = next(item for item in sources["files"] if item["path"].endswith("/DiscoDivas.R"))
    manifest = {
        "arguments": vars(args), "evaluation_protocol": "official_train_val",
        "input_paths": {key: str(value) for key, value in paths.items()}, "input_sha256": hashes,
        "methods": list(COMPARISON_METHODS), "source_ids": list(bank.source_ids),
        "model_ids": bank.model_ids, "reference_groups": list(bank.groups),
        "published_combination": bank.combination.tolist(), "bank_provenance": bank.provenance,
        "reference_centers": "1000G medians as proxies; common external PC loading space",
        "quality_parameters": [1] * len(bank.groups), "inner_folds": 3,
        "ridge_C_grid": [0.1, 1, 10], "wrapper_clip_eps": 0.001,
        "multiplicity_family_size": 6, "n_estimators": 16, "tabpfn_version": "v3",
        "n_context": len(context), "n_query": len(query),
        "n_train_cases": int(context[args.disease].sum()),
        "n_validation_cases": int(query[args.disease].sum()),
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "tabpfn",
        )},
        "excluded_from_matched_bank": sorted({
            r["id"] for r in resources.values() if r["disease"] == args.disease
        } - set(bank.source_ids)),
        "exclusion_reason": {
            "PGS002308": "T2D controlled comparison uses the D-PRISM five-metascore bank",
            "PGS004696": "Mixed tuning cohort without an identified PC median; no arbitrary center",
        },
        "source_sha256": {name: sha256_file(name) for name in (
            "src/decipehr/discodivas.py", "src/decipehr/external_pgs_benchmark.py",
            "scripts/run_official_discodivas.R", "scripts/evaluate_external_pgs_cv.py",
            "src/decipehr/tabpfn_ensemble.py", "src/decipehr/eligibility.py",
            "src/decipehr/metrics.py", "src/decipehr/models.py",
            "src/decipehr/ancestry_performance.py", __file__,
        )},
    }
    write_json(out / "preflight.json", manifest)
    if args.preflight_only:
        print(f"Preflight passed: {args.disease}, train={len(context)}, val={len(query)}, "
              f"{len(bank.groups)} matched inputs.", flush=True)
        return
    with (out / "R_session.txt").open("w") as log:
        subprocess.run(["Rscript", "-e", 'library(rio); sessionInfo()'],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    combined_pcs = pd.concat([context, query])[['sampleid', *REFERENCE_PCS]]
    probes = combined_pcs[["sampleid"]].copy()
    for i, group in enumerate(bank.groups):
        probes[group] = np.sin(np.arange(len(probes)) * (i + 1)) + i
    official_args = dict(
        pcs=combined_pcs, centers=centers, groups=bank.groups,
        r_script=r_script["path"], expected_sha256=r_script["sha256"],
    )
    # Coefficients are functions of the external centers and each patient's PCs only.
    _, coefficients = official_interpolation(
        **official_args, scores=probes, out_dir=out / "official_geometry",
    )
    coefficients.to_csv(out / "interpolation_coefficients.csv", index=False)
    context = context.merge(coefficients, on="sampleid", validate="one_to_one")
    query = query.merge(coefficients, on="sampleid", validate="one_to_one")
    harmonizer = TrainScoreHarmonizer(bank, frozenset(context.sampleid))
    normalized_train, normalized_query, parameters = harmonizer(context, query)
    normalized = pd.concat([normalized_train, normalized_query], ignore_index=True)
    real_scores = normalized[["sampleid", *bank.model_ids]].rename(
        columns=dict(zip(bank.model_ids, bank.groups, strict=True)),
    )
    official_scores, checked_coefficients = official_interpolation(
        **official_args, scores=real_scores, out_dir=out / "official_final_scores",
    )
    reference = normalized.set_index("sampleid").loc[official_scores.sampleid, "discodivas_prs"]
    np.testing.assert_allclose(official_scores.discodivas_prs, reference, atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(coefficients, checked_coefficients)
    normalized[["sampleid", *bank.model_ids, "discodivas_prs"]].to_csv(
        out / "final_harmonized_scores.csv", index=False,
    )
    parameters.to_csv(out / "final_harmonization.csv", index=False)
    print(f"Official interpolation verified for {args.disease}; fitting matched methods.", flush=True)
    result = evaluate_official_validation(
        context, query, disease=args.disease, model_ids=bank.model_ids,
        methods=COMPARISON_METHODS, seed=args.seed,
        tabpfn_factory=_tabpfn_factory(args.device, args.model_path),
        score_transform=harmonizer,
    )
    metrics, _ = summarize_benchmark(result.predictions, n_bootstraps=args.n_bootstraps)
    artifacts = {
        "validation_predictions.csv": result.predictions, "fit_audit.csv": result.fits,
        "preprocessing.csv": result.preprocessing, "summary_metrics.csv": metrics,
        "harmonization_fit_audit.csv": pd.DataFrame(harmonizer.fit_audit),
    }
    for name, frame in artifacts.items():
        frame.to_csv(out / name, index=False)
    manifest.update({"status": "complete", "official_final_scores_verified": True,
                     "output_sha256": {name: sha256_file(out / name) for name in artifacts}})
    for name in ("interpolation_coefficients.csv", "final_harmonized_scores.csv",
                 "final_harmonization.csv", "R_session.txt"):
        manifest["output_sha256"][name] = sha256_file(out / name)
    write_json(out / "run.json", manifest)
    print(f"Completed {args.disease}: {out}", flush=True)


if __name__ == "__main__":
    main()
