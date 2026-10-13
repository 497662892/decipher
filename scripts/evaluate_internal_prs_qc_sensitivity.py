"""Repeat the original patient-only development split without new genotype exclusions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from evaluate_external_pgs_cv import _tabpfn_factory

from decipehr.discodivas import REFERENCE_PCS, comparison_bank
from decipehr.eligibility import load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_pgs_benchmark import prepare_official_validation_data
from decipehr.internal_prs_benchmark import METHODS, evaluate_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/internal_prs/nyu_20261011")
    parser.add_argument("--disease", choices=["T2D", "cad", "bc"], required=True)
    parser.add_argument("--eligibility-config", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.run_dir)
    manifest = json.loads((root / "prepared/manifest.json").read_text())
    for name, digest in manifest["code_sha256"].items():
        if sha256_file(name) != digest:
            raise ValueError(f"Frozen benchmark code changed: {name}")
    for name, digest in manifest["source_sha256"].items():
        if sha256_file(manifest["source_paths"][name]) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    paths = manifest["source_paths"]
    rules = load_eligibility_rules(args.eligibility_config)
    if sha256_file(args.eligibility_config) != manifest["source_sha256"]["eligibility"]:
        raise ValueError("Eligibility config changed")
    full_ids = manifest["panels"][args.disease]["full_ids"]
    context, query = prepare_official_validation_data(
        train_phenotype=pd.read_csv(paths["train"], dtype={"sampleid": str}),
        validation_phenotype=pd.read_csv(paths["development"], dtype={"sampleid": str}),
        scores=pd.read_csv(paths["scores"], dtype={"sampleid": str}), soft_weights=None,
        hard_labels=pd.read_csv(paths["ancestry"], sep="\t", dtype={"IID": str}),
        disease=args.disease, model_ids=full_ids, eligibility_rules=rules,
    )
    pcs = pd.read_csv(paths["reference_pcs"], sep="\t", dtype={"sampleid": str})
    geometry = pd.read_csv(paths[f"geometry_{args.disease}"], dtype={"sampleid": str})
    frames = []
    for frame in (context, query):
        frame = frame.merge(pcs[["sampleid", *REFERENCE_PCS]], on="sampleid", validate="one_to_one")
        frame = frame.merge(geometry, on="sampleid", validate="one_to_one")
        # This deliberately reproduces patient-only membership, NOT kinship protection.
        frame["kinship_group"] = "patient_only_" + frame.sampleid
        frames.append(frame)
    context, query = frames
    out = root / "original_cohort_sensitivity" / args.disease
    out.mkdir(parents=True, exist_ok=False)
    bank = comparison_bank(args.disease, pd.read_csv(paths["mapping"]))

    def save_method(method, frame, selection, fits, parameters):
        frame.to_csv(out / f"{method}.predictions.csv", index=False, float_format="%.17g")
        pd.DataFrame(selection).to_csv(out / "selection.partial.csv", index=False)
        pd.DataFrame(fits).to_csv(out / "fits.partial.csv", index=False)

    prediction, selection, fits, parameters = evaluate_partition(
        context, query, disease=args.disease, full_ids=full_ids, bank=bank, methods=METHODS,
        seed=manifest["seed"], tabpfn_factory=_tabpfn_factory("cuda", paths["checkpoint"]),
        on_method=save_method,
    )
    prediction.to_csv(out / "predictions.csv", index=False, float_format="%.17g")
    selection.to_csv(out / "selection.csv", index=False)
    fits.to_csv(out / "fits.csv", index=False)
    write_json(out / "parameters.json", parameters)
    write_json(out / "COMPLETE.json", {
        "status": "complete", "disease": args.disease, "n_context": len(context), "n_query": len(query),
        "purpose": "sensitivity to original unfiltered patient-only development split; relatedness unresolved",
        "no_new_genotype_qc_exclusions": True, "kinship_protection": False,
        "prepared_manifest_sha256": sha256_file(root / "prepared/manifest.json"),
        "script_sha256": sha256_file(__file__), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_sha256": {name: sha256_file(out / name) for name in (
            "predictions.csv", "selection.csv", "fits.csv", "parameters.json")},
    })
    print(f"Complete original-cohort sensitivity: {args.disease}, train={len(context)}, query={len(query)}")


if __name__ == "__main__":
    main()
