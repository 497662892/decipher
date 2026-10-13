"""Apply the exact audited training score matrices to a disjoint holdout cohort."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from decipehr.external_pgs import (
    collect_score_parts,
    load_weight_lock,
    plink_score_command,
    sha256_file,
    write_json,
)


def variant_axis(path: Path) -> dict:
    """Compare coordinates, IDs and allele orientation, ignoring INFO metadata."""
    digest, count = hashlib.sha256(), 0
    with path.open() as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            values = line.rstrip("\n").split("\t", 5)[:5]
            if line.startswith("#"):
                if values != ["#CHROM", "POS", "ID", "REF", "ALT"]:
                    raise ValueError("Unexpected PVAR leading columns")
                continue
            if len(values) != 5:
                raise ValueError("Malformed PVAR")
            digest.update(("\t".join(values) + "\n").encode())
            count += 1
    if not count:
        raise ValueError("Empty PVAR")
    return {"n_variants": count, "canonical_sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True,
                        help="Completed original scoring directory, including matched score matrices.")
    parser.add_argument("--pfile", required=True)
    parser.add_argument("--reference-freq", required=True)
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=32000)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    target = Path(args.pfile).resolve()
    target_axis = variant_axis(Path(str(target) + ".pvar"))
    frequency_hash = sha256_file(args.reference_freq)
    target_samples = pd.read_csv(str(target) + ".psam", sep="\t", dtype=str)
    target_ids = target_samples["IID" if "IID" in target_samples else "#IID"]
    if target_ids.duplicated().any():
        raise ValueError("Duplicate holdout IDs")
    frames, sources, models, axes = [], [], set(), {}
    for source_index, source_text in enumerate(args.source):
        source = Path(source_text)
        run = json.loads((source / "scoring_run.json").read_text())
        lock_path = source / "weights.scored.lock.json"
        lock = load_weight_lock(lock_path)
        for filename, key in [("weights.scored.lock.json", "weight_lock_sha256"),
                              ("scores.long.csv", "scores_sha256"), ("scoring_qc.csv", "qc_sha256")]:
            if sha256_file(source / filename) != run[key]:
                raise ValueError(f"Changed training scoring input: {filename}")
        if run["reference_frequency_sha256"] != frequency_hash:
            raise ValueError("Reference frequencies differ from training scoring")
        context_pfile = run["pfile"]
        if context_pfile not in axes:
            axes[context_pfile] = variant_axis(Path(context_pfile + ".pvar"))
        if axes[context_pfile] != target_axis:
            raise ValueError("Training and holdout variant axes differ; explicit harmonization required")
        context_samples = pd.read_csv(context_pfile + ".psam", sep="\t", dtype=str)
        context_ids = context_samples["IID" if "IID" in context_samples else "#IID"]
        if set(context_ids) & set(target_ids):
            raise ValueError("Training and holdout patients overlap")
        qc = pd.read_csv(source / "scoring_qc.csv").set_index("model")
        resources = {item["id"]: item for item in lock["resources"]}
        if models & set(resources):
            raise ValueError("Duplicate score across source directories")
        models.update(resources)
        aliases = {qc.loc[key, "matcher_accession"]: key for key in resources}
        parts = {key: [] for key in resources}
        provenance = {"directory": str(source), "scoring_run_sha256": sha256_file(source / "scoring_run.json"),
                      "score_matrices": [], "training_variant_axis": axes[context_pfile]}
        for part_index, matrix in enumerate(sorted((source / "matched").glob("*.scorefile.gz"))):
            with gzip.open(matrix, "rt") as handle:
                columns = handle.readline().rstrip("\n").split("\t")
            selected = [i + 1 for i, col in enumerate(columns) if col in aliases]
            if not selected:
                continue
            extract = source / f"part{part_index}.snplist"
            expected = (source / f"part{part_index}.sscore.vars").read_text().splitlines()
            if extract.read_text().splitlines() != expected:
                # PLINK may emit variant IDs in genomic rather than matrix order.
                if set(extract.read_text().splitlines()) != set(expected):
                    raise ValueError("Training score extraction differs from scored variants")
            prefix = out / f"source{source_index}_part{part_index}"
            command = plink_score_command(
                pfile=str(target), scorefile=str(matrix), frequency_file=args.reference_freq,
                out=str(prefix), plink2=args.plink2, threads=args.threads,
                extract_file=str(extract), score_columns=selected, memory_mb=args.memory_mb,
            )
            matrix_hash = sha256_file(matrix)
            with Path(str(prefix) + ".stdout.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            used = Path(str(prefix) + ".sscore.vars").read_text().splitlines()
            if len(used) != len(expected) or set(used) != set(expected):
                raise ValueError("Holdout used a different variant subset")
            if sha256_file(matrix) != matrix_hash:
                raise ValueError("Score matrix changed during scoring")
            provenance["score_matrices"].append({"path": str(matrix), "sha256": matrix_hash,
                                                  "n_scored_variants": len(used), "command": command})
            for col in columns:
                if col in aliases:
                    parts[aliases[col]].append(Path(str(prefix) + ".sscore"))
        for key, item in resources.items():
            scores = collect_score_parts(parts[key], key, item["disease"],
                                         tool_accession=qc.loc[key, "matcher_accession"])
            if set(scores.sampleid) != set(target_ids):
                raise ValueError("Incomplete holdout score coverage")
            frames.append(scores)
        sources.append(provenance)
        print(f"Scored {len(resources)} frozen models from {source}", flush=True)
    scores_path = out / "scores.long.csv"
    pd.concat(frames, ignore_index=True).to_csv(scores_path, index=False)
    write_json(out / "VERIFIED.json", {
        "pfile": str(target), "variant_axis": target_axis, "reference_frequency_sha256": frequency_hash,
        "n_patients": len(target_ids), "n_models": len(models), "sources": sources,
        "scores_sha256": sha256_file(scores_path), "patient_overlap": 0,
        "holdout_phenotypes_used": False, "coefficient_refitting": False,
        "genotype_imputation": "Frozen external 1000G allele frequencies; PLINK --error-on-freq-calc",
    })


if __name__ == "__main__":
    main()
