"""Project holdout genotypes with the already selected, frozen reference PCA."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files

SELECTION = Path("outputs/ancestry_transfer_20261104/FINAL_PC_SELECTION.json")
PFILE = Path("outputs/final_genotype/qc_final/holdout_test.common")
COVARIATES = Path("data/processed/final/holdout_test_covariates.csv")


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    out = args.out_dir
    if (out / "COMPLETE.json").exists():
        verify_files(out, json.loads((out / "COMPLETE.json").read_text())["output_sha256"])
        return
    selection = json.loads(SELECTION.read_text())
    if selection["selected_mode"] != "reference_pc" or selection["oadp_enabled"]:
        raise ValueError("Expected the selected standard reference PC scheme")
    reference = Path(selection["reference_projection_directory"])
    source = json.loads((reference / "projection.json").read_text())
    verify_files(reference, source["output_sha256"])
    cov = pd.read_csv(COVARIATES, dtype={"sampleid": str})
    if not cov.sampleid.is_unique or set(cov) & {"y", "T2D", "cad", "bc"}:
        raise ValueError("Holdout must have unique patients and no outcomes")
    samples = pd.read_csv(Path(f"{PFILE}.psam"), sep="\t", dtype=str)
    sample_ids = samples["#IID" if "#IID" in samples else "IID"]
    prior = pd.read_csv(reference / "target_pcs.tsv", sep="\t", dtype={"sampleid": str})
    if set(sample_ids) != set(cov.sampleid) or set(cov.sampleid) & set(prior.sampleid):
        raise ValueError("Holdout coverage or patient disjointness failed")
    files = [SELECTION, COVARIATES, Path(f"{PFILE}.psam"), reference / "projection.json",
             *[reference / f"target.{name}" for name in ["snplist", "afreq", "loadings.tsv"]],
             Path(__file__)]
    before = {str(p): sha256_file(p) for p in files}
    out.mkdir(parents=True, exist_ok=True)
    command = [shutil.which("plink2") or "plink2", "--pfile", str(PFILE),
               "--extract", str(reference / "target.snplist"),
               "--read-freq", str(reference / "target.afreq"), "--error-on-freq-calc",
               "--score", str(reference / "target.loadings.tsv"), "1", "2", "header-read",
               "variance-standardize", "cols=+scoresums", "list-variants",
               "--score-col-nums", "3-12", "--threads", str(args.threads), "--memory", "24000",
               "--out", str(out / "holdout")]
    write_json(out / "PROTOCOL.json", {
        "method": "Frozen 1000G PCA projection with PLINK variance-standardized dosage SUM",
        "n_reference": source["n_reference"], "n_holdout": len(cov), "n_pcs": 10,
        "n_expected_variants": source["n_scored_variants"], "reference_pca_refitted": False,
        "oadp": False, "outcomes_used": False, "command": command, "input_sha256": before,
        "genotype_files": {str(Path(f"{PFILE}{s}")): {
            "bytes": Path(f"{PFILE}{s}").stat().st_size,
            "mtime_ns": Path(f"{PFILE}{s}").stat().st_mtime_ns}
            for s in [".pgen", ".pvar", ".psam"]},
        "genotype_provenance": "File stats plus exact scored-marker and sample coverage; whole PGEN/PVAR not rehashed",
    })
    with (out / "plink.stdout.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    variants = (out / "holdout.sscore.vars").read_text().splitlines()
    expected = (reference / "target.snplist").read_text().splitlines()
    if len(variants) != len(set(variants)) or set(variants) != set(expected):
        raise ValueError("Holdout projection changed the frozen marker set")
    scored = pd.read_csv(out / "holdout.sscore", sep="\t", dtype={"#IID": str, "IID": str})
    scored = scored.rename(columns={"#IID": "sampleid", "IID": "sampleid"})
    if not scored.sampleid.is_unique or set(scored.sampleid) != set(cov.sampleid):
        raise ValueError("Incomplete holdout projection")
    scored = scored.set_index("sampleid").loc[cov.sampleid]
    result = scored[[f"PC{i}_SUM" for i in range(1, 11)]].rename(
        columns={f"PC{i}_SUM": f"refPC{i}" for i in range(1, 11)})
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite projected PC")
    result.reset_index().to_csv(out / "holdout_pcs.tsv", sep="\t", index=False)
    verify_files(Path.cwd(), before)
    files = [p for p in out.iterdir() if p.is_file() and p.name != "COMPLETE.json"]
    write_json(out / "COMPLETE.json", {"status": "complete", "n_holdout": len(result),
        "n_scored_variants": len(variants), "pca_refitted": False, "oadp": False,
        "same_reference_coordinates_as_selected_train_models": True,
        "holdout_outcomes_used": False,
        "output_sha256": {p.name: sha256_file(p) for p in files}})
    print(f"Projected {len(result)} holdout patients on {len(variants)} frozen markers", flush=True)


if __name__ == "__main__":
    main()
