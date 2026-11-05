"""Apply the mature bigsnpr/bigutilsr OADP projection to the frozen PLINK PCA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files

ORIGINAL = Path("outputs/external_pgs/discodivas/projection_1000g_refit_v2_20261010")


def dosage_geometry(out):
    """Validate the allele adapter and provide row norms while retaining dosages.

    OADP receives the frozen PLINK projections. Matrix products independently
    verify the adapter; they do not replace the sealed scoring result.
    """
    sys.path.insert(0, str(out.parent / "vendor"))
    import pgenlib

    pcs = [f"PC{i}" for i in range(1, 11)]
    weights = pd.read_csv(ORIGINAL / "reference.loadings.tsv", sep="\t")
    w = weights[pcs].to_numpy()
    norms = np.linalg.norm(w, axis=0)
    v = w / norms
    orthogonal_error = float(np.max(np.abs(v.T @ v - np.eye(10))))
    if orthogonal_error >= 1e-6:
        raise ValueError("Frozen PCA loadings are not orthogonal")
    reference = pd.read_csv(ORIGINAL / "reference.sscore", sep="\t")
    d = np.linalg.norm(reference[[f"{pc}_SUM" for pc in pcs]].to_numpy() / norms, axis=0)
    eigen_error = float(np.max(np.abs(d / np.sqrt(len(w) * np.loadtxt(ORIGINAL / "reference_pca.eigenval")) - 1)))
    if eigen_error >= 1e-4:
        raise ValueError("SVD scale differs from frozen PLINK eigenvalues")
    checks = {}
    for kind in ["reference", "target"]:
        prefix = out / "private" / kind
        variants = pd.read_csv(prefix.with_suffix(".pvar"), sep="\t", comment="#", header=None,
                               names=["CHROM", "POS", "ID", "REF", "ALT"], usecols=range(5), dtype=str)
        samples = pd.read_csv(prefix.with_suffix(".psam"), sep="\t", dtype=str)
        ids = samples["#IID" if "#IID" in samples else "IID"].to_numpy()
        loadings = pd.read_csv(ORIGINAL / f"{kind}.loadings.tsv", sep="\t")
        freq = pd.read_csv(ORIGINAL / f"{kind}.afreq", sep="\t").set_index("ID").loc[loadings.ID]
        if not variants.ID.is_unique or len(variants) != len(w):
            raise ValueError("Variant coverage changed")
        index = pd.Series(np.arange(len(variants)), index=variants.ID).loc[loadings.ID].to_numpy()
        variants = variants.iloc[index].reset_index(drop=True)
        if not np.array_equal(loadings[pcs].to_numpy(), w):
            raise ValueError("Target and reference use different loadings")
        if not ((variants.ALT == loadings.A1) | (variants.REF == loadings.A1)).all():
            raise ValueError("Dosage allele mismatch")
        af = np.where(loadings.A1.to_numpy() == freq.ALT.to_numpy(), freq.ALT_FREQS, 1 - freq.ALT_FREQS)
        scale = np.sqrt(2 * af * (1 - af))
        if not (np.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("Invalid frozen reference frequencies")
        flip = variants.ALT.to_numpy() != loadings.A1.to_numpy()
        xv, row_norm = np.zeros((len(ids), 10)), np.zeros(len(ids))
        noninteger, missing_count = 0, 0
        with pgenlib.PgenReader(os.fsencode(prefix.with_suffix(".pgen")), raw_sample_ct=len(ids)) as reader:
            if reader.get_variant_ct() != len(w) or reader.get_raw_sample_ct() != len(ids):
                raise ValueError("PGEN dimensions differ from frozen contract")
            for start in range(0, len(w), 256):
                stop = min(start + 256, len(w))
                block = np.empty((stop - start, len(ids)), dtype=np.float64)
                for j, variant in enumerate(index[start:stop]):
                    reader.read_dosages(int(variant), block[j])
                missing = block < 0
                noninteger += int(((np.abs(block - np.rint(block)) > 1e-8) & ~missing).sum())
                missing_count += int(missing.sum())
                block[flip[start:stop]] = 2 - block[flip[start:stop]]
                block -= 2 * af[start:stop, None]
                block /= scale[start:stop, None]
                block[missing] = 0
                xv += block.T @ v[start:stop]
                row_norm += np.sum(block * block, axis=0)
        old = pd.read_csv(ORIGINAL / f"{kind}.sscore", sep="\t", dtype={"#IID": str, "IID": str})
        old = old.set_index("#IID" if "#IID" in old else "IID").loc[ids]
        expected = old[[f"{pc}_SUM" for pc in pcs]].to_numpy()
        difference = np.abs(xv * norms - expected)
        half_ulp = .5 * 10 ** (np.floor(np.log10(np.maximum(np.abs(expected), np.finfo(float).eps))) - 5)
        excess = np.maximum(difference - half_ulp, 0).max(axis=0) / expected.std(axis=0, ddof=1)
        if excess.max() >= 1e-7:
            raise ValueError(f"Native dosage replay failed for {kind}: {excess.max()}")
        geometry = pd.DataFrame(expected / norms, columns=pcs)
        geometry.insert(0, "sampleid", ids)
        geometry["X_norm"] = row_norm
        geometry.to_csv(out / "private" / f"{kind}_geometry.tsv", sep="\t", index=False, float_format="%.17g")
        checks[kind] = {"n": len(ids), "noninteger_dosages": noninteger, "missing_dosages": missing_count,
                        "max_replay_error_over_pc_sd": float((difference.max(axis=0) / expected.std(axis=0, ddof=1)).max()),
                        "max_error_beyond_six_digit_rounding_over_pc_sd": float(excess.max())}
        print(f"Native dosage replay passed: {kind}; noninteger dosage calls={noninteger}", flush=True)
    write_json(out / "geometry.json", {"weight_norms": norms.tolist(), "singular_values": d.tolist(),
        "orthonormal_loadings_verified": True, "max_orthogonality_error": orthogonal_error,
        "eigenvalue_scale_relative_error": eigen_error, "dosage_replay": checks,
        "pgenlib_version": importlib.metadata.version("pgenlib"),
        "pgenlib_binary_sha256": sha256_file(Path(pgenlib.__file__))})


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if (out / "COMPLETE.json").exists():
        verify_files(out, json.loads((out / "COMPLETE.json").read_text())["output_sha256"])
        return
    seal = json.loads((ORIGINAL / "projection.json").read_text())
    verify_files(ORIGINAL, seal["output_sha256"])
    out.mkdir(parents=True, exist_ok=True)
    private = out / "private"
    private.mkdir(exist_ok=True)
    (out / "tmp").mkdir(exist_ok=True)
    originals = json.loads((ORIGINAL / "commands.json").read_text())
    protocol = {
        "method": "bigutilsr::pca_OADP_proj2 (OADP implementation used by bigsnpr)",
        "reference": str(ORIGINAL), "n_reference": 2503, "n_target": 8663,
        "n_variants": 53571, "n_pcs": 10,
        "pca_basis_refitted": False, "phenotypes_used": False,
        "scaling": "Normalize frozen PLINK ALT loadings to unit norm; derive SVD singular values from reference-only scores; map OADP scores back to the original PLINK SUM scale",
        "missing_genotypes": "Reference-frequency mean imputation (zero after reference centering)",
        "dosages": "Native PGEN dosages retained with PLINK pgenlib; BED hardcalls rejected by replay check",
        "evaluation": "Frozen full-reference PCA; per-individual OADP, no joint fit on target samples",
        "interpretation": "Correction magnitude is not an independent estimate of true projection bias",
        "sources": ["https://privefl.github.io/bigsnpr/reference/bed_projectSelfPCA.html",
                    "https://privefl.github.io/bigutilsr/reference/pca_OADP_proj.html"],
        "input_sha256": {str(ORIGINAL / name): digest for name, digest in seal["output_sha256"].items()},
        "implementation_sha256": {str(p): sha256_file(p) for p in [Path(__file__), Path("scripts/project_ancestry_oadp.R")]},
    }
    write_json(out / "PROTOCOL.json", protocol)
    commands = []
    for kind, command in [("reference", originals[1]), ("target", originals[2])]:
        prefix = private / kind
        cmd = ["plink2", "--pfile", command[command.index("--pfile") + 1],
               "--extract", str(ORIGINAL / f"{kind}.snplist"), "--make-pgen",
               "--threads", str(args.threads), "--memory", "24000", "--out", str(prefix)]
        commands.append(cmd)
        with (out / f"{kind}_export.stdout.log").open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f"Exported {kind} fixed markers", flush=True)
    dosage_geometry(out)
    cmd = ["Rscript", "scripts/project_ancestry_oadp.R", str(ORIGINAL.resolve()), str(out), str(args.threads)]
    commands.append(cmd)
    write_json(out / "commands.json", commands)
    env = {**os.environ, "TMPDIR": str(out / "tmp")}
    subprocess.run(cmd, env=env, check=True)
    corrected = pd.read_csv(out / "target_pcs.tsv", sep="\t", dtype={"sampleid": str})
    prior = pd.read_csv(ORIGINAL / "target_pcs.tsv", sep="\t", dtype={"sampleid": str})
    if not corrected.sampleid.is_unique or set(corrected.sampleid) != set(prior.sampleid):
        raise ValueError("Corrected PC patient coverage differs")
    corrected = corrected.set_index("sampleid").loc[prior.sampleid]
    if list(corrected.columns) != [f"refPC{i}" for i in range(1, 11)]:
        raise ValueError("PC column contract changed")
    values = corrected.to_numpy(float)
    if not np.isfinite(values).all() or np.array_equal(values, prior.iloc[:, 1:].to_numpy()):
        raise ValueError("Invalid or unchanged OADP coordinates")
    diagnostic = json.loads((out / "diagnostics.json").read_text())
    if not diagnostic["target_order_invariant"] or not diagnostic["orthonormal_loadings_verified"]:
        raise ValueError("Projection checks failed")
    verify_files(ORIGINAL, seal["output_sha256"])
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for p in [Path(__file__), Path("scripts/project_ancestry_oadp.R")]:
        (snapshot / p.name).write_bytes(p.read_bytes())
    files = [p for p in out.rglob("*") if p.is_file() and "tmp" not in p.relative_to(out).parts and p.name != "COMPLETE.json"]
    write_json(out / "COMPLETE.json", {
        "status": "complete", "method": protocol["method"], "n_target": len(corrected),
        "n_reference": seal["n_reference"], "n_scored_variants": seal["n_scored_variants"],
        "reference_basis_unchanged": True, "phenotypes_used": False,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in files},
    })
    print(json.dumps({"status": "complete", "patients": len(corrected), "diagnostics": diagnostic}), flush=True)


if __name__ == "__main__":
    main()
