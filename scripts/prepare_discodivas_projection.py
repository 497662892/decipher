"""Project frozen DiscoDivas PC loadings with PLINK2 in a shared GRCh38 space."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.ancestry_distance import load_panel
from decipehr.external_pgs import sha256_file, write_json


def allele_key(first: pd.Series, second: pd.Series) -> pd.Series:
    return pd.Series(
        ["/".join(sorted((a, b))) for a, b in zip(first, second, strict=True)],
        index=first.index,
    )


def match_loadings(loadings: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Join published rsIDs to Catalog GRCh38 positions with exact allele pairs."""
    if loadings.ID.duplicated().any():
        raise ValueError("Duplicate rsIDs in upstream PC loadings")
    mapping = positions.drop_duplicates().copy()
    mapping = mapping.loc[~mapping.rsID.duplicated(keep=False)]
    joined = loadings.merge(mapping, left_on="ID", right_on="rsID", validate="one_to_one")
    exact = allele_key(joined.REF, joined.ALT).eq(
        allele_key(joined.effect_allele, joined.other_allele),
    )
    exact &= joined["#CHROM"].astype(str).eq(joined.hm_chr.astype(str))
    exact &= joined.A1.eq(joined.REF) | joined.A1.eq(joined.ALT)
    joined = joined.loc[exact].copy()
    joined["allele_key"] = allele_key(joined.REF, joined.ALT)
    joined = joined.loc[~joined.allele_key.isin(["A/T", "C/G"])]
    joined = joined.loc[~joined.duplicated(["hm_chr", "hm_pos", "allele_key"], keep=False)]
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--weight-lock", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--target-pfile", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=48000)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--refit-reference-pca", action="store_true",
                        help="Fit PLINK PCA on 1000G only, using the common upstream SNP panel.")
    parser.add_argument("--plink", default="plink2")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    source = json.loads(Path(args.source_lock).read_text())
    loading_item = next(x for x in source["files"] if x["path"].endswith(".pca.weight"))
    if sha256_file(loading_item["path"]) != loading_item["sha256"]:
        raise ValueError("Changed official PCA loadings")
    loadings = pd.read_csv(loading_item["path"], sep="\t", dtype={"#CHROM": str})
    pcs = [f"PC{i}" for i in range(1, 11)]
    lock = json.loads(Path(args.weight_lock).read_text())
    positions, mapping_sources = [], []
    # Use Catalog's existing rsID harmonization, not a new coordinate liftover.
    for accession in ("PGS002308", "PGS004696", "PGS004869"):
        resource = next(x for x in lock["resources"] if x["id"] == accession)
        if resource["header"]["hmpos_build"] != "GRCh38":
            raise ValueError("Coordinate map must be harmonized to GRCh38")
        if sha256_file(resource["path"]) != resource["sha256"]:
            raise ValueError("Changed Catalog position source")
        columns = ["rsID", "hm_chr", "hm_pos", "effect_allele", "other_allele"]
        table = pd.read_csv(
            resource["path"], sep="\t", comment="#", usecols=columns,
            dtype=str, keep_default_na=False,
        )
        table = table.loc[table.rsID.isin(loadings.ID)].copy()
        # Canonicalize pairs to make swapped source orientations identical.
        pairs = table[["effect_allele", "other_allele"]].to_numpy(copy=True)
        pairs.sort(axis=1)
        table[["effect_allele", "other_allele"]] = pairs
        positions.append(table)
        mapping_sources.append({"id": accession, "sha256": resource["sha256"]})
    joined = match_loadings(loadings, pd.concat(positions, ignore_index=True))
    joined["CHR"] = joined.hm_chr.astype(str)
    joined["BP"] = pd.to_numeric(joined.hm_pos, errors="raise").astype(int)
    reference_run = json.loads(Path(args.reference_run).read_text())
    if reference_run["build"] != "GRCh38":
        raise ValueError("Expected GRCh38 genotype matching resource")
    reference_dir = Path(args.reference_run).parent
    matches = []
    for chunk in pd.read_csv(
        reference_dir / "matched_variants.tsv", sep="\t", chunksize=200000,
        dtype={"CHR": str},
    ):
        keep = chunk.merge(joined, on=["CHR", "BP", "allele_key"], how="inner")
        if not keep.empty:
            matches.append(keep)
    common = pd.concat(matches, ignore_index=True)
    if common.ID.duplicated().any() or common.target_id.duplicated().any():
        raise ValueError("Ambiguous matched PCA variants")
    coverage = len(common) / len(loadings)
    joined.to_csv(out / "mapped_loadings.tsv", sep="\t", index=False)
    common.to_csv(out / "matched_pca_variants.tsv", sep="\t", index=False)
    write_json(out / "matching_qc.json", {
        "n_loadings": len(loadings), "n_coordinate_matches": len(joined),
        "n_common_variants": len(common), "coverage": coverage,
        "minimum_coverage": args.min_coverage,
    })
    if coverage < args.min_coverage and not args.refit_reference_pca:
        raise ValueError(f"PCA loading coverage {coverage:.3%} below {args.min_coverage:.0%}")
    if not np.isfinite(common[pcs].to_numpy()).all():
        raise ValueError("Nonfinite PCA loading")
    commands = []
    if args.refit_reference_pca:
        if len(common) < 10000:
            raise ValueError("Insufficient common markers for a new external-reference PCA")
        common.ref_id.to_csv(out / "pca.snplist", header=False, index=False)
        command = [
            args.plink, "--pfile", reference_run["reference_pfile"],
            "--extract", str(out / "pca.snplist"), "--read-freq",
            str(reference_dir / "reference.afreq"), "--error-on-freq-calc",
            "--pca", "10", "allele-wts", "vcols=chrom,ref,alt",
            "--seed", "42",
            "--threads", str(args.threads), "--memory", str(args.memory_mb),
            "--out", str(out / "reference_pca"),
        ]
        commands.append(command)
        write_json(out / "commands.json", commands)
        with (out / "reference_pca.stdout.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        fitted = pd.read_csv(out / "reference_pca.eigenvec.allele", sep="\t")
        # PLINK writes opposite loadings for both alleles of a biallelic SNP.
        # Select the ALT row once; scoring both would double all coordinates.
        fitted = fitted.loc[fitted.A1.eq(fitted.ALT)].copy()
        fitted = fitted[["ID", "A1", *pcs]].rename(columns={"ID": "ref_id"})
        common = common.drop(columns=["A1", *pcs]).merge(
            fitted, on="ref_id", how="left", validate="one_to_one",
        )
        if not np.isfinite(common[pcs]).all().all() or common.A1.isna().any():
            raise ValueError("Incomplete fitted reference PCA loadings")
        common.to_csv(out / "scoring_pca_variants.tsv", sep="\t", index=False)
    for kind, pfile, id_col, frequency_name in (
        ("reference", reference_run["reference_pfile"], "ref_id", "reference.afreq"),
        ("target", args.target_pfile, "target_id", "reference.target_ids.afreq"),
    ):
        table = common[[id_col, "A1", *pcs]].rename(columns={id_col: "ID"})
        table.to_csv(out / f"{kind}.loadings.tsv", sep="\t", index=False)
        table.ID.to_csv(out / f"{kind}.snplist", index=False, header=False)
        frequencies = []
        for chunk in pd.read_csv(
            reference_dir / frequency_name, sep="\t", chunksize=300000,
        ):
            keep = chunk.loc[chunk.ID.isin(table.ID)]
            if not keep.empty:
                frequencies.append(keep)
        frequencies = pd.concat(frequencies, ignore_index=True)
        if frequencies.ID.duplicated().any() or set(frequencies.ID) != set(table.ID):
            raise ValueError("Incomplete external imputation frequencies")
        frequencies.to_csv(out / f"{kind}.afreq", sep="\t", index=False)
        command = [
            args.plink, "--pfile", str(pfile), "--extract", str(out / f"{kind}.snplist"),
            "--read-freq", str(out / f"{kind}.afreq"), "--error-on-freq-calc",
            "--score", str(out / f"{kind}.loadings.tsv"), "1", "2", "header-read",
            "cols=+scoresums", "list-variants", "--score-col-nums", "3-12",
            "--threads", str(args.threads), "--memory", str(args.memory_mb),
            "--out", str(out / kind),
        ]
        if args.refit_reference_pca:
            command.insert(command.index("cols=+scoresums"), "variance-standardize")
        commands.append(command)
        write_json(out / "commands.json", commands)
        with (out / f"{kind}.stdout.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        actual = pd.read_csv(out / f"{kind}.sscore.vars", header=None)[0].astype(str)
        if len(actual) != len(table) or set(actual) != set(table.ID):
            raise ValueError("PLINK scored a different SNP set")
    id_dtypes = {"IID": str, "#IID": str}
    reference = pd.read_csv(out / "reference.sscore", sep="\t", dtype=id_dtypes)
    target = pd.read_csv(out / "target.sscore", sep="\t", dtype=id_dtypes)
    reference = reference.rename(columns={"#IID": "IID", "#FID": "FID"})
    target = target.rename(columns={"#IID": "IID", "#FID": "FID"})
    score_columns = [f"{pc}_SUM" for pc in pcs]
    panel = load_panel(args.panel)
    ref = reference.merge(panel, on="IID", how="left", validate="one_to_one")
    if ref.super_pop.isna().any():
        raise ValueError("Missing 1000G population annotation")
    if len(ref) != reference_run["n_reference_samples"]:
        raise ValueError("Reference projection lost samples")
    for table in (reference, target):
        if not table.IID.is_unique or not np.isfinite(table[score_columns]).all().all():
            raise ValueError("Invalid projected patient table")
    centers = ref.groupby("super_pop")[score_columns].median().rename(
        columns=dict(zip(score_columns, pcs, strict=True)),
    ).reset_index(names="POP")
    projection_correlations = None
    if args.refit_reference_pca:
        eigenvectors = pd.read_csv(
            out / "reference_pca.eigenvec", sep="\t", dtype=id_dtypes,
        ).rename(columns={"#IID": "IID"})
        checked = reference.merge(eigenvectors[["IID", *pcs]], on="IID", validate="one_to_one")
        projection_correlations = {
            pc: float(np.corrcoef(checked[pc], checked[f"{pc}_SUM"])[0, 1]) for pc in pcs
        }
        # PCA axis signs are arbitrary; both cohorts use the same scored orientation.
        if min(abs(value) for value in projection_correlations.values()) < 0.99:
            raise ValueError("Reference projection disagrees with fitted reference eigenvectors")
    centers.to_csv(out / "reference_medians.tsv", sep="\t", index=False)
    target[["IID", *score_columns]].rename(columns=dict(zip(
        ["IID", *score_columns], ["sampleid", *[f"refPC{i}" for i in range(1, 11)]],
        strict=True,
    ))).to_csv(out / "target_pcs.tsv", sep="\t", index=False)
    outputs = [
        "matched_pca_variants.tsv", "reference.sscore", "target.sscore",
        "reference_medians.tsv", "target_pcs.tsv", "commands.json",
    ]
    if args.refit_reference_pca:
        outputs += ["reference_pca.eigenvec.allele", "reference_pca.eigenvec",
                    "reference_pca.eigenval", "scoring_pca_variants.tsv"]
    write_json(out / "projection.json", {
        "status": "complete", "arguments": vars(args), "n_loading_variants": len(loadings),
        "n_rsids_mapped": len(joined), "n_scored_variants": len(common), "coverage": coverage,
        "genome_build": "GRCh38", "official_loading_sha256": loading_item["sha256"],
        "discodivas_commit": source["commit"], "position_sources": mapping_sources,
        "n_reference": len(reference), "n_target": len(target),
        "reference_projection_correlations": projection_correlations,
        "reference_group_counts": ref.super_pop.value_counts().to_dict(),
        "reference_centers": "1000G population medians recomputed on common SNPs",
        "imputation": "fixed 1000G allele frequencies; target frequencies forbidden",
        "pc_scaling": (
            "PLINK variance-standardized dosage SUM; PCs fitted only on 1000G common markers"
            if args.refit_reference_pca else
            "unstandardized allele-dosage SUM with official PC1-10 loadings"
        ),
        "projection_mode": "1000G_reference_refit" if args.refit_reference_pca else "upstream_loadings",
        "reference_fit_reason": (
            "Original loading coverage failed 90% gate; fit a common-marker external PC space "
            "before outcome evaluation instead of treating the incomplete projection as exact."
            if args.refit_reference_pca else None
        ),
        "output_sha256": {name: sha256_file(out / name) for name in outputs},
        "input_sha256": {key: sha256_file(getattr(args, key)) for key in (
            "source_lock", "weight_lock", "reference_run", "panel",
        )},
    })
    print(f"Projected {len(target)} targets and {len(reference)} references; "
          f"{len(common)} / {len(loadings)} PCA SNPs ({coverage:.2%}).")


if __name__ == "__main__":
    main()
