from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_1000G_GRCH38_RELEASE_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL"
)
DEFAULT_1000G_PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "integrated_call_samples_v3.20130502.ALL.panel"
)
DEFAULT_1000G_MANIFEST_NAME = "20190312_biallelic_SNV_and_INDEL_MANIFEST.txt"
DEFAULT_1000G_README_NAME = "20190312_biallelic_SNV_and_INDEL_README.txt"


def parse_chromosomes(spec: str) -> list[int]:
    """Parse chromosome specs like '1-22' or '1,2,5-8'."""
    chromosomes: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid chromosome range {token!r}")
            chromosomes.extend(range(start, end + 1))
        else:
            chromosomes.append(int(token))

    unique: list[int] = []
    seen: set[int] = set()
    for chrom in chromosomes:
        if chrom < 1 or chrom > 22:
            raise ValueError(f"Only autosomes 1-22 are supported, got chromosome {chrom}")
        if chrom not in seen:
            unique.append(chrom)
            seen.add(chrom)
    if not unique:
        raise ValueError("No chromosomes were provided")
    return unique


def one_kg_vcf_name(chrom: int) -> str:
    return f"ALL.chr{chrom}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"


def normalize_1000g_panel(panel_in: Path, panel_out: Path, keep_out: Path) -> dict[str, Any]:
    """Normalize the 1000G sample panel and write a PLINK --keep file."""
    panel = pd.read_csv(panel_in, sep=r"\s+", dtype=str, engine="python")
    column_by_lower = {column.lower(): column for column in panel.columns}
    required = ["sample", "pop", "super_pop"]
    missing = [column for column in required if column not in column_by_lower]
    if missing:
        raise ValueError(f"1000G panel is missing required columns: {missing}")

    sample_col = column_by_lower["sample"]
    pop_col = column_by_lower["pop"]
    super_pop_col = column_by_lower["super_pop"]
    gender_col = column_by_lower.get("gender")

    out = pd.DataFrame(
        {
            "FID": panel[sample_col],
            "IID": panel[sample_col],
            "pop": panel[pop_col],
            "super_pop": panel[super_pop_col],
        }
    )
    if gender_col is not None:
        out["gender"] = panel[gender_col]
    out = out.dropna(subset=["IID", "pop", "super_pop"]).copy()
    for column in out.columns:
        out[column] = out[column].astype(str).str.strip()
    out = out.loc[
        (out["IID"] != "")
        & (out["pop"] != "")
        & (out["super_pop"] != "")
        & (out["super_pop"] != ".")
    ].copy()
    out = out.drop_duplicates(subset=["IID"], keep="first")
    out = out.sort_values(["super_pop", "pop", "IID"], kind="mergesort").reset_index(drop=True)

    panel_out.parent.mkdir(parents=True, exist_ok=True)
    keep_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(panel_out, sep="\t", index=False)
    out[["FID", "IID"]].to_csv(keep_out, sep="\t", index=False, header=False)

    counts = Counter(out["super_pop"])
    return {
        "input_rows": int(len(panel)),
        "output_rows": int(len(out)),
        "super_pop_counts": {key: int(counts[key]) for key in sorted(counts)},
        "panel": str(panel_out),
        "keep": str(keep_out),
    }


def pfile_paths(prefix: Path) -> tuple[Path, Path, Path]:
    """Return PLINK2 pgen/pvar/psam paths for a file prefix."""
    return tuple(Path(str(prefix) + suffix) for suffix in (".pgen", ".pvar", ".psam"))


def build_plink_convert_command(
    *,
    plink2: str,
    vcf: Path,
    out_prefix: Path,
    threads: int,
    keep_file: Path | None = None,
) -> list[str]:
    """Build the PLINK2 command that converts one 1000G chromosome VCF to SNP-only PGEN."""
    command = [
        plink2,
        "--vcf",
        str(vcf),
        "--double-id",
    ]
    if keep_file is not None:
        command.extend(["--keep", str(keep_file)])
    command.extend(
        [
            "--snps-only",
            "just-acgt",
            "--max-alleles",
            "2",
            "--set-all-var-ids",
            "@:#:$r:$a",
            "--new-id-max-allele-len",
            "23",
            "missing",
            "--rm-dup",
            "exclude-all",
            "--make-pgen",
            "--sort-vars",
            "--threads",
            str(threads),
            "--out",
            str(out_prefix),
        ]
    )
    return command


def build_plink_merge_command(
    *,
    plink2: str,
    merge_list: Path,
    out_prefix: Path,
    threads: int,
) -> list[str]:
    return [
        plink2,
        "--pmerge-list",
        str(merge_list),
        "--sort-vars",
        "--make-pgen",
        "--threads",
        str(threads),
        "--out",
        str(out_prefix),
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
