from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path
from typing import Any

from decipehr.qc_config import load_qc_config, normalize_qc_config


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def parse_chromosomes(value: str) -> list[str]:
    chromosomes: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            chromosomes.extend(str(chrom) for chrom in range(int(start), int(end) + 1))
        else:
            chromosomes.append(part)
    if not chromosomes:
        raise ValueError("At least one chromosome must be specified")
    return chromosomes


def format_under_dir(directory: str | Path, pattern: str, chrom: str) -> str:
    formatted = pattern.format(chr=chrom, CHR=chrom)
    path = Path(formatted)
    if path.is_absolute():
        return str(path)
    return str(Path(directory) / path)


def geno_args(max_missingness: float, mode: str) -> str:
    mode = mode.lower()
    if mode == "dosage":
        return f"--geno {max_missingness:g} dosage"
    if mode == "hardcall":
        return f"--geno {max_missingness:g}"
    if mode == "hh-missing":
        return f"--geno {max_missingness:g} hh-missing"
    raise ValueError("variant_qc.genotype_missingness_mode must be dosage, hardcall, or hh-missing")


def legacy_config(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pfile_pattern or not args.info_pattern or not args.out_dir:
        raise SystemExit(
            "--config is recommended. Without --config, --pfile-pattern, --info-pattern, "
            "and --out-dir are required."
        )
    return normalize_qc_config(
        {
            "variant_qc": {
                "min_info": args.min_info,
                "min_maf": args.min_maf,
                "max_variant_missingness": args.geno,
                "info_score": {
                    "variant_col": "col0",
                    "info_col": "info_IMPUTE",
                    "freq_col": "MAF",
                    "freq_is_maf": True,
                },
            },
            "paths": {
                "plink2": args.plink2,
                "qc_work_dir": args.out_dir,
                "raw_pgen_dir": ".",
                "raw_pgen_pattern": args.pfile_pattern,
                "info_score_dir": ".",
                "info_score_pattern": args.info_pattern,
            },
        }
    )


def build_commands(config: dict[str, Any], chromosomes: list[str], use_legacy_filter_alias: bool) -> list[str]:
    variant_qc = config["variant_qc"]
    sample_qc = config["sample_qc"]
    paths = config["paths"]
    info_score = variant_qc["info_score"]

    if variant_qc.get("apply_hwe"):
        raise ValueError("HWE filtering is intentionally not implemented in V1 minimal QC")
    if sample_qc.get("apply_mind") and not sample_qc.get("keep_file"):
        raise ValueError("sample_qc.apply_mind requires one shared sample_qc.keep_file")

    qc_work_dir = Path(paths["qc_work_dir"])
    lists_dir = qc_work_dir / "lists"
    step1_dir = qc_work_dir / "qc_step1"
    final_dir = qc_work_dir / "qc_final"
    reports_dir = qc_work_dir / "reports"
    logs_dir = qc_work_dir / "logs"
    pmerge_list = final_dir / "qc_pfiles.txt"
    sample_keep = sample_qc.get("keep_file") if sample_qc.get("apply_mind") else None

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated only from paths and thresholds; no phenotype rows are read here.",
        "# V1 minimal PRS-ready QC: variant QC, reports, merge, no default sample removal/HWE.",
        f"mkdir -p {q(lists_dir)} {q(step1_dir)} {q(final_dir)} {q(reports_dir)} {q(logs_dir)}",
        f": > {q(pmerge_list)}",
    ]

    for chrom in chromosomes:
        pfile = format_under_dir(paths["raw_pgen_dir"], paths["raw_pgen_pattern"], chrom)
        info = format_under_dir(paths["info_score_dir"], paths["info_score_pattern"], chrom)
        snplist = lists_dir / f"chr{chrom}.info_maf.snplist"
        step1_prefix = step1_dir / f"chr{chrom}.step1"
        final_prefix = final_dir / f"chr{chrom}.qc"
        filter_info_flag = "--info-score" if use_legacy_filter_alias else "--info-file"
        filter_cmd = (
            f"python scripts/filter_info_scores.py {filter_info_flag} {q(info)} "
            f"--out {q(snplist)} "
            f"--min-info {float(variant_qc['min_info']):g} "
            f"--min-maf {float(variant_qc['min_maf']):g} "
            f"--max-af {float(variant_qc['max_af']):g} "
            f"--variant-id-col {q(info_score['variant_col'])} "
            f"--info-col {q(info_score['info_col'])} "
            f"--freq-col {q(info_score['freq_col'])} "
            f"--freq-is-maf {str(info_score['freq_is_maf']).lower()}"
        )
        plink_filters = []
        if variant_qc.get("keep_snps_only", True):
            plink_filters.append("--snps-only just-acgt")
        if variant_qc.get("biallelic_only", True):
            plink_filters.append("--max-alleles 2")
        plink_filters.append(
            geno_args(
                float(variant_qc["max_variant_missingness"]),
                str(variant_qc["genotype_missingness_mode"]),
            )
        )
        final_keep = f" --keep {q(sample_keep)}" if sample_keep else ""
        lines.extend(
            [
                "",
                f"test -f {q(pfile + '.pgen')} || {{ echo 'Missing {pfile}.pgen' >&2; exit 1; }}",
                f"test -f {q(pfile + '.pvar')} || {{ echo 'Missing {pfile}.pvar' >&2; exit 1; }}",
                f"test -f {q(pfile + '.psam')} || {{ echo 'Missing {pfile}.psam' >&2; exit 1; }}",
                f"test -f {q(info)} || {{ echo 'Missing {info}' >&2; exit 1; }}",
                filter_cmd,
                f"test -s {q(snplist)} || {{ echo 'Empty snplist {snplist}' >&2; exit 1; }}",
                (
                    f"{q(paths['plink2'])} --pfile {q(pfile)} --extract {q(snplist)} "
                    f"{' '.join(plink_filters)} --make-pgen --out {q(step1_prefix)}"
                ),
                f"{q(paths['plink2'])} --pfile {q(step1_prefix)} --missing --out {q(reports_dir / f'chr{chrom}.step1_missingness')}",
                (
                    f"{q(paths['plink2'])} --pfile {q(step1_prefix)}{final_keep} "
                    f"--make-pgen --out {q(final_prefix)}"
                ),
                f"printf '%s\\n' {q(final_prefix)} >> {q(pmerge_list)}",
            ]
        )

    merged_prefix = final_dir / "all_chromosomes.qc"
    lines.extend(
        [
            "",
            f"{q(paths['plink2'])} --pmerge-list {q(pmerge_list)} --sort-vars --make-pgen --out {q(merged_prefix)}",
            (
                f"{q(paths['plink2'])} --pfile {q(merged_prefix)} --missing --freq --hardy "
                f"--pgen-info --out {q(reports_dir / 'all_chromosomes.qc_summary')}"
            ),
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Write PHI-safe PLINK2 genotype QC commands.")
    parser.add_argument("--config", default=None, help="Structured QC config, e.g. configs/qc.yaml.")
    parser.add_argument(
        "--pfile-pattern",
        default=None,
        help="Legacy PLINK2 pfile prefix pattern with {chr}, e.g. data/raw/plink/Testing_Chr{chr}.",
    )
    parser.add_argument(
        "--info-pattern",
        default=None,
        help="Legacy info-score file pattern with {chr}.",
    )
    parser.add_argument("--out-dir", default=None, help="Legacy QC work directory.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--min-info", type=float, default=0.8)
    parser.add_argument("--min-maf", type=float, default=0.01)
    parser.add_argument("--geno", type=float, default=0.05, help="Maximum per-variant missingness.")
    parser.add_argument("--chromosomes", default="1-22")
    args = parser.parse_args()

    config = load_qc_config(args.config) if args.config else legacy_config(args)
    chromosomes = parse_chromosomes(args.chromosomes)
    lines = build_commands(config, chromosomes, use_legacy_filter_alias=args.config is None)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Wrote genotype QC command template for {len(chromosomes)} chromosomes to {path}")


if __name__ == "__main__":
    main()
