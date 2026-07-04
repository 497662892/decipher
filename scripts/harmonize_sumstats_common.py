from __future__ import annotations

import argparse
from pathlib import Path

from decipehr.sumstats_harmonization import (
    harmonize_sumstats_common_from_files,
    write_common_outputs,
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def default_summary_path(out: str | Path) -> Path:
    path = Path(out)
    name = path.name
    for suffix in [".tsv.gz", ".csv.gz", ".tsv", ".csv"]:
        if name.endswith(suffix):
            return path.with_name(f"{name[: -len(suffix)]}.summary.json")
    return path.with_suffix(".summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonize standardized sumstats to target genotype variants."
    )
    parser.add_argument("--sumstats", required=True, help="Standardized sumstats table.")
    parser.add_argument("--target-pvar", required=True, help="QC target PLINK2 .pvar.")
    parser.add_argument("--out", required=True, help="Method-agnostic harmonized sumstats output.")
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--remove-ambiguous", type=parse_bool, default=True)
    parser.add_argument("--keep-only-acgt", type=parse_bool, default=True)
    parser.add_argument("--remove-duplicate-sumstats-variants", type=parse_bool, default=True)
    parser.add_argument("--target-pvar-chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    harmonized, summary = harmonize_sumstats_common_from_files(
        args.sumstats,
        args.target_pvar,
        remove_ambiguous=args.remove_ambiguous,
        keep_only_acgt=args.keep_only_acgt,
        remove_duplicate_sumstats_variants=args.remove_duplicate_sumstats_variants,
        target_pvar_chunksize=args.target_pvar_chunksize,
    )
    summary_out = Path(args.summary_out) if args.summary_out else default_summary_path(args.out)
    write_common_outputs(harmonized, summary, args.out, summary_out)
    if harmonized.empty:
        raise ValueError("Common harmonized sumstats output is empty")
    print(f"Wrote {len(harmonized)} common harmonized variants to {args.out}")
    print(f"Wrote summary to {summary_out}")


if __name__ == "__main__":
    main()
