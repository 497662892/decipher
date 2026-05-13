from __future__ import annotations

import argparse
from pathlib import Path

from decipehr.ancestry_harmonization import (
    match_target_to_reference,
    read_variant_table,
    write_ancestry_overlap_outputs,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build conservative genome-wide target/1000G overlap files for ancestry PCA. "
            "Matches by CHR:BP:unordered alleles and writes separate extract lists plus a "
            "target-to-1000G variant ID rename map."
        )
    )
    parser.add_argument("--target-variants", required=True, help="Target .pvar or .bim file.")
    parser.add_argument("--reference-variants", required=True, help="1000G .pvar or .bim file.")
    parser.add_argument("--target-format", default="auto", choices=["auto", "pvar", "bim"])
    parser.add_argument("--reference-format", default="auto", choices=["auto", "pvar", "bim"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="genome")
    parser.add_argument("--remove-ambiguous", type=parse_bool, default=True)
    args = parser.parse_args()

    target = read_variant_table(args.target_variants, args.target_format)
    reference = read_variant_table(args.reference_variants, args.reference_format)
    matched, summary = match_target_to_reference(
        target,
        reference,
        remove_ambiguous=args.remove_ambiguous,
    )
    paths = write_ancestry_overlap_outputs(
        matched,
        summary,
        out_dir=Path(args.out_dir),
        prefix=args.prefix,
    )
    print(
        "Wrote "
        f"{summary['n_matched']} matched variants to {paths['matched']} "
        f"({summary['n_target_rename_to_reference_id']} target IDs need 1000G renaming)"
    )


if __name__ == "__main__":
    main()
