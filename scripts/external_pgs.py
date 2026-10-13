from __future__ import annotations

import argparse

from decipehr.external_pgs import (
    fetch_reviewed_weights,
    inventory_catalog,
    prepare_reference_frequencies,
    score_frozen_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and score frozen external PGS weights.")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="Snapshot Catalog metadata and create a review template.")
    inventory.add_argument("--manifest", default="configs/external_pgs.yaml")
    inventory.add_argument("--diseases", default="T2D,cad,bc")
    inventory.add_argument("--build", choices=["GRCh37", "GRCh38"], required=True)
    inventory.add_argument("--out-dir", required=True)
    fetch = commands.add_parser("fetch", help="Download reviewed weights and freeze SHA256 checksums.")
    fetch.add_argument("--inventory", required=True)
    fetch.add_argument("--review", required=True)
    fetch.add_argument("--out-dir", required=True)
    reference = commands.add_parser("prepare-reference", help="Estimate external-only frequencies using target IDs.")
    reference.add_argument("--target-pvar", required=True)
    reference.add_argument("--reference-pfile", required=True)
    reference.add_argument("--target-build", choices=["GRCh37", "GRCh38"], required=True)
    reference.add_argument("--reference-build", choices=["GRCh37", "GRCh38"], required=True)
    reference.add_argument("--reference-provenance", required=True)
    reference.add_argument("--out-dir", required=True)
    reference.add_argument("--plink2", default="plink2")
    reference.add_argument("--threads", type=int, default=4)
    reference.add_argument("--memory-mb", type=int, default=16000)
    score = commands.add_parser("score", help="Run PGS Catalog matching and PLINK2 scoring.")
    score.add_argument("--weight-lock", required=True)
    score.add_argument("--pfile", required=True)
    score.add_argument("--target-build", choices=["GRCh37", "GRCh38"], required=True)
    score.add_argument("--reference-freq", required=True, help="External PLINK2 .afreq, using target variant IDs.")
    score.add_argument("--reference-provenance", required=True, help="External frequency source and version.")
    score.add_argument("--out-dir", required=True)
    score.add_argument("--plink2", default="plink2")
    score.add_argument("--threads", type=int, default=4)
    score.add_argument("--memory-mb", type=int, default=16000)
    score.add_argument("--min-overlap", type=float, default=0.75)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory_catalog(
            args.manifest, args.out_dir, build=args.build, diseases=args.diseases.split(","),
        )
        print(f"Inventoried {len(result['resources'])} candidate scores; review is still required.")
    elif args.command == "fetch":
        result = fetch_reviewed_weights(args.inventory, args.review, args.out_dir)
        print(f"Froze {len(result['resources'])} external weight files.")
    elif args.command == "prepare-reference":
        result = prepare_reference_frequencies(
            target_pvar=args.target_pvar, reference_pfile=args.reference_pfile,
            target_build=args.target_build, reference_build=args.reference_build,
            reference_provenance=args.reference_provenance, out_dir=args.out_dir,
            plink2=args.plink2, threads=args.threads, memory_mb=args.memory_mb,
        )
        print(f"Prepared external-only frequencies for {result['n_variants']} target variants.")
    else:
        score_frozen_weights(
            args.weight_lock, pfile=args.pfile, target_build=args.target_build,
            reference_freq=args.reference_freq, reference_provenance=args.reference_provenance,
            out_dir=args.out_dir, plink2=args.plink2, threads=args.threads,
            min_overlap=args.min_overlap, memory_mb=args.memory_mb,
        )
        print(f"Wrote audited external PGS scores to {args.out_dir}")


if __name__ == "__main__":
    main()
