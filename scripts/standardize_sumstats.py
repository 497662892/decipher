from __future__ import annotations

import argparse

from decipehr.sumstats import standardize_sumstats_file, standardize_sumstats_file_streaming


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize GWAS summary statistics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per chunk for large text inputs; set to 0 to load the full file in memory.",
    )
    parser.add_argument(
        "--variant-id-strategy",
        choices=["source", "decipehr"],
        default="source",
        help="Use source IDs, or infer decipEHR PLINK IDs such as c1_817186_snv_G_A.",
    )
    args = parser.parse_args()
    if args.chunksize:
        n_variants = standardize_sumstats_file_streaming(
            args.input,
            args.output,
            chunksize=args.chunksize,
            variant_id_strategy=args.variant_id_strategy,
        )
    else:
        n_variants = len(
            standardize_sumstats_file(
                args.input,
                args.output,
                variant_id_strategy=args.variant_id_strategy,
            )
        )
    print(f"Wrote {n_variants} standardized variants to {args.output}")


if __name__ == "__main__":
    main()
