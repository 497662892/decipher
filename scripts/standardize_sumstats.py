from __future__ import annotations

import argparse

from decipehr.sumstats import standardize_sumstats_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize GWAS summary statistics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = standardize_sumstats_file(args.input, args.output)
    print(f"Wrote {len(out)} standardized variants to {args.output}")


if __name__ == "__main__":
    main()

