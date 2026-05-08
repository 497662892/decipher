from __future__ import annotations

import argparse
import json
from pathlib import Path

from decipehr.config import load_yaml
from decipehr.sumstats import standardize_sumstats_file_streaming
from decipehr.sumstats_harmonization import (
    harmonize_sumstats_common_from_files,
    write_common_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare method-agnostic standardized and harmonized sumstats for one disease."
    )
    parser.add_argument("--disease", required=True)
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument("--sumstats-root", required=True)
    parser.add_argument("--target-pvar", required=True)
    parser.add_argument("--out-dir", default="data/processed/sumstats")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--target-pvar-chunksize", type=int, default=1_000_000)
    parser.add_argument("--reuse-standardized", action="store_true")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest).get("summary_statistics", {})
    if args.disease not in manifest:
        raise ValueError(f"Disease {args.disease!r} is not present in {args.manifest}")
    entry = manifest[args.disease]
    if not isinstance(entry, dict):
        raise ValueError(f"Manifest entry for {args.disease} must be a mapping")

    accession = entry["accession"]
    raw_sumstats = Path(args.sumstats_root) / entry["file"]
    variant_id_strategy = entry.get("variant_id_strategy", "source")
    out_dir = Path(args.out_dir)
    standardized = out_dir / "standardized" / f"{args.disease}.{accession}.standardized.tsv.gz"
    harmonized = out_dir / "harmonized" / f"{args.disease}.{accession}.harmonized.tsv.gz"
    report = out_dir / "reports" / f"{args.disease}.{accession}.harmonized.summary.json"

    if args.reuse_standardized and standardized.exists() and standardized.stat().st_size > 0:
        print(f"Reusing standardized sumstats: {standardized}")
    else:
        n_standardized = standardize_sumstats_file_streaming(
            raw_sumstats,
            standardized,
            chunksize=args.chunksize,
            variant_id_strategy=variant_id_strategy,
        )
        print(f"Wrote {n_standardized} standardized variants to {standardized}")

    common, summary = harmonize_sumstats_common_from_files(
        standardized,
        args.target_pvar,
        target_pvar_chunksize=args.target_pvar_chunksize,
    )
    write_common_outputs(common, summary, harmonized, report)
    if common.empty:
        raise ValueError(f"Common harmonized sumstats output is empty for {args.disease}")
    print(f"Wrote {len(common)} common harmonized variants to {harmonized}")
    print(f"Wrote summary to {report}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
