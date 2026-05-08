from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from decipehr.config import load_yaml


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write commands to prepare official GWAS Catalog sumstats and P+T PRS scripts."
    )
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument("--sumstats-root", required=True)
    parser.add_argument("--pfile-prefix", required=True, help="QC'd genome-wide PLINK2 pfile prefix.")
    parser.add_argument("--processed-dir", default="data/processed/sumstats")
    parser.add_argument("--score-dir", default="outputs/pt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = load_yaml(args.manifest).get("summary_statistics", {})
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Manifest must contain a non-empty summary_statistics mapping")

    processed_dir = Path(args.processed_dir)
    score_dir = Path(args.score_dir)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# This command file prepares public/official summary statistics and PLINK2 scoring commands.",
        "# It does not read phenotype rows. Run the generated per-disease P+T scripts on an approved node.",
        f"mkdir -p {q(processed_dir)} {q(score_dir)}",
    ]

    for disease, entry in manifest.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry for {disease} must be a mapping")
        filename = entry["file"]
        variant_id_strategy = entry.get("variant_id_strategy", "source")
        raw_sumstats = Path(args.sumstats_root) / filename
        standardized = processed_dir / f"{disease}.{entry['accession']}.standardized.tsv.gz"
        commands = score_dir / disease / f"{disease}.pt.commands.sh"
        score_prefix = score_dir / disease / disease
        lines.extend(
            [
                "",
                f"mkdir -p {q(score_dir / disease)}",
                (
                    f"python scripts/standardize_sumstats.py --input {q(raw_sumstats)} "
                    f"--output {q(standardized)} --variant-id-strategy {q(variant_id_strategy)}"
                ),
                (
                    f"python scripts/write_pt_commands.py --genotype-kind pfile "
                    f"--bfile-prefix {q(args.pfile_prefix)} --sumstats {q(standardized)} "
                    f"--score-prefix {q(score_prefix)} --out {q(commands)}"
                ),
            ]
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote official PRS command template for {len(manifest)} diseases to {path}")


if __name__ == "__main__":
    main()
