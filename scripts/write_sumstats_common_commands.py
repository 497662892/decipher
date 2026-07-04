from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

from decipehr.config import load_yaml


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write method-agnostic summary-statistics standardization and harmonization commands."
    )
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument("--sumstats-root", required=True)
    parser.add_argument("--target-pvar", required=True, help="QC target PLINK2 .pvar.")
    parser.add_argument("--out-dir", default="data/processed/sumstats")
    parser.add_argument(
        "--standardized-source-dir",
        default=None,
        help="Optional directory containing precomputed standardized sumstats to reuse.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--target-pvar-chunksize", type=int, default=1_000_000)
    parser.add_argument("--reuse-standardized", action="store_true")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest).get("summary_statistics", {})
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Manifest must contain a non-empty summary_statistics mapping")

    out_dir = Path(args.out_dir)
    standardized_dir = out_dir / "standardized"
    harmonized_dir = out_dir / "harmonized"
    reports_dir = out_dir / "reports"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Method-agnostic common layer only: raw official sumstats -> standardized -> harmonized.",
        "# Downstream P+T, PRS-CS, and LDpred2 inputs should be derived from harmonized/*.tsv.gz.",
        f"mkdir -p {q(standardized_dir)} {q(harmonized_dir)} {q(reports_dir)}",
        f"test -s {q(args.target_pvar)} || {{ echo 'Missing target pvar: {args.target_pvar}' >&2; exit 1; }}",
    ]

    for disease, entry in manifest.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry for {disease} must be a mapping")
        raw_sumstats = Path(args.sumstats_root) / entry["file"]
        command = (
            f"python scripts/prepare_sumstats_common.py --disease {q(disease)} "
            f"--manifest {q(args.manifest)} --sumstats-root {q(args.sumstats_root)} "
            f"--target-pvar {q(args.target_pvar)} --out-dir {q(out_dir)} "
            f"--chunksize {args.chunksize} --target-pvar-chunksize {args.target_pvar_chunksize}"
        )
        if args.standardized_source_dir:
            accession = entry["accession"]
            standardized_source = (
                Path(args.standardized_source_dir) / f"{disease}.{accession}.standardized.tsv.gz"
            )
            command += f" --standardized-source-dir {q(args.standardized_source_dir)}"
        else:
            standardized_source = None
        if args.reuse_standardized:
            command += " --reuse-standardized"

        lines.append("")
        if standardized_source is not None:
            lines.append(
                f"test -s {q(standardized_source)} || "
                f"{{ echo 'Missing standardized source: {standardized_source}' >&2; exit 1; }}"
            )
        else:
            lines.append(
                f"test -s {q(raw_sumstats)} || "
                f"{{ echo 'Missing sumstats: {raw_sumstats}' >&2; exit 1; }}"
            )
        lines.append(command)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Wrote common sumstats command template for {len(manifest)} diseases to {path}")


if __name__ == "__main__":
    main()
