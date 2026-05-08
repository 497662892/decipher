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
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunksize", type=int, default=250_000)
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
        lines.extend(
            [
                "",
                f"test -s {q(raw_sumstats)} || {{ echo 'Missing sumstats: {raw_sumstats}' >&2; exit 1; }}",
                (
                    f"python scripts/prepare_sumstats_common.py --disease {q(disease)} "
                    f"--manifest {q(args.manifest)} --sumstats-root {q(args.sumstats_root)} "
                    f"--target-pvar {q(args.target_pvar)} --out-dir {q(out_dir)} "
                    f"--chunksize {args.chunksize}"
                ),
            ]
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Wrote common sumstats command template for {len(manifest)} diseases to {path}")


if __name__ == "__main__":
    main()
