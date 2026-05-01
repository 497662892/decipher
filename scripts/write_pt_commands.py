from __future__ import annotations

import argparse
from pathlib import Path

from decipehr.prs import pt_parameter_grid
from decipehr.schema import parse_csv_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Write PLINK2 P+T command templates.")
    parser.add_argument("--bfile-prefix", required=True, help="PLINK bed/bim/fam prefix or pfile prefix.")
    parser.add_argument("--genotype-kind", choices=["bfile", "pfile"], default="pfile")
    parser.add_argument("--sumstats", required=True, help="Standardized sumstats with SNP/A1/BETA/P columns.")
    parser.add_argument("--score-prefix", required=True)
    parser.add_argument("--p-thresholds", default="1,0.5,0.05,0.0005,0.000005,0.00000005")
    parser.add_argument("--r2-thresholds", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--window-kb", type=int, default=250)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    p_thresholds = [float(x) for x in parse_csv_list(args.p_thresholds)]
    r2_thresholds = [float(x) for x in parse_csv_list(args.r2_thresholds)]
    grid = pt_parameter_grid(p_thresholds, r2_thresholds)
    genotype_flag = "--pfile" if args.genotype_kind == "pfile" else "--bfile"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated command template. Inspect paths and columns before running.",
        "# Assumes standardized sumstats columns: SNP CHR BP A1 A2 P BETA.",
        "# PLINK2 .clumps output stores variant ID in column 3.",
    ]
    for _, row in grid.iterrows():
        tag = f"p{row.p_threshold:g}_r2{row.clump_r2:g}".replace(".", "p").replace("-", "m")
        clumped = f"{args.score_prefix}.{tag}.clumped"
        score_out = f"{args.score_prefix}.{tag}"
        lines.extend(
            [
                "",
                (
                    f"plink2 {genotype_flag} {args.bfile_prefix} "
                    f"--clump {args.sumstats} --clump-id-field SNP --clump-p-field P "
                    f"--clump-p1 {row.p_threshold} --clump-r2 {row.clump_r2} "
                    f"--clump-kb {args.window_kb} --out {clumped}"
                ),
                f"awk 'NR > 1 && $3 != \"\" {{print $3}}' {clumped}.clumps > {clumped}.snplist",
                (
                    f"plink2 {genotype_flag} {args.bfile_prefix} "
                    f"--score {args.sumstats} 1 4 7 header-read cols=+scoresums "
                    f"--extract {clumped}.snplist "
                    f"--out {score_out}"
                ),
            ]
        )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(grid)} P+T command pairs to {path}")


if __name__ == "__main__":
    main()

