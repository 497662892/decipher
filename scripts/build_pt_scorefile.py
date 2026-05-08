from __future__ import annotations

import argparse
import json
from pathlib import Path

from decipehr.pt import build_multithreshold_score_file
from decipehr.schema import parse_csv_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one PLINK2 P+T score file with one beta column per p-value threshold."
    )
    parser.add_argument("--harmonized", required=True)
    parser.add_argument("--clumps", required=True, help="PLINK2 .clumps output.")
    parser.add_argument("--p-thresholds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    p_thresholds = [float(value) for value in parse_csv_list(args.p_thresholds)]
    summary = build_multithreshold_score_file(
        args.harmonized,
        args.clumps,
        args.out,
        p_thresholds,
    )
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote P+T multi-threshold score file to {args.out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
