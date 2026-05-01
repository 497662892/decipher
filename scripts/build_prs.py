from __future__ import annotations

import argparse

from decipehr.io import read_table, write_table
from decipehr.prs import compute_prs_from_dosage


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a small-matrix PRS prototype.")
    parser.add_argument("--genotype", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--score-col", default="prs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    score = compute_prs_from_dosage(
        read_table(args.genotype),
        read_table(args.weights),
        patient_col=args.patient_col,
    )
    if args.score_col != "prs":
        score = score.rename(columns={"prs": args.score_col})
    write_table(score, args.out)
    print(f"Wrote PRS for {len(score)} patients to {args.out}")


if __name__ == "__main__":
    main()

