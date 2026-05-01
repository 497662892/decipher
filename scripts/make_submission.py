from __future__ import annotations

import argparse

from decipehr.io import read_table, write_table
from decipehr.submission import validate_probability_columns, wide_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert long predictions into a wide submission file.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--probability-col", default="y_prob")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pred = read_table(args.predictions)
    sub = wide_submission(pred, args.patient_col, args.disease_col, args.probability_col)
    validate_probability_columns(sub, args.patient_col)
    write_table(sub, args.out)
    print(f"Wrote submission to {args.out}")


if __name__ == "__main__":
    main()

