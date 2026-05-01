from __future__ import annotations

import argparse

from decipehr.io import read_table, write_table
from decipehr.qc import label_balance, missingness_report
from decipehr.schema import parse_csv_list, require_columns, validate_binary_labels, validate_unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and copy phenotype/covariate data.")
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--covariates", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--missingness-report", default=None)
    parser.add_argument("--label-report", default=None)
    args = parser.parse_args()

    labels = parse_csv_list(args.labels)
    covariates = parse_csv_list(args.covariates)
    df = read_table(args.phenotype)
    require_columns(df, [args.patient_col, *labels, *covariates], "phenotype")
    validate_unique(df, args.patient_col, "phenotype")
    validate_binary_labels(df, labels)
    write_table(df, args.out)
    if args.missingness_report:
        write_table(missingness_report(df), args.missingness_report)
    if args.label_report:
        write_table(label_balance(df, labels), args.label_report)
    print(f"Wrote validated phenotype table to {args.out}")


if __name__ == "__main__":
    main()

