from __future__ import annotations

import argparse
import json

from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list
from decipehr.splits import make_patient_folds, summarize_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level stratified CV folds.")
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = parse_csv_list(args.labels)
    phenotype = read_table(args.phenotype)
    folds = make_patient_folds(phenotype, args.patient_col, labels, args.n_folds, args.seed)
    write_table(folds, args.out)
    summary = summarize_folds(phenotype, folds, args.patient_col, labels)
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

