from __future__ import annotations

import argparse
import json

from decipehr.io import read_table
from decipehr.sample_sets import build_sample_set_tables, write_sample_set_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split QC genotype samples into labeled training and unlabelled validation sets."
    )
    parser.add_argument("--genotype-psam", required=True, help="QC genotype .psam file.")
    parser.add_argument("--phenotype", required=True, help="Prepared phenotype table.")
    parser.add_argument(
        "--validation-covariates",
        default=None,
        help="Optional official validation covariate table. If provided, validation keep is QC samples in this table.",
    )
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument(
        "--validation-patient-col",
        default=None,
        help="Patient ID column in --validation-covariates. Defaults to --patient-col.",
    )
    parser.add_argument(
        "--genotype-sample-col",
        default=None,
        help="Sample ID column in the .psam. Defaults to #IID/IID/sampleid/patient_id auto-detection.",
    )
    parser.add_argument("--out-dir", default="data/processed/sample_sets")
    args = parser.parse_args()

    genotype_samples = read_table(args.genotype_psam)
    phenotype = read_table(args.phenotype)
    validation_samples = read_table(args.validation_covariates) if args.validation_covariates else None
    tables = build_sample_set_tables(
        genotype_samples,
        phenotype,
        patient_col=args.patient_col,
        genotype_sample_col=args.genotype_sample_col,
        validation_samples=validation_samples,
        validation_patient_col=args.validation_patient_col,
    )
    paths = write_sample_set_tables(tables, args.out_dir, patient_col=args.patient_col)
    print(json.dumps(tables.counts.to_dict("records"), indent=2))
    print("Wrote sample set files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
