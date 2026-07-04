from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from decipehr.eligibility import load_eligibility_rules
from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns
from decipehr.splits import (
    disease_fold_balance_report,
    make_disease_specific_folds,
    make_patient_folds,
    summarize_disease_folds,
    summarize_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level stratified CV folds.")
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disease-specific", action="store_true")
    parser.add_argument("--ancestry-clusters", default=None)
    parser.add_argument("--cluster-col", default="ancestry_cluster")
    parser.add_argument("--age-col", default=None)
    parser.add_argument("--sex-col", default=None)
    parser.add_argument("--sex-balance-labels", default="cad,T2D")
    parser.add_argument("--eligibility-config", default=None)
    parser.add_argument("--seed-candidates", type=int, default=100)
    parser.add_argument("--diagnostics-out", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = parse_csv_list(args.labels)
    phenotype = read_table(args.phenotype)
    if args.ancestry_clusters:
        clusters = read_table(args.ancestry_clusters)
        require_columns(clusters, [args.patient_col, args.cluster_col], "ancestry_clusters")
        phenotype = phenotype.merge(clusters[[args.patient_col, args.cluster_col]], on=args.patient_col)

    if args.disease_specific:
        if not args.age_col:
            raise ValueError("--age-col is required with --disease-specific")
        folds = make_disease_specific_folds(
            phenotype=phenotype,
            patient_col=args.patient_col,
            label_cols=labels,
            cluster_col=args.cluster_col,
            age_col=args.age_col,
            sex_col=args.sex_col,
            sex_balance_labels=parse_csv_list(args.sex_balance_labels),
            eligibility_rules=load_eligibility_rules(args.eligibility_config),
            n_folds=args.n_folds,
            seed=args.seed,
            n_seed_candidates=args.seed_candidates,
        )
        write_table(folds, args.out)
        summaries = summarize_disease_folds(phenotype, folds, args.patient_col, labels)
        summary_rows = [asdict(summary) for summary in summaries]
        if args.diagnostics_out:
            write_table(
                disease_fold_balance_report(
                    phenotype,
                    folds,
                    args.patient_col,
                    labels,
                    cluster_col=args.cluster_col,
                    age_col=args.age_col,
                    sex_col=args.sex_col,
                ),
                args.diagnostics_out,
            )
        print(json.dumps(summary_rows, indent=2, sort_keys=True))
    else:
        folds = make_patient_folds(phenotype, args.patient_col, labels, args.n_folds, args.seed)
        write_table(folds, args.out)
        summary = summarize_folds(phenotype, folds, args.patient_col, labels)
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
