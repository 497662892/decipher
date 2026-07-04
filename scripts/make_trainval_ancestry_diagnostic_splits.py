from __future__ import annotations

import argparse
import json
from pathlib import Path

from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns, validate_unique
from decipehr.splits import (
    make_balanced_patient_folds,
    patient_fold_balance_report,
    summarize_folds,
)


def _read_hard_labels(path: str, patient_col: str, hard_col: str):
    hard = read_table(path)
    if patient_col not in hard.columns and "IID" in hard.columns:
        hard = hard.rename(columns={"IID": patient_col})
    optional = [column for column in ["nearest_1000G_superpop", "mahal_margin"] if column in hard.columns]
    require_columns(hard, [patient_col, hard_col], "hard ancestry labels")
    validate_unique(hard, patient_col, "hard ancestry labels")
    return hard[[patient_col, hard_col, *optional]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create one shared train-val 5-fold split balanced by 1000G hard ancestry "
            "label and the multi-disease diagnostic label pattern."
        )
    )
    parser.add_argument("--phenotype", default="data/processed/final/train_val_phenotype.csv")
    parser.add_argument(
        "--hard-labels",
        default=(
            "outputs/final_ancestry_1000g/combined/"
            "target_1000G_hard_labels_pc10_margin0.5.train_val.tsv"
        ),
    )
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--labels", default="cad,bc,T2D")
    parser.add_argument("--hard-col", default="hard_label")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-candidates", type=int, default=1000)
    parser.add_argument(
        "--out",
        default="data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.csv",
    )
    parser.add_argument(
        "--diagnostics-out",
        default="data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.diagnostics.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.summary.json",
    )
    args = parser.parse_args()

    labels = parse_csv_list(args.labels)
    phenotype = read_table(args.phenotype)
    hard = _read_hard_labels(args.hard_labels, args.patient_col, args.hard_col)
    require_columns(phenotype, [args.patient_col, *labels], "train-val phenotype")
    validate_unique(phenotype, args.patient_col, "train-val phenotype")

    merged = phenotype.merge(hard, on=args.patient_col, how="left", validate="one_to_one")
    missing = merged[args.hard_col].isna()
    if missing.any():
        examples = merged.loc[missing, args.patient_col].head(5).astype(str).tolist()
        raise ValueError(f"Phenotype rows missing hard ancestry labels: {examples}")

    folds = make_balanced_patient_folds(
        merged,
        patient_col=args.patient_col,
        label_cols=labels,
        categorical_cols=[args.hard_col],
        n_folds=args.n_folds,
        seed=args.seed,
        n_seed_candidates=args.seed_candidates,
    )
    out = folds.merge(
        merged[[args.patient_col, args.hard_col, *labels]],
        on=args.patient_col,
        how="left",
        validate="one_to_one",
    )
    out["diagnostic_label"] = out[labels].astype(int).astype(str).agg("_".join, axis=1)
    out["diagnostic_burden"] = out[labels].astype(int).sum(axis=1)
    ordered_cols = [
        args.patient_col,
        "fold",
        args.hard_col,
        "diagnostic_label",
        "diagnostic_burden",
        *labels,
    ]
    out = out[ordered_cols].sort_values([args.patient_col]).reset_index(drop=True)

    out_path = Path(args.out)
    diagnostics_path = Path(args.diagnostics_out)
    summary_path = Path(args.summary_out)
    write_table(out, out_path)
    diagnostics = patient_fold_balance_report(
        out,
        out[[args.patient_col, "fold"]],
        patient_col=args.patient_col,
        label_cols=labels,
        categorical_cols=[args.hard_col, "diagnostic_label"],
    )
    write_table(diagnostics, diagnostics_path)
    summary = summarize_folds(out, out[[args.patient_col, "fold"]], args.patient_col, labels)
    summary_payload = {
        "phenotype": args.phenotype,
        "hard_labels": args.hard_labels,
        "patient_col": args.patient_col,
        "labels": labels,
        "hard_col": args.hard_col,
        "n_patients": summary.n_patients,
        "n_folds": summary.n_folds,
        "label_counts": summary.label_counts,
        "hard_label_counts": out[args.hard_col].value_counts().sort_index().astype(int).to_dict(),
        "fold_counts": out["fold"].value_counts().sort_index().astype(int).to_dict(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote train-val ancestry/diagnostic folds to {out_path}")
    print(f"Wrote split diagnostics to {diagnostics_path}")
    print(f"Wrote split summary to {summary_path}")


if __name__ == "__main__":
    main()
