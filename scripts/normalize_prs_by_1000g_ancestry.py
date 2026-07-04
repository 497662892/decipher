from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from decipehr.ancestry_normalization import (
    DEFAULT_1000G_GROUPS,
    infer_score_columns,
    normalize_by_hard_label,
    normalize_by_soft_weights,
)
from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns


def _patient_id_frame(path: str, patient_col: str) -> pd.DataFrame:
    frame = read_table(path)
    if patient_col not in frame.columns and "IID" in frame.columns:
        frame = frame.rename(columns={"IID": patient_col})
    require_columns(frame, [patient_col], path)
    return frame


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize wide PRS score files by 1000G ancestry labels. Hard normalization "
            "uses hard-label-specific training mean/sd. Soft normalization uses weighted "
            "1000G group moments and each patient's soft weights."
        )
    )
    parser.add_argument("--train-prs", required=True)
    parser.add_argument("--validation-prs", required=True)
    parser.add_argument("--hard-labels", required=True)
    parser.add_argument("--soft-weights", required=True)
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--score-cols", default="")
    parser.add_argument("--hard-col", default="hard_label")
    parser.add_argument("--groups", default=",".join(DEFAULT_1000G_GROUPS))
    parser.add_argument("--min-hard-group-size", type=int, default=20)
    parser.add_argument("--min-soft-effective-weight", type=float, default=20.0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    train_scores = read_table(args.train_prs)
    validation_scores = read_table(args.validation_prs)
    score_cols = infer_score_columns(
        train_scores,
        patient_col=args.patient_col,
        explicit=parse_csv_list(args.score_cols),
    )
    require_columns(validation_scores, [args.patient_col, *score_cols], "validation PRS scores")

    hard_labels = _patient_id_frame(args.hard_labels, args.patient_col)
    soft_weights = _patient_id_frame(args.soft_weights, args.patient_col)
    groups = tuple(parse_csv_list(args.groups))
    if not groups:
        raise ValueError("--groups must not be empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hard = normalize_by_hard_label(
        train_scores=train_scores,
        validation_scores=validation_scores,
        hard_labels=hard_labels,
        patient_col=args.patient_col,
        score_cols=score_cols,
        hard_col=args.hard_col,
        min_group_size=args.min_hard_group_size,
    )
    soft = normalize_by_soft_weights(
        train_scores=train_scores,
        validation_scores=validation_scores,
        soft_weights=soft_weights,
        patient_col=args.patient_col,
        score_cols=score_cols,
        groups=groups,
        min_effective_weight=args.min_soft_effective_weight,
    )

    write_table(hard.train, out_dir / "hard_z.train_prs_scores.csv")
    write_table(hard.validation, out_dir / "hard_z.validation_prs_scores.csv")
    write_table(soft.train, out_dir / "soft_z.train_prs_scores.csv")
    write_table(soft.validation, out_dir / "soft_z.validation_prs_scores.csv")
    parameters = pd.concat([hard.parameters, soft.parameters], ignore_index=True)
    write_table(parameters, out_dir / "normalization_parameters.csv")
    _write_summary(
        out_dir / "normalization_summary.json",
        {
            "train_prs": args.train_prs,
            "validation_prs": args.validation_prs,
            "hard_labels": args.hard_labels,
            "soft_weights": args.soft_weights,
            "score_cols": score_cols,
            "hard": hard.summary,
            "soft": soft.summary,
        },
    )
    print(f"Wrote hard/soft ancestry-normalized PRS scores to {out_dir}")


if __name__ == "__main__":
    main()
