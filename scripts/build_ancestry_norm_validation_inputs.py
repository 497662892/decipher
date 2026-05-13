from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns, validate_unique

DEFAULT_EXCLUDE_COLS = {
    "FID",
    "#FID",
    "IID",
    "#IID",
    "sampleid",
    "patient_id",
    "sample_id",
    "split",
    "hard_label",
    "nearest_1000G_superpop",
    "soft_nearest_group",
    "max_weight",
    "weight_entropy",
}


def _patient_series(frame: pd.DataFrame, patient_col: str, context: str) -> pd.Series:
    if patient_col in frame.columns:
        return frame[patient_col].astype("string")
    for fallback in ["IID", "#IID", "sampleid", "patient_id", "sample_id"]:
        if fallback in frame.columns:
            return frame[fallback].astype("string")
    raise ValueError(f"{context} is missing patient column {patient_col!r} and common fallbacks")


def _score_columns(frame: pd.DataFrame, explicit: list[str], patient_col: str, context: str) -> list[str]:
    if explicit:
        require_columns(frame, explicit, context)
        return explicit
    excluded = set(DEFAULT_EXCLUDE_COLS)
    excluded.add(patient_col)
    score_cols = []
    for column in frame.columns:
        if column in excluded or column.startswith("w_") or column.startswith("d_mahal_"):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            score_cols.append(column)
    if not score_cols:
        raise ValueError(f"No numeric score columns could be inferred from {context}; pass --score-cols")
    return score_cols


def _setting_to_long(
    path: str | Path,
    *,
    setting: str,
    disease: str,
    patient_col: str,
    score_cols: list[str],
    context: str,
) -> pd.DataFrame:
    frame = read_table(path)
    patients = _patient_series(frame, patient_col, context)
    normalized = frame.copy()
    normalized[patient_col] = patients
    validate_unique(normalized, patient_col, context)
    cols = _score_columns(normalized, score_cols, patient_col, context)

    rows = []
    for column in cols:
        values = pd.to_numeric(normalized[column], errors="coerce")
        out = pd.DataFrame(
            {
                patient_col: normalized[patient_col],
                "disease": disease,
                "prs": values,
                "model": f"{column}__{setting}",
                "base_model": column,
                "normalization": setting,
                "source_score_file": str(path),
                "source_score_column": column,
            }
        )
        rows.append(out)
    long = pd.concat(rows, ignore_index=True)
    bad = long["prs"].isna()
    if bad.any():
        examples = long.loc[bad, [patient_col, "model", "source_score_column"]].head(5).to_dict(
            "records"
        )
        raise ValueError(f"{context} has nonnumeric or missing PRS values: {examples}")
    return long


def _parse_setting(raw: list[str]) -> tuple[str, str, str]:
    if len(raw) != 3:
        raise ValueError("--setting requires NAME TRAIN_SCORES VALIDATION_SCORES")
    name, train_path, validation_path = raw
    if not name.strip():
        raise ValueError("Setting name must not be empty")
    return name.strip(), train_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build validation-ready long PRS inputs from raw, hard-normalized, and "
            "soft-normalized wide score tables."
        )
    )
    parser.add_argument(
        "--setting",
        nargs=3,
        action="append",
        metavar=("NAME", "TRAIN_SCORES", "VALIDATION_SCORES"),
        required=True,
        help="Repeat for each normalization setting, e.g. raw train.tsv validation.tsv.",
    )
    parser.add_argument("--disease", required=True)
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument(
        "--score-cols",
        default="",
        help="Comma-separated score columns. If omitted, numeric non-ID columns are inferred.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-out", default=None)
    parser.add_argument("--validation-out", default=None)
    args = parser.parse_args()

    score_cols = parse_csv_list(args.score_cols)
    train_parts = []
    validation_parts = []
    for raw_setting in args.setting:
        setting, train_path, validation_path = _parse_setting(raw_setting)
        train_parts.append(
            _setting_to_long(
                train_path,
                setting=setting,
                disease=args.disease,
                patient_col=args.patient_col,
                score_cols=score_cols,
                context=f"{setting} training scores",
            )
        )
        validation_parts.append(
            _setting_to_long(
                validation_path,
                setting=setting,
                disease=args.disease,
                patient_col=args.patient_col,
                score_cols=score_cols,
                context=f"{setting} validation scores",
            )
        )

    out_dir = Path(args.out_dir)
    train_out = Path(args.train_out) if args.train_out else out_dir / f"{args.disease}.train_long.csv"
    validation_out = (
        Path(args.validation_out)
        if args.validation_out
        else out_dir / f"{args.disease}.validation_long.csv"
    )
    write_table(pd.concat(train_parts, ignore_index=True), train_out)
    write_table(pd.concat(validation_parts, ignore_index=True), validation_out)
    print(f"Wrote ancestry-normalization training long scores to {train_out}")
    print(f"Wrote ancestry-normalization validation long scores to {validation_out}")


if __name__ == "__main__":
    main()
