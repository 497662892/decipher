from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import require_columns


def _coerce_numeric(frame: pd.DataFrame, columns: list[str], context: str) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    parameter_cols = [column for column in columns if column != "prs"]
    invalid = out[parameter_cols].isna().any(axis=1)
    if invalid.any():
        examples = out.loc[invalid, parameter_cols].head(5).to_dict("records")
        raise ValueError(f"Invalid numeric LDpred2 values in {context}: {examples}")
    missing_prs = out["prs"].isna()
    if missing_prs.any():
        print(
            f"Dropped {int(missing_prs.sum())} LDpred2 rows with missing PRS in {context}",
            file=sys.stderr,
        )
        out = out.loc[~missing_prs].copy()
    if out.empty:
        raise ValueError(f"No finite LDpred2 PRS rows remain in {context}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LDpred2 long score files.")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument(
        "--score-file-column",
        default="score_file",
        help="Manifest column containing the LDpred2 score CSV to collect.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = read_table(args.task_manifest)
    require_columns(manifest, ["disease", args.score_file_column], "LDpred2 task manifest")

    frames = []
    for task in manifest.to_dict("records"):
        score_file = Path(task[args.score_file_column])
        if not score_file.exists() or score_file.stat().st_size == 0:
            raise FileNotFoundError(f"Missing LDpred2 score output: {score_file}")
        frame = read_table(score_file)
        require_columns(
            frame,
            [args.patient_col, "disease", "h2", "p", "sparse", "prs"],
            f"LDpred2 score file {score_file}",
        )
        frame = _coerce_numeric(frame, ["h2", "p", "sparse", "prs"], str(score_file))
        duplicate_keys = [args.patient_col, "disease", "h2", "p", "sparse"]
        if "model" in frame.columns:
            duplicate_keys.append("model")
        if frame.duplicated(duplicate_keys).any():
            examples = frame.loc[frame.duplicated(duplicate_keys), duplicate_keys].head(5)
            raise ValueError(f"Duplicate LDpred2 score rows in {score_file}: {examples.to_dict('records')}")
        if "model_scope" not in frame.columns:
            frame["model_scope"] = task.get("model_scope", "unknown")
        frame["source_score_file"] = str(score_file)
        frames.append(frame)

    if not frames:
        raise ValueError("No LDpred2 score rows were collected")
    out = pd.concat(frames, ignore_index=True)
    columns = [
        args.patient_col,
        "disease",
        "model_scope",
        "h2",
        "p",
        "sparse",
        "prs",
        "source_score_file",
    ]
    if "fold" in out.columns:
        columns.insert(3, "fold")
    if "model" in out.columns:
        columns.insert(columns.index("prs"), "model")
    for optional in ["p_init", "chain"]:
        if optional in out.columns:
            columns.insert(columns.index("prs"), optional)
    write_table(out[columns], args.out)
    print(f"Wrote collected LDpred2 scores to {args.out}")


if __name__ == "__main__":
    main()
