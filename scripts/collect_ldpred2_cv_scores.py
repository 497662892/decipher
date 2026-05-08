from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import require_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect fold-specific LDpred2 score files.")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = read_table(args.task_manifest)
    require_columns(manifest, ["disease", "fold", "score_file"], "LDpred2 task manifest")

    frames = []
    for task in manifest.to_dict("records"):
        score_file = Path(task["score_file"])
        if not score_file.exists() or score_file.stat().st_size == 0:
            raise FileNotFoundError(f"Missing LDpred2 score output: {score_file}")
        frame = read_table(score_file)
        require_columns(
            frame,
            [args.patient_col, "disease", "fold", "h2", "p", "sparse", "prs"],
            f"LDpred2 score file {score_file}",
        )
        frame = frame.copy()
        frame["source_score_file"] = str(score_file)
        frames.append(frame)

    if not frames:
        raise ValueError("No LDpred2 score rows were collected")
    out = pd.concat(frames, ignore_index=True)
    out["prs"] = pd.to_numeric(out["prs"], errors="coerce")
    out["h2"] = pd.to_numeric(out["h2"], errors="coerce")
    out["p"] = pd.to_numeric(out["p"], errors="coerce")
    out["sparse"] = pd.to_numeric(out["sparse"], errors="coerce").astype("Int64")
    out = out.dropna(subset=[args.patient_col, "prs", "h2", "p", "sparse"])
    columns = [
        args.patient_col,
        "disease",
        "fold",
        "h2",
        "p",
        "sparse",
        "prs",
        "source_score_file",
    ]
    if "model" in out.columns:
        columns.insert(6, "model")
    write_table(out[columns], args.out)
    print(f"Wrote collected LDpred2 CV scores to {args.out}")


if __name__ == "__main__":
    main()
