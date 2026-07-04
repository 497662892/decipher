from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.pt import p_threshold_column, plink_sscore_columns, read_plink_sscore
from decipehr.schema import require_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PLINK2 P+T .sscore files.")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument(
        "--sscore-column",
        default="sscore",
        help="Manifest column containing the PLINK2 .sscore path to collect.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = read_table(args.task_manifest)
    require_columns(
        manifest,
        ["disease", "clump_r2", "p_thresholds", args.sscore_column],
        "P+T task manifest",
    )

    rows = []
    for task in manifest.to_dict("records"):
        sscore = Path(task[args.sscore_column])
        if not sscore.exists() or sscore.stat().st_size == 0:
            raise FileNotFoundError(f"Missing PLINK2 score output: {sscore}")
        p_thresholds = [float(value) for value in str(task["p_thresholds"]).split(",") if value]
        frame = read_plink_sscore(sscore, patient_col=args.patient_col)
        score_names = [p_threshold_column(p_threshold) for p_threshold in p_thresholds]
        score_cols = plink_sscore_columns(list(frame.columns), len(p_thresholds), score_names)
        for p_threshold, score_col in zip(p_thresholds, score_cols, strict=True):
            part = frame[[args.patient_col, score_col]].copy()
            part = part.rename(columns={score_col: "prs"})
            part["disease"] = task["disease"]
            part["model_scope"] = task.get("model_scope", "unknown")
            part["clump_r2"] = float(task["clump_r2"])
            part["p_threshold"] = float(p_threshold)
            part["sscore"] = str(sscore)
            rows.append(part)

    if not rows:
        raise ValueError("No P+T score rows were collected")
    out = pd.concat(rows, ignore_index=True)
    out["prs"] = pd.to_numeric(out["prs"], errors="coerce")
    out = out.dropna(subset=[args.patient_col, "prs"])
    write_table(
        out[
            [
                args.patient_col,
                "disease",
                "model_scope",
                "clump_r2",
                "p_threshold",
                "prs",
                "sscore",
            ]
        ],
        args.out,
    )
    print(f"Wrote collected P+T scores to {args.out}")


if __name__ == "__main__":
    main()
