from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.pt import plink_sscore_columns, read_plink_sscore
from decipehr.schema import require_columns


def _score_column(columns: list[str]) -> str:
    for score_name in ["BETA", "prs"]:
        try:
            return plink_sscore_columns(columns, 1, [score_name])[0]
        except ValueError:
            pass
    sum_cols = [col for col in columns if col.upper().endswith("_SUM")]
    avg_cols = [col for col in columns if col.upper().endswith("_AVG")]
    for col in sum_cols + avg_cols:
        if col.upper() not in {"NAMED_ALLELE_DOSAGE_SUM", "ALLELE_CT"}:
            return col
    raise ValueError("PLINK .sscore output is missing a PRS score column")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PLINK2 .sscore files from MegaPRS tasks.")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--sscore-column", default="sscore")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = read_table(args.task_manifest)
    require_columns(
        manifest,
        ["disease", "model_scope", "model", args.sscore_column],
        "MegaPRS task manifest",
    )

    rows = []
    for task in manifest.to_dict("records"):
        sscore = Path(task[args.sscore_column])
        if not sscore.exists() or sscore.stat().st_size == 0:
            raise FileNotFoundError(f"Missing PLINK2 score output: {sscore}")
        frame = read_plink_sscore(sscore, patient_col=args.patient_col)
        score_col = _score_column(list(frame.columns))
        part = frame[[args.patient_col, score_col]].copy().rename(columns={score_col: "prs"})
        part["disease"] = task["disease"]
        part["model_scope"] = task["model_scope"]
        part["model"] = task["model"]
        part["sscore"] = str(sscore)
        rows.append(part)

    if not rows:
        raise ValueError("No MegaPRS score rows were collected")
    out = pd.concat(rows, ignore_index=True)
    out["prs"] = pd.to_numeric(out["prs"], errors="coerce")
    out = out.dropna(subset=[args.patient_col, "prs"])
    write_table(out[[args.patient_col, "disease", "model_scope", "model", "prs", "sscore"]], args.out)
    print(f"Wrote collected MegaPRS scores to {args.out}")


if __name__ == "__main__":
    main()
