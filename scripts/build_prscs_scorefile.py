from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from decipehr.io import read_table
from decipehr.schema import require_columns


def _json_dumps(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def read_prscs_effects(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        effect_path = Path(path)
        if not effect_path.exists() or effect_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing PRS-CS effect file: {effect_path}")
        frame = pd.read_csv(
            effect_path,
            sep=r"\s+",
            header=None,
            names=["CHR", "rsid", "BP", "A1", "A2", "BETA"],
            usecols=[0, 1, 2, 3, 4, 5],
        )
        frame["effect_file"] = str(effect_path)
        frames.append(frame)
    if not frames:
        raise ValueError("No PRS-CS effect files were provided")
    return pd.concat(frames, ignore_index=True)


def expand_effect_files(patterns: list[str]) -> list[str]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map PRS-CS posterior effects from rsID back to target IDs for PLINK2 scoring."
    )
    parser.add_argument("--effect-files", nargs="+", required=True)
    parser.add_argument("--target-id-map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    effect_files = expand_effect_files(args.effect_files)
    effects = read_prscs_effects(effect_files)
    target_map = read_table(args.target_id_map)
    require_columns(target_map, ["rsid", "target_snp"], "PRS-CS target ID map")

    merged = effects.merge(target_map, on="rsid", how="inner")
    duplicated = merged["target_snp"].duplicated(keep=False)
    merged = merged.loc[~duplicated].copy()
    if merged.empty:
        raise ValueError("No PRS-CS effects could be mapped to target IDs")

    score = merged[["target_snp", "A1", "BETA"]].rename(columns={"target_snp": "SNP"})
    score["BETA"] = pd.to_numeric(score["BETA"], errors="coerce")
    score = score.dropna(subset=["SNP", "A1", "BETA"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    score.to_csv(args.out, sep="\t", index=False)

    summary = {
        "n_effect_files": len(effect_files),
        "n_effect_rows": int(len(effects)),
        "n_mapped_rows_before_duplicate_filter": int(len(merged) + duplicated.sum()),
        "n_duplicate_target_rows_removed": int(duplicated.sum()),
        "n_score_rows": int(len(score)),
        "score_file": str(args.out),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(_json_dumps(summary), encoding="utf-8")
    print(f"Wrote PRS-CS PLINK2 score file to {args.out}")


if __name__ == "__main__":
    main()
