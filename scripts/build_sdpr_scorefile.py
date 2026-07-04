from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def _json_dumps(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def expand_effect_files(patterns: list[str]) -> list[str]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return paths


def read_sdpr_effect(path: str | Path) -> pd.DataFrame:
    effect_path = Path(path)
    if not effect_path.exists() or effect_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing SDPR effect file: {effect_path}")

    frame = pd.read_csv(effect_path, sep=r"\s+", engine="python")
    lower_to_original = {col.lower(): col for col in frame.columns}
    if {"snp", "a1", "beta"}.issubset(lower_to_original):
        out = frame[
            [lower_to_original["snp"], lower_to_original["a1"], lower_to_original["beta"]]
        ].copy()
        out.columns = ["SNP", "A1", "BETA"]
    else:
        frame = pd.read_csv(
            effect_path,
            sep=r"\s+",
            header=None,
            names=["SNP", "A1", "BETA"],
            usecols=[0, 1, 2],
            engine="python",
        )
        out = frame

    out["A1"] = out["A1"].astype("string").str.upper()
    out["BETA"] = pd.to_numeric(out["BETA"], errors="coerce")
    out["effect_file"] = str(effect_path)
    return out.dropna(subset=["SNP", "A1", "BETA"])


def map_effects_to_target_ids(effects: pd.DataFrame, target_id_map: str | Path | None) -> pd.DataFrame:
    if target_id_map is None:
        return effects
    mapping = pd.read_csv(target_id_map, sep="\t", dtype="string")
    required = {"rsid", "target_snp"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"SDPR target ID map missing columns: {sorted(missing)}")
    mapped = effects.merge(mapping[["rsid", "target_snp"]], left_on="SNP", right_on="rsid", how="inner")
    if mapped.empty:
        raise ValueError("No SDPR effects could be mapped from rsID to target IDs")
    mapped["SNP"] = mapped["target_snp"]
    return mapped.drop(columns=["rsid", "target_snp"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PLINK2 score file from SDPR chromosome effects.")
    parser.add_argument("--effect-files", nargs="+", required=True)
    parser.add_argument("--target-id-map", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    effect_files = expand_effect_files(args.effect_files)
    effects = pd.concat([read_sdpr_effect(path) for path in effect_files], ignore_index=True)
    n_effect_rows_before_mapping = len(effects)
    effects = map_effects_to_target_ids(effects, args.target_id_map)
    duplicated = effects["SNP"].duplicated(keep=False)
    score = effects.loc[~duplicated, ["SNP", "A1", "BETA"]].copy()
    if score.empty:
        raise ValueError("No SDPR effects remained after duplicate filtering")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    score.to_csv(args.out, sep="\t", index=False)
    summary = {
        "n_effect_files": len(effect_files),
        "n_effect_rows": int(n_effect_rows_before_mapping),
        "n_effect_rows_after_target_mapping": int(len(effects)),
        "target_id_map": args.target_id_map,
        "n_duplicate_snp_rows_removed": int(duplicated.sum()),
        "n_score_rows": int(len(score)),
        "score_file": str(args.out),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(_json_dumps(summary), encoding="utf-8")
    print(f"Wrote SDPR PLINK2 score file to {args.out}")


if __name__ == "__main__":
    main()
