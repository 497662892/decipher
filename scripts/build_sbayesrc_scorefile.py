from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SNP_ALIASES = ["Name", "SNP", "ID", "Id", "rsid"]
EFFECT_ALLELE_ALIASES = ["A1", "Allele1", "EffectAllele"]
EFFECT_ALIASES = ["A1Effect", "Effect", "BETA", "Beta", "b"]


def _json_dumps(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _first_existing(columns: list[str], aliases: list[str], context: str) -> str:
    for alias in aliases:
        if alias in columns:
            return alias
    lower = {col.lower(): col for col in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise ValueError(f"{context} missing any of columns: {aliases}")


def read_sbayesrc_snpres(path: str | Path) -> pd.DataFrame:
    snpres = Path(path)
    if not snpres.exists() or snpres.stat().st_size == 0:
        raise FileNotFoundError(f"Missing sBayesRC snpRes file: {snpres}")

    frame = pd.read_csv(snpres, sep=r"\s+", engine="python")
    try:
        snp_col = _first_existing(list(frame.columns), SNP_ALIASES, "sBayesRC snpRes")
        a1_col = _first_existing(
            list(frame.columns),
            EFFECT_ALLELE_ALIASES,
            "sBayesRC snpRes",
        )
        beta_col = _first_existing(list(frame.columns), EFFECT_ALIASES, "sBayesRC snpRes")
        out = frame[[snp_col, a1_col, beta_col]].copy()
        out.columns = ["SNP", "A1", "BETA"]
    except ValueError:
        frame = pd.read_csv(
            snpres,
            sep=r"\s+",
            header=None,
            usecols=[1, 4, 7],
            names=["SNP", "A1", "BETA"],
            engine="python",
        )
        out = frame

    out["SNP"] = out["SNP"].astype("string")
    out["A1"] = out["A1"].astype("string").str.upper()
    out["BETA"] = pd.to_numeric(out["BETA"], errors="coerce")
    out["effect_file"] = str(snpres)
    return out.dropna(subset=["SNP", "A1", "BETA"])


def map_effects_to_target_ids(effects: pd.DataFrame, target_id_map: str | Path | None) -> pd.DataFrame:
    if target_id_map is None:
        return effects
    mapping = pd.read_csv(target_id_map, sep="\t", dtype="string")
    required = {"rsid", "target_snp"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"sBayesRC target ID map missing columns: {sorted(missing)}")
    mapped = effects.merge(mapping[["rsid", "target_snp"]], left_on="SNP", right_on="rsid", how="inner")
    if mapped.empty:
        raise ValueError("No sBayesRC effects could be mapped from rsID to target IDs")
    mapped["SNP"] = mapped["target_snp"]
    return mapped.drop(columns=["rsid", "target_snp"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PLINK2 score file from GCTB sBayesRC snpRes.")
    parser.add_argument("--snpres", required=True)
    parser.add_argument("--target-id-map", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    effects = read_sbayesrc_snpres(args.snpres)
    n_effect_rows_before_mapping = len(effects)
    effects = map_effects_to_target_ids(effects, args.target_id_map)
    duplicated = effects["SNP"].duplicated(keep=False)
    score = effects.loc[~duplicated, ["SNP", "A1", "BETA"]].copy()
    if score.empty:
        raise ValueError("No sBayesRC effects remained after duplicate filtering")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    score.to_csv(args.out, sep="\t", index=False)
    summary = {
        "snpres": args.snpres,
        "target_id_map": args.target_id_map,
        "n_effect_rows": int(n_effect_rows_before_mapping),
        "n_effect_rows_after_target_mapping": int(len(effects)),
        "n_duplicate_snp_rows_removed": int(duplicated.sum()),
        "n_score_rows": int(len(score)),
        "score_file": str(args.out),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(_json_dumps(summary), encoding="utf-8")
    print(f"Wrote sBayesRC PLINK2 score file to {args.out}")


if __name__ == "__main__":
    main()
