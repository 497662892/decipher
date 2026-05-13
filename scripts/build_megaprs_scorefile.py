from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PREDICTOR_ALIASES = ["Predictor", "SNP", "ID", "rsid"]
EFFECT_ALLELE_ALIASES = ["A1", "Allele1", "EffectAllele"]
EFFECT_ALIASES = ["Effect", "BETA", "Beta", "beta", "A1Effect"]
NON_EFFECT_COLUMNS = {"Predictor", "SNP", "ID", "rsid", "A1", "A2", "Centre", "Center"}


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


def _effect_column(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    for alias in EFFECT_ALIASES:
        if alias in columns:
            return alias
    lower = {col.lower(): col for col in columns}
    for alias in EFFECT_ALIASES:
        if alias.lower() in lower:
            return lower[alias.lower()]
    numeric_candidates = []
    for col in columns:
        if col in NON_EFFECT_COLUMNS:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().any():
            numeric_candidates.append(col)
    if numeric_candidates:
        return numeric_candidates[-1]
    raise ValueError(f"MegaPRS effects missing any effect column: {EFFECT_ALIASES}")


def read_megaprs_effects(path: str | Path) -> pd.DataFrame:
    effects_path = Path(path)
    if not effects_path.exists() or effects_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing MegaPRS effects file: {effects_path}")

    frame = pd.read_csv(effects_path, sep=r"\s+", engine="python")
    predictor_col = _first_existing(list(frame.columns), PREDICTOR_ALIASES, "MegaPRS effects")
    a1_col = _first_existing(list(frame.columns), EFFECT_ALLELE_ALIASES, "MegaPRS effects")
    beta_col = _effect_column(frame)
    out = frame[[predictor_col, a1_col, beta_col]].copy()
    out.columns = ["SNP", "A1", "BETA"]
    out["SNP"] = out["SNP"].astype("string")
    out["A1"] = out["A1"].astype("string").str.upper()
    out["BETA"] = pd.to_numeric(out["BETA"], errors="coerce")
    out["effect_file"] = str(effects_path)
    return out.dropna(subset=["SNP", "A1", "BETA"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PLINK2 score file from LDAK MegaPRS effects.")
    parser.add_argument("--effects", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    effects = read_megaprs_effects(args.effects)
    duplicated = effects["SNP"].duplicated(keep=False)
    score = effects.loc[~duplicated, ["SNP", "A1", "BETA"]].copy()
    if score.empty:
        raise ValueError("No MegaPRS effects remained after duplicate filtering")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    score.to_csv(args.out, sep="\t", index=False)
    summary = {
        "effects": args.effects,
        "n_effect_rows": int(len(effects)),
        "n_duplicate_snp_rows_removed": int(duplicated.sum()),
        "n_score_rows": int(len(score)),
        "score_file": str(args.out),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(_json_dumps(summary), encoding="utf-8")
    print(f"Wrote MegaPRS PLINK2 score file to {args.out}")


if __name__ == "__main__":
    main()
