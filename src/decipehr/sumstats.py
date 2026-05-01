from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.io import read_table, write_table

ALIASES = {
    "SNP": ["SNP", "ID", "RSID", "RS_ID", "MARKER", "MARKERNAME", "VARIANT", "VARIANT_ID"],
    "CHR": ["CHR", "CHROM", "CHROMOSOME", "#CHROM"],
    "BP": ["BP", "POS", "POSITION", "BASE_PAIR_LOCATION"],
    "A1": ["A1", "EA", "EFFECT_ALLELE", "ALT", "ALLELE1"],
    "A2": ["A2", "NEA", "OTHER_ALLELE", "NON_EFFECT_ALLELE", "REF", "ALLELE0"],
    "BETA": ["BETA", "EFFECT", "EFFECT_SIZE", "LOG_OR"],
    "OR": ["OR", "ODDS_RATIO"],
    "SE": ["SE", "STDERR", "STANDARD_ERROR"],
    "P": ["P", "PVAL", "P_VALUE", "PVALUE"],
    "N": ["N", "N_TOTAL", "TOTAL_N", "SAMPLESIZE", "SAMPLE_SIZE"],
}


def _canonical_lookup(columns: list[str]) -> dict[str, str]:
    upper_to_original = {col.upper(): col for col in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias.upper() in upper_to_original:
                mapping[canonical] = upper_to_original[alias.upper()]
                break
    return mapping


def standardize_sumstats(df: pd.DataFrame) -> pd.DataFrame:
    mapping = _canonical_lookup(list(df.columns))
    required = ["SNP", "CHR", "BP", "A1", "A2", "P"]
    missing = [col for col in required if col not in mapping]
    if missing:
        raise ValueError(f"Summary statistics missing required columns: {missing}")
    if "BETA" not in mapping and "OR" not in mapping:
        raise ValueError("Summary statistics need either BETA or OR")

    out = pd.DataFrame()
    for col in required:
        out[col] = df[mapping[col]]
    if "BETA" in mapping:
        out["BETA"] = pd.to_numeric(df[mapping["BETA"]], errors="coerce")
    else:
        out["BETA"] = np.log(pd.to_numeric(df[mapping["OR"]], errors="coerce"))
    if "SE" in mapping:
        out["SE"] = pd.to_numeric(df[mapping["SE"]], errors="coerce")
    if "N" in mapping:
        out["N"] = pd.to_numeric(df[mapping["N"]], errors="coerce")

    out["SNP"] = out["SNP"].astype(str)
    out["CHR"] = out["CHR"].astype(str).str.replace("chr", "", case=False, regex=False)
    out["BP"] = pd.to_numeric(out["BP"], errors="coerce").astype("Int64")
    out["A1"] = out["A1"].astype(str).str.upper()
    out["A2"] = out["A2"].astype(str).str.upper()
    out["P"] = pd.to_numeric(out["P"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["SNP", "CHR", "BP", "A1", "A2", "BETA", "P"])
    out = out.drop_duplicates(subset=["SNP"], keep="first")
    return out.sort_values(["CHR", "BP", "SNP"]).reset_index(drop=True)


def standardize_sumstats_file(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    standardized = standardize_sumstats(read_table(input_path))
    write_table(standardized, output_path)
    return standardized


def inventory_sumstats(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    suffixes = {".txt", ".tsv", ".csv", ".gz", ".sumstats"}
    rows = []
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix.lower() in suffixes or path.name.endswith(".tsv.gz")):
            rows.append({"path": str(path), "name": path.name, "size_bytes": path.stat().st_size})
    return pd.DataFrame(rows)

