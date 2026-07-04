from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.io import read_table, write_table

ALIASES = {
    "SNP": [
        "SNP",
        "ID",
        "RSID",
        "RS_ID",
        "MARKER",
        "MARKERNAME",
        "VARIANT",
        "HM_VARIANT_ID",
        "VARIANT_ID",
    ],
    "CHR": ["CHR", "CHROM", "CHROMOSOME", "#CHROM", "HM_CHROM"],
    "BP": ["BP", "POS", "POSITION", "BASE_PAIR_LOCATION", "HM_POS"],
    "A1": ["A1", "EA", "EFFECT_ALLELE", "HM_EFFECT_ALLELE", "ALT", "ALLELE1"],
    "A2": ["A2", "NEA", "OTHER_ALLELE", "HM_OTHER_ALLELE", "NON_EFFECT_ALLELE", "REF", "ALLELE0"],
    "BETA": ["BETA", "HM_BETA", "EFFECT", "EFFECT_SIZE", "LOG_OR"],
    "OR": ["OR", "ODDS_RATIO"],
    "SE": ["SE", "STDERR", "STANDARD_ERROR"],
    "P": ["P", "PVAL", "P_VALUE", "PVALUE"],
    "N": ["N", "N_TOTAL", "TOTAL_N", "SAMPLESIZE", "SAMPLE_SIZE"],
}


def _column_lookup(columns: list[str]) -> dict[str, str]:
    return {col.upper(): col for col in columns}


def _canonical_lookup(columns: list[str]) -> dict[str, str]:
    upper_to_original = _column_lookup(columns)
    mapping: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias.upper() in upper_to_original:
                mapping[canonical] = upper_to_original[alias.upper()]
                break
    return mapping


def _variant_kind(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1:
        return "snv"
    if len(ref) < len(alt):
        return "ins"
    if len(ref) > len(alt):
        return "del"
    return "var"


def _decipehr_ids_from_hm_variant_id(df: pd.DataFrame, hm_variant_col: str) -> pd.Series:
    parts = df[hm_variant_col].astype("string").str.split("_", n=3, expand=True)
    if parts.shape[1] != 4:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    chrom = parts[0].astype("string").str.replace("chr", "", case=False, regex=False)
    pos = parts[1].astype("string")
    ref = parts[2].astype("string").str.upper()
    alt = parts[3].astype("string").str.upper()
    kind = [_variant_kind(str(r), str(a)) for r, a in zip(ref, alt, strict=False)]
    return "c" + chrom + "_" + pos + "_" + pd.Series(kind, index=df.index) + "_" + ref + "_" + alt


def _decipehr_ids_from_alt_column(
    out: pd.DataFrame,
    df: pd.DataFrame,
    alt_col: str,
) -> pd.Series:
    alt = df[alt_col].astype("string").str.upper()
    a1 = out["A1"].astype("string").str.upper()
    a2 = out["A2"].astype("string").str.upper()
    ref = pd.Series(pd.NA, index=df.index, dtype="string")
    ref = ref.where(alt.ne(a1), a2)
    ref = ref.where(alt.ne(a2), a1)
    has_ref_alt = ref.notna() & alt.notna() & ref.ne(alt)

    ids = pd.Series(pd.NA, index=df.index, dtype="string")
    kind = [_variant_kind(str(r), str(a)) for r, a in zip(ref, alt, strict=False)]
    ids.loc[has_ref_alt] = (
        "c"
        + out.loc[has_ref_alt, "CHR"].astype("string")
        + "_"
        + out.loc[has_ref_alt, "BP"].astype("string")
        + "_"
        + pd.Series(kind, index=df.index).loc[has_ref_alt]
        + "_"
        + ref.loc[has_ref_alt]
        + "_"
        + alt.loc[has_ref_alt]
    )
    return ids


def infer_decipehr_variant_ids(df: pd.DataFrame, standardized: pd.DataFrame) -> pd.Series:
    lookup = _column_lookup(list(df.columns))
    if "HM_VARIANT_ID" in lookup:
        return _decipehr_ids_from_hm_variant_id(df, lookup["HM_VARIANT_ID"])
    if "ALT" in lookup:
        return _decipehr_ids_from_alt_column(standardized, df, lookup["ALT"])
    return pd.Series(pd.NA, index=df.index, dtype="string")


def standardize_sumstats(
    df: pd.DataFrame,
    sort: bool = True,
    variant_id_strategy: str = "source",
) -> pd.DataFrame:
    if variant_id_strategy not in {"source", "decipehr"}:
        raise ValueError("variant_id_strategy must be 'source' or 'decipehr'")
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

    out["CHR"] = out["CHR"].astype(str).str.replace("chr", "", case=False, regex=False)
    out["BP"] = pd.to_numeric(out["BP"], errors="coerce").astype("Int64")
    out["A1"] = out["A1"].astype(str).str.upper()
    out["A2"] = out["A2"].astype(str).str.upper()
    if variant_id_strategy == "decipehr":
        out["SNP"] = infer_decipehr_variant_ids(df, out)
    out["SNP"] = out["SNP"].astype("string").str.strip()
    out.loc[out["SNP"].isin(["", "nan", "<NA>"]), "SNP"] = pd.NA
    out["P"] = pd.to_numeric(out["P"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["SNP", "CHR", "BP", "A1", "A2", "BETA", "P"])
    out = out.drop_duplicates(subset=["SNP"], keep="first")
    if sort:
        out = out.sort_values(["CHR", "BP", "SNP"])
    return out.reset_index(drop=True)


def standardize_sumstats_file(
    input_path: str | Path,
    output_path: str | Path,
    variant_id_strategy: str = "source",
) -> pd.DataFrame:
    standardized = standardize_sumstats(read_table(input_path), variant_id_strategy=variant_id_strategy)
    write_table(standardized, output_path)
    return standardized


def standardize_sumstats_file_streaming(
    input_path: str | Path,
    output_path: str | Path,
    chunksize: int = 250_000,
    variant_id_strategy: str = "source",
) -> int:
    """Standardize large summary-statistics files without holding all rows in memory.

    Streaming output preserves input chunk order instead of applying a global sort.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1")

    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    total = 0
    first_chunk = True
    for chunk in read_table(input_path, chunksize=chunksize):
        standardized = standardize_sumstats(
            chunk,
            sort=False,
            variant_id_strategy=variant_id_strategy,
        )
        write_table(
            standardized,
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        total += len(standardized)
        first_chunk = False
    return total


def inventory_sumstats(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    suffixes = {".txt", ".tsv", ".csv", ".gz", ".sumstats"}
    rows = []
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix.lower() in suffixes or path.name.endswith(".tsv.gz")):
            rows.append({"path": str(path), "name": path.name, "size_bytes": path.stat().st_size})
    return pd.DataFrame(rows)
