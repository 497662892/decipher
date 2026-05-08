from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import require_columns

VALID_ALLELES = {"A", "C", "G", "T"}
REQUIRED_SUMSTATS_COLUMNS = ["SNP", "CHR", "BP", "A1", "A2", "P", "BETA"]
TARGET_COLUMNS = [
    "target_snp",
    "target_chr",
    "target_bp",
    "target_ref",
    "target_alt",
    "CHR_NORM",
    "BP_INT",
]


def normalize_chrom(value: Any) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    text = text.upper()
    if text == "M":
        return "MT"
    return text


def normalize_allele(value: Any) -> str:
    return str(value).strip().upper()


def is_ambiguous(a1: str, a2: str) -> bool:
    pair = {a1, a2}
    return pair == {"A", "T"} or pair == {"C", "G"}


def read_target_pvar(path: str | Path) -> pd.DataFrame:
    """Read a PLINK2 .pvar and normalize the columns needed for sumstats matching."""
    path = Path(path)
    skiprows = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("##"):
                skiprows += 1
                continue
            break
    pvar = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=skiprows)
    if "CHROM" in pvar.columns and "#CHROM" not in pvar.columns:
        pvar = pvar.rename(columns={"CHROM": "#CHROM"})
    require_columns(pvar, ["#CHROM", "POS", "ID", "REF", "ALT"], "target pvar")

    out = pd.DataFrame(
        {
            "target_snp": pvar["ID"].astype("string").str.strip(),
            "target_chr": pvar["#CHROM"],
            "target_bp": pd.to_numeric(pvar["POS"], errors="coerce").astype("Int64"),
            "target_ref": pvar["REF"].map(normalize_allele),
            "target_alt": pvar["ALT"].map(normalize_allele),
        }
    )
    out["CHR_NORM"] = out["target_chr"].map(normalize_chrom)
    out["BP_INT"] = out["target_bp"]
    out.loc[out["target_snp"].isin(["", ".", "nan", "<NA>"]), "target_snp"] = pd.NA
    return out.dropna(subset=["target_snp", "CHR_NORM", "BP_INT"]).reset_index(drop=True)


def _pvar_skiprows(path: str | Path) -> int:
    skiprows = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("##"):
                skiprows += 1
                continue
            break
    return skiprows


def _iter_raw_pvar_chunks(path: str | Path, chunksize: int) -> tuple[pd.DataFrame, int]:
    if chunksize < 1:
        raise ValueError("target pvar chunksize must be at least 1")
    total = 0
    for chunk in pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=_pvar_skiprows(path),
        chunksize=chunksize,
    ):
        total += len(chunk)
        yield chunk, total


def _sumstats_id_set(sumstats: pd.DataFrame) -> set[str]:
    require_columns(sumstats, ["SNP"], "standardized summary statistics")
    values = sumstats["SNP"].astype("string").str.strip()
    values = values.dropna()
    values = values.loc[~values.isin(["", ".", "nan", "<NA>"])]
    return set(values.astype(str).tolist())


def _sumstats_position_index(sumstats: pd.DataFrame) -> pd.MultiIndex:
    require_columns(sumstats, ["CHR", "BP"], "standardized summary statistics")
    positions = pd.DataFrame(
        {
            "CHR_NORM": sumstats["CHR"].map(normalize_chrom),
            "BP_INT": pd.to_numeric(sumstats["BP"], errors="coerce").astype("Int64"),
        }
    ).dropna()
    positions["BP_INT"] = positions["BP_INT"].astype(int)
    positions = positions.drop_duplicates()
    return pd.MultiIndex.from_frame(positions[["CHR_NORM", "BP_INT"]])


def read_relevant_target_pvar(
    path: str | Path,
    sumstats: pd.DataFrame,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, int]:
    """Read only target pvar rows that can match standardized sumstats by ID or CHR:BP."""
    source_ids = _sumstats_id_set(sumstats)
    source_positions = _sumstats_position_index(sumstats)
    chunks = []
    n_total = 0
    for raw_chunk, n_seen in _iter_raw_pvar_chunks(path, chunksize):
        n_total = n_seen
        target = read_target_pvar_from_frame(raw_chunk)
        if target.empty:
            continue
        target_positions = pd.MultiIndex.from_frame(
            target[["CHR_NORM", "BP_INT"]].astype({"BP_INT": int})
        )
        mask = target["target_snp"].isin(source_ids) | target_positions.isin(source_positions)
        if mask.any():
            chunks.append(target.loc[mask].copy())
    if not chunks:
        empty = pd.DataFrame(columns=TARGET_COLUMNS)
        return empty, n_total
    return pd.concat(chunks, ignore_index=True).drop_duplicates().reset_index(drop=True), n_total


def _base_summary(sumstats: pd.DataFrame, target_pvar: pd.DataFrame) -> dict[str, int | float | bool]:
    duplicated_target_ids = int(
        target_pvar["target_snp"].astype("string").duplicated(keep=False).sum()
    )
    return {
        "n_sumstats_input": int(len(sumstats)),
        "n_target_pvar_input": int(len(target_pvar)),
        "n_after_required_columns": 0,
        "n_after_duplicate_removal": 0,
        "n_duplicate_sumstats_variant_rows": 0,
        "n_duplicate_target_pvar_ids": duplicated_target_ids,
        "n_removed_non_acgt": 0,
        "n_removed_ambiguous": 0,
        "n_removed_duplicate_or_multimatch": 0,
        "n_removed_allele_mismatch": 0,
        "n_unmatched_to_target_pvar": 0,
        "n_matched_to_target_pvar": 0,
        "n_matched_by_id": 0,
        "n_matched_by_position_alleles": 0,
        "n_harmonized_output": 0,
        "match_rate_after_duplicate_removal": 0.0,
        "strand_flipping_enabled": False,
    }


def _prepare_sumstats(
    sumstats: pd.DataFrame,
    remove_duplicate_sumstats_variants: bool,
    keep_only_acgt: bool,
    remove_ambiguous: bool,
    summary: dict[str, int | float | bool],
) -> pd.DataFrame:
    require_columns(sumstats, REQUIRED_SUMSTATS_COLUMNS, "standardized summary statistics")
    frame = sumstats.dropna(subset=REQUIRED_SUMSTATS_COLUMNS).copy()
    frame["BETA"] = pd.to_numeric(frame["BETA"], errors="coerce")
    frame["P"] = pd.to_numeric(frame["P"], errors="coerce")
    frame["BP_INT"] = pd.to_numeric(frame["BP"], errors="coerce").astype("Int64")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["BETA", "P", "BP_INT"])
    summary["n_after_required_columns"] = int(len(frame))

    frame["_source_index"] = np.arange(len(frame))
    frame["source_snp"] = frame["SNP"].astype("string").str.strip()
    frame["source_chr"] = frame["CHR"]
    frame["source_bp"] = frame["BP_INT"]
    frame["A1"] = frame["A1"].map(normalize_allele)
    frame["A2"] = frame["A2"].map(normalize_allele)
    frame["CHR_NORM"] = frame["CHR"].map(normalize_chrom)
    frame.loc[frame["source_snp"].isin(["", ".", "nan", "<NA>"]), "source_snp"] = pd.NA
    frame = frame.dropna(subset=["source_snp", "CHR_NORM", "BP_INT"])

    if remove_duplicate_sumstats_variants:
        duplicated = frame["source_snp"].duplicated(keep=False)
        summary["n_duplicate_sumstats_variant_rows"] = int(duplicated.sum())
        summary["n_removed_duplicate_or_multimatch"] = int(
            summary["n_removed_duplicate_or_multimatch"]
        ) + int(duplicated.sum())
        frame = frame.loc[~duplicated].copy()
    summary["n_after_duplicate_removal"] = int(len(frame))

    if keep_only_acgt:
        valid = frame["A1"].isin(VALID_ALLELES) & frame["A2"].isin(VALID_ALLELES)
        summary["n_removed_non_acgt"] = int(summary["n_removed_non_acgt"]) + int((~valid).sum())
        frame = frame.loc[valid].copy()
    if remove_ambiguous:
        ambiguous = frame.apply(lambda row: is_ambiguous(str(row["A1"]), str(row["A2"])), axis=1)
        summary["n_removed_ambiguous"] = int(summary["n_removed_ambiguous"]) + int(ambiguous.sum())
        frame = frame.loc[~ambiguous].copy()
    return frame


def _allele_pair_matches(frame: pd.DataFrame) -> pd.Series:
    same_order = (frame["A1"] == frame["target_ref"]) & (frame["A2"] == frame["target_alt"])
    reverse_order = (frame["A1"] == frame["target_alt"]) & (frame["A2"] == frame["target_ref"])
    return same_order | reverse_order


def _format_harmonized_rows(frame: pd.DataFrame, matched_by: str) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "SNP": frame["target_snp"].astype("string"),
            "CHR": frame["target_chr"],
            "BP": frame["target_bp"].astype("Int64"),
            "A1": frame["A1"],
            "A2": frame["A2"],
            "P": frame["P"],
            "BETA": frame["BETA"],
            "source_snp": frame["source_snp"],
            "source_chr": frame["source_chr"],
            "source_bp": frame["source_bp"].astype("Int64"),
            "target_snp": frame["target_snp"].astype("string"),
            "target_ref": frame["target_ref"],
            "target_alt": frame["target_alt"],
            "matched_by": matched_by,
        }
    )
    for optional in ["SE", "N"]:
        if optional in frame.columns:
            rows[optional] = frame[optional]
    return rows


def _valid_target_acgt(frame: pd.DataFrame) -> pd.Series:
    return frame["target_ref"].isin(VALID_ALLELES) & frame["target_alt"].isin(VALID_ALLELES)


def _harmonize_by_id(
    sumstats: pd.DataFrame,
    target_pvar: pd.DataFrame,
    keep_only_acgt: bool,
    summary: dict[str, int | float | bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_counts = target_pvar["target_snp"].value_counts(dropna=True)
    duplicate_target_ids = set(target_counts[target_counts > 1].index.astype(str))
    duplicate_source = sumstats["source_snp"].isin(duplicate_target_ids)
    summary["n_removed_duplicate_or_multimatch"] = int(
        summary["n_removed_duplicate_or_multimatch"]
    ) + int(duplicate_source.sum())
    candidates = sumstats.loc[~duplicate_source].copy()

    unique_targets = target_pvar.loc[
        ~target_pvar["target_snp"].duplicated(keep=False),
        ["target_snp", "target_chr", "target_bp", "target_ref", "target_alt"],
    ]
    merged = candidates.merge(
        unique_targets,
        left_on="source_snp",
        right_on="target_snp",
        how="left",
        sort=False,
    )
    has_id_match = merged["target_snp"].notna()
    id_matches = merged.loc[has_id_match].copy()
    unmatched = merged.loc[~has_id_match, sumstats.columns].copy()

    if id_matches.empty:
        return pd.DataFrame(), unmatched

    if keep_only_acgt:
        valid_target = _valid_target_acgt(id_matches)
        summary["n_removed_non_acgt"] = int(summary["n_removed_non_acgt"]) + int(
            (~valid_target).sum()
        )
        id_matches = id_matches.loc[valid_target].copy()
    allele_match = _allele_pair_matches(id_matches)
    summary["n_removed_allele_mismatch"] = int(summary["n_removed_allele_mismatch"]) + int(
        (~allele_match).sum()
    )
    id_matches = id_matches.loc[allele_match].copy()
    summary["n_matched_by_id"] = int(summary["n_matched_by_id"]) + int(len(id_matches))
    return _format_harmonized_rows(id_matches, "id"), unmatched


def _harmonize_by_position(
    sumstats: pd.DataFrame,
    target_pvar: pd.DataFrame,
    keep_only_acgt: bool,
    summary: dict[str, int | float | bool],
) -> pd.DataFrame:
    if sumstats.empty:
        return pd.DataFrame()
    target_cols = TARGET_COLUMNS
    merged = sumstats.merge(
        target_pvar[target_cols],
        on=["CHR_NORM", "BP_INT"],
        how="left",
        sort=False,
    )

    has_candidate = merged["target_snp"].notna()
    candidate_source_indexes = merged.loc[has_candidate, "_source_index"].unique()
    summary["n_unmatched_to_target_pvar"] = int(summary["n_unmatched_to_target_pvar"]) + int(
        (~sumstats["_source_index"].isin(candidate_source_indexes)).sum()
    )

    candidates = merged.loc[has_candidate].copy()
    if candidates.empty:
        return pd.DataFrame()

    if keep_only_acgt:
        valid_target = _valid_target_acgt(candidates)
        summary["n_removed_non_acgt"] = int(summary["n_removed_non_acgt"]) + int(
            (~valid_target).sum()
        )
        candidates = candidates.loc[valid_target].copy()
        if candidates.empty:
            return pd.DataFrame()

    matches = candidates.loc[_allele_pair_matches(candidates)].copy()
    summary["n_removed_allele_mismatch"] = int(summary["n_removed_allele_mismatch"]) + int(
        candidates["_source_index"].nunique() - matches["_source_index"].nunique()
    )
    if matches.empty:
        return pd.DataFrame()

    match_counts = matches.groupby("_source_index", sort=False)["_source_index"].transform("size")
    multimatch = match_counts > 1
    summary["n_removed_duplicate_or_multimatch"] = int(
        summary["n_removed_duplicate_or_multimatch"]
    ) + int(multimatch.sum())
    matched = matches.loc[~multimatch].copy()
    if matched.empty:
        return pd.DataFrame()

    summary["n_matched_by_position_alleles"] = int(summary["n_matched_by_position_alleles"]) + int(
        len(matched)
    )
    return _format_harmonized_rows(matched, "position_alleles")


def harmonize_sumstats_common(
    sumstats: pd.DataFrame,
    target_pvar: pd.DataFrame,
    remove_ambiguous: bool = True,
    keep_only_acgt: bool = True,
    remove_duplicate_sumstats_variants: bool = True,
) -> tuple[pd.DataFrame, dict[str, int | float | bool]]:
    """Harmonize standardized summary statistics to a target PLINK2 pvar.

    The output is method-agnostic: it keeps target IDs/coordinates, summary-statistic
    P/BETA columns, and source-vs-target audit columns. Method-specific scripts
    should derive P+T, PRS-CS, or LDpred2 inputs from this table.
    """
    target = target_pvar.copy()
    if not set(TARGET_COLUMNS).issubset(target.columns):
        require_columns(target, ["#CHROM", "POS", "ID", "REF", "ALT"], "target pvar")
        target = read_target_pvar_from_frame(target)
    summary = _base_summary(sumstats, target)
    prepared = _prepare_sumstats(
        sumstats,
        remove_duplicate_sumstats_variants=remove_duplicate_sumstats_variants,
        keep_only_acgt=keep_only_acgt,
        remove_ambiguous=remove_ambiguous,
        summary=summary,
    )
    id_rows, unmatched = _harmonize_by_id(prepared, target, keep_only_acgt, summary)
    position_rows = _harmonize_by_position(unmatched, target, keep_only_acgt, summary)
    harmonized = pd.concat([id_rows, position_rows], ignore_index=True)
    if not harmonized.empty:
        duplicated_output = harmonized["SNP"].duplicated(keep=False)
        summary["n_removed_duplicate_or_multimatch"] = int(
            summary["n_removed_duplicate_or_multimatch"]
        ) + int(duplicated_output.sum())
        harmonized = harmonized.loc[~duplicated_output].copy()
        harmonized = harmonized.sort_values(["CHR", "BP", "SNP"]).reset_index(drop=True)
    summary["n_harmonized_output"] = int(len(harmonized))
    summary["n_matched_to_target_pvar"] = int(len(harmonized))
    denominator = int(summary["n_after_duplicate_removal"])
    if denominator:
        summary["match_rate_after_duplicate_removal"] = len(harmonized) / denominator
    return harmonized, summary


def read_target_pvar_from_frame(pvar: pd.DataFrame) -> pd.DataFrame:
    if "CHROM" in pvar.columns and "#CHROM" not in pvar.columns:
        pvar = pvar.rename(columns={"CHROM": "#CHROM"})
    require_columns(pvar, ["#CHROM", "POS", "ID", "REF", "ALT"], "target pvar")
    out = pd.DataFrame(
        {
            "target_snp": pvar["ID"].astype("string").str.strip(),
            "target_chr": pvar["#CHROM"],
            "target_bp": pd.to_numeric(pvar["POS"], errors="coerce").astype("Int64"),
            "target_ref": pvar["REF"].map(normalize_allele),
            "target_alt": pvar["ALT"].map(normalize_allele),
        }
    )
    out["CHR_NORM"] = out["target_chr"].map(normalize_chrom)
    out["BP_INT"] = out["target_bp"]
    out.loc[out["target_snp"].isin(["", ".", "nan", "<NA>"]), "target_snp"] = pd.NA
    return out.dropna(subset=["target_snp", "CHR_NORM", "BP_INT"]).reset_index(drop=True)


def write_common_outputs(
    harmonized: pd.DataFrame,
    summary: dict[str, int | float | bool],
    out: str | Path,
    summary_out: str | Path,
) -> None:
    write_table(harmonized, out)
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_out).write_text(_json_dumps(summary), encoding="utf-8")


def harmonize_sumstats_common_from_files(
    sumstats_path: str | Path,
    target_pvar_path: str | Path,
    remove_ambiguous: bool = True,
    keep_only_acgt: bool = True,
    remove_duplicate_sumstats_variants: bool = True,
    target_pvar_chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, dict[str, int | float | bool]]:
    """Read standardized sumstats and harmonize using chunk-filtered target pvar rows."""
    sumstats = read_table(sumstats_path)
    target, n_target_total = read_relevant_target_pvar(
        target_pvar_path,
        sumstats,
        chunksize=target_pvar_chunksize,
    )
    harmonized, summary = harmonize_sumstats_common(
        sumstats,
        target,
        remove_ambiguous=remove_ambiguous,
        keep_only_acgt=keep_only_acgt,
        remove_duplicate_sumstats_variants=remove_duplicate_sumstats_variants,
    )
    summary["n_target_pvar_input"] = int(n_target_total)
    summary["n_target_pvar_relevant"] = int(len(target))
    return harmonized, summary


def _json_dumps(summary: dict[str, int | float | bool]) -> str:
    import json

    return json.dumps(summary, indent=2, sort_keys=True) + "\n"
