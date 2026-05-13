from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from decipehr.io import ensure_parent, write_table
from decipehr.schema import require_columns
from decipehr.sumstats_harmonization import (
    VALID_ALLELES,
    is_ambiguous,
    normalize_allele,
    normalize_chrom,
)

AUTOSOMES = {str(chrom) for chrom in range(1, 23)}
VARIANT_COLUMNS = ["variant_id", "chrom", "bp", "allele1", "allele2"]


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _pvar_skiprows(path: str | Path) -> int:
    skiprows = 0
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("##"):
                skiprows += 1
                continue
            break
    return skiprows


def read_pvar_variants(path: str | Path) -> pd.DataFrame:
    """Read a PLINK2 .pvar as an unordered variant/allele table for reference matching."""
    pvar = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=_pvar_skiprows(path))
    if "CHROM" in pvar.columns and "#CHROM" not in pvar.columns:
        pvar = pvar.rename(columns={"CHROM": "#CHROM"})
    require_columns(pvar, ["#CHROM", "POS", "ID", "REF", "ALT"], "pvar")
    return pd.DataFrame(
        {
            "variant_id": pvar["ID"],
            "chrom": pvar["#CHROM"],
            "bp": pvar["POS"],
            "allele1": pvar["REF"],
            "allele2": pvar["ALT"],
        }
    )


def read_bim_variants(path: str | Path) -> pd.DataFrame:
    """Read a PLINK1 .bim as an unordered variant/allele table for reference matching."""
    bim = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["chrom", "variant_id", "cm", "bp", "allele1", "allele2"],
    )
    return bim[VARIANT_COLUMNS].copy()


def read_variant_table(path: str | Path, file_format: str = "auto") -> pd.DataFrame:
    """Read .pvar or .bim variants into normalized matching input columns."""
    path = Path(path)
    normalized = file_format.lower()
    if normalized == "auto":
        suffixes = [suffix.lower() for suffix in path.suffixes]
        if ".pvar" in suffixes:
            normalized = "pvar"
        elif ".bim" in suffixes:
            normalized = "bim"
        else:
            raise ValueError(f"Cannot infer variant file format from {path}; pass --*-format")
    if normalized == "pvar":
        return read_pvar_variants(path)
    if normalized == "bim":
        return read_bim_variants(path)
    raise ValueError("file_format must be auto, pvar, or bim")


def _blank_or_missing_id(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.isna() | text.isin(["", ".", "nan", "<NA>"])


def _allele_key(a1: pd.Series, a2: pd.Series) -> pd.Series:
    left = a1.astype(str)
    right = a2.astype(str)
    first = left.where(left <= right, right)
    second = right.where(left <= right, left)
    return first + "/" + second


def prepare_reference_variants(
    variants: pd.DataFrame,
    *,
    context: str,
    remove_ambiguous: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter variants to a conservative autosomal A/C/G/T biallelic SNP matching set."""
    require_columns(variants, VARIANT_COLUMNS, context)
    frame = variants.copy()
    summary: dict[str, int] = {
        "n_input": int(len(frame)),
        "n_removed_bad_id": 0,
        "n_removed_non_autosomal": 0,
        "n_removed_bad_position": 0,
        "n_removed_non_acgt_or_multiallelic": 0,
        "n_removed_same_allele": 0,
        "n_removed_ambiguous": 0,
        "n_removed_duplicate_ids": 0,
        "n_removed_duplicate_position_alleles": 0,
        "n_ready": 0,
    }

    frame["variant_id"] = frame["variant_id"].astype("string").str.strip()
    bad_id = _blank_or_missing_id(frame["variant_id"])
    summary["n_removed_bad_id"] = int(bad_id.sum())
    frame = frame.loc[~bad_id].copy()

    frame["CHR_NORM"] = frame["chrom"].map(normalize_chrom)
    non_autosomal = ~frame["CHR_NORM"].isin(AUTOSOMES)
    summary["n_removed_non_autosomal"] = int(non_autosomal.sum())
    frame = frame.loc[~non_autosomal].copy()
    frame["CHR_INT"] = frame["CHR_NORM"].astype(int)

    frame["BP_INT"] = pd.to_numeric(frame["bp"], errors="coerce").astype("Int64")
    bad_position = frame["BP_INT"].isna()
    summary["n_removed_bad_position"] = int(bad_position.sum())
    frame = frame.loc[~bad_position].copy()
    frame["BP_INT"] = frame["BP_INT"].astype(int)

    frame["A1_NORM"] = frame["allele1"].map(normalize_allele)
    frame["A2_NORM"] = frame["allele2"].map(normalize_allele)
    valid_alleles = frame["A1_NORM"].isin(VALID_ALLELES) & frame["A2_NORM"].isin(VALID_ALLELES)
    summary["n_removed_non_acgt_or_multiallelic"] = int((~valid_alleles).sum())
    frame = frame.loc[valid_alleles].copy()

    same_allele = frame["A1_NORM"] == frame["A2_NORM"]
    summary["n_removed_same_allele"] = int(same_allele.sum())
    frame = frame.loc[~same_allele].copy()

    if remove_ambiguous:
        ambiguous = frame.apply(lambda row: is_ambiguous(str(row["A1_NORM"]), str(row["A2_NORM"])), axis=1)
        summary["n_removed_ambiguous"] = int(ambiguous.sum())
        frame = frame.loc[~ambiguous].copy()

    duplicate_ids = frame["variant_id"].duplicated(keep=False)
    summary["n_removed_duplicate_ids"] = int(duplicate_ids.sum())
    frame = frame.loc[~duplicate_ids].copy()

    frame["allele_key"] = _allele_key(frame["A1_NORM"], frame["A2_NORM"])
    duplicate_position = frame.duplicated(["CHR_NORM", "BP_INT", "allele_key"], keep=False)
    summary["n_removed_duplicate_position_alleles"] = int(duplicate_position.sum())
    frame = frame.loc[~duplicate_position].copy()

    out = frame[
        ["variant_id", "CHR_NORM", "CHR_INT", "BP_INT", "A1_NORM", "A2_NORM", "allele_key"]
    ].copy()
    out = out.sort_values(["CHR_INT", "BP_INT", "variant_id"], kind="mergesort").reset_index(
        drop=True
    )
    summary["n_ready"] = int(len(out))
    return out, summary


def match_target_to_reference(
    target_variants: pd.DataFrame,
    reference_variants: pd.DataFrame,
    *,
    remove_ambiguous: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match target variants to reference variants by chromosome, position, and allele pair."""
    target, target_summary = prepare_reference_variants(
        target_variants,
        context="target variants",
        remove_ambiguous=remove_ambiguous,
    )
    reference, reference_summary = prepare_reference_variants(
        reference_variants,
        context="reference variants",
        remove_ambiguous=remove_ambiguous,
    )

    merged = target.merge(
        reference,
        on=["CHR_NORM", "CHR_INT", "BP_INT", "allele_key"],
        how="inner",
        suffixes=("_target", "_ref"),
        sort=False,
    )
    if merged.empty:
        matched = pd.DataFrame(
            columns=[
                "target_id",
                "ref_id",
                "CHR",
                "BP",
                "target_allele1",
                "target_allele2",
                "ref_allele1",
                "ref_allele2",
                "allele_key",
                "same_allele_order",
                "same_variant_id",
            ]
        )
    else:
        matched = pd.DataFrame(
            {
                "target_id": merged["variant_id_target"],
                "ref_id": merged["variant_id_ref"],
                "CHR": merged["CHR_NORM"],
                "BP": merged["BP_INT"],
                "target_allele1": merged["A1_NORM_target"],
                "target_allele2": merged["A2_NORM_target"],
                "ref_allele1": merged["A1_NORM_ref"],
                "ref_allele2": merged["A2_NORM_ref"],
                "allele_key": merged["allele_key"],
            }
        )
        matched["same_allele_order"] = (
            (matched["target_allele1"] == matched["ref_allele1"])
            & (matched["target_allele2"] == matched["ref_allele2"])
        )
        matched["same_variant_id"] = matched["target_id"] == matched["ref_id"]
        matched["_CHR_INT"] = merged["CHR_INT"]
        matched = matched.sort_values(["_CHR_INT", "BP", "target_id"], kind="mergesort").reset_index(
            drop=True
        )
        matched = matched.drop(columns=["_CHR_INT"])

    summary: dict[str, Any] = {
        "match_key": "CHR:BP:unordered_alleles",
        "remove_ambiguous": bool(remove_ambiguous),
        "target": target_summary,
        "reference": reference_summary,
        "n_matched": int(len(matched)),
        "n_same_variant_id": int(matched["same_variant_id"].sum()) if not matched.empty else 0,
        "n_target_rename_to_reference_id": int((~matched["same_variant_id"]).sum())
        if not matched.empty
        else 0,
        "match_rate_target_ready": float(len(matched) / len(target)) if len(target) else 0.0,
        "match_rate_reference_ready": float(len(matched) / len(reference)) if len(reference) else 0.0,
    }
    return matched, summary


def write_ancestry_overlap_outputs(
    matched: pd.DataFrame,
    summary: dict[str, Any],
    *,
    out_dir: str | Path,
    prefix: str = "genome",
) -> dict[str, Path]:
    """Write matched audit table, extract lists, target rename map, and summary JSON."""
    out_dir = Path(out_dir)
    paths = {
        "matched": out_dir / f"{prefix}.matched_variants.tsv",
        "target_extract": out_dir / f"{prefix}.target_extract.snplist",
        "reference_extract": out_dir / f"{prefix}.1000g_extract.snplist",
        "target_rename": out_dir / f"{prefix}.target_rename_to_1000g.tsv",
        "summary": out_dir / f"{prefix}.summary.json",
    }
    for path in paths.values():
        ensure_parent(path)

    write_table(matched, paths["matched"])
    paths["target_extract"].write_text(
        "\n".join(matched["target_id"].astype(str).tolist()) + ("\n" if len(matched) else ""),
        encoding="utf-8",
    )
    paths["reference_extract"].write_text(
        "\n".join(matched["ref_id"].astype(str).tolist()) + ("\n" if len(matched) else ""),
        encoding="utf-8",
    )
    rename = matched[["target_id", "ref_id"]].rename(
        columns={"target_id": "old_id", "ref_id": "new_id"}
    )
    write_table(rename, paths["target_rename"])
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths
