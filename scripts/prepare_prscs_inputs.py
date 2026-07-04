from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.io import read_table
from decipehr.schema import parse_csv_list, require_columns
from decipehr.sumstats import _canonical_lookup, infer_decipehr_variant_ids

RSID_ALIASES = ["HM_RSID", "RSID", "RS_ID", "VARIANT_ID", "SNP"]


def _json_dumps(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _find_rsid_column(columns: list[str]) -> str:
    upper_to_original = {col.upper(): col for col in columns}
    for alias in RSID_ALIASES:
        if alias in upper_to_original:
            return upper_to_original[alias]
    raise ValueError(f"Raw summary statistics have no rsID column among {RSID_ALIASES}")


def _minimal_standardized_frame(raw: pd.DataFrame) -> pd.DataFrame:
    mapping = _canonical_lookup(list(raw.columns))
    missing = [col for col in ["CHR", "BP", "A1", "A2"] if col not in mapping]
    if missing:
        raise ValueError(f"Raw summary statistics missing columns needed for decipEHR IDs: {missing}")

    out = pd.DataFrame(index=raw.index)
    out["CHR"] = raw[mapping["CHR"]].astype("string").str.replace("chr", "", case=False, regex=False)
    out["BP"] = pd.to_numeric(raw[mapping["BP"]], errors="coerce").astype("Int64")
    out["A1"] = raw[mapping["A1"]].astype("string").str.upper()
    out["A2"] = raw[mapping["A2"]].astype("string").str.upper()
    return out


def _clean_rsid(values: pd.Series) -> pd.Series:
    rsid = values.astype("string").str.strip()
    rsid = rsid.mask(rsid.isin(["", ".", "NA", "#NA", "nan", "<NA>"]))
    return rsid.mask(~rsid.str.match(r"^rs[0-9]+$", na=False))


def build_target_rsid_map(raw_sumstats: str | Path, chunksize: int) -> tuple[pd.DataFrame, dict[str, int]]:
    pieces = []
    n_raw = 0
    n_with_decipehr_id = 0
    n_with_rsid = 0
    rsid_col: str | None = None

    for chunk in read_table(raw_sumstats, chunksize=chunksize):
        n_raw += len(chunk)
        if rsid_col is None:
            rsid_col = _find_rsid_column(list(chunk.columns))
        minimal = _minimal_standardized_frame(chunk)
        target_snp = infer_decipehr_variant_ids(chunk, minimal)
        rsid = _clean_rsid(chunk[rsid_col])
        frame = pd.DataFrame({"target_snp": target_snp, "rsid": rsid}).dropna()
        n_with_decipehr_id += int(target_snp.notna().sum())
        n_with_rsid += int(rsid.notna().sum())
        if not frame.empty:
            pieces.append(frame.drop_duplicates())

    if not pieces:
        raise ValueError("No decipEHR target ID to rsID mappings could be derived")

    mapping = pd.concat(pieces, ignore_index=True).drop_duplicates()
    duplicate_target = mapping["target_snp"].duplicated(keep=False)
    duplicate_rsid = mapping["rsid"].duplicated(keep=False)
    n_duplicate_target = int(duplicate_target.sum())
    n_duplicate_rsid = int(duplicate_rsid.sum())
    mapping = mapping.loc[~duplicate_target & ~duplicate_rsid].reset_index(drop=True)
    summary = {
        "n_raw_sumstats_rows": int(n_raw),
        "n_raw_rows_with_decipehr_id": int(n_with_decipehr_id),
        "n_raw_rows_with_rsid": int(n_with_rsid),
        "n_duplicate_target_mappings_removed": n_duplicate_target,
        "n_duplicate_rsid_mappings_removed": n_duplicate_rsid,
        "n_unique_target_rsid_mappings": int(len(mapping)),
    }
    return mapping, summary


def _chromosome_filter(values: pd.Series, chromosomes: set[str] | None) -> pd.Series:
    chrom = values.astype("string").str.replace(r"\.0$", "", regex=True)
    if chromosomes is None:
        return pd.Series(True, index=values.index)
    return chrom.isin(chromosomes)


def _format_chr(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True)


def _append_table(frame: pd.DataFrame, path: Path, first_chunk: bool, header: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        mode="w" if first_chunk else "a",
        header=header and first_chunk,
    )


def _append_suffix(path: str | Path, suffix: str) -> Path:
    return Path(f"{path}{suffix}")


def _has_positive_numeric_column(path: str | Path, column: str, chunksize: int) -> bool:
    for chunk in read_table(path, usecols=[column], chunksize=chunksize):
        values = pd.to_numeric(chunk[column], errors="coerce")
        if bool((np.isfinite(values) & (values > 0)).any()):
            return True
    return False


def write_prscs_inputs(
    *,
    harmonized: str | Path,
    target_rsid_map: pd.DataFrame,
    out_prefix: str | Path,
    summary_out: str | Path,
    chromosomes: set[str] | None,
    mode: str,
    chunksize: int,
) -> dict[str, int | str]:
    out_prefix = Path(out_prefix)
    sst_path = _append_suffix(out_prefix, ".sst")
    bim_path = _append_suffix(out_prefix, ".bim")
    map_path = _append_suffix(out_prefix, ".target_id_map.tsv")
    for path in [sst_path, bim_path, map_path]:
        if path.exists():
            path.unlink()

    header = read_table(harmonized, nrows=0)
    required = ["SNP", "CHR", "BP", "A1", "A2", "BETA", "P", "target_ref", "target_alt"]
    require_columns(header, required, "harmonized summary statistics")
    rsid_join_col = "source_snp" if "source_snp" in header.columns else "SNP"
    has_se = "SE" in header.columns
    if mode == "beta-se" and not has_se:
        raise ValueError("Requested beta-se mode, but harmonized summary statistics lack SE")
    if mode == "beta-se":
        value_col = "SE"
    elif mode == "auto" and has_se and _has_positive_numeric_column(harmonized, "SE", chunksize):
        value_col = "SE"
    else:
        value_col = "P"

    n_harmonized = 0
    n_after_chrom_filter = 0
    n_after_rsid_map = 0
    n_written = 0
    first_sst = True
    first_bim = True
    written_map_rows = []
    merge_map = target_rsid_map.rename(columns={"target_snp": "map_source_snp"})

    for chunk in read_table(harmonized, chunksize=chunksize):
        n_harmonized += len(chunk)
        keep_chrom = _chromosome_filter(chunk["CHR"], chromosomes)
        chunk = chunk.loc[keep_chrom].copy()
        n_after_chrom_filter += len(chunk)
        if chunk.empty:
            continue

        merged = chunk.merge(merge_map, left_on=rsid_join_col, right_on="map_source_snp", how="inner")
        n_after_rsid_map += len(merged)
        if merged.empty:
            continue

        merged["BETA"] = pd.to_numeric(merged["BETA"], errors="coerce")
        merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce")
        finite = np.isfinite(merged["BETA"]) & np.isfinite(merged[value_col]) & (merged[value_col] > 0)
        merged = merged.loc[finite].drop_duplicates(subset=["rsid"]).copy()
        if merged.empty:
            continue

        sst = merged[["rsid", "A1", "A2", "BETA", value_col]].rename(
            columns={"rsid": "SNP", value_col: value_col.upper()}
        )
        bim = pd.DataFrame(
            {
                "CHR": _format_chr(merged["CHR"]),
                "SNP": merged["rsid"],
                "CM": 0,
                "BP": pd.to_numeric(merged["BP"], errors="coerce").astype("Int64"),
                "A1": merged["target_alt"],
                "A2": merged["target_ref"],
            }
        ).dropna()
        id_map = merged[["rsid", "SNP"]].rename(columns={"SNP": "target_snp"}).drop_duplicates()

        _append_table(sst, sst_path, first_sst)
        first_sst = False
        bim_path.parent.mkdir(parents=True, exist_ok=True)
        bim.to_csv(bim_path, sep="\t", index=False, header=False, mode="w" if first_bim else "a")
        first_bim = False
        written_map_rows.append(id_map)
        n_written += len(sst)

    if n_written == 0:
        raise ValueError("No PRS-CS rows were written; check rsID mapping and chromosome filter")

    target_map = pd.concat(written_map_rows, ignore_index=True).drop_duplicates()
    target_map.to_csv(map_path, sep="\t", index=False)

    summary = {
        "mode": "beta-se" if value_col == "SE" else "beta-p",
        "n_harmonized_rows": int(n_harmonized),
        "n_after_chromosome_filter": int(n_after_chrom_filter),
        "n_after_rsid_map": int(n_after_rsid_map),
        "n_prscs_rows_written": int(n_written),
        "rsid_join_column": rsid_join_col,
        "sst_file": str(sst_path),
        "bim_file": str(bim_path),
        "target_id_map": str(map_path),
    }
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_out).write_text(_json_dumps(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare rsID-based PRS-CS sumstats, BIM, and target-ID map."
    )
    parser.add_argument("--raw-sumstats", required=True)
    parser.add_argument("--harmonized", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--chromosomes", default=None)
    parser.add_argument("--mode", choices=["auto", "beta-se", "beta-p"], default="auto")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    chromosomes = set(parse_csv_list(args.chromosomes)) if args.chromosomes else None
    target_rsid_map, map_summary = build_target_rsid_map(args.raw_sumstats, args.chunksize)
    summary = write_prscs_inputs(
        harmonized=args.harmonized,
        target_rsid_map=target_rsid_map,
        out_prefix=args.out_prefix,
        summary_out=args.summary_out,
        chromosomes=chromosomes,
        mode=args.mode,
        chunksize=args.chunksize,
    )
    combined = {**map_summary, **summary}
    Path(args.summary_out).write_text(_json_dumps(combined), encoding="utf-8")
    print(f"Wrote PRS-CS inputs to {args.out_prefix}.sst and {args.out_prefix}.bim")


if __name__ == "__main__":
    main()
