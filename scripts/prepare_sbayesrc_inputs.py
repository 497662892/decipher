from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_prscs_inputs import build_target_rsid_map
from scipy.stats import norm

from decipehr.io import read_table
from decipehr.schema import parse_csv_list, require_columns

FREQ_ID_COLUMNS = ["ID", "SNP", "#ID"]
FREQ_REF_COLUMNS = ["REF", "A2"]
FREQ_ALT_COLUMNS = ["ALT", "ALT1", "A1"]
FREQ_VALUE_COLUMNS = ["ALT1_FREQ", "ALT_FREQS", "ALT_FREQ", "A1_FREQ"]
N_COLUMNS = ["N", "N_EFF", "NEFF", "N_eff", "n_eff"]


def _json_dumps(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _first_present(columns: list[str], candidates: list[str], context: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"{context} missing any of columns: {candidates}")


def _chromosome_filter(values: pd.Series, chromosomes: set[str] | None) -> pd.Series:
    chrom = values.astype("string").str.replace(r"\.0$", "", regex=True)
    if chromosomes is None:
        return pd.Series(True, index=values.index)
    return chrom.isin(chromosomes)


def _clean_allele(values: pd.Series) -> pd.Series:
    return values.astype("string").str.upper()


def _clean_frequency(values: pd.Series) -> pd.Series:
    text = values.astype("string")
    # PLINK2 ALT_FREQS can be comma-delimited for multiallelic variants. This
    # project QC keeps biallelic SNPs, so comma rows are treated as unusable.
    text = text.mask(text.str.contains(",", regex=False, na=False))
    return pd.to_numeric(text, errors="coerce")


def read_plink2_frequency(path: str | Path, chunksize: int) -> pd.DataFrame:
    """Read PLINK2 .afreq output as target SNP -> REF/ALT/ALT frequency."""
    header = pd.read_csv(path, sep=r"\s+", nrows=0, engine="python")
    columns = list(header.columns)
    id_col = _first_present(columns, FREQ_ID_COLUMNS, "PLINK2 frequency file")
    ref_col = _first_present(columns, FREQ_REF_COLUMNS, "PLINK2 frequency file")
    alt_col = _first_present(columns, FREQ_ALT_COLUMNS, "PLINK2 frequency file")
    freq_col = _first_present(columns, FREQ_VALUE_COLUMNS, "PLINK2 frequency file")

    pieces = []
    for chunk in pd.read_csv(
        path,
        sep=r"\s+",
        usecols=[id_col, ref_col, alt_col, freq_col],
        chunksize=chunksize,
        engine="python",
    ):
        frame = pd.DataFrame(
            {
                "target_snp": chunk[id_col].astype("string"),
                "target_ref_freq": _clean_allele(chunk[ref_col]),
                "target_alt_freq": _clean_allele(chunk[alt_col]),
                "target_alt_frequency": _clean_frequency(chunk[freq_col]),
            }
        )
        frame = frame.dropna(subset=["target_snp", "target_alt_frequency"])
        frame = frame.loc[
            frame["target_ref_freq"].isin(["A", "C", "G", "T"])
            & frame["target_alt_freq"].isin(["A", "C", "G", "T"])
            & np.isfinite(frame["target_alt_frequency"])
            & (frame["target_alt_frequency"] >= 0)
            & (frame["target_alt_frequency"] <= 1)
        ]
        if not frame.empty:
            pieces.append(frame.drop_duplicates(subset=["target_snp"]))

    if not pieces:
        raise ValueError(f"No usable PLINK2 frequency rows were read from {path}")
    freq = pd.concat(pieces, ignore_index=True)
    duplicated = freq["target_snp"].duplicated(keep=False)
    return freq.loc[~duplicated].reset_index(drop=True)


def _target_column(columns: list[str]) -> str:
    return "target_snp" if "target_snp" in columns else "SNP"


def _n_values(chunk: pd.DataFrame, default_n: float | None) -> tuple[pd.Series, int]:
    n_col = next((col for col in N_COLUMNS if col in chunk.columns), None)
    if n_col is not None:
        values = pd.to_numeric(chunk[n_col], errors="coerce")
        missing = ~(np.isfinite(values) & (values > 0))
        if default_n is not None:
            values = values.mask(missing, default_n)
            return values, int(missing.sum())
        return values, 0
    if default_n is None:
        raise ValueError("sBayesRC .ma inputs need N/N_eff in sumstats or --default-n")
    return pd.Series(float(default_n), index=chunk.index), len(chunk)


def _se_values(chunk: pd.DataFrame) -> tuple[pd.Series, str]:
    beta = pd.to_numeric(chunk["BETA"], errors="coerce")
    p = pd.to_numeric(chunk["P"], errors="coerce")
    z = pd.Series(norm.isf(p / 2), index=chunk.index)
    derived = beta.abs() / z
    if "SE" in chunk.columns:
        se = pd.to_numeric(chunk["SE"], errors="coerce")
        valid = np.isfinite(se) & (se > 0)
        if bool(valid.all()):
            return se, "SE"
        if bool(valid.any()):
            return se.mask(~valid, derived), "SE_with_beta_p_fallback"
    return derived, "derived_from_beta_p"


def _effect_frequency(frame: pd.DataFrame) -> pd.Series:
    a1 = _clean_allele(frame["A1"])
    target_ref = _clean_allele(frame["target_ref_freq"])
    target_alt = _clean_allele(frame["target_alt_freq"])
    alt_freq = pd.to_numeric(frame["target_alt_frequency"], errors="coerce")
    freq = pd.Series(np.nan, index=frame.index, dtype=float)
    freq = freq.mask(a1 == target_alt, alt_freq)
    freq = freq.mask(a1 == target_ref, 1 - alt_freq)
    return freq


def _append_table(frame: pd.DataFrame, path: Path, first_chunk: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, mode="w" if first_chunk else "a", header=first_chunk)


def write_sbayesrc_inputs(
    *,
    harmonized: str | Path,
    freq: str | Path,
    out: str | Path,
    target_id_map_out: str | Path,
    summary_out: str | Path,
    chromosomes: set[str] | None,
    default_n: float | None,
    chunksize: int,
    raw_sumstats: str | Path | None = None,
) -> dict[str, int | float | str | None]:
    out = Path(out)
    target_id_map_out = Path(target_id_map_out)
    summary_out = Path(summary_out)
    for path in [out, target_id_map_out]:
        if path.exists():
            path.unlink()

    header = read_table(harmonized, nrows=0)
    require_columns(header, ["SNP", "CHR", "A1", "A2", "BETA", "P"], "harmonized sumstats")
    target_col = _target_column(list(header.columns))
    rsid_join_col = "source_snp" if "source_snp" in header.columns else target_col
    if target_col != "SNP":
        require_columns(header, [target_col], "harmonized sumstats")
    if rsid_join_col != target_col:
        require_columns(header, [rsid_join_col], "harmonized sumstats")

    target_rsid_map = None
    map_summary: dict[str, int | str] = {}
    if raw_sumstats is not None:
        target_rsid_map, map_summary = build_target_rsid_map(raw_sumstats, chunksize)
    elif not header["SNP"].astype("string").str.match(r"^rs[0-9]+$", na=False).all():
        raise ValueError("raw_sumstats is required unless harmonized SNP IDs are rsIDs")

    frequency = read_plink2_frequency(freq, chunksize)
    merge_map = (
        None
        if target_rsid_map is None
        else target_rsid_map.rename(columns={"target_snp": "map_target_snp"})
    )
    target_map_rows = []
    seen_rsids: set[str] = set()

    n_harmonized = 0
    n_after_chromosome_filter = 0
    n_after_rsid_map = 0
    n_after_frequency = 0
    n_rows_with_default_n = 0
    n_written = 0
    n_allele_frequency_mismatch = 0
    se_source = "SE" if "SE" in header.columns else "derived_from_beta_p"
    first_chunk = True

    for chunk in read_table(harmonized, chunksize=chunksize):
        n_harmonized += len(chunk)
        chunk = chunk.loc[_chromosome_filter(chunk["CHR"], chromosomes)].copy()
        n_after_chromosome_filter += len(chunk)
        if chunk.empty:
            continue

        chunk["target_snp_for_freq"] = chunk[target_col].astype("string")
        if merge_map is not None:
            chunk = chunk.merge(
                merge_map,
                left_on=rsid_join_col,
                right_on="map_target_snp",
                how="inner",
            )
            n_after_rsid_map += len(chunk)
        else:
            chunk["rsid"] = chunk["SNP"].astype("string")
            n_after_rsid_map += len(chunk)
        if chunk.empty:
            continue

        chunk = chunk.merge(
            frequency,
            left_on="target_snp_for_freq",
            right_on="target_snp",
            how="inner",
        )
        n_after_frequency += len(chunk)
        if chunk.empty:
            continue

        beta = pd.to_numeric(chunk["BETA"], errors="coerce")
        p = pd.to_numeric(chunk["P"], errors="coerce")
        se, se_source = _se_values(chunk)
        n_values, n_defaulted = _n_values(chunk, default_n)
        n_rows_with_default_n += n_defaulted
        effect_freq = _effect_frequency(chunk)
        n_allele_frequency_mismatch += int(effect_freq.isna().sum())

        ma = pd.DataFrame(
            {
                "SNP": chunk["rsid"].astype("string"),
                "A1": _clean_allele(chunk["A1"]),
                "A2": _clean_allele(chunk["A2"]),
                "freq": effect_freq,
                "b": beta,
                "se": se,
                "p": p,
                "N": n_values,
            }
        )
        finite = (
            ma["SNP"].notna()
            & ma["A1"].isin(["A", "C", "G", "T"])
            & ma["A2"].isin(["A", "C", "G", "T"])
            & np.isfinite(ma["freq"])
            & (ma["freq"] > 0)
            & (ma["freq"] < 1)
            & np.isfinite(ma["b"])
            & np.isfinite(ma["se"])
            & (ma["se"] > 0)
            & np.isfinite(ma["p"])
            & (ma["p"] > 0)
            & (ma["p"] <= 1)
            & np.isfinite(ma["N"])
            & (ma["N"] > 0)
        )
        ma = ma.loc[finite].drop_duplicates(subset=["SNP"])
        is_new = ~ma["SNP"].isin(seen_rsids)
        ma = ma.loc[is_new]
        if ma.empty:
            continue
        seen_rsids.update(ma["SNP"].astype(str).tolist())

        target_map_rows.append(
            pd.DataFrame(
                {
                    "rsid": ma["SNP"].astype("string"),
                    "target_snp": chunk.loc[ma.index, "target_snp_for_freq"].astype("string"),
                }
            )
        )
        _append_table(ma, out, first_chunk)
        first_chunk = False
        n_written += len(ma)

    if n_written == 0:
        raise ValueError("No sBayesRC .ma rows were written")

    target_map = pd.concat(target_map_rows, ignore_index=True).drop_duplicates()
    target_id_map_out.parent.mkdir(parents=True, exist_ok=True)
    target_map.to_csv(target_id_map_out, sep="\t", index=False)

    summary = {
        **map_summary,
        "n_harmonized_rows": int(n_harmonized),
        "n_after_chromosome_filter": int(n_after_chromosome_filter),
        "n_after_rsid_map": int(n_after_rsid_map),
        "n_after_frequency_join": int(n_after_frequency),
        "n_rows_with_default_n": int(n_rows_with_default_n),
        "n_allele_frequency_mismatches": int(n_allele_frequency_mismatch),
        "n_sbayesrc_rows_written": int(n_written),
        "n_target_id_map_rows": int(len(target_map)),
        "default_n": None if default_n is None else float(default_n),
        "se_source": se_source,
        "rsid_join_column": rsid_join_col,
        "harmonized": str(harmonized),
        "freq": str(freq),
        "sbayesrc_ma": str(out),
        "target_id_map": str(target_id_map_out),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(_json_dumps(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare GCTB sBayesRC .ma summary statistics from harmonized PRS sumstats."
    )
    parser.add_argument("--harmonized", required=True)
    parser.add_argument("--freq", required=True, help="PLINK2 .afreq from training samples.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-id-map-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--raw-sumstats", default=None)
    parser.add_argument("--chromosomes", default=None)
    parser.add_argument("--default-n", type=float, default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    chromosomes = set(parse_csv_list(args.chromosomes)) if args.chromosomes else None
    summary = write_sbayesrc_inputs(
        harmonized=args.harmonized,
        freq=args.freq,
        out=args.out,
        target_id_map_out=args.target_id_map_out,
        summary_out=args.summary_out,
        chromosomes=chromosomes,
        default_n=args.default_n,
        chunksize=args.chunksize,
        raw_sumstats=args.raw_sumstats,
    )
    print(
        f"Wrote {summary['n_sbayesrc_rows_written']} sBayesRC .ma rows to {args.out}",
    )


if __name__ == "__main__":
    main()
