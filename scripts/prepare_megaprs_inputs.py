from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from decipehr.io import read_table
from decipehr.schema import parse_csv_list, require_columns

FREQ_ID_COLUMNS = ["ID", "SNP", "#ID"]
FREQ_REF_COLUMNS = ["REF", "A2"]
FREQ_ALT_COLUMNS = ["ALT", "ALT1", "A1"]
FREQ_VALUE_COLUMNS = ["ALT1_FREQ", "ALT_FREQS", "ALT_FREQ", "A1_FREQ"]
N_COLUMNS = ["N", "N_EFF", "NEFF", "N_eff", "n_eff", "n"]
AMBIGUOUS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


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
    text = text.mask(text.str.contains(",", regex=False, na=False))
    return pd.to_numeric(text, errors="coerce")


def read_plink2_frequency(path: str | Path, chunksize: int) -> pd.DataFrame:
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
        raise ValueError("MegaPRS inputs need N/N_eff/n in sumstats or --default-n")
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


def _is_ambiguous(a1: pd.Series, a2: pd.Series) -> pd.Series:
    pairs = pd.Series(list(zip(a1.astype(str), a2.astype(str), strict=False)), index=a1.index)
    return pairs.isin(AMBIGUOUS)


def _append_table(frame: pd.DataFrame, path: Path, first_chunk: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, mode="w" if first_chunk else "a", header=first_chunk)


def _append_predictors(values: pd.Series, path: Path, first_chunk: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(path, sep="\t", index=False, mode="w" if first_chunk else "a", header=False)


def write_megaprs_inputs(
    *,
    harmonized: str | Path,
    freq: str | Path,
    out: str | Path,
    predictors_out: str | Path | None,
    summary_out: str | Path,
    chromosomes: set[str] | None,
    default_n: float | None,
    chunksize: int,
    drop_ambiguous: bool,
) -> dict[str, int | float | str | None]:
    out = Path(out)
    predictors_path = Path(predictors_out) if predictors_out else None
    summary_out = Path(summary_out)
    if out.exists():
        out.unlink()
    if predictors_path is not None and predictors_path.exists():
        predictors_path.unlink()

    header = read_table(harmonized, nrows=0)
    require_columns(header, ["SNP", "CHR", "A1", "A2", "BETA", "P"], "harmonized sumstats")
    target_col = _target_column(list(header.columns))

    frequency = read_plink2_frequency(freq, chunksize)
    seen_predictors: set[str] = set()

    n_harmonized = 0
    n_after_chromosome_filter = 0
    n_after_frequency = 0
    n_rows_with_default_n = 0
    n_allele_frequency_mismatch = 0
    n_ambiguous_removed = 0
    n_written = 0
    se_source = "SE" if "SE" in header.columns else "derived_from_beta_p"
    first_chunk = True

    for chunk in read_table(harmonized, chunksize=chunksize):
        n_harmonized += len(chunk)
        chunk = chunk.loc[_chromosome_filter(chunk["CHR"], chromosomes)].copy()
        n_after_chromosome_filter += len(chunk)
        if chunk.empty:
            continue

        chunk["target_snp_for_freq"] = chunk[target_col].astype("string")
        chunk = chunk.merge(
            frequency,
            left_on="target_snp_for_freq",
            right_on="target_snp",
            how="inner",
        )
        n_after_frequency += len(chunk)
        if chunk.empty:
            continue

        a1 = _clean_allele(chunk["A1"])
        a2 = _clean_allele(chunk["A2"])
        if drop_ambiguous:
            ambiguous = _is_ambiguous(a1, a2)
            n_ambiguous_removed += int(ambiguous.sum())
            chunk = chunk.loc[~ambiguous].copy()
            a1 = a1.loc[chunk.index]
            a2 = a2.loc[chunk.index]
            if chunk.empty:
                continue

        beta = pd.to_numeric(chunk["BETA"], errors="coerce")
        se, se_source = _se_values(chunk)
        n_values, n_defaulted = _n_values(chunk, default_n)
        n_rows_with_default_n += n_defaulted
        a1_freq = _effect_frequency(chunk)
        n_allele_frequency_mismatch += int(a1_freq.isna().sum())

        megaprs = pd.DataFrame(
            {
                "Predictor": chunk["target_snp_for_freq"].astype("string"),
                "A1": a1,
                "A2": a2,
                "BETA": beta,
                "SE": se,
                "n": n_values,
                "A1Freq": a1_freq,
            }
        )
        finite = (
            megaprs["Predictor"].notna()
            & megaprs["A1"].isin(["A", "C", "G", "T"])
            & megaprs["A2"].isin(["A", "C", "G", "T"])
            & (megaprs["A1"] != megaprs["A2"])
            & np.isfinite(megaprs["BETA"])
            & np.isfinite(megaprs["SE"])
            & (megaprs["SE"] > 0)
            & np.isfinite(megaprs["n"])
            & (megaprs["n"] > 0)
            & np.isfinite(megaprs["A1Freq"])
            & (megaprs["A1Freq"] > 0)
            & (megaprs["A1Freq"] < 1)
        )
        megaprs = megaprs.loc[finite].drop_duplicates(subset=["Predictor"])
        is_new = ~megaprs["Predictor"].isin(seen_predictors)
        megaprs = megaprs.loc[is_new]
        if megaprs.empty:
            continue
        seen_predictors.update(megaprs["Predictor"].astype(str).tolist())
        _append_table(megaprs, out, first_chunk)
        if predictors_path is not None:
            _append_predictors(megaprs["Predictor"], predictors_path, first_chunk)
        first_chunk = False
        n_written += len(megaprs)

    if n_written == 0:
        raise ValueError("No MegaPRS summary-statistic rows were written")

    summary = {
        "n_harmonized_rows": int(n_harmonized),
        "n_after_chromosome_filter": int(n_after_chromosome_filter),
        "n_after_frequency_join": int(n_after_frequency),
        "n_rows_with_default_n": int(n_rows_with_default_n),
        "n_allele_frequency_mismatches": int(n_allele_frequency_mismatch),
        "n_ambiguous_removed": int(n_ambiguous_removed),
        "n_megaprs_rows_written": int(n_written),
        "default_n": None if default_n is None else float(default_n),
        "drop_ambiguous": bool(drop_ambiguous),
        "se_source": se_source,
        "harmonized": str(harmonized),
        "freq": str(freq),
        "megaprs_sumstats": str(out),
        "predictors": None if predictors_path is None else str(predictors_path),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(_json_dumps(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LDAK MegaPRS summary statistics.")
    parser.add_argument("--harmonized", required=True)
    parser.add_argument("--freq", required=True, help="PLINK2 .afreq from training samples.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictors-out", default=None)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--chromosomes", default=None)
    parser.add_argument("--default-n", type=float, default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--drop-ambiguous", action="store_true")
    args = parser.parse_args()

    chromosomes = set(parse_csv_list(args.chromosomes)) if args.chromosomes else None
    summary = write_megaprs_inputs(
        harmonized=args.harmonized,
        freq=args.freq,
        out=args.out,
        predictors_out=args.predictors_out,
        summary_out=args.summary_out,
        chromosomes=chromosomes,
        default_n=args.default_n,
        chunksize=args.chunksize,
        drop_ambiguous=args.drop_ambiguous,
    )
    print(f"Wrote {summary['n_megaprs_rows_written']} MegaPRS rows to {args.out}")


if __name__ == "__main__":
    main()
