from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_prscs_inputs import build_target_rsid_map

from decipehr.io import read_table
from decipehr.schema import parse_csv_list, require_columns

SDPR_MIN_P = 1e-300


def _json_dumps(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _chromosome_filter(values: pd.Series, chromosomes: set[str] | None) -> pd.Series:
    chrom = values.astype("string").str.replace(r"\.0$", "", regex=True)
    if chromosomes is None:
        return pd.Series(True, index=values.index)
    return chrom.isin(chromosomes)


def _append_table(frame: pd.DataFrame, path: Path, first_chunk: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, mode="w" if first_chunk else "a", header=first_chunk)


def _format_chr(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True)


def _write_rsid_validation_bim(
    *,
    target_bim: str | Path,
    target_rsid_map: pd.DataFrame,
    out: str | Path,
) -> dict[str, int | str]:
    bim = pd.read_csv(
        target_bim,
        sep=r"\s+",
        header=None,
        names=["CHR", "target_snp", "CM", "BP", "A1", "A2"],
        engine="python",
    )
    merged = bim.merge(target_rsid_map, on="target_snp", how="inner")
    duplicated_target = merged["target_snp"].duplicated(keep=False)
    duplicated_rsid = merged["rsid"].duplicated(keep=False)
    merged = merged.loc[~duplicated_target & ~duplicated_rsid].copy()
    rsid_bim = merged[["CHR", "rsid", "CM", "BP", "A1", "A2"]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rsid_bim.to_csv(out, sep="\t", header=False, index=False)
    return {
        "target_bim": str(target_bim),
        "valid_bim": str(out),
        "n_target_bim_rows": int(len(bim)),
        "n_valid_bim_rows": int(len(rsid_bim)),
    }


def write_sdpr_sumstats(
    *,
    harmonized: str | Path,
    out: str | Path,
    summary_out: str | Path,
    chromosomes: set[str] | None,
    default_n: float | None,
    chunksize: int,
    raw_sumstats: str | Path | None = None,
    target_id_map_out: str | Path | None = None,
    target_bim: str | Path | None = None,
    valid_bim_out: str | Path | None = None,
) -> dict[str, int | float | str | None]:
    out = Path(out)
    if out.exists():
        out.unlink()
    if target_id_map_out is not None and Path(target_id_map_out).exists():
        Path(target_id_map_out).unlink()
    if valid_bim_out is not None and Path(valid_bim_out).exists():
        Path(valid_bim_out).unlink()

    header = read_table(harmonized, nrows=0)
    require_columns(header, ["SNP", "CHR", "A1", "A2", "BETA", "P"], "harmonized summary statistics")
    n_col = next((col for col in ["N", "N_EFF", "NEFF", "N_eff", "n_eff"] if col in header.columns), None)
    target_rsid_map = None
    map_summary: dict[str, int | str] = {}
    rsid_join_col = None
    target_map_rows = []
    if raw_sumstats is not None:
        target_rsid_map, map_summary = build_target_rsid_map(raw_sumstats, chunksize)
        rsid_join_col = "source_snp" if "source_snp" in header.columns else "SNP"

    n_harmonized = 0
    n_after_chromosome_filter = 0
    n_after_rsid_map = 0
    n_with_default_n = 0
    n_p_clipped = 0
    n_written = 0
    first_chunk = True
    seen_snps: set[str] = set()

    for chunk in read_table(harmonized, chunksize=chunksize):
        n_harmonized += len(chunk)
        chunk = chunk.loc[_chromosome_filter(chunk["CHR"], chromosomes)].copy()
        n_after_chromosome_filter += len(chunk)
        if chunk.empty:
            continue
        if target_rsid_map is not None and rsid_join_col is not None:
            chunk = chunk.merge(
                target_rsid_map.rename(columns={"target_snp": "map_source_snp"}),
                left_on=rsid_join_col,
                right_on="map_source_snp",
                how="inner",
            )
            n_after_rsid_map += len(chunk)
            if chunk.empty:
                continue

        chunk["BETA"] = pd.to_numeric(chunk["BETA"], errors="coerce")
        chunk["P"] = pd.to_numeric(chunk["P"], errors="coerce")
        if n_col is not None:
            n_eff = pd.to_numeric(chunk[n_col], errors="coerce")
            if default_n is not None and np.isfinite(default_n) and default_n > 0:
                missing_n = ~(np.isfinite(n_eff) & (n_eff > 0))
                n_with_default_n += int(missing_n.sum())
                n_eff = n_eff.mask(missing_n, default_n)
        elif default_n is not None and np.isfinite(default_n) and default_n > 0:
            n_eff = pd.Series(default_n, index=chunk.index)
            n_with_default_n += len(chunk)
        else:
            raise ValueError("SDPR inputs need N/N_eff in harmonized sumstats or --default-n")

        frame = pd.DataFrame(
            {
                "SNP": chunk["rsid"].astype("string") if target_rsid_map is not None else chunk["SNP"].astype("string"),
                "A1": chunk["A1"].astype("string").str.upper(),
                "A2": chunk["A2"].astype("string").str.upper(),
                "BETA": chunk["BETA"],
                "P": chunk["P"],
                "N": n_eff,
            }
        )
        finite = (
            frame["SNP"].notna()
            & frame["A1"].isin(["A", "C", "G", "T"])
            & frame["A2"].isin(["A", "C", "G", "T"])
            & np.isfinite(frame["BETA"])
            & np.isfinite(frame["P"])
            & (frame["P"] > 0)
            & (frame["P"] <= 1)
            & np.isfinite(frame["N"])
            & (frame["N"] > 0)
        )
        frame = frame.loc[finite].drop_duplicates(subset=["SNP"])
        is_new = ~frame["SNP"].isin(seen_snps)
        frame = frame.loc[is_new]
        if frame.empty:
            continue
        below_min_p = frame["P"] < SDPR_MIN_P
        n_p_clipped += int(below_min_p.sum())
        frame.loc[below_min_p, "P"] = SDPR_MIN_P
        seen_snps.update(frame["SNP"].astype(str).tolist())
        if target_rsid_map is not None:
            target_map_rows.append(
                chunk.loc[frame.index, ["rsid", "SNP"]]
                .rename(columns={"SNP": "target_snp"})
                .drop_duplicates()
            )

        _append_table(frame, out, first_chunk)
        first_chunk = False
        n_written += len(frame)

    if n_written == 0:
        raise ValueError("No SDPR summary-statistic rows were written")

    target_map = None
    if target_map_rows:
        target_map = pd.concat(target_map_rows, ignore_index=True).drop_duplicates()
        if target_id_map_out is not None:
            Path(target_id_map_out).parent.mkdir(parents=True, exist_ok=True)
            target_map.to_csv(target_id_map_out, sep="\t", index=False)

    valid_bim_summary: dict[str, int | str] = {}
    if target_map is not None and target_bim is not None and valid_bim_out is not None:
        valid_bim_summary = _write_rsid_validation_bim(
            target_bim=target_bim,
            target_rsid_map=target_map,
            out=valid_bim_out,
        )

    summary = {
        **map_summary,
        "n_harmonized_rows": int(n_harmonized),
        "n_after_chromosome_filter": int(n_after_chromosome_filter),
        "n_after_rsid_map": int(n_after_rsid_map),
        "n_rows_with_default_n": int(n_with_default_n),
        "n_p_values_clipped_to_sdpr_min": int(n_p_clipped),
        "sdpr_min_p": SDPR_MIN_P,
        "n_sdpr_rows_written": int(n_written),
        "default_n": None if default_n is None else float(default_n),
        "rsid_join_column": rsid_join_col,
        "sdpr_sumstats": str(out),
        "target_id_map": None if target_id_map_out is None else str(target_id_map_out),
        **valid_bim_summary,
    }
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_out).write_text(_json_dumps(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SDPR summary statistics from harmonized PRS sumstats.")
    parser.add_argument("--harmonized", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--chromosomes", default=None)
    parser.add_argument("--default-n", type=float, default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--raw-sumstats", default=None)
    parser.add_argument("--target-id-map-out", default=None)
    parser.add_argument("--target-bim", default=None)
    parser.add_argument("--valid-bim-out", default=None)
    args = parser.parse_args()

    chromosomes = set(parse_csv_list(args.chromosomes)) if args.chromosomes else None
    summary = write_sdpr_sumstats(
        harmonized=args.harmonized,
        out=args.out,
        summary_out=args.summary_out,
        chromosomes=chromosomes,
        default_n=args.default_n,
        chunksize=args.chunksize,
        raw_sumstats=args.raw_sumstats,
        target_id_map_out=args.target_id_map_out,
        target_bim=args.target_bim,
        valid_bim_out=args.valid_bim_out,
    )
    print(f"Wrote {summary['n_sdpr_rows_written']} SDPR summary-statistic rows to {args.out}")


if __name__ == "__main__":
    main()
