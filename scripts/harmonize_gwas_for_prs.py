from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from decipehr.io import read_table

VALID_ALLELES = {"A", "C", "G", "T"}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def default_summary_path(out: Path) -> Path:
    if out.name.endswith(".score.tsv"):
        return out.with_name(f"{out.name[:-len('.score.tsv')]}.summary.json")
    if out.suffix:
        return out.with_suffix(".summary.json")
    return out.with_name(f"{out.name}.summary.json")


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


def required_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def read_pvar(path: str | Path) -> pd.DataFrame:
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
    required_columns(pvar, ["#CHROM", "POS", "ID", "REF", "ALT"], "target pvar")
    pvar = pvar.copy()
    pvar["CHROM_NORM"] = pvar["#CHROM"].map(normalize_chrom)
    pvar["POS_INT"] = pd.to_numeric(pvar["POS"], errors="coerce").astype("Int64")
    pvar["REF_NORM"] = pvar["REF"].map(normalize_allele)
    pvar["ALT_NORM"] = pvar["ALT"].map(normalize_allele)
    pvar["ID_STR"] = pvar["ID"].astype("string")
    return pvar


def build_target_indexes(pvar: pd.DataFrame) -> tuple[dict[str, pd.Series], set[str], dict[tuple[str, int], list[pd.Series]]]:
    id_counts = pvar["ID_STR"].value_counts(dropna=True)
    duplicate_ids = {str(variant_id) for variant_id, count in id_counts.items() if count > 1}
    unique_ids = set(id_counts[id_counts == 1].index.astype(str))
    by_id: dict[str, pd.Series] = {}
    by_position: dict[tuple[str, int], list[pd.Series]] = defaultdict(list)
    for _, row in pvar.iterrows():
        variant_id = str(row["ID_STR"])
        if variant_id in unique_ids and variant_id not in {"", ".", "nan", "<NA>"}:
            by_id[variant_id] = row
        if not pd.isna(row["POS_INT"]):
            by_position[(str(row["CHROM_NORM"]), int(row["POS_INT"]))].append(row)
    return by_id, duplicate_ids, by_position


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def clean_required_gwas_rows(
    gwas: pd.DataFrame,
    columns: list[str],
    beta_col: str,
) -> pd.DataFrame:
    clean = gwas.dropna(subset=columns).copy()
    clean[beta_col] = pd.to_numeric(clean[beta_col], errors="coerce")
    return clean.dropna(subset=[beta_col])


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize GWAS summary statistics for PLINK2 PRS scoring.")
    parser.add_argument("--gwas", required=True)
    parser.add_argument("--pvar", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--snp-col", default="SNP")
    parser.add_argument("--ea-col", default="EA")
    parser.add_argument("--oa-col", default="OA")
    parser.add_argument("--beta-col", default="BETA")
    parser.add_argument("--chr-col", default=None)
    parser.add_argument("--bp-col", default=None)
    parser.add_argument("--remove-ambiguous", type=parse_bool, default=True)
    parser.add_argument("--keep-only-acgt", type=parse_bool, default=True)
    parser.add_argument("--remove-duplicate-gwas-variants", type=parse_bool, default=True)
    parser.add_argument("--match-strategy", default="id_then_position_alleles")
    parser.add_argument("--allow-strand-flip", type=parse_bool, default=False)
    parser.add_argument("--sep", default=None, help="Optional GWAS delimiter override.")
    args = parser.parse_args()

    if args.allow_strand_flip:
        raise ValueError("Strand-flip rescue is intentionally not implemented in V1")
    if args.match_strategy != "id_then_position_alleles":
        raise ValueError("V1 only supports --match-strategy id_then_position_alleles")

    gwas = read_table(args.gwas, sep=args.sep) if args.sep else read_table(args.gwas)
    required = [args.snp_col, args.ea_col, args.oa_col, args.beta_col]
    required_columns(gwas, required, "GWAS summary statistics")
    if (args.chr_col is None) != (args.bp_col is None):
        raise ValueError("--chr-col and --bp-col must be provided together")
    has_position = args.chr_col is not None and args.bp_col is not None
    if has_position:
        required_columns(gwas, [args.chr_col, args.bp_col], "GWAS summary statistics")

    pvar = read_pvar(args.pvar)
    by_id, duplicate_target_ids, by_position = build_target_indexes(pvar)

    summary = {
        "n_gwas_input": int(len(gwas)),
        "n_target_pvar_input": int(len(pvar)),
        "n_after_required_columns": 0,
        "n_after_duplicate_removal": 0,
        "n_duplicate_gwas_variant_rows": 0,
        "n_duplicate_target_pvar_ids": int(len(duplicate_target_ids)),
        "n_matched_to_target_pvar": 0,
        "n_matched_by_id": 0,
        "n_matched_by_position_alleles": 0,
        "n_removed_non_acgt": 0,
        "n_removed_ambiguous": 0,
        "n_removed_duplicate_or_multimatch": 0,
        "n_removed_allele_mismatch": 0,
        "n_unmatched_to_target_pvar": 0,
        "n_harmonized_output": 0,
        "strand_flipping_enabled": False,
    }

    gwas = clean_required_gwas_rows(gwas, required, args.beta_col)
    summary["n_after_required_columns"] = int(len(gwas))
    if args.remove_duplicate_gwas_variants:
        duplicate_gwas = gwas[args.snp_col].astype("string").duplicated(keep=False)
        n_duplicate_gwas = int(duplicate_gwas.sum())
        summary["n_duplicate_gwas_variant_rows"] = n_duplicate_gwas
        summary["n_removed_duplicate_or_multimatch"] += n_duplicate_gwas
        gwas = gwas.loc[~duplicate_gwas].copy()
    summary["n_after_duplicate_removal"] = int(len(gwas))

    rows: list[dict[str, Any]] = []
    for row_dict in gwas.to_dict("records"):
        snp = str(row_dict[args.snp_col])
        ea = normalize_allele(row_dict[args.ea_col])
        oa = normalize_allele(row_dict[args.oa_col])
        beta = row_dict[args.beta_col]

        if args.keep_only_acgt and (ea not in VALID_ALLELES or oa not in VALID_ALLELES):
            summary["n_removed_non_acgt"] += 1
            continue
        if args.remove_ambiguous and is_ambiguous(ea, oa):
            summary["n_removed_ambiguous"] += 1
            continue

        target: pd.Series | None = None
        matched_by = None
        if snp in duplicate_target_ids:
            summary["n_removed_duplicate_or_multimatch"] += 1
            continue
        if snp in by_id:
            target = by_id[snp]
            matched_by = "id"
        elif has_position:
            chrom = normalize_chrom(row_dict[args.chr_col])
            bp = pd.to_numeric(pd.Series([row_dict[args.bp_col]]), errors="coerce").iloc[0]
            if not pd.isna(bp):
                candidates = by_position.get((chrom, int(bp)), [])
                pair = {ea, oa}
                matches = [candidate for candidate in candidates if {candidate["REF_NORM"], candidate["ALT_NORM"]} == pair]
                if len(matches) == 1:
                    target = matches[0]
                    matched_by = "position"
                elif len(matches) > 1:
                    summary["n_removed_duplicate_or_multimatch"] += 1
                    continue
                elif candidates:
                    summary["n_removed_allele_mismatch"] += 1
                    continue
        if target is None:
            summary["n_unmatched_to_target_pvar"] += 1
            continue

        ref = str(target["REF_NORM"])
        alt = str(target["ALT_NORM"])
        if args.keep_only_acgt and (ref not in VALID_ALLELES or alt not in VALID_ALLELES):
            summary["n_removed_non_acgt"] += 1
            continue
        if {ea, oa} != {ref, alt}:
            summary["n_removed_allele_mismatch"] += 1
            continue

        target_id = str(target["ID_STR"])
        if target_id in {"", ".", "nan", "<NA>"}:
            summary["n_removed_duplicate_or_multimatch"] += 1
            continue
        summary["n_matched_to_target_pvar"] += 1
        if matched_by == "id":
            summary["n_matched_by_id"] += 1
        elif matched_by == "position":
            summary["n_matched_by_position_alleles"] += 1
        rows.append({"SNP": target_id, "A1": ea, "BETA": beta})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["SNP", "A1", "BETA"]).to_csv(out, sep="\t", index=False)
    summary["n_harmonized_output"] = int(len(rows))
    summary_out = Path(args.summary_out) if args.summary_out else default_summary_path(out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if summary["n_harmonized_output"] == 0:
        raise ValueError("Harmonized score file is empty")
    if summary["n_harmonized_output"] < 50_000:
        warn("harmonized score file contains <50,000 variants")
    comparable = summary["n_matched_to_target_pvar"] + summary["n_removed_allele_mismatch"]
    if comparable:
        mismatch_rate = summary["n_removed_allele_mismatch"] / comparable
        if mismatch_rate > 0.5:
            warn("allele mismatch removed >50% of matched GWAS variants")
    if summary["n_duplicate_target_pvar_ids"]:
        warn(f"target pvar contains {summary['n_duplicate_target_pvar_ids']} duplicated variant IDs")
    if summary["n_duplicate_gwas_variant_rows"]:
        warn(f"GWAS summary statistics contain {summary['n_duplicate_gwas_variant_rows']} duplicated variant rows")
    id_matchable = max(
        0,
        summary["n_after_duplicate_removal"]
        - summary["n_removed_non_acgt"]
        - summary["n_removed_ambiguous"],
    )
    if not has_position and id_matchable > 0:
        id_match_rate = summary["n_matched_by_id"] / id_matchable
        if id_match_rate < 0.1:
            warn("ID matching is <10% and CHR/BP fallback columns were not provided")
    print(f"Wrote {summary['n_harmonized_output']} harmonized variants to {out}")
    print(f"Wrote summary to {summary_out}")


if __name__ == "__main__":
    main()
