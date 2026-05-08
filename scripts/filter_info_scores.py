from __future__ import annotations

import argparse
import json
from pathlib import Path

from decipehr.io import read_table
from decipehr.qc import info_score_filter_masks


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
    if out.name.endswith(".snplist"):
        return out.with_name(f"{out.name[:-len('.snplist')]}.summary.json")
    return out.with_suffix(f"{out.suffix}.summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter decipEHR variant IDs by INFO and MAF.")
    parser.add_argument("--info-file", "--info-score", dest="info_file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--min-info", type=float, default=0.8)
    parser.add_argument("--min-maf", type=float, default=0.01)
    parser.add_argument("--max-af", type=float, default=0.99)
    parser.add_argument("--variant-id-col", "--variant-col", dest="variant_id_col", default="col0")
    parser.add_argument("--info-col", default="info_IMPUTE")
    parser.add_argument("--freq-col", "--maf-col", dest="freq_col", default="MAF")
    parser.add_argument("--freq-is-maf", type=parse_bool, default=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_out = Path(args.summary_out) if args.summary_out else default_summary_path(out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_file": args.info_file,
        "n_total": 0,
        "n_pass_info": 0,
        "n_pass_maf": 0,
        "n_pass_af": 0,
        "n_pass_all": 0,
        "min_info": args.min_info,
        "min_maf": args.min_maf,
        "max_af": args.max_af,
        "variant_id_col": args.variant_id_col,
        "info_col": args.info_col,
        "freq_col": args.freq_col,
        "freq_is_maf": args.freq_is_maf,
    }
    with out.open("w", encoding="utf-8") as fh:
        for chunk in read_table(args.info_file, sep=r"\s+", chunksize=args.chunksize):
            masks = info_score_filter_masks(
                chunk,
                min_info=args.min_info,
                min_maf=args.min_maf,
                max_af=args.max_af,
                variant_col=args.variant_id_col,
                info_col=args.info_col,
                freq_col=args.freq_col,
                freq_is_maf=args.freq_is_maf,
            )
            keep = chunk.loc[masks["pass_all"], args.variant_id_col].astype("string").dropna()
            for variant_id in keep:
                fh.write(f"{variant_id}\n")
            summary["n_total"] += int(len(chunk))
            summary["n_pass_info"] += int(masks["pass_info"].sum())
            summary["n_pass_maf"] += int(masks["pass_maf"].sum())
            summary["n_pass_af"] += int(masks["pass_af"].sum())
            summary["n_pass_all"] += int(masks["pass_all"].sum())
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary['n_pass_all']} variant IDs to {out}")
    print(f"Wrote summary to {summary_out}")


if __name__ == "__main__":
    main()
