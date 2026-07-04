from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import require_columns


def _read_keep(path: str | Path) -> list[str]:
    frame = read_table(path)
    column = "#IID" if "#IID" in frame.columns else "IID"
    require_columns(frame, [column], f"keep file {path}")
    values = frame[column].astype("string").str.strip()
    if values.isna().any() or (values == "").any():
        raise ValueError(f"keep file {path} has blank IDs")
    if values.duplicated().any():
        examples = values.loc[values.duplicated()].head(5).tolist()
        raise ValueError(f"keep file {path} has duplicate IDs: {examples}")
    return values.tolist()


def _load_split_file(path: str | Path, *, split: str, expected_ids: list[str]) -> pd.DataFrame:
    frame = read_table(path)
    require_columns(frame, ["IID"], f"1000G ancestry output {path}")
    out = frame.copy()
    out["sampleid"] = out["IID"].astype("string")
    out.insert(0, "split", split)
    observed = set(out["sampleid"].astype(str))
    expected = set(expected_ids)
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(
            f"{path} sample mismatch for {split}: "
            f"n_missing={len(missing)}, missing_examples={sorted(missing)[:5]}, "
            f"n_extra={len(extra)}, extra_examples={sorted(extra)[:5]}"
        )
    return out


def _paths(root: Path, prefix: str, n_pcs: int, hard_margin: float, soft_tau: float) -> dict[str, Path]:
    return {
        "distances": root / f"{prefix}_distances_pc{n_pcs}.tsv",
        "hard_labels": root / f"{prefix}_hard_labels_pc{n_pcs}_margin{hard_margin:g}.tsv",
        "soft_weights": root / f"{prefix}_soft_weights_pc{n_pcs}_tau{soft_tau:g}.tsv",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine separately projected train-val and holdout-test 1000G ancestry outputs."
    )
    parser.add_argument("--train-ancestry-dir", required=True)
    parser.add_argument("--test-ancestry-dir", required=True)
    parser.add_argument("--prefix", default="target_1000G")
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--hard-margin", type=float, default=0.5)
    parser.add_argument("--soft-tau", type=float, default=1.0)
    parser.add_argument("--train-keep", default="data/processed/sample_sets/train_val.keep.psam")
    parser.add_argument("--test-keep", default="data/processed/sample_sets/holdout_test.keep.psam")
    parser.add_argument("--out-dir", default="outputs/final_ancestry_1000g/combined")
    args = parser.parse_args()

    train_ids = _read_keep(args.train_keep)
    test_ids = _read_keep(args.test_keep)
    train_paths = _paths(Path(args.train_ancestry_dir), args.prefix, args.n_pcs, args.hard_margin, args.soft_tau)
    test_paths = _paths(Path(args.test_ancestry_dir), args.prefix, args.n_pcs, args.hard_margin, args.soft_tau)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    summaries = {}
    for name in ["distances", "hard_labels", "soft_weights"]:
        train = _load_split_file(train_paths[name], split="train_val", expected_ids=train_ids)
        test = _load_split_file(test_paths[name], split="holdout_test", expected_ids=test_ids)
        combined = pd.concat([train, test], ignore_index=True)
        source_stem = train_paths[name].stem
        all_out = out_dir / f"{source_stem}.combined.tsv"
        train_out = out_dir / f"{source_stem}.train_val.tsv"
        test_out = out_dir / f"{source_stem}.holdout_test.tsv"
        write_table(combined, all_out)
        write_table(train, train_out)
        write_table(test, test_out)
        outputs[name] = {
            "combined": str(all_out),
            "train_val": str(train_out),
            "holdout_test": str(test_out),
        }
        summaries[name] = {
            "n_train_val": int(len(train)),
            "n_holdout_test": int(len(test)),
            "n_combined": int(len(combined)),
            "train_source": str(train_paths[name]),
            "test_source": str(test_paths[name]),
        }

    summary = {
        "ancestry_source": "1000G_projected_PCA_distances",
        "projection_mode": "train_val_and_holdout_projected_separately_against_same_1000G_reference",
        "outputs": outputs,
        "summaries": summaries,
    }
    summary_out = out_dir / f"{args.prefix}_combined_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
