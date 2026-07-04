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


def _with_split(frame: pd.DataFrame, train_ids: list[str], test_ids: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    require_columns(frame, ["IID"], "1000G ancestry output")
    train_set = set(train_ids)
    test_set = set(test_ids)
    out = frame.copy()
    out["sampleid"] = out["IID"].astype("string")
    out["split"] = pd.NA
    out.loc[out["sampleid"].isin(train_set), "split"] = "train_val"
    out.loc[out["sampleid"].isin(test_set), "split"] = "holdout_test"
    overlap = train_set & test_set
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(f"train and test keep files overlap: {examples}")
    missing_train = train_set - set(out["sampleid"].astype(str))
    missing_test = test_set - set(out["sampleid"].astype(str))
    summary = {
        "n_rows": int(len(out)),
        "n_train_val": int((out["split"] == "train_val").sum()),
        "n_holdout_test": int((out["split"] == "holdout_test").sum()),
        "n_unassigned": int(out["split"].isna().sum()),
        "n_missing_train_val": int(len(missing_train)),
        "n_missing_holdout_test": int(len(missing_test)),
    }
    if missing_train or missing_test:
        examples = {"train_val": sorted(missing_train)[:5], "holdout_test": sorted(missing_test)[:5]}
        raise ValueError(f"1000G ancestry output is missing keep-file samples: {examples}")
    return out, summary


def _write_split_tables(frame: pd.DataFrame, out_dir: Path, stem: str) -> dict[str, str]:
    paths = {
        "all": out_dir / f"{stem}.with_split.tsv",
        "train_val": out_dir / f"{stem}.train_val.tsv",
        "holdout_test": out_dir / f"{stem}.holdout_test.tsv",
    }
    write_table(frame, paths["all"])
    write_table(frame.loc[frame["split"] == "train_val"].reset_index(drop=True), paths["train_val"])
    write_table(frame.loc[frame["split"] == "holdout_test"].reset_index(drop=True), paths["holdout_test"])
    return {name: str(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split 1000G ancestry outputs into final train-val and holdout-test tables."
    )
    parser.add_argument("--ancestry-dir", default="outputs/final_ancestry_1000g/ancestry")
    parser.add_argument("--prefix", default="target_1000G")
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--hard-margin", type=float, default=0.5)
    parser.add_argument("--soft-tau", type=float, default=1.0)
    parser.add_argument("--train-keep", default="data/processed/sample_sets/train_val.keep.psam")
    parser.add_argument("--test-keep", default="data/processed/sample_sets/holdout_test.keep.psam")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    ancestry_dir = Path(args.ancestry_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ancestry_dir / "split"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = _read_keep(args.train_keep)
    test_ids = _read_keep(args.test_keep)

    files = {
        "distances": ancestry_dir / f"{args.prefix}_distances_pc{args.n_pcs}.tsv",
        "hard_labels": ancestry_dir
        / f"{args.prefix}_hard_labels_pc{args.n_pcs}_margin{args.hard_margin:g}.tsv",
        "soft_weights": ancestry_dir / f"{args.prefix}_soft_weights_pc{args.n_pcs}_tau{args.soft_tau:g}.tsv",
    }
    summaries = {}
    output_paths = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame, summary = _with_split(read_table(path), train_ids, test_ids)
        summaries[name] = {"source": str(path), **summary}
        output_paths[name] = _write_split_tables(frame, out_dir, path.stem)

    summary = {
        "ancestry_source": "1000G_projected_PCA_distances",
        "train_keep": str(args.train_keep),
        "test_keep": str(args.test_keep),
        "outputs": output_paths,
        "summaries": summaries,
    }
    summary_out = out_dir / f"{args.prefix}_split_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
