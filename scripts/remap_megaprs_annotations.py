from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def _json_dumps(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _read_predictors(path: Path, id_column: int | None) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictor file: {path}")
    column = id_column
    if column is None:
        column = 2 if path.name.endswith(".bim") or path.name.endswith(".cors.bim") else 1
    if column < 1:
        raise ValueError("--predictor-id-column is 1-based and must be >= 1")

    predictors: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < column:
                continue
            value = fields[column - 1]
            if value in {"", ".", "Predictor", "SNP", "ID"}:
                continue
            predictors.add(value)
    if not predictors:
        raise ValueError(f"No predictors were read from {path}")
    return predictors


def _load_target_map(
    path: Path,
    predictors: set[str] | None,
) -> tuple[dict[str, str], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing target ID map: {path}")

    rsid_to_target: dict[str, str | None] = {}
    n_rows = 0
    n_after_predictor_filter = 0
    n_duplicate_rsids = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Target ID map has no header: {path}")
        missing = {"rsid", "target_snp"}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Target ID map missing columns: {sorted(missing)}")

        for row in reader:
            n_rows += 1
            rsid = (row.get("rsid") or "").strip()
            target = (row.get("target_snp") or "").strip()
            if not rsid or not target or rsid == "." or target == ".":
                continue
            if predictors is not None and target not in predictors:
                continue
            n_after_predictor_filter += 1
            existing = rsid_to_target.get(rsid)
            if existing is None and rsid in rsid_to_target:
                continue
            if existing is not None:
                if existing != target:
                    rsid_to_target[rsid] = None
                    n_duplicate_rsids += 1
                continue
            rsid_to_target[rsid] = target

    unique_map = {rsid: target for rsid, target in rsid_to_target.items() if target is not None}
    if not unique_map:
        raise ValueError(f"No rsID-to-target mappings remained after filtering {path}")
    summary = {
        "n_target_map_rows": n_rows,
        "n_target_map_rows_after_predictor_filter": n_after_predictor_filter,
        "n_duplicate_rsids_removed": n_duplicate_rsids,
        "n_unique_rsid_to_target_mappings": len(unique_map),
    }
    return unique_map, summary


def _remap_annotation_file(
    *,
    in_file: Path,
    out_file: Path,
    rsid_to_target: dict[str, str],
) -> dict[str, int | str]:
    n_rows = 0
    n_mapped = 0
    n_duplicate_targets = 0
    seen_targets: set[str] = set()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with in_file.open("r", encoding="utf-8") as in_handle, out_file.open(
        "w", encoding="utf-8", newline=""
    ) as out_handle:
        for line in in_handle:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            n_rows += 1
            target = rsid_to_target.get(fields[0])
            if target is None:
                continue
            if target in seen_targets:
                n_duplicate_targets += 1
                continue
            seen_targets.add(target)
            out_handle.write(" ".join([target, *fields[1:]]) + "\n")
            n_mapped += 1
    return {
        "annotation_file": str(in_file),
        "out_file": str(out_file),
        "n_rows": n_rows,
        "n_mapped_rows": n_mapped,
        "n_duplicate_target_rows_removed": n_duplicate_targets,
    }


def remap_annotations(
    *,
    annotation_prefix: Path,
    annotation_number: int,
    target_id_map: Path,
    out_prefix: Path,
    predictors: Path | None,
    predictor_id_column: int | None,
    summary_out: Path,
    allow_empty: bool,
) -> dict[str, object]:
    predictor_set = None if predictors is None else _read_predictors(predictors, predictor_id_column)
    rsid_to_target, map_summary = _load_target_map(target_id_map, predictor_set)

    annotations = []
    for index in range(1, annotation_number + 1):
        in_file = Path(f"{annotation_prefix}{index}")
        out_file = Path(f"{out_prefix}{index}")
        if not in_file.exists():
            raise FileNotFoundError(f"Missing annotation file: {in_file}")
        annotations.append(
            _remap_annotation_file(
                in_file=in_file,
                out_file=out_file,
                rsid_to_target=rsid_to_target,
            )
        )

    empty = [item["out_file"] for item in annotations if item["n_mapped_rows"] == 0]
    if empty and not allow_empty:
        raise ValueError(
            "Some remapped MegaPRS annotation files are empty; LDAK will fail. "
            f"Examples: {empty[:5]}"
        )

    names_file = Path(f"{annotation_prefix}.names")
    if names_file.exists():
        shutil.copyfile(names_file, Path(f"{out_prefix}.names"))

    mapped_counts = [int(item["n_mapped_rows"]) for item in annotations]
    summary: dict[str, object] = {
        **map_summary,
        "annotation_prefix": str(annotation_prefix),
        "annotation_number": annotation_number,
        "target_id_map": str(target_id_map),
        "predictors": None if predictors is None else str(predictors),
        "out_prefix": str(out_prefix),
        "n_annotations": len(annotations),
        "n_empty_annotations": len(empty),
        "min_mapped_rows_per_annotation": min(mapped_counts),
        "max_mapped_rows_per_annotation": max(mapped_counts),
        "n_total_annotation_rows": sum(int(item["n_rows"]) for item in annotations),
        "n_total_mapped_rows": sum(mapped_counts),
        "annotations": annotations,
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(_json_dumps(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap rsID-named LDAK MegaPRS annotation files to decipEHR target IDs."
    )
    parser.add_argument("--annotation-prefix", required=True)
    parser.add_argument("--annotation-number", type=int, default=86)
    parser.add_argument("--target-id-map", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--predictors", default=None)
    parser.add_argument("--predictor-id-column", type=int, default=None)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    summary = remap_annotations(
        annotation_prefix=Path(args.annotation_prefix),
        annotation_number=args.annotation_number,
        target_id_map=Path(args.target_id_map),
        out_prefix=Path(args.out_prefix),
        predictors=None if args.predictors is None else Path(args.predictors),
        predictor_id_column=args.predictor_id_column,
        summary_out=Path(args.summary_out),
        allow_empty=args.allow_empty,
    )
    print(
        "Wrote remapped MegaPRS annotations to "
        f"{args.out_prefix}1..{args.out_prefix}{args.annotation_number}; "
        f"mapped_rows={summary['n_total_mapped_rows']}"
    )


if __name__ == "__main__":
    main()
