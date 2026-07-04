from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path


@dataclass(frozen=True)
class VariantRow:
    chrom: str
    pos: str
    variant_id: str
    ref: str
    alt: str

    def as_list(self) -> list[str]:
        return [self.chrom, self.pos, self.variant_id, self.ref, self.alt]


def _iter_pvar_exact(path: str | Path):
    path = Path(path)
    with path.open("rt", encoding="utf-8") as handle:
        header: list[str] | None = None
        indexes: dict[str, int] | None = None
        for line in handle:
            if not line.strip() or line.startswith("##"):
                continue
            fields = line.rstrip("\n").split()
            if header is None:
                header = [field.lstrip("#") for field in fields]
                indexes = {name: header.index(name) for name in ["CHROM", "POS", "ID", "REF", "ALT"]}
                continue
            assert indexes is not None
            yield VariantRow(
                chrom=fields[indexes["CHROM"]],
                pos=fields[indexes["POS"]],
                variant_id=fields[indexes["ID"]],
                ref=fields[indexes["REF"]],
                alt=fields[indexes["ALT"]],
            )


def _read_id_set(path: str | Path) -> set[str]:
    return {row.variant_id for row in _iter_pvar_exact(path)}


def _empty_row() -> list[str]:
    return ["", "", "", "", ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether two PLINK2 .pvar files define the same variants.")
    parser.add_argument("--left-pvar", required=True)
    parser.add_argument("--right-pvar", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    mismatch_path = Path(args.out)
    mismatch_path.parent.mkdir(parents=True, exist_ok=True)

    n_left = 0
    n_right = 0
    n_mismatch_rows = 0
    columns = ["CHROM", "POS", "ID", "REF", "ALT"]
    with mismatch_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["row_index"]
            + [f"{args.left_name}_{column}" for column in columns]
            + [f"{args.right_name}_{column}" for column in columns]
        )
        for row_index, (left_row, right_row) in enumerate(
            zip_longest(_iter_pvar_exact(args.left_pvar), _iter_pvar_exact(args.right_pvar))
        ):
            if left_row is not None:
                n_left += 1
            if right_row is not None:
                n_right += 1
            if left_row != right_row:
                n_mismatch_rows += 1
                writer.writerow(
                    [row_index]
                    + (left_row.as_list() if left_row is not None else _empty_row())
                    + (right_row.as_list() if right_row is not None else _empty_row())
                )

    exact_ordered_match = n_left == n_right and n_mismatch_rows == 0
    if exact_ordered_match:
        n_common_ids = n_left
        n_left_only_ids = 0
        n_right_only_ids = 0
    else:
        left_ids = _read_id_set(args.left_pvar)
        right_ids = _read_id_set(args.right_pvar)
        n_common_ids = len(left_ids & right_ids)
        n_left_only_ids = len(left_ids - right_ids)
        n_right_only_ids = len(right_ids - left_ids)

    summary = {
        "left_pvar": str(args.left_pvar),
        "right_pvar": str(args.right_pvar),
        "left_name": args.left_name,
        "right_name": args.right_name,
        "n_left": n_left,
        "n_right": n_right,
        "n_common_ids": n_common_ids,
        "n_left_only_ids": n_left_only_ids,
        "n_right_only_ids": n_right_only_ids,
        "n_mismatch_rows": n_mismatch_rows,
        "exact_ordered_match": exact_ordered_match,
    }

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_mismatch and not summary["exact_ordered_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
