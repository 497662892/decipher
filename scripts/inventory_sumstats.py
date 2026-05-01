from __future__ import annotations

import argparse

from decipehr.io import write_table
from decipehr.sumstats import inventory_sumstats


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory candidate summary-statistics files.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    inventory = inventory_sumstats(args.root)
    write_table(inventory, args.out)
    print(f"Wrote {len(inventory)} candidate summary-statistics files to {args.out}")


if __name__ == "__main__":
    main()

