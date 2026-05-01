from __future__ import annotations

import argparse

import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.metrics import binary_metrics, odds_ratio_per_sd, risk_stratification_or


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate out-of-fold prediction table.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pred = read_table(args.predictions)
    rows = []
    for (disease, method), frame in pred.groupby(["disease", "method"]):
        base = binary_metrics(frame["y_true"], frame["y_prob"])
        row = {
            "disease": disease,
            "method": method,
            "auc": base.auc,
            "auprc": base.auprc,
            "brier": base.brier,
            "prevalence": base.prevalence,
            "n": base.n,
            "or_per_sd": odds_ratio_per_sd(frame["y_true"], frame["y_prob"]),
        }
        row.update(risk_stratification_or(frame["y_true"], frame["y_prob"]))
        rows.append(row)
    write_table(pd.DataFrame(rows), args.out)
    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()

