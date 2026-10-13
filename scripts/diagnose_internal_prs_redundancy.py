"""Training-only score redundancy diagnostics for the ancestry-gated student design."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from decipehr.discodivas import REFERENCE_PCS
from decipehr.internal_prs_benchmark import DISEASES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/internal_prs/nyu_20261011")
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.run_dir)
    manifest = json.loads((root / "prepared/manifest.json").read_text())
    out = root / "score_redundancy"
    out.mkdir(exist_ok=False)
    summary, pairs = [], []
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for j, disease in enumerate(DISEASES):
        frame = pd.read_parquet(root / "prepared" / f"{disease}.training.parquet")
        columns = manifest["panels"][disease]["full_ids"]
        raw = frame[columns].to_numpy(dtype=float)
        residual = raw - LinearRegression().fit(frame[REFERENCE_PCS], raw).predict(frame[REFERENCE_PCS])
        for i, (kind, values) in enumerate((("raw", raw), ("reference_pc_residual", residual))):
            z = StandardScaler().fit_transform(values)
            correlation = np.corrcoef(z.T)
            eigenvalues = np.linalg.eigvalsh(correlation).clip(min=0)[::-1]
            upper = np.abs(correlation[np.triu_indices(len(columns), k=1)])
            summary.append({"disease": disease, "kind": kind, "n_training": len(frame), "n_scores": len(columns),
                            "participation_rank": eigenvalues.sum()**2 / np.square(eigenvalues).sum(),
                            "pc1_variance_fraction": eigenvalues[0] / eigenvalues.sum(),
                            "n_components_for_95pct": int(np.searchsorted(np.cumsum(eigenvalues)/eigenvalues.sum(), .95) + 1),
                            "max_absolute_correlation": upper.max(),
                            "n_pairs_abs_r_gt_0_95": int((upper > .95).sum()),
                            "n_pairs_abs_r_gt_0_99": int((upper > .99).sum()),
                            "n_pairs": len(upper)})
            pd.DataFrame(correlation, index=columns, columns=columns).to_csv(out / f"{disease}.{kind}.correlation.csv")
            for a in range(len(columns)):
                for b in range(a+1, len(columns)):
                    pairs.append({"disease": disease, "kind": kind, "score_a": columns[a], "score_b": columns[b],
                                  "correlation": correlation[a, b]})
            ax = axes[i, j]
            shown = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_title(f"{disease}: {kind}")
            ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=6)
            ax.set_yticks(range(len(columns)), columns, fontsize=6)
    fig.colorbar(shown, ax=axes, shrink=.5, label="Pearson r (training only)")
    fig.savefig(out / "score_correlations.svg")
    fig.savefig(out / "score_correlations.pdf")
    plt.close(fig)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "score_pairs.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
