"""Aggregate-only visualization of the frozen outcome-blind genotype QC audit."""

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/internal_prs/nyu_20261011")
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.run_dir)
    qc = pd.read_csv(root / "prepared/sample_qc.csv")
    thresholds = pd.read_csv(root / "prepared/qc_thresholds.csv")
    groups = pd.read_csv(root / "prepared/kinship_groups.csv", dtype={"sampleid": str})
    sizes = groups.kinship_group.value_counts()
    out = root / "qc_report"
    out.mkdir(exist_ok=True)
    thresholds.assign(flagged_fraction=thresholds.n_flagged / thresholds.n).to_csv(
        out / "ancestry_qc_counts.csv", index=False)
    sizes.value_counts().sort_index().rename_axis("component_size").rename("n_components").to_csv(
        out / "kinship_component_sizes.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for label, frame in qc.groupby("hard_label", observed=True):
        axes[0].hist(frame.heterozygosity, bins=np.linspace(.25, 1, 151),
                     histtype="step", label=label, linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Heterozygous fraction among called marker genotypes")
    axes[0].set_ylabel("Patients (log scale)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("53,571 frozen ancestry markers")
    axes[1].barh(thresholds.hard_label, 100 * thresholds.n_flagged / thresholds.n)
    axes[1].set_xlabel("Flagged within group (%)")
    axes[1].set_title("Upper median + 6 robust SD")
    axes[1].set_xlim(0, float((100 * thresholds.n_flagged / thresholds.n).max()) + .65)
    for i, row in enumerate(thresholds.itertuples()):
        axes[1].text(100 * row.n_flagged / row.n + .05, i, f"{row.n_flagged}/{row.n}", va="center", fontsize=8)
    counts = sizes.value_counts().sort_index()
    axes[2].scatter(counts.index, counts)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Retained inferred group size (log scale)")
    axes[2].set_ylabel("Number of groups (log scale)")
    axes[2].set_title("KING >= 0.0884; computational groups")
    fig.savefig(out / "genotype_qc.svg")
    fig.savefig(out / "genotype_qc.pdf")
    plt.close(fig)
    print(f"Aggregate QC plots: {out}")


if __name__ == "__main__":
    main()
