from __future__ import annotations

import argparse

from decipehr.ancestry import (
    DEFAULT_CLUSTER_COL,
    DEFAULT_PC_COLS,
    ancestry_cluster_report,
    cluster_quality_report,
    make_ancestry_clusters,
    write_pc_scatter_svg,
)
from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster patients by ancestry PCs.")
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--pc-cols", default=",".join(DEFAULT_PC_COLS))
    parser.add_argument("--labels", default="")
    parser.add_argument("--age-col", default=None)
    parser.add_argument("--sex-col", default=None)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--cluster-col", default=DEFAULT_CLUSTER_COL)
    parser.add_argument("--clusters-out", required=True)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--quality-out", default=None)
    parser.add_argument("--plot-out", default=None)
    parser.add_argument("--quality-k-min", type=int, default=2)
    parser.add_argument("--quality-k-max", type=int, default=6)
    args = parser.parse_args()

    pc_cols = parse_csv_list(args.pc_cols)
    labels = parse_csv_list(args.labels)
    phenotype = read_table(args.phenotype)
    clusters = make_ancestry_clusters(
        phenotype,
        patient_col=args.patient_col,
        pc_cols=pc_cols,
        n_clusters=args.n_clusters,
        seed=args.seed,
        n_init=args.n_init,
        cluster_col=args.cluster_col,
    )
    write_table(clusters, args.clusters_out)

    if args.report_out:
        report = ancestry_cluster_report(
            phenotype,
            clusters,
            patient_col=args.patient_col,
            cluster_col=args.cluster_col,
            pc_cols=pc_cols,
            label_cols=labels,
            age_col=args.age_col,
            sex_col=args.sex_col,
        )
        write_table(report, args.report_out)
    if args.quality_out:
        quality = cluster_quality_report(
            phenotype,
            pc_cols=pc_cols,
            k_min=args.quality_k_min,
            k_max=args.quality_k_max,
            seed=args.seed,
            n_init=args.n_init,
        )
        write_table(quality, args.quality_out)
    if args.plot_out:
        write_pc_scatter_svg(
            phenotype,
            clusters,
            args.plot_out,
            patient_col=args.patient_col,
            cluster_col=args.cluster_col,
            pc1_col=pc_cols[0],
            pc2_col=pc_cols[1],
        )

    print(f"Wrote ancestry clusters for {len(clusters)} patients to {args.clusters_out}")


if __name__ == "__main__":
    main()
