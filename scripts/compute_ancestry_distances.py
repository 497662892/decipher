from __future__ import annotations

import argparse
from pathlib import Path

from decipehr.ancestry_distance import (
    DEFAULT_GROUPS,
    compute_1000g_distances,
    load_panel,
    load_sscore,
    make_hard_labels,
    make_soft_weights,
    write_distance_outputs,
    write_pc1_pc2_assignment_svg,
)
from decipehr.schema import parse_csv_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-patient distances to 1000G super-population centers, hard "
            "1000G-like assignments, soft similarity weights, and a PC1-PC2 audit plot."
        )
    )
    parser.add_argument("--ref-sscore", required=True, help="Projected 1000G .sscore file.")
    parser.add_argument("--target-sscore", required=True, help="Projected target .sscore file.")
    parser.add_argument("--panel", required=True, help="1000G panel with IID/sample and super_pop.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="target_1000G")
    parser.add_argument("--pc-cols", default="", help="Optional comma-separated projected PC columns.")
    parser.add_argument("--n-pcs", type=int, default=10, help="Used only when --pc-cols is omitted.")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--hard-margin", type=float, default=0.5)
    parser.add_argument("--soft-tau", type=float, default=1.0)
    parser.add_argument("--plot-out", default=None)
    args = parser.parse_args()

    groups = tuple(parse_csv_list(args.groups))
    pc_cols = parse_csv_list(args.pc_cols)
    if not pc_cols:
        pc_cols = [f"PC{i}_AVG" for i in range(1, args.n_pcs + 1)]

    ref = load_sscore(args.ref_sscore)
    target = load_sscore(args.target_sscore)
    panel = load_panel(args.panel)
    distances, summary = compute_1000g_distances(
        ref,
        target,
        panel,
        pc_cols=pc_cols,
        groups=groups,
    )
    hard = make_hard_labels(distances, margin_threshold=args.hard_margin, groups=groups)
    soft = make_soft_weights(distances, tau=args.soft_tau, groups=groups)

    out_dir = Path(args.out_dir)
    distances_out = out_dir / f"{args.prefix}_distances_pc{len(pc_cols)}.tsv"
    hard_out = out_dir / f"{args.prefix}_hard_labels_pc{len(pc_cols)}_margin{args.hard_margin:g}.tsv"
    soft_out = out_dir / f"{args.prefix}_soft_weights_pc{len(pc_cols)}_tau{args.soft_tau:g}.tsv"
    summary_out = out_dir / f"{args.prefix}_distances_pc{len(pc_cols)}.summary.json"
    plot_out = (
        Path(args.plot_out)
        if args.plot_out
        else out_dir / f"{args.prefix}_PC1_PC2_1000G_centers_hard_assignments.svg"
    )

    write_distance_outputs(
        distances=distances,
        hard_labels=hard,
        soft_weights=soft,
        summary=summary,
        distances_out=distances_out,
        hard_labels_out=hard_out,
        soft_weights_out=soft_out,
        summary_out=summary_out,
    )
    write_pc1_pc2_assignment_svg(
        ref_sscore=ref,
        target_sscore=target,
        panel=panel,
        hard_labels=hard,
        path=plot_out,
        pc_cols=pc_cols,
        groups=groups,
    )

    print(f"Wrote target-to-1000G distances to {distances_out}")
    print(f"Wrote hard 1000G-like assignments to {hard_out}")
    print(f"Wrote soft 1000G similarity weights to {soft_out}")
    print(f"Wrote PC1-PC2 1000G-center assignment plot to {plot_out}")


if __name__ == "__main__":
    main()
