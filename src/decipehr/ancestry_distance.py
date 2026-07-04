from __future__ import annotations

import json
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from decipehr.io import read_table, write_table
from decipehr.schema import require_columns

DEFAULT_GROUPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
AMBIGUOUS_LABEL = "AMBIGUOUS"
GROUP_COLORS = {
    "AFR": "#4C78A8",
    "AMR": "#F58518",
    "EAS": "#54A24B",
    "EUR": "#E45756",
    "SAS": "#72B7B2",
    "AMBIGUOUS": "#8A8A8A",
}


def load_sscore(path: str | Path) -> pd.DataFrame:
    """Read PLINK2 .sscore output and normalize leading-# ID columns."""
    frame = read_table(path, sep=r"\s+", engine="python")
    if "#FID" in frame.columns and "FID" not in frame.columns:
        frame = frame.rename(columns={"#FID": "FID"})
    if "#IID" in frame.columns and "IID" not in frame.columns:
        frame = frame.rename(columns={"#IID": "IID"})
    require_columns(frame, ["IID"], f"sscore {path}")
    return frame


def resolve_pc_columns(frame: pd.DataFrame, pc_cols: Sequence[str] | None = None, n_pcs: int = 10) -> list[str]:
    """Resolve projected PC columns from explicit names or common PLINK2 score suffixes."""
    if pc_cols:
        columns = list(pc_cols)
        require_columns(frame, columns, "projected PC score table")
        return columns

    candidates = [
        [f"PC{i}_AVG" for i in range(1, n_pcs + 1)],
        [f"PC{i}" for i in range(1, n_pcs + 1)],
        [f"SCORE{i}_AVG" for i in range(1, n_pcs + 1)],
        [f"SCORE{i}_SUM" for i in range(1, n_pcs + 1)],
    ]
    for columns in candidates:
        if set(columns).issubset(frame.columns):
            return columns
    raise ValueError(
        "Could not infer projected PC columns. Pass --pc-cols explicitly. "
        f"Available columns include: {frame.columns[:20].tolist()}"
    )


def load_panel(path: str | Path) -> pd.DataFrame:
    """Read a 1000G panel table with IID/sample and super_pop columns."""
    panel = read_table(path)
    if "sample" in panel.columns and "IID" not in panel.columns:
        panel = panel.rename(columns={"sample": "IID"})
    require_columns(panel, ["IID", "super_pop"], f"1000G panel {path}")
    if "pop" not in panel.columns:
        panel = panel.copy()
        panel["pop"] = ""
    return panel[["IID", "pop", "super_pop"]].copy()


def _numeric_pc_frame(frame: pd.DataFrame, pc_cols: Sequence[str], context: str) -> pd.DataFrame:
    pcs = frame[list(pc_cols)].apply(pd.to_numeric, errors="coerce")
    if pcs.isna().any().any():
        missing = pcs.columns[pcs.isna().any()].tolist()
        raise ValueError(f"{context} has nonnumeric or missing PC values in columns: {missing}")
    return pcs


def compute_1000g_distances(
    ref_sscore: pd.DataFrame,
    target_sscore: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    pc_cols: Sequence[str] | None = None,
    groups: Sequence[str] = DEFAULT_GROUPS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute target-to-1000G super-population center distances in projected PC space."""
    ref_cols = resolve_pc_columns(ref_sscore, pc_cols)
    target_cols = resolve_pc_columns(target_sscore, pc_cols or ref_cols)
    groups = tuple(groups)

    panel = panel.copy()
    require_columns(panel, ["IID", "super_pop"], "1000G panel")
    if "pop" not in panel.columns:
        panel["pop"] = ""
    ref = ref_sscore.merge(panel[["IID", "pop", "super_pop"]], on="IID", how="inner")
    ref = ref.loc[ref["super_pop"].isin(groups)].copy()
    missing_groups = sorted(set(groups) - set(ref["super_pop"].dropna().astype(str)))
    if missing_groups:
        raise ValueError(f"Reference projected scores are missing 1000G groups: {missing_groups}")
    if ref.empty:
        raise ValueError("No 1000G reference samples remained after joining the panel")

    ref_pcs = _numeric_pc_frame(ref, ref_cols, "1000G projected scores")
    target_pcs = _numeric_pc_frame(target_sscore, target_cols, "target projected scores")
    ref_for_math = ref.copy()
    ref_for_math[ref_cols] = ref_pcs
    target_values = target_pcs.to_numpy(dtype=float)

    centers = ref_for_math.groupby("super_pop", sort=True)[ref_cols].mean()
    out_cols = ["FID", "IID"] if "FID" in target_sscore.columns else ["IID"]
    out = target_sscore[out_cols].copy()

    ref_sd = ref_pcs.std(axis=0, ddof=0).replace(0, np.nan).to_numpy(dtype=float)
    for group in groups:
        center = centers.loc[group, ref_cols].to_numpy(dtype=float)
        diff = (target_values - center) / ref_sd
        out[f"d_euclid_{group}"] = np.sqrt(np.nansum(diff**2, axis=1))

    covariance = LedoitWolf().fit(ref_pcs.to_numpy(dtype=float)).covariance_
    inv_covariance = np.linalg.pinv(covariance)
    for group in groups:
        center = centers.loc[group, ref_cols].to_numpy(dtype=float)
        diff = target_values - center
        d2 = np.einsum("ij,jk,ik->i", diff, inv_covariance, diff)
        out[f"d_mahal_{group}"] = np.sqrt(np.maximum(d2, 0.0))

    mahal_cols = [f"d_mahal_{group}" for group in groups]
    distances = out[mahal_cols].to_numpy(dtype=float)
    nearest_idx = np.argmin(distances, axis=1)
    sorted_distances = np.sort(distances, axis=1)
    out["nearest_1000G_superpop"] = [groups[index] for index in nearest_idx]
    out["min_mahal"] = sorted_distances[:, 0]
    out["second_min_mahal"] = sorted_distances[:, 1] if len(groups) > 1 else np.nan
    out["mahal_margin"] = out["second_min_mahal"] - out["min_mahal"]

    summary = {
        "n_reference_projected": int(len(ref_sscore)),
        "n_reference_with_panel": int(len(ref)),
        "n_target_projected": int(len(target_sscore)),
        "pc_cols": ref_cols,
        "groups": list(groups),
        "reference_group_counts": {
            str(group): int(count) for group, count in ref["super_pop"].value_counts().sort_index().items()
        },
        "distance_metric": "pooled_mahalanobis_ledoit_wolf",
    }
    return out, summary


def make_hard_labels(
    distances: pd.DataFrame,
    *,
    margin_threshold: float = 0.5,
    groups: Sequence[str] = DEFAULT_GROUPS,
) -> pd.DataFrame:
    """Make hard 1000G genetic-similarity labels from nearest-center distances."""
    require_columns(
        distances,
        ["IID", "nearest_1000G_superpop", "min_mahal", "second_min_mahal", "mahal_margin"],
        "distance table",
    )
    id_cols = ["FID", "IID"] if "FID" in distances.columns else ["IID"]
    keep_cols = [
        *id_cols,
        "nearest_1000G_superpop",
        "min_mahal",
        "second_min_mahal",
        "mahal_margin",
        *[f"d_mahal_{group}" for group in groups if f"d_mahal_{group}" in distances.columns],
    ]
    out = distances[keep_cols].copy()
    out["hard_label"] = out["nearest_1000G_superpop"].astype(str) + "-like"
    out.loc[out["mahal_margin"] < margin_threshold, "hard_label"] = AMBIGUOUS_LABEL
    columns = [*id_cols, "nearest_1000G_superpop", "hard_label", "min_mahal", "second_min_mahal", "mahal_margin"]
    columns.extend([col for col in out.columns if col.startswith("d_mahal_")])
    return out[columns]


def make_soft_weights(
    distances: pd.DataFrame,
    *,
    tau: float = 1.0,
    groups: Sequence[str] = DEFAULT_GROUPS,
) -> pd.DataFrame:
    """Convert 1000G center distances to RBF soft genetic-similarity weights."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    distance_cols = [f"d_mahal_{group}" for group in groups]
    require_columns(distances, ["IID", *distance_cols], "distance table")
    id_cols = ["FID", "IID"] if "FID" in distances.columns else ["IID"]
    matrix = distances[distance_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Distance table contains nonfinite Mahalanobis distances")

    log_s = -0.5 * (matrix / tau) ** 2
    log_s = log_s - log_s.max(axis=1, keepdims=True)
    scores = np.exp(log_s)
    weights = scores / scores.sum(axis=1, keepdims=True)

    out = distances[id_cols].copy()
    for index, group in enumerate(groups):
        out[f"w_{group}"] = weights[:, index]
    out["max_weight"] = weights.max(axis=1)
    out["soft_nearest_group"] = [f"{groups[index]}-like" for index in weights.argmax(axis=1)]
    out["weight_entropy"] = -(weights * np.log(weights + 1e-12)).sum(axis=1)
    return out


def write_distance_outputs(
    *,
    distances: pd.DataFrame,
    hard_labels: pd.DataFrame,
    soft_weights: pd.DataFrame,
    summary: dict[str, Any],
    distances_out: str | Path,
    hard_labels_out: str | Path,
    soft_weights_out: str | Path,
    summary_out: str | Path | None = None,
) -> None:
    write_table(distances, distances_out)
    write_table(hard_labels, hard_labels_out)
    write_table(soft_weights, soft_weights_out)
    if summary_out:
        Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_out).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def write_pc1_pc2_assignment_svg(
    *,
    ref_sscore: pd.DataFrame,
    target_sscore: pd.DataFrame,
    panel: pd.DataFrame,
    hard_labels: pd.DataFrame,
    path: str | Path,
    pc_cols: Sequence[str] | None = None,
    groups: Sequence[str] = DEFAULT_GROUPS,
    width: int = 980,
    height: int = 720,
) -> None:
    """Write an SVG audit plot with 1000G centers and target hard assignments."""
    resolved = resolve_pc_columns(ref_sscore, pc_cols)
    if len(resolved) < 2:
        raise ValueError("At least two PC columns are required for the PC1-PC2 plot")
    target_cols = resolve_pc_columns(target_sscore, resolved)
    pc1_ref, pc2_ref = resolved[0], resolved[1]
    pc1_target, pc2_target = target_cols[0], target_cols[1]

    panel = panel.copy()
    require_columns(panel, ["IID", "super_pop"], "1000G panel")
    ref = ref_sscore.merge(panel[["IID", "super_pop"]], on="IID", how="inner")
    ref = ref.loc[ref["super_pop"].isin(groups)].copy()
    require_columns(hard_labels, ["IID", "hard_label"], "hard labels")
    target = target_sscore.merge(hard_labels[["IID", "hard_label"]], on="IID", how="left")

    ref_pcs = _numeric_pc_frame(ref, [pc1_ref, pc2_ref], "1000G projected scores")
    target_pcs = _numeric_pc_frame(target, [pc1_target, pc2_target], "target projected scores")
    ref = ref.assign(__pc1=ref_pcs[pc1_ref].to_numpy(), __pc2=ref_pcs[pc2_ref].to_numpy())
    target = target.assign(
        __pc1=target_pcs[pc1_target].to_numpy(),
        __pc2=target_pcs[pc2_target].to_numpy(),
        hard_label=target["hard_label"].fillna(AMBIGUOUS_LABEL),
    )
    centers = ref.groupby("super_pop", sort=True)[["__pc1", "__pc2"]].mean()

    all_x = pd.concat([ref["__pc1"], target["__pc1"]], ignore_index=True)
    all_y = pd.concat([ref["__pc2"], target["__pc2"]], ignore_index=True)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    x_pad = (x_max - x_min) * 0.06 or 1.0
    y_pad = (y_max - y_min) * 0.06 or 1.0
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    left, right, top, bottom = 82, 210, 42, 78
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:Arial,sans-serif;fill:#222}"
        ".axis{stroke:#333;stroke-width:1.2}"
        ".grid{stroke:#ddd;stroke-width:0.8}"
        ".center{stroke:#111;stroke-width:2.2;stroke-linecap:round}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{width / 2:.1f}" y="25" text-anchor="middle" font-size="18">'
        "1000G projection with target hard assignments</text>",
    ]
    for i in range(6):
        tx = left + i / 5 * plot_w
        ty = top + i / 5 * plot_h
        parts.append(f'<line class="grid" x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{top + plot_h}"/>')
        parts.append(f'<line class="grid" x1="{left}" y1="{ty:.1f}" x2="{left + plot_w}" y2="{ty:.1f}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')

    for _, row in ref.iterrows():
        group = str(row["super_pop"])
        color = GROUP_COLORS.get(group, "#777777")
        parts.append(
            f'<circle cx="{sx(float(row["__pc1"])):.2f}" cy="{sy(float(row["__pc2"])):.2f}" '
            f'r="2.1" fill="{color}" fill-opacity="0.28"/>'
        )
    for _, row in target.iterrows():
        label = str(row["hard_label"])
        group = label.removesuffix("-like")
        color = GROUP_COLORS.get(group, GROUP_COLORS[AMBIGUOUS_LABEL])
        parts.append(
            f'<circle cx="{sx(float(row["__pc1"])):.2f}" cy="{sy(float(row["__pc2"])):.2f}" '
            f'r="3.0" fill="{color}" stroke="#111" stroke-width="0.35" fill-opacity="0.82"/>'
        )

    for group, row in centers.iterrows():
        color = GROUP_COLORS.get(str(group), "#111111")
        x = sx(float(row["__pc1"]))
        y = sy(float(row["__pc2"]))
        parts.extend(
            [
                f'<line class="center" x1="{x - 8:.2f}" y1="{y:.2f}" x2="{x + 8:.2f}" y2="{y:.2f}" stroke="{color}"/>',
                f'<line class="center" x1="{x:.2f}" y1="{y - 8:.2f}" x2="{x:.2f}" y2="{y + 8:.2f}" stroke="{color}"/>',
                f'<text x="{x + 10:.2f}" y="{y - 8:.2f}" font-size="12">{escape(str(group))} center</text>',
            ]
        )

    for i in range(6):
        value = x_min + i / 5 * (x_max - x_min)
        tx = left + i / 5 * plot_w
        parts.append(
            f'<text x="{tx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11">'
            f"{value:.3g}</text>"
        )
        value = y_max - i / 5 * (y_max - y_min)
        ty = top + i / 5 * plot_h
        parts.append(
            f'<text x="{left - 10}" y="{ty + 4:.1f}" text-anchor="end" font-size="11">'
            f"{value:.3g}</text>"
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 26}" text-anchor="middle" font-size="14">'
        f"{escape(pc1_ref)}</text>"
    )
    parts.append(
        f'<text x="23" y="{top + plot_h / 2:.1f}" transform="rotate(-90 23 {top + plot_h / 2:.1f})" '
        f'text-anchor="middle" font-size="14">{escape(pc2_ref)}</text>'
    )

    legend_x = left + plot_w + 24
    legend_y = top + 12
    parts.append(f'<text x="{legend_x}" y="{legend_y}" font-size="13" font-weight="bold">1000G / target colors</text>')
    for index, group in enumerate([*groups, AMBIGUOUS_LABEL]):
        y = legend_y + 24 + index * 22
        color = GROUP_COLORS.get(str(group), "#777777")
        label = str(group) if group == AMBIGUOUS_LABEL else f"{group}-like"
        parts.append(f'<circle cx="{legend_x + 7}" cy="{y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{y + 4}" font-size="12">{escape(label)}</text>')
    y = legend_y + 24 + (len(groups) + 1) * 22
    parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 15}" y2="{y}" stroke="#111" stroke-width="2.2"/>')
    parts.append(f'<line x1="{legend_x + 7.5}" y1="{y - 7}" x2="{legend_x + 7.5}" y2="{y + 7}" stroke="#111" stroke-width="2.2"/>')
    parts.append(f'<text x="{legend_x + 20}" y="{y + 4}" font-size="12">1000G center</text>')
    parts.append("</svg>")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
