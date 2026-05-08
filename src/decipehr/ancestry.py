from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from decipehr.io import ensure_parent
from decipehr.schema import require_columns

DEFAULT_PC_COLS = [f"PC{i}_sd" for i in range(1, 11)]
DEFAULT_CLUSTER_COL = "ancestry_cluster"


def _pc_matrix(frame: pd.DataFrame, pc_cols: Sequence[str]) -> np.ndarray:
    require_columns(frame, pc_cols, "phenotype")
    pcs = frame[list(pc_cols)].apply(pd.to_numeric, errors="coerce")
    if pcs.isna().any().any():
        missing = pcs.columns[pcs.isna().any()].tolist()
        raise ValueError(f"PC columns must be numeric and complete; missing values in {missing}")
    return pcs.to_numpy(dtype=float)


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    values, counts = np.unique(labels, return_counts=True)
    order = sorted(zip(values.tolist(), counts.tolist(), strict=True), key=lambda item: (-item[1], item[0]))
    mapping = {old: new for new, (old, _) in enumerate(order)}
    return np.array([mapping[int(label)] for label in labels], dtype=int)


def make_ancestry_clusters(
    phenotype: pd.DataFrame,
    patient_col: str,
    pc_cols: Sequence[str] = DEFAULT_PC_COLS,
    n_clusters: int = 3,
    seed: int = 42,
    n_init: int = 50,
    cluster_col: str = DEFAULT_CLUSTER_COL,
) -> pd.DataFrame:
    """Cluster patients using standardized ancestry PCs."""
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    require_columns(phenotype, [patient_col], "phenotype")
    if phenotype[patient_col].duplicated().any():
        examples = phenotype.loc[phenotype[patient_col].duplicated(), patient_col].head(5).tolist()
        raise ValueError(f"phenotype has duplicate {patient_col} values, examples={examples}")

    scaled = StandardScaler().fit_transform(_pc_matrix(phenotype, pc_cols))
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init).fit_predict(scaled)
    out = phenotype[[patient_col]].copy()
    out[cluster_col] = _relabel_by_size(labels)
    return out


def cluster_quality_report(
    phenotype: pd.DataFrame,
    pc_cols: Sequence[str] = DEFAULT_PC_COLS,
    k_min: int = 2,
    k_max: int = 6,
    seed: int = 42,
    n_init: int = 50,
    silhouette_sample_size: int | None = 5000,
) -> pd.DataFrame:
    """Return inertia and silhouette diagnostics for candidate cluster counts."""
    if k_min < 2 or k_max < k_min:
        raise ValueError("Require 2 <= k_min <= k_max")
    scaled = StandardScaler().fit_transform(_pc_matrix(phenotype, pc_cols))
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(scaled)
        sample_size = silhouette_sample_size
        if sample_size is not None and sample_size >= len(scaled):
            sample_size = None
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(
                    silhouette_score(
                        scaled,
                        model.labels_,
                        sample_size=sample_size,
                        random_state=seed,
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _safe_name(value: object) -> str:
    text = str(value).strip()
    safe = "".join(char if char.isalnum() else "_" for char in text)
    return safe or "missing"


def ancestry_cluster_report(
    phenotype: pd.DataFrame,
    clusters: pd.DataFrame,
    patient_col: str,
    cluster_col: str = DEFAULT_CLUSTER_COL,
    pc_cols: Sequence[str] = DEFAULT_PC_COLS,
    label_cols: Sequence[str] | None = None,
    age_col: str | None = None,
    sex_col: str | None = None,
) -> pd.DataFrame:
    """Summarize cluster sizes and key covariate/label distributions."""
    label_cols = list(label_cols or [])
    required = [patient_col, *pc_cols, *label_cols]
    if age_col:
        required.append(age_col)
    if sex_col:
        required.append(sex_col)
    require_columns(phenotype, required, "phenotype")
    require_columns(clusters, [patient_col, cluster_col], "clusters")

    merged = phenotype.merge(clusters[[patient_col, cluster_col]], on=patient_col, how="inner")
    sex_values = sorted(merged[sex_col].dropna().astype(str).unique().tolist()) if sex_col else []
    rows = []
    for cluster, frame in merged.groupby(cluster_col, sort=True):
        row: dict[str, float | int | str] = {cluster_col: int(cluster), "n": int(len(frame))}
        for pc_col in pc_cols:
            row[f"{pc_col}_mean"] = float(pd.to_numeric(frame[pc_col], errors="coerce").mean())
        for label_col in label_cols:
            label = pd.to_numeric(frame[label_col], errors="coerce")
            row[f"{label_col}_cases"] = int(label.sum())
            row[f"{label_col}_prevalence"] = float(label.mean())
        if age_col:
            age = pd.to_numeric(frame[age_col], errors="coerce")
            row[f"{age_col}_mean"] = float(age.mean())
            row[f"{age_col}_q10"] = float(age.quantile(0.1))
            row[f"{age_col}_q50"] = float(age.quantile(0.5))
            row[f"{age_col}_q90"] = float(age.quantile(0.9))
        if sex_col:
            counts = frame[sex_col].astype(str).value_counts()
            for value in sex_values:
                name = _safe_name(value)
                count = int(counts.get(value, 0))
                row[f"{sex_col}_{name}_n"] = count
                row[f"{sex_col}_{name}_prop"] = count / len(frame) if len(frame) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def write_pc_scatter_svg(
    phenotype: pd.DataFrame,
    clusters: pd.DataFrame,
    path: str | Path,
    patient_col: str,
    cluster_col: str = DEFAULT_CLUSTER_COL,
    pc1_col: str = "PC1_sd",
    pc2_col: str = "PC2_sd",
    width: int = 900,
    height: int = 650,
) -> None:
    """Write a lightweight PC1-PC2 cluster scatter plot as SVG."""
    require_columns(phenotype, [patient_col, pc1_col, pc2_col], "phenotype")
    require_columns(clusters, [patient_col, cluster_col], "clusters")
    merged = phenotype.merge(clusters[[patient_col, cluster_col]], on=patient_col, how="inner")
    svg = _pc_scatter_svg(merged, cluster_col, pc1_col, pc2_col, width, height)
    path = Path(path)
    ensure_parent(path)
    path.write_text(svg, encoding="utf-8")


def _pc_scatter_svg(
    frame: pd.DataFrame,
    cluster_col: str,
    pc1_col: str,
    pc2_col: str,
    width: int,
    height: int,
) -> str:
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    left, right, top, bottom = 78, 30, 35, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    x = pd.to_numeric(frame[pc1_col], errors="coerce")
    y = pd.to_numeric(frame[pc2_col], errors="coerce")
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_pad = (x_max - x_min) * 0.04 or 1.0
    y_pad = (y_max - y_min) * 0.04 or 1.0
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

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
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-size="18">'
        "PC1-PC2 ancestry clusters</text>",
    ]

    for i in range(6):
        tx = left + i / 5 * plot_w
        ty = top + i / 5 * plot_h
        parts.append(f'<line class="grid" x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{top + plot_h}"/>')
        parts.append(f'<line class="grid" x1="{left}" y1="{ty:.1f}" x2="{left + plot_w}" y2="{ty:.1f}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')

    for _, row in frame.sort_values(cluster_col).iterrows():
        cluster = int(row[cluster_col])
        color = colors[cluster % len(colors)]
        parts.append(
            f'<circle cx="{sx(float(row[pc1_col])):.2f}" cy="{sy(float(row[pc2_col])):.2f}" '
            f'r="2.2" fill="{color}" fill-opacity="0.48"/>'
        )

    for i in range(6):
        value = x_min + i / 5 * (x_max - x_min)
        tx = left + i / 5 * plot_w
        parts.append(
            f'<text x="{tx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11">'
            f"{value:.2f}</text>"
        )
        value = y_max - i / 5 * (y_max - y_min)
        ty = top + i / 5 * plot_h
        parts.append(
            f'<text x="{left - 10}" y="{ty + 4:.1f}" text-anchor="end" font-size="11">'
            f"{value:.2f}</text>"
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-size="14">'
        f"{escape(pc1_col)}</text>"
    )
    parts.append(
        f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" '
        f'text-anchor="middle" font-size="14">{escape(pc2_col)}</text>'
    )

    legend_x = left + plot_w - 135
    legend_y = top + 16
    parts.append(f'<rect x="{legend_x - 10}" y="{legend_y - 18}" width="145" height="82" fill="#fff" opacity="0.86"/>')
    for i, cluster in enumerate(sorted(frame[cluster_col].unique())):
        color = colors[int(cluster) % len(colors)]
        y_pos = legend_y + i * 22
        count = int((frame[cluster_col] == cluster).sum())
        parts.append(f'<circle cx="{legend_x}" cy="{y_pos}" r="5" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 12}" y="{y_pos + 4}" font-size="12">'
            f"cluster {int(cluster)} (n={count})</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
