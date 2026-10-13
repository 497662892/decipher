from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DEFAULT_RESULTS_DIR = Path("outputs/ancestry_stratified_best_single_vs_tabpfn3")
GROUP_ORDER = [
    "ALL",
    "AFR-like",
    "AMR-like",
    "EAS-like",
    "EUR-like",
    "SAS-like",
    "AMBIGUOUS",
]
DISEASE_ORDER = ["cad", "bc", "T2D"]
DISEASE_DISPLAY = {"cad": "CAD", "bc": "BC", "T2D": "T2D"}
MODEL_DISPLAY = {
    "best_single_prs": "Best single PRS (before)",
    "tabpfn3_best_single": "TabPFN-3 + best single PRS (bridge)",
    "tabpfn3_full": "TabPFN-3 full (after)",
}
METRICS = {
    "auc": ("AUC", True, 3),
    "nagelkerke_r2": ("Nagelkerke R²", True, 3),
    "top10_vs_lower90_or": ("OR90", True, 2),
}
SINGLE_COLOR = "#E69F00"
BRIDGE_COLOR = "#6F42C1"
TABPFN_COLOR = "#0072B2"
BETTER_COLOR = "#087F5B"
WORSE_COLOR = "#C92A2A"
TIE_COLOR = "#667085"
CONNECTOR_COLOR = "#98A2B3"


def _metric_frame(metrics: pd.DataFrame, disease: str, metric: str) -> pd.DataFrame:
    frame = metrics.loc[metrics["disease"].eq(disease)].copy()
    frame["ancestry_group"] = pd.Categorical(
        frame["ancestry_group"],
        categories=GROUP_ORDER,
        ordered=True,
    )
    frame = (
        frame.pivot(
            index=["ancestry_group", "n"],
            columns="model",
            values=metric,
        )
        .reset_index()
        .sort_values("ancestry_group")
    )
    frame["ancestry_group"] = frame["ancestry_group"].astype(str)
    return frame


def _delta_lookup(deltas: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    return {
        (str(row.disease), str(row.ancestry_group), str(row.metric)): row
        for _, row in deltas.iterrows()
    }


def _axis_limits(frame: pd.DataFrame) -> tuple[float, float]:
    values = frame[["best_single_prs", "tabpfn3_full"]].to_numpy(dtype=float)
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    span = upper - lower
    if span == 0:
        span = max(abs(upper), 1.0) * 0.1
    padding = span * 0.14
    return lower - padding, upper + padding


def _group_labels(frame: pd.DataFrame) -> list[str]:
    labels = []
    for row in frame.itertuples(index=False):
        group = "Total" if row.ancestry_group == "ALL" else row.ancestry_group
        labels.append(f"{group}  (n={int(row.n):,})")
    return labels


def _draw_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    disease: str,
    metric: str,
    delta_rows: dict[tuple[str, str, str], pd.Series],
    show_group_labels: bool,
) -> None:
    title, higher_is_better, digits = METRICS[metric]
    y = np.arange(len(frame))
    single = frame["best_single_prs"].to_numpy(dtype=float)
    tabpfn = frame["tabpfn3_full"].to_numpy(dtype=float)
    delta = tabpfn - single

    ax.axhspan(-0.45, 0.45, color="#F2F4F7", zorder=0)
    for index, (single_value, tabpfn_value) in enumerate(zip(single, tabpfn, strict=True)):
        ax.plot(
            [single_value, tabpfn_value],
            [index, index],
            color=CONNECTOR_COLOR,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(single, y, s=55, color=SINGLE_COLOR, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(tabpfn, y, s=55, color=TABPFN_COLOR, edgecolor="white", linewidth=0.8, zorder=3)

    for index, row in enumerate(frame.itertuples(index=False)):
        single_value = float(row.best_single_prs)
        tabpfn_value = float(row.tabpfn3_full)
        delta_value = float(delta[index])
        ax.annotate(
            f"{single_value:.{digits}f}",
            (single_value, index),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.8,
            color=SINGLE_COLOR,
        )
        ax.annotate(
            f"{tabpfn_value:.{digits}f}",
            (tabpfn_value, index),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=TABPFN_COLOR,
        )
        is_better = delta_value > 0 if higher_is_better else delta_value < 0
        is_tie = np.isclose(delta_value, 0, atol=1e-12, rtol=0)
        delta_color = TIE_COLOR if is_tie else (BETTER_COLOR if is_better else WORSE_COLOR)
        delta_row = delta_rows.get((disease, row.ancestry_group, metric))
        significant = bool(
            delta_row is not None
            and (
                (float(delta_row.ci_low) > 0 and float(delta_row.ci_high) > 0)
                or (float(delta_row.ci_low) < 0 and float(delta_row.ci_high) < 0)
            )
        )
        marker = "*" if significant else ""
        ax.text(
            1.025,
            index,
            f"Δ {delta_value:+.{digits}f}{marker}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9.2,
            fontweight="bold" if significant else "normal",
            color=delta_color,
            clip_on=False,
        )

    ax.set_xlim(*_axis_limits(frame))
    ax.set_ylim(-0.65, len(frame) - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(y)
    if show_group_labels:
        ax.set_yticklabels(_group_labels(frame), fontsize=9.2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5, colors="#475467")
    ax.xaxis.grid(True, color="#EAECF0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D0D5DD")
    ax.set_title(f"{DISEASE_DISPLAY[disease]} · {title}", fontsize=12, fontweight="bold", pad=18)
    ax.text(
        1.025,
        -0.58,
        "After − before",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#667085",
        clip_on=False,
    )


def _legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SINGLE_COLOR,
            markeredgecolor="white",
            markersize=8,
            label=MODEL_DISPLAY["best_single_prs"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TABPFN_COLOR,
            markeredgecolor="white",
            markersize=8,
            label=MODEL_DISPLAY["tabpfn3_full"],
        ),
        Line2D(
            [0],
            [0],
            color=BETTER_COLOR,
            linewidth=0,
            marker="$+$",
            markersize=10,
            label="Green delta = TabPFN better",
        ),
        Line2D(
            [0],
            [0],
            color=WORSE_COLOR,
            linewidth=0,
            marker="$-$",
            markersize=10,
            label="Red delta = TabPFN worse",
        ),
    ]


def _three_arm_delta_lookup(
    comparisons: pd.DataFrame,
) -> dict[tuple[str, str, str], pd.Series]:
    auc = comparisons.loc[comparisons["metric"].eq("auc")]
    return {
        (str(row.disease), str(row.ancestry_group), str(row.comparison)): row
        for _, row in auc.iterrows()
    }


def _draw_three_arm_auc_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    disease: str,
    delta_rows: dict[tuple[str, str, str], pd.Series],
) -> None:
    y = np.arange(len(frame))
    a = frame["best_single_prs"].to_numpy(dtype=float)
    b = frame["tabpfn3_best_single"].to_numpy(dtype=float)
    c = frame["tabpfn3_full"].to_numpy(dtype=float)
    for index, values in enumerate(zip(a, b, c, strict=True)):
        ax.plot(
            values,
            [index, index, index],
            color=CONNECTOR_COLOR,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(a, y, s=58, color=SINGLE_COLOR, edgecolor="white", zorder=3)
    ax.scatter(b, y, s=58, color=BRIDGE_COLOR, edgecolor="white", zorder=3)
    ax.scatter(c, y, s=58, color=TABPFN_COLOR, edgecolor="white", zorder=3)

    for index, row in enumerate(frame.itertuples(index=False)):
        ax.annotate(
            f"{row.best_single_prs:.3f}",
            (row.best_single_prs, index),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.4,
            color=SINGLE_COLOR,
        )
        ax.annotate(
            f"{row.tabpfn3_best_single:.3f}",
            (row.tabpfn3_best_single, index),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=BRIDGE_COLOR,
        )
        ax.annotate(
            f"{row.tabpfn3_full:.3f}",
            (row.tabpfn3_full, index),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.4,
            color=TABPFN_COLOR,
        )
        bridge_delta = delta_rows.get(
            (
                disease,
                row.ancestry_group,
                "tabpfn3_best_single_vs_best_single",
            )
        )
        panel_delta = delta_rows.get(
            (
                disease,
                row.ancestry_group,
                "tabpfn3_full_vs_tabpfn3_best_single",
            )
        )
        if bridge_delta is not None and panel_delta is not None:
            ax.text(
                1.025,
                index,
                (
                    f"B-A {bridge_delta.candidate_minus_reference:+.3f}\n"
                    f"C-B {panel_delta.candidate_minus_reference:+.3f}"
                ),
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8.2,
                color="#344054",
                clip_on=False,
            )

    values = frame[
        ["best_single_prs", "tabpfn3_best_single", "tabpfn3_full"]
    ].to_numpy(dtype=float)
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    span = max(upper - lower, 0.02)
    ax.set_xlim(lower - 0.14 * span, upper + 0.14 * span)
    ax.set_ylim(-0.65, len(frame) - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(_group_labels(frame), fontsize=9.2)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5, colors="#475467")
    ax.xaxis.grid(True, color="#EAECF0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D0D5DD")
    ax.set_title(f"{DISEASE_DISPLAY[disease]} · AUC", fontsize=12, fontweight="bold")


def plot_three_arm_auc(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    out_dir: Path,
) -> None:
    delta_rows = _three_arm_delta_lookup(comparisons)
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 16))
    for ax, disease in zip(axes, DISEASE_ORDER, strict=True):
        _draw_three_arm_auc_panel(
            ax,
            _metric_frame(metrics, disease, "auc"),
            disease,
            delta_rows,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=MODEL_DISPLAY[model],
        )
        for model, color in [
            ("best_single_prs", SINGLE_COLOR),
            ("tabpfn3_best_single", BRIDGE_COLOR),
            ("tabpfn3_full", TABPFN_COLOR),
        ]
    ]
    fig.suptitle(
        "Ancestry-stratified AUC: single PRS → bridge → full TabPFN-3",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Official validation; every arm uses the same sex + age + PC1-PC10 "
            "wrapper. Right labels show paired point deltas."
        ),
        ha="center",
        fontsize=9,
        color="#475467",
    )
    fig.subplots_adjust(left=0.19, right=0.82, top=0.91, bottom=0.05, hspace=0.42)
    _save_figure(fig, out_dir / "three_arm_auc")


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_auc(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    out_dir: Path,
) -> None:
    delta_rows = _delta_lookup(deltas)
    fig, axes = plt.subplots(3, 1, figsize=(12, 15.5))
    for ax, disease in zip(axes, DISEASE_ORDER, strict=True):
        _draw_panel(
            ax,
            _metric_frame(metrics, disease, "auc"),
            disease,
            "auc",
            delta_rows,
            show_group_labels=True,
        )
    fig.suptitle(
        "Ancestry-stratified AUC: best single PRS before vs TabPFN-3 after",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Official validation; both arms use the same sex + age + PC1–PC10 wrapper. "
            "BC is female-only. * paired 95% bootstrap CI excludes 0. "
            "Groups are 1000G-reference genetic-similarity labels."
        ),
        ha="center",
        fontsize=9,
        color="#475467",
    )
    fig.subplots_adjust(left=0.19, right=0.84, top=0.89, bottom=0.06, hspace=0.5)
    _save_figure(fig, out_dir / "before_after_auc")


def plot_standard_metrics(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    out_dir: Path,
) -> None:
    delta_rows = _delta_lookup(deltas)
    metric_order = ["auc", "nagelkerke_r2", "top10_vs_lower90_or"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 16.5))
    for row_index, disease in enumerate(DISEASE_ORDER):
        for column_index, metric in enumerate(metric_order):
            _draw_panel(
                axes[row_index, column_index],
                _metric_frame(metrics, disease, metric),
                disease,
                metric,
                delta_rows,
                show_group_labels=column_index == 0,
            )
    fig.suptitle(
        "Ancestry-stratified standard performance: before vs after TabPFN-3",
        fontsize=19,
        fontweight="bold",
        y=0.992,
    )
    fig.legend(
        handles=_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Official validation; same standard covariate-adjusted wrapper for both arms. "
            "* paired 95% bootstrap CI excludes 0. OR90 is volatile in small subgroups. "
            "Groups are 1000G-reference genetic-similarity labels."
        ),
        ha="center",
        fontsize=9,
        color="#475467",
    )
    fig.subplots_adjust(left=0.075, right=0.94, top=0.925, bottom=0.05, hspace=0.48, wspace=0.5)
    _save_figure(fig, out_dir / "before_after_standard_metrics")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot best single PRS before versus TabPFN-3 after by ancestry-like group."
    )
    parser.add_argument(
        "--metrics",
        default=str(DEFAULT_RESULTS_DIR / "ancestry_group_metrics.csv"),
    )
    parser.add_argument(
        "--deltas",
        default=str(DEFAULT_RESULTS_DIR / "paired_bootstrap_delta_summary.csv"),
    )
    parser.add_argument(
        "--comparisons",
        default=str(DEFAULT_RESULTS_DIR / "paired_three_arm_comparison_summary.csv"),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR / "plots"))
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    deltas = pd.read_csv(args.deltas)
    comparisons = pd.read_csv(args.comparisons)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_auc(metrics, deltas, out_dir)
    plot_standard_metrics(metrics, deltas, out_dir)
    plot_three_arm_auc(metrics, comparisons, out_dir)
    print(f"Wrote before/after and three-arm plots to {out_dir}")


if __name__ == "__main__":
    main()
