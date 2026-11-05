"""Build an offline academic view from sealed aggregate results, without refitting."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

TRAITS = {"T2D": "Type 2 diabetes", "cad": "Coronary artery disease", "bc": "Breast cancer (female)"}
COHORTS = {
    "original_val": "Development validation",
    "nested_oof": "Nested training OOF",
    "exclude_shared_group": "Validation: shared kinship groups excluded",
    "exclude_direct_0177": "Validation: direct KING >= 0.177 excluded",
}
ARMS = {
    "M0_KD": "M0 distilled student",
    "H_SELECTED": "Selected student",
    "TabPFN_v3": "TabPFN v3 teacher",
    "M0_SUP": "M0 supervised",
    "LR_ALL": "All-PRS LR",
    "LR_SINGLE_BEST": "Single-best PRS LR",
}
METRICS = {
    "auroc": ("AUROC", True),
    "log_loss": ("Log loss", False),
    "brier": ("Brier score", False),
    "nagelkerke_r2": ("Nagelkerke pseudo-R²", True),
}
MODES = {"official_pc": "Official PCs", "reference_pc": "Reference PCs"}
COLORS = {"M0_KD": "#007f78", "H_SELECTED": "#56859a", "TabPFN_v3": "#283f60"}
TABLES = ["metrics", "paired_differences", "cohort_counts", "single_prs_selections", "subgroup_metrics"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(root, expected):
    for name, digest in expected.items():
        if sha(root / name) != digest:
            raise ValueError(f"Sealed source changed: {name}")


def forest(gaps, counts, cohort, metric, kind, mode, assets):
    """Three disease panels share an x-axis, preserving raw contrast signs."""
    arms = list(ARMS) if kind == "transfer" else list(ARMS)[:3]
    rows = []
    for trait in TRAITS:
        for arm in arms:
            challenger = f"reference_pc/{arm}" if kind == "transfer" else f"{mode}/{arm}"
            reference = f"official_pc/{arm}" if kind == "transfer" else f"{mode}/LR_SINGLE_BEST"
            rows.append(gaps.loc[(trait, cohort, challenger, reference, metric)])
    bound = max(abs(float(row[k])) for row in rows for k in ["ci_low", "ci_high", "delta"])
    limit = math.ceil(bound * 1.18 * 1000) / 1000
    label, higher = METRICS[metric]
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 3.65 if kind == "transfer" else 3.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.16, right=.98, top=.78 if kind == "transfer" else .70,
                        bottom=.23 if kind == "transfer" else .27, wspace=.20)
    for i, (ax, trait) in enumerate(zip(axes, TRAITS, strict=True)):
        n = counts.loc[(trait, cohort), "n"]
        ax.set_title(f"{TRAITS[trait]}\nn = {n:,}", fontsize=10, color="#24364b", pad=12, linespacing=1.6)
        ax.axvline(0, color="#8c9aa6", linewidth=.9, linestyle=(0, (3, 3)), zorder=1)
        ax.axhspan(-.42, .42, color="#edf7f5", zorder=0)
        for j, arm in enumerate(arms):
            row = rows[i * len(arms) + j]
            color = COLORS.get(arm, "#8c99a7")
            ax.plot([row.ci_low, row.ci_high], [j, j], color=color, lw=2 if arm == "M0_KD" else 1.4)
            ax.plot([row.ci_low, row.ci_high], [j, j], "|", color=color, markersize=5)
            ax.plot(row.delta, j, "D" if arm == "M0_KD" else "o", color=color, markersize=5, zorder=3)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(len(arms) - .5, -.5)
        ax.set_yticks(range(len(arms)), [ARMS[a] for a in arms], fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
        ax.tick_params(axis="both", length=0, labelsize=9, pad=7, colors="#526377")
        ax.grid(axis="x", color="#edf0f3", linewidth=.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    direction = "Positive favors reference PCs" if kind == "transfer" else "Positive favors student / teacher"
    if not higher:
        direction = direction.replace("Positive", "Negative")
    contrast = "Reference PCs minus official PCs" if kind == "transfer" else f"Model minus single-best LR | {MODES[mode]}"
    fig.text(.16, .96, f"{COHORTS[cohort]}  |  {contrast}", fontsize=9, color="#526377", va="top")
    fig.text(.57, .105 if kind == "transfer" else .12, f"Δ {label}  ·  {direction}", ha="center", fontsize=9, color="#24364b")
    fig.text(.16, .025, "Points: paired differences. Bars: 95% kinship-group bootstrap CIs (1,000 replicates).\nConditional on fitted models; no multiplicity adjustment.", fontsize=7.5, color="#647487")
    key = "__".join([kind, cohort, metric] + ([mode] if kind == "baseline" else []))
    result = {"key": key}
    for fmt, mime in [("svg", "image/svg+xml"), ("pdf", "application/pdf")]:
        buf = io.BytesIO()
        metadata = {"Date": None} if fmt == "svg" else {"CreationDate": None, "ModDate": None}
        fig.savefig(buf, format=fmt, metadata=metadata, facecolor="white")
        raw = buf.getvalue()
        (assets / f"{key}.{fmt}").write_bytes(raw)
        result[fmt] = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
    plt.close(fig)
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Existing sealed analysis run")
    args = parser.parse_args()
    root = args.out_dir.resolve()
    seal = json.loads((root / "VERIFIED.json").read_text())
    expected = seal["output_sha256"]
    verify_seal(root, expected)
    report = root / "report"
    assets = report / "academic_assets"
    assets.mkdir(exist_ok=True)
    frames = {name: pd.read_csv(report / f"{name}.csv") for name in TABLES}
    measured = frames["metrics"].set_index(["trait", "cohort", "mode", "arm"])
    gaps = frames["paired_differences"].set_index(["trait", "cohort", "arm", "reference", "metric"])
    counts = frames["cohort_counts"].set_index(["trait", "cohort"])
    if not all(frame.index.is_unique for frame in [measured, gaps, counts]):
        raise ValueError("Aggregate tables contain duplicate keys")
    for (trait, cohort, arm, reference, metric), row in gaps.iterrows():
        a_mode, a_arm = arm.split("/")
        b_mode, b_arm = reference.split("/")
        delta = measured.loc[(trait, cohort, a_mode, a_arm), metric] - measured.loc[(trait, cohort, b_mode, b_arm), metric]
        if not math.isclose(delta, row.delta, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Paired difference disagrees with measured metrics")
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "ancestry-report", "pdf.fonttype": 42})
    figures = {}
    for cohort in COHORTS:
        for metric in METRICS:
            for kind, modes in [("transfer", ["reference_pc"]), ("baseline", list(MODES))]:
                for mode in modes:
                    figure = forest(gaps, counts, cohort, metric, kind, mode, assets)
                    figures[figure.pop("key")] = figure
        print(f"Rendered {cohort}", flush=True)
    payload = {name: json.loads(frame.to_json(orient="records", double_precision=15)) for name, frame in frames.items()}
    payload["figures"] = figures
    payload["csv"] = {name: (report / f"{name}.csv").read_text() for name in TABLES}
    payload["sources"] = json.loads((report / "sources.json").read_text())
    templates = Path(__file__).parent / "templates"
    source_paths = [Path(__file__), *(templates / f"ancestry_transfer_report.{ext}" for ext in ["html", "css", "js"])]
    text = source_paths[1].read_text()
    for marker, value in {
        "@@STYLE@@": source_paths[2].read_text(),
        "@@SCRIPT@@": source_paths[3].read_text(),
        "@@DATA@@": json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"),
    }.items():
        text = text.replace(marker, value)
    output = report / "report_academic_zh.html"
    output.write_text(text)
    verify_seal(root, expected)
    snapshot = assets / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in source_paths:
        (snapshot / path.name).write_bytes(path.read_bytes())
    receipt = {
        "status": "rendered_source_hashes_verified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models_refitted": False,
        "bootstrap_recomputed": False,
        "original_sealed_outputs_unchanged": len(expected),
        "original_verified_json_sha256": sha(root / "VERIFIED.json"),
        "input_sha256": expected,
        "table_rows": {name: len(frame) for name, frame in frames.items()},
        "all_1056_deltas_rechecked": True,
        "figure_count": len(figures),
        "source_sha256": {str(path.relative_to(Path.cwd())): sha(path) for path in source_paths},
        "output_sha256": {str(path.relative_to(root)): sha(path) for path in [output, *sorted(assets.glob("*.svg")), *sorted(assets.glob("*.pdf"))]},
        "browser_qa": "See academic_assets/qa/BROWSER_VERIFIED.json",
    }
    (assets / "PRESENTATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"html": str(output), "bytes": output.stat().st_size, "figures": len(figures), "sealed_outputs_unchanged": len(expected)}))


if __name__ == "__main__":
    main()
