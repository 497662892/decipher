"""Consolidate sealed OADP evaluation tables and build an offline comparison."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_ancestry_transfer_report import COHORTS, METRICS, TRAITS
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from run_ancestry_oadp import BASELINE
from run_ancestry_transfer import read

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files

ARMS = {"TabPFN_v3": "TabPFN v3 teacher", "M0_KD": "M0 distilled student",
        "H_SELECTED": "Selected student", "M0_SUP": "M0 supervised",
        "LR_ALL": "All-PRS LR", "LR_SINGLE_BEST": "Single-best PRS LR"}
TABLES = ["metrics", "paired_differences", "cohort_counts", "subgroup_metrics", "single_prs_selections"]


def plot(gaps, counts, cohort, metric, assets):
    rows = [gaps.loc[(trait, cohort, f"oadp_pc/{arm}", f"reference_pc/{arm}", metric)]
            for trait in TRAITS for arm in ARMS]
    bound = max(abs(float(r[c])) for r in rows for c in ["ci_low", "ci_high", "delta"])
    limit = max(.001, math.ceil(bound * 1.15 * 10000) / 10000)
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 3.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.16, right=.98, top=.78, bottom=.23, wspace=.20)
    for i, (ax, trait) in enumerate(zip(axes, TRAITS, strict=True)):
        ax.set_title(f"{TRAITS[trait]}\nn = {counts.loc[(trait, cohort), 'n']:,}", fontsize=10, color="#24364b", pad=12, linespacing=1.6)
        ax.axvline(0, color="#8c9aa6", linewidth=.9, linestyle=(0, (3, 3)))
        ax.axhspan(-.42, .42, color="#f0f4f8", zorder=0)
        for j, arm in enumerate(ARMS):
            r = rows[i * len(ARMS) + j]
            color = {"TabPFN_v3": "#283f60", "M0_KD": "#007f78", "H_SELECTED": "#56859a"}.get(arm, "#8c99a7")
            ax.plot([r.ci_low, r.ci_high], [j, j], color=color, lw=1.8)
            ax.plot([r.ci_low, r.ci_high], [j, j], "|", color=color, markersize=5)
            ax.plot(r.delta, j, "D" if arm in ["TabPFN_v3", "M0_KD"] else "o", color=color, markersize=5)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(len(ARMS) - .5, -.5)
        ax.set_yticks(range(len(ARMS)), list(ARMS.values()))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
        ax.tick_params(axis="both", length=0, labelsize=9, pad=7, colors="#526377")
        ax.grid(axis="x", color="#edf0f3", linewidth=.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.16, .96, f"{COHORTS[cohort]} | OADP minus uncorrected reference PCs", fontsize=9, color="#526377", va="top")
    direction = "Positive" if METRICS[metric][1] else "Negative"
    fig.text(.57, .105, f"Δ {METRICS[metric][0]} · {direction} favors OADP", ha="center", fontsize=9, color="#24364b")
    fig.text(.16, .025, "Points: paired differences. Bars: 95% kinship-group bootstrap CIs (1,000 replicates).\nConditional on fitted models; no multiplicity adjustment.", fontsize=7.5, color="#647487")
    key = f"{cohort}__{metric}"
    result = {}
    for fmt, mime in [("svg", "image/svg+xml"), ("pdf", "application/pdf")]:
        stream = io.BytesIO()
        metadata = {"Date": None} if fmt == "svg" else {"CreationDate": None, "ModDate": None}
        fig.savefig(stream, format=fmt, metadata=metadata, facecolor="white")
        (assets / f"{key}.{fmt}").write_bytes(stream.getvalue())
        result[fmt] = f"data:{mime};base64," + base64.b64encode(stream.getvalue()).decode()
    plt.close(fig)
    return key, result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.out_dir
    report = root / "report"
    for trait in TRAITS:
        verify_files(report / trait, read(report / trait / "COMPLETE.json")["output_sha256"])
    verify_files(BASELINE, read(BASELINE / "VERIFIED.json")["output_sha256"])
    tables = {name: pd.concat([pd.read_csv(report / trait / f"{name}.csv") for trait in TRAITS], ignore_index=True)
              for name in TABLES}
    for name, table in tables.items():
        table.to_csv(report / f"{name}.csv", index=False)
    tables["correction_by_pc"] = pd.read_csv(root / "projection/correction_by_pc.csv")
    tables["correction_by_pc"].to_csv(report / "correction_by_pc.csv", index=False)
    measured = tables["metrics"].set_index(["trait", "cohort", "mode", "arm"])
    gaps = tables["paired_differences"].set_index(["trait", "cohort", "arm", "reference", "metric"])
    counts = tables["cohort_counts"].set_index(["trait", "cohort"])
    if not all(table.index.is_unique for table in [measured, gaps, counts]):
        raise ValueError("Duplicate aggregate keys")
    if len(measured) != 216 or len(gaps) != 576 or len(counts) != 12:
        raise ValueError("Incomplete comparison coverage")
    for (trait, cohort, arm, reference, metric), row in gaps.iterrows():
        a_mode, a_arm = arm.split("/")
        b_mode, b_arm = reference.split("/")
        delta = measured.loc[(trait, cohort, a_mode, a_arm), metric] - measured.loc[(trait, cohort, b_mode, b_arm), metric]
        np.testing.assert_allclose(delta, row.delta, atol=1e-12, rtol=0)
    assets = report / "assets"
    assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "oadp-report", "pdf.fonttype": 42})
    figures = dict(plot(gaps, counts, cohort, metric, assets) for cohort in COHORTS for metric in METRICS)
    payload = {name: json.loads(table.to_json(orient="records", double_precision=15)) for name, table in tables.items()}
    payload["figures"] = figures
    payload["csv"] = {name: (report / f"{name}.csv").read_text() for name in tables}
    templates = Path("scripts/templates")
    sources = [Path(__file__), templates / "ancestry_oadp_report.html", templates / "ancestry_oadp_report.js", templates / "ancestry_transfer_report.css"]
    text = sources[1].read_text().replace("@@STYLE@@", sources[3].read_text()).replace("@@SCRIPT@@", sources[2].read_text())
    text = text.replace("@@DATA@@", json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"))
    (report / "report_zh.html").write_text(text)
    snapshot = root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    artifacts = [report / "report_zh.html", *[report / f"{name}.csv" for name in tables], *sorted(assets.glob("*.svg")), *sorted(assets.glob("*.pdf"))]
    write_json(root / "VERIFIED.json", {"status": "evaluation_complete_and_verified", "baseline_unchanged": True,
        "metric_rows": len(measured), "paired_contrasts": len(gaps), "cohorts": len(counts),
        "all_deltas_match_point_estimates": True, "bootstrap_replicates": 1000,
        "prediction_seals": {trait: sha256_file(root / "private" / trait / "COMPLETE.json") for trait in TRAITS},
        "projection_seal": sha256_file(root / "projection/COMPLETE.json"),
        "report_source_sha256": {str(p): sha256_file(p) for p in sources},
        "output_sha256": {str(p.relative_to(root)): sha256_file(p) for p in artifacts}})
    print(f"Rendered {report / 'report_zh.html'}; 216 metrics, 576 paired intervals, 16 figures", flush=True)


if __name__ == "__main__":
    main()
