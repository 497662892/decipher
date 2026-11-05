"""Report paired holdout prediction agreement without accessing holdout outcomes."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from run_ancestry_holdout_agreement import SCOPES
from run_ancestry_transfer import TRAITS, read

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prediction_agreement import agreement, paired_probabilities
from decipehr.prs_distillation.data import verify_files

LABELS = {"T2D": "Type 2 diabetes", "cad": "Coronary artery disease", "bc": "Breast cancer (female)"}
STUDENTS = ["M0_KD", "H_SELECTED"]


def figure(predictions, metrics, scope, student, kind, assets):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.89, bottom=.21, top=.81, wspace=.13)
    bins, largest_difference = [], 0
    for ax, trait in zip(axes, TRAITS, strict=True):
        frame = predictions.loc[predictions.trait.eq(trait) & predictions.scope.eq(scope)]
        _, x, y = paired_probabilities(frame, student)
        delta = y-x
        largest_difference = max(largest_difference, float(np.abs(delta).max()))
        row = metrics.loc[(trait, scope, student)]
        ax.set_title(f"{LABELS[trait]}\nn = {len(x):,}", fontsize=10, pad=12, linespacing=1.5)
        if kind == "scatter":
            bins.append(ax.hexbin(x, y, gridsize=34, extent=(0, 1, 0, 1), mincnt=1, cmap="viridis", linewidths=0))
            ax.plot([0, 1], [0, 1], "--", color="#788b9d", lw=.8)
            ax.text(.04, .96, f"Pearson r = {row.pearson_probability:.3f}\nSpearman ρ = {row.spearman:.3f}",
                    transform=ax.transAxes, va="top", fontsize=9, color="#24364b",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 3})
        else:
            bins.append(ax.hexbin(x, delta, gridsize=34, mincnt=1, cmap="viridis", linewidths=0))
            ax.axhline(0, color="#788b9d", lw=.8, ls="--")
            ax.axhline(delta.mean(), color="#bb6d28", lw=1)
            ax.text(.04, .96, f"Mean bias = {100*delta.mean():+.2f} pp\nMAE = {100*row.mae_probability:.2f} pp",
                    transform=ax.transAxes, va="top", fontsize=9, color="#24364b",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 3})
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=9, length=3, colors="#526377")
        ax.set_xlabel("TabPFN v3 probability", fontsize=9, labelpad=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c8d1d9")
    maximum = max(float(h.get_array().max()) for h in bins)
    for h in bins:
        h.set_norm(LogNorm(vmin=1, vmax=max(2, maximum)))
    if kind == "scatter":
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Student probability", fontsize=9)
    else:
        limit = max(.02, np.ceil(largest_difference*20)/20)
        axes[0].set_ylim(-limit, limit)
        axes[0].set_ylabel("Student minus TabPFN probability", fontsize=9)
    bar = fig.colorbar(bins[0], cax=fig.add_axes([.92, .25, .013, .49]))
    bar.set_label("Patients per hexagon", fontsize=8)
    bar.ax.tick_params(labelsize=8)
    scope_label = "Frozen train models" if scope == "train_only" else "Train + validation refit"
    student_label = "M0 distilled student" if student == "M0_KD" else "Previously selected student"
    fig.text(.065, .97, f"{scope_label} | {student_label} | Standard reference PCs", fontsize=10, va="top", color="#24364b")
    fig.text(.065, .045, "Same holdout patients for both models. Descriptive prediction agreement; outcome labels unavailable.", fontsize=8, color="#647487")
    key = f"{scope}__{student}__{kind}"
    output = {}
    for fmt, mime in [("svg", "image/svg+xml"), ("pdf", "application/pdf")]:
        stream = io.BytesIO()
        metadata = {"Date": None} if fmt == "svg" else {"CreationDate": None, "ModDate": None}
        fig.savefig(stream, format=fmt, facecolor="white", metadata=metadata)
        (assets / f"{key}.{fmt}").write_bytes(stream.getvalue())
        output[fmt] = f"data:{mime};base64," + base64.b64encode(stream.getvalue()).decode()
    plt.close(fig)
    return key, output


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.out_dir
    report, assets = root / "report", root / "report/assets"
    assets.mkdir(parents=True, exist_ok=True)
    protocol = read(root / "PROTOCOL.json")
    if protocol["holdout_outcomes_used"] or protocol["oadp"]:
        raise ValueError("Unexpected analysis scope")
    frames, receipts = [], {}
    for trait in TRAITS:
        for scope in SCOPES:
            folder = root / "private" / trait / scope
            receipt = read(folder / "COMPLETE.json")
            verify_files(folder, {"predictions.parquet": receipt["output_sha256"]["predictions.parquet"]})
            if receipt["holdout_outcomes_used"]:
                raise ValueError("Holdout outcomes were used")
            frames.append(pd.read_parquet(folder / "predictions.parquet"))
            receipts[f"{trait}/{scope}"] = sha256_file(folder / "COMPLETE.json")
    predictions = pd.concat(frames, ignore_index=True)
    availability = pd.read_csv(root / "availability.csv")
    selected = availability.set_index("trait").selected_student.to_dict()
    rows, cross = [], []
    for trait in TRAITS:
        for scope in SCOPES:
            frame = predictions.loc[predictions.trait.eq(trait) & predictions.scope.eq(scope)]
            cache = {}
            for arm in STUDENTS:
                student_model = "M0_KD" if arm == "M0_KD" else selected[trait]
                ids, x, y = paired_probabilities(frame, arm)
                if len(ids) != availability.set_index("trait").loc[trait, "n_holdout"]:
                    raise ValueError("Incomplete holdout coverage")
                if student_model not in cache:
                    cache[student_model] = agreement(x, y, ids)
                rows.append({"trait": trait, "scope": scope, "student": arm,
                             "student_model": student_model, **cache[student_model]})
            print(f"Agreement complete: {trait}/{scope}", flush=True)
        for arm in ["TabPFN_v3", *STUDENTS]:
            frame = predictions.loc[predictions.trait.eq(trait) & predictions.arm.eq(arm)].copy()
            frame["arm"] = frame.scope
            ids, x, y = paired_probabilities(frame, "trainval_refit", "train_only")
            cross.append({"trait": trait, "model": arm, "comparison": "trainval_refit_minus_train_only",
                          **agreement(x, y, ids, replicates=0)})
    tables = {"agreement": pd.DataFrame(rows), "cross_scope_stability": pd.DataFrame(cross), "availability": availability}
    for name, table in tables.items():
        table.to_csv(report / f"{name}.csv", index=False)
    measured = tables["agreement"].set_index(["trait", "scope", "student"])
    if not measured.index.is_unique or len(measured) != 12:
        raise ValueError("Incomplete model/context comparison")
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "holdout-agreement", "pdf.fonttype": 42})
    figures = dict(figure(predictions, measured, scope, student, kind, assets)
                   for scope in SCOPES for student in STUDENTS for kind in ["scatter", "difference"])
    payload = {name: json.loads(table.to_json(orient="records", double_precision=15)) for name, table in tables.items()}
    payload.update(figures=figures, csv={name: (report / f"{name}.csv").read_text() for name in tables})
    templates = Path("scripts/templates")
    sources = [Path(__file__), Path("src/decipehr/prediction_agreement.py"),
               templates / "ancestry_holdout_agreement.html", templates / "ancestry_holdout_agreement.js",
               templates / "ancestry_transfer_report.css"]
    html = sources[2].read_text().replace("@@STYLE@@", sources[4].read_text()).replace("@@SCRIPT@@", sources[3].read_text())
    html = html.replace("@@DATA@@", json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"))
    (report / "report_zh.html").write_text(html)
    snapshot = root / "source_snapshot"
    for path in sources:
        (snapshot / path.name).write_bytes(path.read_bytes())
    outputs = [report / "report_zh.html", *[report / f"{name}.csv" for name in tables],
               *sorted(assets.glob("*.svg")), *sorted(assets.glob("*.pdf"))]
    write_json(root / "VERIFIED.json", {"status": "agreement_analysis_complete",
        "comparison_rows": 12, "unique_student_teacher_pairs": 10, "holdout_patient_disease_rows_per_model": 10031,
        "same_holdout_patients_across_scopes_and_models": True, "holdout_outcomes_used": False,
        "bootstrap_replicates": 1000, "bootstrap_unit": "paired patient", "bootstrap_kinship_modeled": False,
        "bootstrap_conditions_on_fitted_models": True, "multiplicity_adjusted": False,
        "prediction_receipts_sha256": receipts, "source_sha256": {str(p): sha256_file(p) for p in sources},
        "output_sha256": {str(p.relative_to(root)): sha256_file(p) for p in outputs}})
    print(f"Rendered {report / 'report_zh.html'}", flush=True)


if __name__ == "__main__":
    main()
