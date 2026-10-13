#!/usr/bin/env python3
"""Assemble an English BC/CAD/T2D report from the sealed May OOF analyses.

Uses aggregate tables and existing English vector figures. No model fitting,
patient-level prediction export, or changes to the original analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BC_ROOT = REPO / "outputs/may_final_oof_prediction_analysis"
OTHER_ROOT = REPO / "outputs/may_final_cad_t2d_oof_prediction_analysis"
DEFAULT_OUTPUT = REPO / "outputs/may_final_three_disease_oof_analysis_en"
MODEL_NAMES = {
    "submitted": "Submitted TabPFN configuration",
    "covariates": "Covariates-only logistic baseline",
}
METRICS = {
    "top10_vs_lower90_or": "OR90",
    "nagelkerke_r2": "Nagelkerke R²",
    "auc": "ROC AUC",
    "brier": "Brier score",
    "log_loss": "Log loss",
}
FAMILY_NAMES = {
    "pt": "P+T", "ldpred2": "LDpred2", "lassosum2": "lassosum2",
    "prscs": "PRS-CS", "sdpr": "SDPR", "megaprs_noannot": "MegaPRS (unannotated)",
    "megaprs_annotated": "MegaPRS (annotated)", "sbayesrc": "SBayesRC",
}
TABLE_NAMES = [
    "pooled_metrics", "fold_metrics", "deciles", "intervals", "paired_deltas",
    "thresholds", "tail_bands", "histograms", "roc_pr_curves",
    "fold_prediction_distributions", "cohort_audit", "score_transform_audit",
]
BC_PREFIXED = {
    "deciles", "intervals", "paired_deltas", "thresholds", "tail_bands",
    "histograms", "roc_pr_curves",
}

CSS = """
:root{--ink:#203849;--muted:#536c7a;--teal:#087f8c;--gold:#b97918;--line:#dce7ed}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:90px}
body{margin:0;background:#f2f6f8;color:var(--ink);font:16px/1.75 -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif}
a{color:#076f80;text-underline-offset:3px}a:hover{color:#034b58}
.nav{position:sticky;top:0;z-index:10;background:#163646;color:white;padding:12px 24px;display:flex;flex-wrap:wrap;gap:10px 24px;box-shadow:0 2px 6px #0002}
.nav a{color:white;font-size:14px;text-decoration:none}.nav a:hover{text-decoration:underline}.nav strong{margin-right:auto}
main{max-width:1180px;margin:28px auto;background:white;padding:42px 48px;border-radius:14px}
h1{font-size:38px;line-height:1.22;letter-spacing:-.035em;margin:10px 0 22px}
h2{font-size:28px;line-height:1.35;margin:48px 0 18px}h3{font-size:21px;line-height:1.4;margin:34px 0 12px}
.eyebrow{color:var(--teal);font-size:12px;font-weight:750;letter-spacing:.13em;text-transform:uppercase}
.lead{font-size:20px;line-height:1.65;color:var(--muted);max-width:960px}
.small,figcaption{font-size:13px;color:var(--muted)}.small{line-height:1.6}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}
.card{background:#edf6f7;border-top:3px solid var(--teal);padding:20px;border-radius:7px}.card h3{margin:0 0 10px}.card p{margin:8px 0}.card .value{font-size:28px;font-weight:700;line-height:1.3}
.note,.takeaway{padding:17px 20px;margin:22px 0;border-left:4px solid #c29942;background:#fff8e9;border-radius:0 7px 7px 0}
.takeaway{border-color:var(--teal);background:#eff8f8}.note p,.takeaway p{margin:0}
.chapter{border-top:3px solid var(--teal);margin-top:56px;padding-top:12px}.chapter>h2{margin-top:14px}
figure{margin:26px 0 30px}figure svg{width:100%;height:auto;display:block}figcaption{margin-top:10px;line-height:1.6}
.table-wrap{overflow-x:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;line-height:1.6}
td,th{border-bottom:1px solid var(--line);padding:11px 12px;text-align:left;vertical-align:top}th{background:#edf3f7;color:#254354}tbody tr:nth-child(even){background:#f9fbfc}
.formula{background:#eff4f8;padding:18px 22px;margin:20px 0;border-radius:7px;overflow-x:auto;font-family:Georgia,'Times New Roman',serif;font-size:18px;line-height:1.9}
.formula small{font:13px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;display:block;margin-top:8px}
.explorer{background:#eaf5f6;padding:22px;border-radius:9px;margin:22px 0}.controls{display:flex;gap:20px;flex-wrap:wrap}.explorer label{display:flex;flex-direction:column;font-size:13px;font-weight:650;gap:6px}
select{font-family:inherit;font-size:15px;line-height:1.5;padding:9px 12px;border:1px solid #b5cbd2;border-radius:5px;background:white;color:var(--ink);max-width:100%}.explorer output{display:block;font-size:17px;margin:17px 0 5px;line-height:1.8}
details{margin:20px 0;padding:12px 17px;background:#f7fafc;border:1px solid var(--line);border-radius:6px}summary{cursor:pointer;font-weight:650}details p:last-child{margin-bottom:4px}
code{font-size:.87em;background:#edf2f6;overflow-wrap:anywhere}pre{font:12px/1.7 monospace;background:#edf2f6;padding:18px;white-space:pre-wrap;overflow-wrap:anywhere}
.download{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.download a{border:1px solid #b4ccd4;border-radius:6px;padding:8px 13px;font-size:14px;text-decoration:none}
.inline-nav{font-size:14px;display:flex;gap:10px 22px;flex-wrap:wrap;margin:18px 0}.divider{border-top:1px solid var(--line);margin-top:36px;padding-top:18px}
@media(max-width:760px){main{padding:24px 18px;margin:0;border-radius:0}h1{font-size:29px}h2{font-size:25px}.cards{grid-template-columns:1fr}.nav{padding:9px 14px;gap:6px 15px}.nav strong{width:100%}table{font-size:12px}td,th{padding:8px}.lead{font-size:18px}}
@media print{body{background:white;font-size:11pt}.nav,.download,.explorer,.inline-nav{display:none}main{margin:0;padding:0;max-width:none}.chapter{break-before:page}h2,h3,figcaption{break-after:avoid}figure,.cards,.note,.takeaway{break-inside:avoid}a{color:inherit;text-decoration:none}}
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_originals() -> dict:
    """Verify the sealed analyses without exporting patient-level source data."""
    cache = {}
    result = {}
    for root in [BC_ROOT, OTHER_ROOT]:
        seal_path = root / "VERIFIED.json"
        seal = json.loads(seal_path.read_text())
        if seal["status"] != "PASS":
            raise ValueError(f"Original report is not verified: {root}")
        for key, base in [("source_sha256", REPO), ("artifact_sha256", root)]:
            for name, expected in seal[key].items():
                path = (base / name).resolve()
                if path not in cache:
                    cache[path] = sha256(path)
                if cache[path] != expected:
                    raise ValueError(f"Original sealed file changed: {path}")
        result[str(root.relative_to(REPO))] = {
            "status": "PASS", "seal_sha256": sha256(seal_path),
            "original_source_files_verified": len(seal["source_sha256"]),
            "original_artifacts_verified": len(seal["artifact_sha256"]),
        }
    return result


@dataclass
class Disease:
    key: str
    title: str
    root: Path
    tables: dict[str, pd.DataFrame]
    record: dict
    figures: dict[str, str]

    @property
    def label(self):
        return self.key.upper()

    @property
    def point(self):
        return self.tables["pooled_metrics"].set_index("model").loc["submitted"]

    @property
    def baseline(self):
        return self.tables["pooled_metrics"].set_index("model").loc["covariates"]

    @property
    def deciles(self):
        frame = self.tables["deciles"]
        return frame.loc[frame.model.eq("submitted")].sort_values("decile")

    @property
    def selection(self):
        return self.record.get("prs_selection", self.record.get("bc_prs_selection"))


def load_disease(key: str, title: str, root: Path) -> Disease:
    tables = {}
    for name in TABLE_NAMES + ([] if key == "bc" else ["top10_membership"]):
        prefix = "bc_" if key == "bc" and name in BC_PREFIXED else ""
        frame = pd.read_csv(root / f"{prefix}{name}.csv")
        if "disease" in frame:
            frame = frame.loc[frame.disease.str.lower().eq(key)].copy()
        if frame.empty:
            raise ValueError(f"Empty aggregate table: {key}/{name}")
        tables[name] = frame
    manifest = json.loads((root / "figure_manifest.json").read_text())
    figures = {row["name"][:2]: row["name"] for row in manifest}
    result = Disease(key, title, root, tables,
                     json.loads((root / "analysis_record.json").read_text()), figures)
    p, b, d = result.point, result.baseline, result.deciles
    assert p.n == b.n and p.cases == b.cases
    assert int(p.n) == int(d.n.sum()) and int(p.cases) == int(d.cases.sum())
    assert p.top10_n == math.ceil(p.n / 10)
    assert d.iloc[-1].n == p.top10_n and d.iloc[-1].cases == p.top10_cases
    assert set(tables["fold_metrics"].fold) == set(range(5))
    assert tables["intervals"].n_valid.eq(2000).all()
    ll_null = p.cases * math.log(p.prevalence) + (p.n-p.cases)*math.log1p(-p.prevalence)
    from_gains = -math.expm1(-2*d.ll_gain_null.sum()/p.n) / -math.expm1(2*ll_null/p.n)
    assert math.isclose(from_gains, p.nagelkerke_r2, abs_tol=1e-12)
    assert math.isclose(d.ll_gain_covariates.sum(), p.n*(b.log_loss-p.log_loss), abs_tol=1e-9)
    return result


def table(frame: pd.DataFrame, name: str) -> str:
    return f'<div class="table-wrap" data-table="{name}">' + frame.to_html(
        index=False, border=0, float_format=lambda x: f"{x:.4f}"
    ) + "</div>"


def ci_text(disease: Disease, model: str, metric: str) -> str:
    ci = disease.tables["intervals"].set_index(["model", "metric"]).loc[(model, metric)]
    p = disease.tables["pooled_metrics"].set_index("model").loc[model, metric]
    digits = 3 if metric == "top10_vs_lower90_or" else 4
    return f"{p:.{digits}f} ({ci.low:.{digits}f}–{ci.high:.{digits}f})"


def figure(disease: Disease, code: str, caption: str, out: Path, pages: list) -> str:
    name = disease.figures[code]
    dest = out / "figures" / disease.key
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in ["svg", "png", "pdf"]:
        shutil.copyfile(disease.root / "figures" / f"{name}.{suffix}", dest / f"{name}.{suffix}")
    svg = (dest / f"{name}.svg").read_text()
    svg = svg[svg.index("<svg"):]
    for identifier in set(re.findall(r'id="([^"]+)"', svg)):
        target = f"{disease.key}_{name}_{identifier}"
        svg = svg.replace(f'id="{identifier}"', f'id="{target}"')
        svg = svg.replace(f'href="#{identifier}"', f'href="#{target}"')
        svg = svg.replace(f'url(#{identifier})', f'url(#{target})')
    number = len(pages) + 1
    caption_id = f"caption-{number}"
    svg = svg.replace("<svg ", f'<svg role="img" aria-labelledby="{caption_id}" ', 1)
    relative = (dest / name).relative_to(out).as_posix()
    pages.append({"page": number, "disease": disease.label, "figure": name,
                  "pdf": f"{relative}.pdf", "caption": caption})
    return (f'<figure id="figure-{number}">{svg}<figcaption id="{caption_id}">'
            f'<strong>Figure {number}.</strong> {caption} '
            f'<a href="{relative}.svg">SVG</a> · <a href="{relative}.pdf">PDF</a>'
            '</figcaption></figure>')


def summary_frame(diseases: list[Disease]) -> pd.DataFrame:
    rows = []
    for dis in diseases:
        p, b, d = dis.point, dis.baseline, dis.deciles
        row = {"disease": dis.label, "n": int(p.n), "cases": int(p.cases),
               "case_fraction": p.prevalence}
        for metric in METRICS:
            row[metric] = p[metric]
            row[f"baseline_{metric}"] = b[metric]
            row[f"delta_{metric}"] = p[metric] - b[metric]
        row.update({"top10_n": int(p.top10_n), "top10_cases": int(p.top10_cases),
                    "top10_case_fraction": p.top10_case_rate,
                    "top10_sensitivity": p.top10_sensitivity,
                    "extra_top10_cases": int(p.top10_cases-b.top10_cases)})
        for column in ["ll_gain_null", "ll_gain_covariates"]:
            row[column] = d[column].sum()
            row[f"low20_{column}_share"] = d.loc[d.decile.le(2), column].sum()/d[column].sum()
            row[f"top10_{column}_share"] = d.loc[d.decile.eq(10), column].sum()/d[column].sum()
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_explorer(dis: Disease) -> str:
    p = dis.point
    options = "".join(
        f'<option value="{value:g}"{" selected" if value == .1 else ""}>{100*value:g}%</option>'
        for value in sorted(dis.tables["thresholds"].top_fraction.unique())
    )
    return f"""<div class="explorer" data-explorer="{dis.key}">
      <div class="controls"><label for="{dis.key}-model">Prediction model
      <select id="{dis.key}-model"><option value="submitted">Submitted configuration</option>
      <option value="covariates">Covariates-only baseline</option></select></label>
      <label for="{dis.key}-fraction">Highest-scoring patients selected
      <select id="{dis.key}-fraction">{options}</select></label></div>
      <output id="{dis.key}-result" aria-live="polite">Top 10%: {int(p.top10_cases):,} cases
      among {int(p.top10_n):,} selected; case fraction {p.top10_case_rate:.1%};
      all cases captured {p.top10_sensitivity:.1%}; OR {p.top10_vs_lower90_or:.2f}.</output>
      <p class="small">Select a model and threshold to inspect the existing predictions.
      This is descriptive threshold analysis; no model is refitted and no threshold is selected using these results.
      The primary competition threshold remains the highest 10%.</p>
      <noscript>The default top-10% result is shown above. Enable JavaScript to change the selection.</noscript>
    </div>"""


def disease_chapter(dis: Disease, out: Path, pages: list) -> str:
    p, b, d = dis.point, dis.baseline, dis.deciles
    low, high = d.loc[d.decile.le(2)], d.iloc[-1]
    gain_null, gain_cov = d.ll_gain_null.sum(), d.ll_gain_covariates.sum()
    low_null, high_null = low.ll_gain_null.sum()/gain_null, high.ll_gain_null/gain_null
    low_cov, high_cov = low.ll_gain_covariates.sum()/gain_cov, high.ll_gain_covariates/gain_cov
    fold = dis.tables["fold_metrics"]
    fold_sub = fold.loc[fold.model.eq("submitted")].set_index("fold")
    fold_base = fold.loc[fold.model.eq("covariates")].set_index("fold")
    improved = all((fold_sub[m] > fold_base[m]).all() for m in list(METRICS)[:3])
    assert improved
    mean_probability = (d.n*d.predicted).sum()/p.n
    intro = {
        "bc": ("BC shows useful improvement over covariates alone, with limited absolute discrimination and a modest OR90. "
               "The lower end of the risk distribution contributes substantially to the gain over a constant-probability reference, "
               "while the most enriched high-risk subgroup is smaller than the full top decile."),
        "cad": ("CAD combines strong overall discrimination with substantial enrichment in the highest-scoring 10%. "
                "The covariates-only baseline is already strong; the submitted configuration adds a smaller, "
                "consistent improvement across the risk distribution."),
        "t2d": ("T2D improves both overall probability prediction and high-risk enrichment over covariates alone. "
                "The highest-scoring decile is an important source of incremental log-likelihood gain, "
                "and replacing part of the baseline top decile increases the number of cases captured."),
    }[dis.key]
    sections = [
        f'<section class="chapter" id="{dis.key}" data-disease="{dis.key}">',
        f'<p class="eyebrow">Disease analysis · {dis.label}</p><h2>{dis.title} ({dis.label})</h2>',
        f'<p class="lead">{intro}</p>',
        f'<div class="cards"><div class="card"><div class="value">{int(p.n):,}</div><p>OOF patients'
        + (' · eligible females only' if dis.key == 'bc' else '') + '</p></div>'
        f'<div class="card"><div class="value">{int(p.cases):,}</div><p>Cases · {p.prevalence:.2%} of this cohort</p></div>'
        '<div class="card"><div class="value">5 folds</div><p>One held-out prediction per patient</p></div></div>',
        f'<div class="inline-nav"><a href="#{dis.key}-performance">Performance</a><a href="#{dis.key}-distribution">Distributions & calibration</a>'
        f'<a href="#{dis.key}-gains">Likelihood gains</a><a href="#{dis.key}-tail">High-risk selection</a><a href="#{dis.key}-folds">Fold consistency</a></div>',
        f'<h3 id="{dis.key}-performance">1. Pooled OOF performance and uncertainty</h3>',
    ]
    rows = []
    for model in MODEL_NAMES:
        row = {"Model": MODEL_NAMES[model]}
        for metric, label in list(METRICS.items())[:3]:
            row[f"{label} (95% CI)"] = ci_text(dis, model, metric)
        mp = dis.tables["pooled_metrics"].set_index("model").loc[model]
        row["Cases in top 10%"] = f"{int(mp.top10_cases):,} / {int(mp.top10_n):,}"
        rows.append(row)
    sections += [table(pd.DataFrame(rows), f"{dis.key}-performance"),
                 '<p>These estimates pool all held-out patients. Confidence intervals use 2,000 paired patient bootstrap samples '
                 'with frozen predictions; each resample is ranked again. The primary estimate is not the average of the five fold estimates.</p>',
                 figure(dis, "07", "Pooled estimates and percentile 95% bootstrap intervals. Uncertainty is conditional on the saved predictions; it does not include model refitting or feature selection.", out, pages)]
    deltas = dis.tables["paired_deltas"].set_index("metric")
    paired = []
    for metric, label in METRICS.items():
        dr = deltas.loc[metric]
        paired.append({"Metric": label, "Submitted − baseline": f"{dr.delta:+.4f}",
                       "Paired 95% CI": f"{dr.low:+.4f} to {dr.high:+.4f}",
                       "Improvement direction": "Lower" if metric in {"brier", "log_loss"} else "Higher"})
    sections += [
        '<details><summary>Paired improvements over the covariates-only baseline</summary>'
        + table(pd.DataFrame(paired), f"{dis.key}-paired-differences")
        + '<p class="small">The paired interval uses the same resampled patients for both models. It is the interval for their difference, not the difference between two confidence limits.</p></details>',
        f'<h3 id="{dis.key}-distribution">2. Prediction distributions, risk deciles, and calibration</h3>',
        figure(dis, "01", "Cases and controls are each normalized to unit density. The submitted score is a standardized clipped logit; it is not a probability. Native probabilities are the saved TabPFN outputs.", out, pages),
        f'<p>The lowest decile has a case fraction of <strong>{d.iloc[0].observed:.2%}</strong>. '
        f'The highest decile contains <strong>{int(high.cases):,}/{int(high.n):,} cases ({high.observed:.2%})</strong>, '
        f'compared with <strong>{p.lower90_case_rate:.2%}</strong> in the remaining 90%. '
        f'Selecting the top decile captures <strong>{p.top10_sensitivity:.2%}</strong> of all cases.</p>',
        figure(dis, "02", "Decile 1 is lowest risk and decile 10 is highest risk. Labels give cases/patients; error bars are Wilson 95% intervals. The dashed predicted line is the mean of individual native probabilities. Each model uses its own score deciles in the calibration panel.", out, pages),
        f'<p>The overall mean native probability is <strong>{mean_probability:.2%}</strong>, versus the observed cohort case fraction of '
        f'<strong>{p.prevalence:.2%}</strong>. In the top decile, mean predicted probability is '
        f'<strong>{high.predicted:.2%}</strong>, versus <strong>{high.observed:.2%}</strong> observed. '
        f'Brier score is <strong>{p.brier:.4f}</strong> versus {b.brier:.4f} for the baseline; '
        f'log loss is <strong>{p.log_loss:.4f}</strong> versus {b.log_loss:.4f}. Lower values indicate better probability predictions.</p>',
        '<p class="small">In the calibration panel, points above the diagonal indicate underprediction; points below it indicate overprediction. '
        'These are descriptive checks of the existing probabilities. Cohort case fractions should not be read as population incidence or lifetime disease risk. '
        '<a href="#reading-guide">See the shared reading guide.</a></p>',
        f'<h3 id="{dis.key}-gains">3. Contributions across the risk distribution</h3>',
        f'<p>Relative to the prevalence-only reference, the total log-likelihood gain is <strong>{gain_null:.2f} nats</strong>. '
        f'The lowest 20% contribute <strong>{low_null:.1%}</strong>, and the highest 10% contribute <strong>{high_null:.1%}</strong>. '
        'A model can improve the overall probability score by recognizing controls at the low-risk end, as well as by enriching cases at the high-risk end.</p>',
        figure(dis, "03", "Bars decompose additive log-likelihood gains, not R² itself. Left: comparison with the constant cohort case fraction. Right: comparison with the matched covariates-only OOF probabilities. Both panels group patients by the submitted score.", out, pages),
        f'<p>Relative to the covariates-only model, the additional log-likelihood gain is <strong>{gain_cov:.2f} nats</strong>. '
        f'The lowest 20% contribute <strong>{low_cov:.1%}</strong>, and the highest 10% contribute <strong>{high_cov:.1%}</strong>. '
        'This second comparison answers where the improvement beyond the baseline arises. '
        '<a href="#likelihood-method">The exact calculation and its connection to Nagelkerke R² are given below.</a></p>',
    ]
    contributions = []
    for title, column in [("Prevalence-only reference", "ll_gain_null"), ("Covariates-only OOF", "ll_gain_covariates")]:
        total = d[column].sum()
        contributions.append({"Reference": title, "Total gain (nats)": f"{total:.2f}",
                              "Lowest 20%": f"{d.loc[d.decile.le(2), column].sum()/total:.1%}",
                              "Middle 70%": f"{d.loc[d.decile.between(3, 9), column].sum()/total:.1%}",
                              "Highest 10%": f"{d.loc[d.decile.eq(10), column].sum()/total:.1%}"})
    sections += [table(pd.DataFrame(contributions), f"{dis.key}-likelihood-shares")]
    if dis.key == "bc":
        sections += [f'<div class="takeaway"><p><strong>Why BC can have a useful R² gain with a modest OR90.</strong> '
                     f'OR90 compares just the top-decile boundary with the rest of the cohort. Nagelkerke R² uses every patient’s probability. '
                     f'The lowest 20% account for {low_null:.1%} of the gain over a constant reference, but only {low_cov:.1%} of the gain beyond covariates. '
                     'Thus, low-risk recognition is a major part of the overall model signal; it does not explain the entire incremental improvement over the baseline. '
                     f'The absolute R² remains {p.nagelkerke_r2:.4f}.</p></div>']
    bands = dis.tables["tail_bands"]
    one = bands.iloc[0]
    next_cases, next_n = int(p.top10_cases-one.cases), int(p.top10_n-one.n)
    sections += [
        f'<h3 id="{dis.key}-tail">4. High-risk thresholds and case yield</h3>',
        f'<p>The highest-scoring 1% contain <strong>{int(one.cases)}/{int(one.n)} cases ({one.observed:.1%})</strong>. '
        f'The next 1–10% contain <strong>{next_cases}/{next_n} cases ({next_cases/next_n:.1%})</strong>. '
        'This distinguishes a highly enriched extreme tail from the broader group used by OR90.</p>',
        figure(dis, "04", "Left: OR for each selected fraction versus everyone remaining, with pointwise Wald intervals from the fixed 2×2 table. Right: non-overlapping rank bands and Wilson intervals. The 10% threshold defines OR90.", out, pages),
        threshold_explorer(dis),
    ]
    if "top10_membership" in dis.tables:
        member = dis.tables["top10_membership"].set_index("group")
        entered, left = member.loc["model_only"], member.loc["baseline_only"]
        sections += [
            f'<p>Both models select {int(p.top10_n):,} patients. The submitted model newly selects '
            f'<strong>{int(entered.n)} patients, including {int(entered.cases)} cases ({entered.observed:.1%})</strong>, '
            f'while the {int(left.n)} patients selected only by the baseline include {int(left.cases)} cases ({left.observed:.1%}). '
            f'The net gain is <strong>{int(entered.cases-left.cases)} cases</strong> at the same selection capacity.</p>',
            figure(dis, "08", "Both = selected by both models; Model only = selected only by the submitted configuration; Baseline only = selected only by covariates; Neither = selected by neither. Left: group case fractions and Wilson intervals. Right: exact numbers of cases captured.", out, pages),
        ]
    else:
        sections += [f'<p>At the same selection capacity of {int(p.top10_n):,} patients, the submitted model captures '
                     f'<strong>{int(p.top10_cases)} cases</strong> versus <strong>{int(b.top10_cases)}</strong> for covariates alone: '
                     f'<strong>{int(p.top10_cases-b.top10_cases)} additional cases</strong>. '
                     'The distribution and threshold plots show that the most enriched tail is smaller than the entire top decile.</p>']
    sections += [
        f'<h3 id="{dis.key}-folds">5. Fold consistency and overall discrimination</h3>',
        f'<p>Across the five held-out folds, submitted-model OR90 ranges from <strong>{fold_sub.top10_vs_lower90_or.min():.2f} '
        f'to {fold_sub.top10_vs_lower90_or.max():.2f}</strong>, and R² ranges from '
        f'<strong>{fold_sub.nagelkerke_r2.min():.4f} to {fold_sub.nagelkerke_r2.max():.4f}</strong>. '
        'The submitted configuration improves OR90, R², and AUC over the baseline in all five folds. '
        'The training sets overlap, so these folds are not five independent experiments.</p>',
        figure(dis, "05", "Each fold uses its own top-decile boundary. The vertical dashed line and shaded band show the pooled estimate and its bootstrap interval, not fold-specific confidence intervals.", out, pages),
        figure(dis, "06", "ROC and precision–recall curves summarize ranking across all thresholds. The PR summary is average precision (AP); its no-information reference is the cohort case fraction. Score- and native-probability AUC/AP were checked for agreement.", out, pages),
        f'<p>ROC AUC is <strong>{p.auc:.4f}</strong> versus {b.auc:.4f}; average precision is '
        f'<strong>{p.auprc:.4f}</strong> versus {b.auprc:.4f}. These describe discrimination. '
        'Probability calibration and high-risk case yield address additional, distinct aspects of performance.</p>',
    ]
    fd = dis.tables["fold_prediction_distributions"].copy()
    fd["fold"] = fd["fold"] + 1
    fd = fd.drop(columns=["disease"], errors="ignore").rename(columns={
        "fold": "Fold", "n": "Patients", "cases": "Cases", "score_mean": "Mean score",
        "score_sd": "Score SD", "probability_mean": "Mean native probability",
        "pooled_top10_n": "Patients in pooled top 10%", "pooled_top10_cases": "Cases in pooled top 10%",
    })
    sections += [
        '<details><summary>Fold distributions and contributions to the pooled top decile</summary>'
        + table(fd, f"{dis.key}-fold-distributions")
        + '<p class="small">Displayed fold numbers are 1–5; source CSVs retain the original 0–4 labels. Pooled top-decile counts use one shared disease-level threshold and need not be equal across folds.</p></details>',
        f'<p class="small">PRS inputs: {html.escape(", ".join(FAMILY_NAMES[row["family"]] for row in dis.selection))}, '
        'combined with metadata in the submitted TabPFN model. '
        '<a href="#evaluation-boundaries">Read the common evaluation boundaries.</a></p></section>',
    ]
    return "\n".join(sections)


def overview(diseases: list[Disease], out: Path, pages: list) -> str:
    cards, rows, gains = [], [], []
    short = {
        "bc": "Low-risk recognition contributes strongly to the overall probability gain; OR90 remains modest.",
        "cad": "Strong absolute performance builds on a strong covariates-only baseline.",
        "t2d": "Both whole-distribution prediction and high-risk enrichment improve substantially over the baseline.",
    }
    for dis in diseases:
        p, b = dis.point, dis.baseline
        cards.append(f'<div class="card"><h3><a href="#{dis.key}">{dis.label}</a></h3>'
                     f'<p>{short[dis.key]}</p><p class="small">OR90 <strong>{p.top10_vs_lower90_or:.2f}</strong> · '
                     f'R² <strong>{p.nagelkerke_r2:.4f}</strong> · AUC <strong>{p.auc:.4f}</strong></p></div>')
        rows.append({"Disease": dis.label, "Patients / cases": f"{int(p.n):,} / {int(p.cases):,}",
                     "Cohort case fraction": f"{p.prevalence:.2%}", "OR90": f"{p.top10_vs_lower90_or:.3f}",
                     "Nagelkerke R²": f"{p.nagelkerke_r2:.4f}", "ROC AUC": f"{p.auc:.4f}",
                     "Cases in top 10%": f"{int(p.top10_cases)} / {int(p.top10_n)} ({p.top10_case_rate:.1%})"})
        gains.append({"Disease": dis.label, "Δ OR90": f"{p.top10_vs_lower90_or-b.top10_vs_lower90_or:+.3f}",
                      "Δ R²": f"{p.nagelkerke_r2-b.nagelkerke_r2:+.4f}", "Δ AUC": f"{p.auc-b.auc:+.4f}",
                      "Additional cases in top 10%": f"{int(p.top10_cases-b.top10_cases):+d}"})
    return "\n".join([
        '<header id="overview"><p class="eyebrow">decipEHR · May final audited submission · English integrated report</p>',
        '<h1>Understanding prediction performance<br>across BC, CAD, and T2D</h1>',
        '<p class="lead">A systematic analysis of pooled five-fold out-of-fold predictions: '
        'risk distributions, calibration, high-risk enrichment, and the log-likelihood gains underlying Nagelkerke R².</p>',
        '<p>The analysis uses the confirmed <strong>Go Gene! audited pipeline with trainval 5cv</strong> package. '
        'Each patient was excluded from the TabPFN fit that generated their prediction. '
        'BC includes eligible female patients only; CAD and T2D use their audited eligible cohorts.</p>',
        '<div class="note"><p><strong>Evaluation scope.</strong> These results evaluate the selected submission configuration through its saved OOF predictions. '
        'PRS parameters and panels were selected using validation information, so this is not fully nested evaluation. '
        'Final holdout labels are unavailable. <a href="#evaluation-boundaries">Full interpretation boundaries</a>.</p></div>',
        '<div class="download"><a href="OOF_ANALYSIS_FIGURES_EN.pdf">24-page vector figure collection</a>'
        '<a href="disease_summary.csv">Three-disease summary CSV</a>'
        '<a href="#reading-guide">How to read the figures</a></div></header>',
        '<div class="cards">' + "".join(cards) + '</div>',
        '<h2>Three diseases, different performance profiles</h2>',
        table(pd.DataFrame(rows), "overview-performance"),
        '<p>Absolute performance and improvement over a baseline answer different questions. '
        'CAD has the highest absolute values here, while its covariates-only baseline is already strong. '
        'T2D shows the largest absolute R² increment over that baseline. BC improves over covariates alone, '
        'but its absolute discrimination and top-decile enrichment remain more limited.</p>',
        table(pd.DataFrame(gains), "overview-differences"),
        figure(diseases[0], "08", "Submitted configurations and matched covariates-only baselines across the three diseases. The original comparison figure is retained. Disease chapters below give bootstrap intervals and paired improvements.", out, pages),
        '<p class="small">Cross-disease comparisons are descriptive: eligibility, case fractions, and baseline predictability differ. '
        'These values do not rank the biological or genetic contribution to the three diseases. '
        'All Δ values compare the submitted configuration with its own matched covariates-only baseline.</p>',
    ])


def shared_methods(diseases: list[Disease]) -> str:
    bc = diseases[0]
    p0 = bc.point.prevalence
    low = bc.deciles.loc[bc.deciles.decile.le(2)]
    control_example = math.log(.98/(1-p0))
    case_example = math.log(.02/p0)
    null_ll = bc.point.cases*math.log(p0)+(bc.point.n-bc.point.cases)*math.log1p(-p0)
    model_ll = null_ll+bc.deciles.ll_gain_null.sum()
    return f"""
<section id="reading-guide" class="chapter">
<p class="eyebrow">Shared reading guide</p>
<h2>What the plots and metric labels mean</h2>
<h3>Risk deciles and “mean native probability”</h3>
<p>Patients are ranked by the submitted score and divided into ten approximately equal-sized groups.
Decile 1 contains the lowest scores; decile 10 contains the highest scores. The highest decile has exactly
<code>ceil(0.1 × N)</code> patients. Stable ordering resolves ties, matching the original analysis.</p>
<div class="formula">Mean native probability in group D = (1 / n<sub>D</sub>) ∑<sub>i ∈ D</sub> p<sub>i</sub><br>
Observed case fraction in group D = (1 / n<sub>D</sub>) ∑<sub>i ∈ D</sub> y<sub>i</sub></div>
<p><strong>Native</strong> means the probability saved directly from the fold-specific TabPFN model, before the submission logit transformation
and without fitting an additional evaluation model. The plotted mean averages patients within that group; it does not average five fold-level metrics.
For example, the BC top decile has mean native probability {bc.deciles.iloc[-1].predicted:.2%} and observed case fraction
{int(bc.point.top10_cases)}/{int(bc.point.top10_n)} = {bc.point.top10_case_rate:.2%}.</p>
<p>The risk-decile panel shows how observed case frequency changes along the ranking. The calibration panel compares predicted probabilities
with observed case fractions: agreement with the diagonal indicates local calibration. Each model has its own decile memberships in that panel,
so the two models’ points are not necessarily the same patients. No recalibration is fitted to these OOF labels.</p>
<h3>OR90: enrichment in the highest-scoring 10%</h3>
<div class="formula">OR90 = [a / b] / [c / d]
<small>a = cases in the top 10%; b = controls in the top 10%; c = cases in the remaining 90%; d = controls in the remaining 90%.</small></div>
<p>OR90 compares <strong>odds</strong> in the top 10% with odds in the remaining 90%. It is not the ratio of their case fractions,
and it does not compare the top decile with only the bottom decile. The selected case fraction is precision;
the fraction of all cases captured is sensitivity (recall). A high OR can coexist with limited recall because only a small fraction of patients is selected.</p>
<h3>Other probability and ranking summaries</h3>
<p>ROC AUC summarizes case–control ranking across thresholds. Average precision summarizes the precision–recall curve and depends on the cohort case fraction.
Brier score is the mean squared probability error, <code>mean((y − p)²)</code>. Log loss is the mean negative binary log score.
Lower Brier score and lower log loss indicate better probability prediction. These quantities complement enrichment and discrimination.</p>
</section>
<section id="likelihood-method" class="chapter">
<p class="eyebrow">Calculation and theoretical basis</p>
<h2>How the gain across the risk distribution relates to R²</h2>
<p>Here, R² always means <strong>Nagelkerke pseudo-R²</strong>. It is a normalized likelihood-based summary, not the fraction of variance explained in linear regression
and not a measure of heritability. The decomposition plots operate on its additive log-likelihood building blocks.</p>
<h3>Step 1 · Calculate each patient’s binary log score</h3>
<div class="formula">ℓ<sub>i</sub>(p<sub>i</sub>) = y<sub>i</sub> log(p<sub>i</sub>) + (1 − y<sub>i</sub>) log(1 − p<sub>i</sub>)</div>
<p>For a case, a higher predicted probability improves the score. For a control, a lower predicted probability improves it.
Confident mistakes receive a large penalty. Natural logarithms are used, so gains are measured in nats.
Probabilities are clipped only at numerical boundaries when evaluating logarithms.</p>
<h3>Step 2 · Compare with a clearly specified reference</h3>
<div class="formula">Δ<sub>i</sub> = ℓ<sub>i</sub>(p<sub>i</sub>) − ℓ<sub>i</sub>(q<sub>i</sub>)<br>
G<sub>k</sub> = ∑<sub>i ∈ decile k</sub> Δ<sub>i</sub> &nbsp;;&nbsp; G = ∑<sub>k=1…10</sub> G<sub>k</sub></div>
<p>In the <strong>left panel</strong>, every q<sub>i</sub> equals the observed cohort case fraction p<sub>0</sub>.
This asks where the model improves over predicting the same probability for everyone.
In the <strong>right panel</strong>, q<sub>i</sub> is that patient’s saved covariates-only OOF probability.
This asks where the additional improvement beyond covariates occurs. Both panels use the submitted model’s score deciles.</p>
<p>The reported percentage for a group is its summed gain divided by the total gain, for example
<code>(G₁ + G₂) / G</code> for the lowest 20%. These are <strong>net log-likelihood gain shares</strong>.
Individual groups can have negative contributions, and a share can exceed 100% if other groups offset it.
The computation uses every individual probability, not the decile’s mean probability; the logarithm is nonlinear.</p>
<h3>Step 3 · Convert the total gain over the constant reference to Nagelkerke R²</h3>
<div class="formula">LL<sub>null</sub> = ∑<sub>i</sub> ℓ<sub>i</sub>(p<sub>0</sub>) &nbsp;;&nbsp;
G = LL<sub>model</sub> − LL<sub>null</sub><br>
R²<sub>N</sub> = [1 − exp(−2G/n)] / [1 − exp(2LL<sub>null</sub>/n)]</div>
<p>This applies Nagelkerke’s normalization of the Cox–Snell likelihood-based coefficient.
Within the same cohort, n and LL<sub>null</sub> are fixed, so larger total log-likelihood gain implies larger R².
The nonlinear transformation means the decile contributions cannot be called additive “R² contributions.”
For the right-hand panel, replacing the null likelihood with the covariates likelihood in this formula would define a different statistic;
the report instead retains the common prevalence-only denominator and reports the actual difference between the two models’ R² values.
See <a href="https://doi.org/10.1093/biomet/78.3.691">Nagelkerke (1991)</a>.</p>
<h3>A worked BC example using the saved OOF results</h3>
<p>There are {int(bc.point.n):,} patients and {int(bc.point.cases):,} cases, giving p<sub>0</sub> = {p0:.6f}.
The constant-reference log-likelihood is {null_ll:.6f}, and the submitted-model log-likelihood is {model_ll:.6f}.
Their difference is <strong>{bc.deciles.ll_gain_null.sum():.6f} nats</strong>, which gives
<strong>R² = {bc.point.nagelkerke_r2:.6f}</strong> using the formula above.</p>
<p>The lowest 20% contain <strong>{int(low.n.sum()):,} patients and {int(low.cases.sum())} cases</strong>.
Their summed gain is <strong>{low.ll_gain_null.sum():.6f} nats</strong>:
{low.ll_gain_null.sum():.6f} / {bc.deciles.ll_gain_null.sum():.6f} =
<strong>{low.ll_gain_null.sum()/bc.deciles.ll_gain_null.sum():.1%}</strong> of the total gain over the constant reference.</p>
<p>As a hypothetical individual example, a probability of 2% instead of the BC reference of {p0:.2%}
adds {control_example:+.3f} nats if the patient is a control, but {case_example:+.3f} nats if the patient is a case.
Thus, a low probability helps only when it is supported by the observed outcomes; predicting low probabilities for everyone would not produce the same benefit.</p>
<h3>Why this has a statistical basis</h3>
<p>The logarithmic score is a strictly proper scoring rule: its expected value is maximized by reporting the true event probability.
If the true conditional probability is r, the expected score advantage of reporting r rather than q is the nonnegative
Kullback–Leibler divergence between Bernoulli(r) and Bernoulli(q).
This explains why low-risk and high-risk predictions can both contribute to an overall probability assessment.
It does not guarantee that every observed subgroup in a finite sample has a positive gain.
See <a href="https://doi.org/10.1198/016214506000001437">Gneiting and Raftery (2007)</a>.</p>
<p class="small">The OOF sum here is an accumulated held-out log score from different fold models, used in the Nagelkerke formula as a predictive summary.
It is not an in-sample maximized likelihood for one fitted model, so a likelihood-ratio chi-square test does not follow automatically.
Estimating p<sub>0</sub> from the evaluation cohort defines the null reference; it does not refit the submitted predictor.</p>
</section>
"""


def provenance(diseases: list[Disease]) -> str:
    rows = []
    for dis in diseases:
        audit = dis.tables["score_transform_audit"].iloc[0]
        membership_column = ("native_and_submitted_top10_identical" if dis.key == "bc"
                             else "native_and_submitted_deciles_identical")
        rows.append({"Disease": dis.label, "Clipped scores": int(audit.n_clipped),
                     "R² from native probabilities": f"{audit.r2_native_probability:.6f}",
                     "R² from recovered clipped probabilities": f"{audit.r2_probability_recovered_from_submitted_score:.6f}",
                     "Identical top-decile membership": "Yes" if audit[membership_column] else "No"})
    sections = [
        '<section id="evaluation-boundaries" class="chapter"><p class="eyebrow">Study scope and provenance</p>',
        '<h2>Evaluation boundaries and reproducibility</h2>',
        '<p><strong>What is held out.</strong> Each patient appears in one shared patient-level fold and is excluded from the TabPFN fit producing that patient’s OOF prediction. '
        'The original audits verified eligible patients, disease labels, fold assignments, and exact equality of the submitted scores with the final TXT and ZIP files. '
        'The covariates-only OOF baseline uses exactly the same patients, labels, and folds.</p>',
        '<p><strong>What was selected earlier.</strong> PRS parameters and panels were chosen using validation information. '
        'Therefore, these are OOF results for the selected configuration, with model selection outside a fully nested evaluation. '
        'They do not establish an unbiased estimate of the complete selection pipeline, and they are not the performance of the single all-trainval fitted instance on final holdout patients.</p>',
        '<p><strong>What the baseline difference means.</strong> The final TabPFN models combine PRS and metadata; the baseline is a covariates-only logistic model. '
        'Improvements can reflect PRS information, nonlinear modeling, and interactions. They cannot be attributed exclusively to genetic effects or interpreted causally.</p>',
        '<p><strong>Uncertainty.</strong> Pooled model intervals and paired differences use 2,000 non-stratified paired patient bootstrap samples, seed 20261005, '
        'with 2.5th/97.5th percentiles. Rankings are recalculated within resamples; models are not refitted. '
        'These intervals exclude refitting and selection uncertainty. Threshold-curve intervals instead use fixed-group 2×2-table Wald intervals; '
        'decile, rank-band, and membership fractions use Wilson intervals. These pointwise intervals are not simultaneous confidence bands.</p>',
        '<p><strong>Score conventions.</strong> Ranking uses the submitted standardized clipped logits. R², Brier score, and log loss use the saved native probabilities. '
        'There is no additional logistic evaluation wrapper. If a 2×2 table contains a zero cell, the repository convention adds 0.5 to all four cells before computing its OR. '
        'The clipping check below supports using the submitted ranking with the native probability metrics in these analyses.</p>',
        table(pd.DataFrame(rows), "clipping-audit"),
        '<details><summary>Selected PRS inputs and source of selection</summary>',
    ]
    for dis in diseases:
        selections = [{"PRS family": FAMILY_NAMES[r["family"]],
                       "Selection source": "Only available setting" if r["selection_metric"] == "unique_new_trainval_available_setting"
                       else "Official validation AUC after new trainval fitting"} for r in dis.selection]
        sections += [f'<h3>{dis.label}</h3>', table(pd.DataFrame(selections), f"{dis.key}-selection")]
    sections += [
        '</details>',
        '<h3 id="sources">Source files and report contents</h3>',
        '<p>Confirmed package: <code>submissions/final_phase/Go Gene!_audited_pipeline_with_trainval_5cv</code>. '
        'The saved <code>trainval_5cv_scores_long.csv</code> provides the submitted scores and native probabilities. '
        'The baseline is <code>outputs/final_stage_single_prs_performance/trainval_final_score_5fold_covariate_oof_predictions.csv</code>. '
        'BC eligibility follows <code>configs/disease_eligibility.yaml</code>.</p>',
        '<p>This English report assembles the previously verified BC analysis and CAD/T2D analyses. '
        'Original source and artifact hashes are checked before and after assembly. '
        'The report includes the 24 existing English figures, matching aggregate CSV tables, and a vector PDF collection in report order. '
        'Its HTML embeds all figures and interaction data, so the report itself can be read offline as a single file.</p>',
        '<p><a href="report_manifest.json">Assembly manifest and figure index</a> · '
        '<a href="source_integrity.json">Original-analysis integrity checks</a> · '
        '<a href="VERIFIED.json">English-report verification record</a></p>',
        '<p>Aggregate tables are organized under <code>tables/bc/</code>, <code>tables/cad/</code>, and <code>tables/t2d/</code>. '
        'Patient-level predictions and identifiers remain in their original protected analysis directories and are not copied into this report package.</p>',
        '<details><summary>Rebuild the English report from the existing analyses</summary>'
        '<pre>UV_CACHE_DIR=outputs/uv-cache/prediction_analysis OPENBLAS_NUM_THREADS=1 \\\n'
        'uv run --offline --no-sync python scripts/build_may_final_three_disease_report_en.py</pre>'
        '<p class="small">Run from the repository root. Assembly uses the project’s pandas dependency and the local pdfunite utility. '
        'It verifies the original artifact seals and does not rerun prediction, bootstrap resampling, or model fitting.</p></details>',
        '<h3>Statistical references</h3>',
        '<p>Nagelkerke NJD (1991). A note on a general definition of the coefficient of determination. '
        '<em>Biometrika</em> 78(3):691–692. <a href="https://doi.org/10.1093/biomet/78.3.691">doi:10.1093/biomet/78.3.691</a>.</p>',
        '<p>Gneiting T, Raftery AE (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. '
        '<em>Journal of the American Statistical Association</em> 102(477):359–378. '
        '<a href="https://doi.org/10.1198/016214506000001437">doi:10.1198/016214506000001437</a>.</p>',
        '<p class="small divider">decipEHR · English integrated OOF analysis · BC / CAD / T2D · '
        'Prepared from the confirmed May audited submission and verified analysis artifacts.</p></section>',
    ]
    return "\n".join(sections)


JAVASCRIPT = """
const thresholdData = THRESHOLD_DATA;
function updateExplorer(key) {
  const model = document.getElementById(key + '-model').value;
  const fraction = Number(document.getElementById(key + '-fraction').value);
  const row = thresholdData[key].find(r => r.model === model && Math.abs(r.top_fraction - fraction) < 1e-8);
  const count = n => Number(n).toLocaleString('en-US');
  const percent = n => (100 * n).toFixed(1) + '%';
  document.getElementById(key + '-result').textContent =
    count(row.cases_top) + ' cases among ' + count(row.n_top) + ' selected patients; ' +
    'case fraction ' + percent(row.case_rate) + '; all cases captured ' + percent(row.sensitivity) +
    '; OR ' + row.or.toFixed(2) + ' (pointwise Wald 95% CI ' + row.or_low.toFixed(2) + '–' + row.or_high.toFixed(2) + ').';
}
for (const key of Object.keys(thresholdData)) {
  document.getElementById(key + '-model').addEventListener('change', () => updateExplorer(key));
  document.getElementById(key + '-fraction').addEventListener('change', () => updateExplorer(key));
  updateExplorer(key);
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(REPO / "outputs") or out == REPO / "outputs":
        raise ValueError("The English report must be written to its own folder under outputs/")
    for source in [BC_ROOT, OTHER_ROOT]:
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("The English output folder must be separate from the original analyses")
    if not shutil.which("pdfunite"):
        raise RuntimeError("pdfunite is required to merge the existing vector figures")
    os.umask(0o077)
    originals = verify_originals()
    diseases = [load_disease("bc", "Breast cancer", BC_ROOT),
                load_disease("cad", "Coronary artery disease", OTHER_ROOT / "cad"),
                load_disease("t2d", "Type 2 diabetes", OTHER_ROOT / "t2d")]
    out.mkdir(parents=True, exist_ok=True)
    for dis in diseases:
        folder = out / "tables" / dis.key
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in dis.tables.items():
            frame.to_csv(folder / f"{name}.csv", index=False)
        (folder / "analysis_record.json").write_text(json.dumps(dis.record, indent=2)+"\n")
    summary_frame(diseases).to_csv(out / "disease_summary.csv", index=False)
    pages = []
    parts = [overview(diseases, out, pages)]
    parts.extend(disease_chapter(dis, out, pages) for dis in diseases)
    parts += [shared_methods(diseases), provenance(diseases)]
    thresholds = {dis.key: json.loads(dis.tables["thresholds"].to_json(orient="records", double_precision=15)) for dis in diseases}
    script = JAVASCRIPT.replace("THRESHOLD_DATA", json.dumps(thresholds))
    body = '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    body += '<meta name="viewport" content="width=device-width, initial-scale=1">'
    body += '<title>BC, CAD, and T2D | May final submission OOF analysis</title>'
    body += '<meta name="description" content="English integrated analysis of BC, CAD, and T2D: distributions, calibration, enrichment, and Nagelkerke R-squared.">'
    body += '<style>' + CSS + '</style></head><body>'
    body += '<nav class="nav" aria-label="Report sections"><strong>decipEHR · OOF</strong><a href="#overview">Overview</a><a href="#bc">BC</a><a href="#cad">CAD</a><a href="#t2d">T2D</a><a href="#reading-guide">Reading guide</a><a href="#likelihood-method">R² calculation</a><a href="#evaluation-boundaries">Methods & scope</a></nav>'
    body += '<main>' + "\n".join(parts) + '</main><script>' + script + '</script></body></html>'
    (out / "OOF_ANALYSIS_EN.html").write_text(body, encoding="utf-8")
    subprocess.run(["pdfunite", *[str(out / row["pdf"]) for row in pages],
                    str(out / "OOF_ANALYSIS_FIGURES_EN.pdf")], check=True)
    after = verify_originals()
    if originals != after:
        raise ValueError("Original analysis integrity changed during report assembly")
    (out / "source_integrity.json").write_text(json.dumps({
        "status": "PASS", "checked_before_and_after_assembly": True, "analyses": originals,
    }, indent=2)+"\n")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "language": "English", "diseases": [dis.label for dis in diseases],
        "scope": "Confirmed May final audited configuration; pooled five-fold OOF; not fully nested model selection",
        "training_performed": False, "patient_level_data_exported": False,
        "figure_count": len(pages), "figures": pages,
        "source_analyses": originals,
        "builder": str(Path(__file__).relative_to(REPO)), "builder_sha256": sha256(Path(__file__)),
        "self_contained_html": "OOF_ANALYSIS_EN.html", "figure_pdf": "OOF_ANALYSIS_FIGURES_EN.pdf",
    }
    (out / "report_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out / "README.txt").write_text(
        "BC, CAD, and T2D: May final submission OOF analysis (English)\n\n"
        "Open OOF_ANALYSIS_EN.html in a browser. All figures and threshold data are embedded; no internet is required.\n"
        "OOF_ANALYSIS_FIGURES_EN.pdf contains the 24 English vector figures in report order.\n"
        "figures/ provides individual SVG, PNG, and PDF files. tables/ contains disease-specific aggregate CSVs.\n"
        "disease_summary.csv contains the three-disease overview. See report_manifest.json for the figure index.\n"
        "VERIFIED.json records the final English-report checks. Original analyses are unchanged.\n"
        "No patient identifiers or patient-level predictions are included.\n\n"
        "Evaluation: selected May submission configuration using pooled five-fold OOF predictions.\n"
        "PRS/panel selection was not fully nested. Final holdout labels were unavailable.\n"
    )
    print(json.dumps({"output": str(out), "figures": len(pages),
                      "diseases": manifest["diseases"], "originals_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
