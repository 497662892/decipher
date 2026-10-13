"""Translate the verified PRS review and redraw its four figures in English."""
from __future__ import annotations

import base64
import html
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from decipehr.external_pgs import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/external_pgs/progress_review_20261012"
OUT = BASE / "html"
FIG = OUT / "figures/en"
SOURCE = OUT / "External_PRS_Review_20261012_CN.html"
TARGET = OUT / "External_PRS_Review_20261012_EN.html"
TRANSLATIONS = ROOT / "scripts/assets/external_prs_review_20261012_en.json"
NAMES = {"T2D": "T2D", "cad": "CAD", "bc": "Female breast cancer"}
COLORS = {"T2D": "#16786c", "cad": "#315f9c", "bc": "#9c4d70"}
GROUPS = ["AFR-like", "AMR-like", "EAS-like", "EUR-like", "SAS-like", "AMBIGUOUS"]
SOURCES = {}
CJK = re.compile(r"[\u4e00-\u9fff]")


def load(path):
    SOURCES[str(path.relative_to(ROOT))] = sha256_file(path)
    return pd.read_csv(path, float_precision="round_trip")


def save(fig, name):
    fig.savefig(FIG / f"{name}_en.svg", bbox_inches="tight", metadata={"Creator": "decipEHR"})
    plt.close(fig)


def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "prs-review-20261012-en"})
    metrics = load(BASE / "academic_revision_20261012/development_metrics.csv")
    pairs = load(BASE / "primary_disco_revision_20261012/primary_paired_comparisons.csv")
    methods = [("covariates_lr", "Covariates only (logistic)"),
               ("best_single_lr", "Single-best PRS (logistic)"),
               ("full_lr", "Multi-PRS (logistic)"), ("tabpfn_full", "Multi-PRS (TabPFN)")]
    fig, axes = plt.subplots(3, 2, figsize=(12.3, 8.5))
    for i, disease in enumerate(NAMES):
        for j, metric in enumerate(["auc", "nagelkerke_r2"]):
            ax = axes[i, j]
            rows = metrics.loc[metrics.disease.eq(disease) & metrics.ancestry_group.eq("ALL")].set_index("method")
            values = [rows.loc[method, metric] for method, _ in methods]
            span = max(max(values) - min(values), .012)
            for y, (method, _), value in zip(range(3, -1, -1), methods, values, strict=True):
                ax.scatter(value, y, s=48, color=COLORS[disease] if method == "tabpfn_full" else "#657681", zorder=3)
                ax.annotate(f"{value:.4f}", (value, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
            ax.set(yticks=range(3, -1, -1), yticklabels=[label for _, label in methods] if j == 0 else [""]*4,
                   xlim=(min(values)-span*.12, max(values)+span*.45), ylim=(-.6, 3.6),
                   title=f"{NAMES[disease]} · {'AUC' if j == 0 else 'Nagelkerke R²'}")
            ax.grid(axis="x", alpha=.17)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=.25, right=.98, top=.95, bottom=.06, hspace=.54, wspace=.17)
    save(fig, "07_primary_baselines")
    for comparison, name, reference in [("full_vs_single_lr", "08_ancestry_single", "Single-best PRS logistic"),
                                        ("full_vs_disco", "09_ancestry_disco", "Disco")]:
        fig, axes = plt.subplots(3, 2, figsize=(12.3, 10.0))
        for i, disease in enumerate(NAMES):
            for j, metric in enumerate(["auc", "nagelkerke_r2"]):
                ax = axes[i, j]
                ax.axvline(0, color="#6e7d87", lw=1)
                rows = pairs.loc[pairs.disease.eq(disease) & pairs.comparison.eq(comparison) & pairs.metric.eq(metric)].set_index("ancestry_group")
                labels = []
                for y, group in zip(range(5, -1, -1), GROUPS, strict=True):
                    row = rows.loc[group]
                    ax.plot([row.ci_low, row.ci_high], [y, y], color=COLORS[disease], lw=1.5)
                    ax.scatter(row.candidate_minus_reference, y, color=COLORS[disease], s=29)
                    flag = "*" if min(row.cases, row.n-row.cases) < 20 else ""
                    labels.append(f"{group}{flag}  ({int(row.n)}/{int(row.cases)})")
                ax.set(yticks=range(5, -1, -1), yticklabels=labels if j == 0 else [""]*6,
                       ylim=(-.7, 5.7), title=f"{NAMES[disease]} · {'ΔAUC' if j == 0 else 'ΔNagelkerke R²'}",
                       xlabel="Full TabPFN − " + reference)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.grid(axis="x", alpha=.17)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.tick_params(axis="y", length=0)
        fig.subplots_adjust(left=.24, right=.98, top=.96, bottom=.06, hspace=.48, wspace=.16)
        save(fig, name)
    auc = load(BASE / "current_auc_comparisons.csv")
    r2 = load(OUT / "paired_r2.csv")
    hard = load(BASE / "hard_ancestry_comparisons.csv")
    soft = load(BASE / "soft_ancestry_comparisons.csv")
    definitions = [("full_vs_covariates", "Add PRS to covariates"),
                   ("full_vs_single_tabpfn", "Full panel vs single PRS"),
                   ("continuous_vs_no_pc", "Continuous PCs vs no PCs"),
                   ("varying_vs_fixed", "Varying vs fixed coefficients"),
                   ("hard_vs_continuous", "Hard labels replacing PCs"),
                   ("full_vs_pc", "Add soft ancestry to PCs")]
    fig, axes = plt.subplots(3, 2, figsize=(12.3, 10.0))
    for i, disease in enumerate(NAMES):
        for j, metric in enumerate(["auc", "nagelkerke_r2"]):
            ax = axes[i, j]
            ax.axvline(0, color="#6e7d87", lw=1)
            for y, (comparison, _) in zip(range(5, -1, -1), definitions, strict=True):
                if metric == "nagelkerke_r2":
                    frame, key = r2, comparison
                elif comparison == "hard_vs_continuous":
                    frame, key = hard, "hard_minus_pc"
                else:
                    frame, key = (soft if comparison == "full_vs_pc" else auc), comparison
                selected = frame.loc[frame.disease.eq(disease) & frame.comparison.eq(key)]
                assert len(selected) == 1
                row = selected.iloc[0]
                delta = row.get("candidate_minus_reference", row.get("delta"))
                ax.plot([row.ci_low, row.ci_high], [y, y], color=COLORS[disease], lw=1.5)
                ax.scatter(delta, y, color=COLORS[disease], s=29)
            ax.set(yticks=range(5, -1, -1), yticklabels=[label for _, label in definitions] if j == 0 else [""]*6,
                   ylim=(-.7, 5.7), title=f"{NAMES[disease]} · {'ΔAUC' if j == 0 else 'ΔNagelkerke R²'}",
                   xlabel="Change − reference (nominal 95% CI)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(axis="x", alpha=.17)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=.27, right=.98, top=.96, bottom=.06, hspace=.48, wspace=.16)
    save(fig, "05_ablations")


class EnglishHTML(HTMLParser):
    def __init__(self, translations, assets):
        super().__init__(convert_charrefs=False)
        self.translations = translations
        self.assets = assets
        self.parts = []
        self.context = ""
        self.skip_pdf_link = False
        self.translated_nodes = 0

    def translated(self, value):
        if CJK.search(value):
            assert value in self.translations, f"Untranslated text: {value[:150]}"
            self.translated_nodes += 1
            return self.translations[value]
        return value.translate(str.maketrans({"：": ": ", "；": "; ", "，": ", ", "。": ". ", "、": ", ", "（": "(", "）": ")"}))

    def handle_decl(self, decl):
        self.parts.append(f"<!{decl}>")

    def handle_starttag(self, tag, attrs):
        if tag == "a" and dict(attrs).get("download") == "External_PRS_Figures.pdf":
            self.skip_pdf_link = True
            return
        if tag in ("style", "script"):
            self.context = tag
        rendered = []
        for key, value in attrs:
            if value is not None:
                if tag == "html" and key == "lang":
                    value = "en"
                value = self.assets.get(value, value)
                if key in ("alt", "title", "aria-label"):
                    value = self.translated(value)
                rendered.append(f'{key}="{html.escape(value, quote=True)}"')
            else:
                rendered.append(key)
        self.parts.append("<" + tag + (" " if rendered else "") + " ".join(rendered) + ">")

    def handle_endtag(self, tag):
        if tag == "a" and self.skip_pdf_link:
            self.skip_pdf_link = False
            return
        self.parts.append(f"</{tag}>")
        if tag in ("style", "script"):
            self.context = ""

    def handle_data(self, data):
        if self.skip_pdf_link:
            return
        if self.context == "script":
            self.parts.append(data.replace("收起明细", "Collapse details").replace("展开全部明细", "Expand all details"))
        elif self.context == "style":
            self.parts.append(data)
        else:
            self.parts.append(html.escape(self.translated(data), quote=False))

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")


def main():
    original_digest = sha256_file(SOURCE)
    manifest = json.loads((OUT / "BUILD_MANIFEST.json").read_text())
    assert original_digest == manifest["html_sha256"]
    for name, digest in manifest["source_sha256"].items():
        assert sha256_file(ROOT / name) == digest, name
    make_figures()
    assets = {}
    for name in manifest["figures"]:
        original = OUT / "figures" / name
        translated = FIG / name.replace(".svg", "_en.svg")
        prefix = "data:image/svg+xml;base64,"
        assets[prefix + base64.b64encode(original.read_bytes()).decode()] = prefix + base64.b64encode(translated.read_bytes()).decode()
        assets[name] = translated.name
        assert not CJK.search(translated.read_text())
    translations = json.loads(TRANSLATIONS.read_text())
    parser = EnglishHTML(translations, assets)
    parser.feed(SOURCE.read_text())
    document = "".join(parser.parts)
    assert not CJK.search(document)
    TARGET.write_text(document)
    assert sha256_file(SOURCE) == original_digest
    write_json(OUT / "EN_BUILD_MANIFEST.json", {
        "edition": "english_translation_of_primary_disco_history_ablation_revision_20261012",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chinese_source_sha256": original_digest, "chinese_report_unchanged": True,
        "html": TARGET.name, "html_sha256": sha256_file(TARGET),
        "translations_sha256": sha256_file(TRANSLATIONS), "translated_nodes": parser.translated_nodes,
        "source_sha256": SOURCES, "english_figure_sha256": {p.name: sha256_file(p) for p in FIG.glob("*.svg")},
        "script_sha256": sha256_file(Path(__file__)), "new_model_fits": 0,
        "patient_data_embedded": False, "standalone_html": True,
    })
    print(json.dumps({"html": str(TARGET), "bytes": TARGET.stat().st_size,
                      "translated_nodes": parser.translated_nodes, "english_figures": 4}))


if __name__ == "__main__":
    main()
