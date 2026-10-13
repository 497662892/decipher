#!/usr/bin/env python3
"""Verify the assembled English report, its aggregate exports, and interactions."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
from build_may_final_three_disease_report_en import (
    BC_PREFIXED,
    BC_ROOT,
    DEFAULT_OUTPUT,
    OTHER_ROOT,
    REPO,
    sha256,
    verify_originals,
)


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.assets = []
        self.scripts = []
        self.in_script = False
        self.svg_count = 0
        self.language = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if "href" in attrs:
            self.links.append(attrs["href"])
        if "xlink:href" in attrs:
            self.links.append(attrs["xlink:href"])
        if tag in {"script", "img", "image", "link", "iframe"}:
            self.assets.extend(attrs[k] for k in ["src", "href", "xlink:href"] if k in attrs)
        if tag == "html":
            self.language = attrs.get("lang")
        if tag == "svg":
            self.svg_count += 1
        if tag == "script":
            self.in_script = True
            self.scripts.append("")

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.scripts[-1] += data


NODE_CHECK = """
const fs = require('fs');
const vm = require('vm');
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const elements = {};
for (const key of Object.keys(input.expected)) {
  for (const field of ['model', 'fraction', 'result']) {
    elements[key + '-' + field] = {
      value: field === 'model' ? 'submitted' : '0.1', textContent: '', listeners: {},
      addEventListener(event, handler) { this.listeners[event] = handler; }
    };
  }
}
const document = {getElementById: id => {
  if (!elements[id]) throw new Error('Unknown element: ' + id);
  return elements[id];
}};
const context = vm.createContext({document});
vm.runInContext(input.script, context);
let checked = 0;
const count = n => Number(n).toLocaleString('en-US');
const percent = n => (100 * n).toFixed(1) + '%';
for (const [key, rows] of Object.entries(input.expected)) {
  if (!elements[key + '-result'].textContent.includes('selected patients')) throw new Error('Initial output missing');
  for (const row of rows) {
    elements[key + '-model'].value = row.model;
    elements[key + '-fraction'].value = String(row.top_fraction);
    const expected = count(row.cases_top) + ' cases among ' + count(row.n_top) + ' selected patients; ' +
      'case fraction ' + percent(row.case_rate) + '; all cases captured ' + percent(row.sensitivity) +
      '; OR ' + row.or.toFixed(2) + ' (pointwise Wald 95% CI ' + row.or_low.toFixed(2) + '–' + row.or_high.toFixed(2) + ').';
    for (const field of ['model', 'fraction']) {
      elements[key + '-' + field].listeners.change();
      if (elements[key + '-result'].textContent !== expected) throw new Error(key + ' state does not match source CSV');
    }
    checked += 1;
  }
}
process.stdout.write(JSON.stringify({status: 'PASS', unique_states: checked, control_events_checked: 2*checked}));
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.umask(0o077)
    out = args.output_dir.resolve()
    if not out.is_relative_to(REPO / "outputs") or out == REPO / "outputs":
        raise ValueError("Verify a report folder under outputs/")
    content = (out / "OOF_ANALYSIS_EN.html").read_text()
    manifest = json.loads((out / "report_manifest.json").read_text())
    assert manifest["builder_sha256"] == sha256(REPO / manifest["builder"])
    parser_html = ReportParser()
    parser_html.feed(content)
    assert parser_html.language == "en"
    assert not re.search(r"[\u3400-\u9fff]", content), "Unexpected untranslated CJK text"
    assert parser_html.svg_count == manifest["figure_count"] == 24
    assert len(parser_html.ids) == len(set(parser_html.ids)), "Duplicate HTML/SVG IDs"
    assert len(parser_html.scripts) == 1
    assert not parser_html.assets, "Report should have no external asset dependencies"
    assert all(identifier in parser_html.ids for identifier in
               ["bc", "cad", "t2d", "reading-guide", "likelihood-method", "evaluation-boundaries"])
    for reference in re.findall(r"url\(#([^)]*)\)", content):
        assert reference in parser_html.ids, f"Missing SVG target: {reference}"
    for link in parser_html.links:
        if link.startswith("#"):
            assert link[1:] in parser_html.ids, f"Broken anchor: {link}"
        elif not link.startswith("https://") and link != "VERIFIED.json":
            assert (out / link).is_file(), f"Missing linked file: {link}"

    roots = {"bc": BC_ROOT, "cad": OTHER_ROOT / "cad", "t2d": OTHER_ROOT / "t2d"}
    table_count = 0
    expected = {}
    summary = pd.read_csv(out / "disease_summary.csv").set_index("disease")
    for key, root in roots.items():
        for exported in sorted((out / "tables" / key).glob("*.csv")):
            prefix = "bc_" if key == "bc" and exported.stem in BC_PREFIXED else ""
            original = pd.read_csv(root / f"{prefix}{exported.name}")
            if "disease" in original:
                original = original.loc[original.disease.str.lower().eq(key)].reset_index(drop=True)
            restored = pd.read_csv(exported)
            pd.testing.assert_frame_equal(original, restored, check_exact=False, atol=1e-12, rtol=1e-12)
            assert not {"sampleid", "patient_id", "iid", "direct_probability"}.intersection(restored.columns)
            table_count += 1
        p = pd.read_csv(out / "tables" / key / "pooled_metrics.csv").set_index("model")
        d = pd.read_csv(out / "tables" / key / "deciles.csv")
        d = d.loc[d.model.eq("submitted")]
        s, b, row = p.loc["submitted"], p.loc["covariates"], summary.loc[key.upper()]
        assert row.n == s.n and row.cases == s.cases
        assert int(s.top10_n) == math.ceil(s.n*.1)
        a, controls_top, c, controls_rest = s.top10_cases, s.top10_n-s.top10_cases, s.lower90_cases, s.lower90_n-s.lower90_cases
        independent_or = (a/controls_top)/(c/controls_rest)
        assert math.isclose(independent_or, row.top10_vs_lower90_or, abs_tol=1e-12)
        ll_null = s.cases*math.log(s.cases/s.n)+(s.n-s.cases)*math.log1p(-s.cases/s.n)
        gain = -s.n*s.log_loss-ll_null
        independent_r2 = (1-math.exp(-2*gain/s.n))/(1-math.exp(2*ll_null/s.n))
        assert math.isclose(independent_r2, row.nagelkerke_r2, abs_tol=1e-12)
        assert math.isclose(gain, d.ll_gain_null.sum(), abs_tol=1e-9)
        assert math.isclose(s.n*(b.log_loss-s.log_loss), d.ll_gain_covariates.sum(), abs_tol=1e-9)
        for metric in ["auc", "nagelkerke_r2", "top10_vs_lower90_or", "brier", "log_loss"]:
            assert math.isclose(row[f"delta_{metric}"], s[metric]-b[metric], abs_tol=1e-12)
        for column in ["ll_gain_null", "ll_gain_covariates"]:
            assert math.isclose(row[f"low20_{column}_share"], d.loc[d.decile.le(2), column].sum()/d[column].sum(), abs_tol=1e-12)
            assert math.isclose(row[f"top10_{column}_share"], d.loc[d.decile.eq(10), column].sum()/d[column].sum(), abs_tol=1e-12)
        expected[key] = pd.read_csv(out / "tables" / key / "thresholds.csv").to_dict("records")
    for row in manifest["figures"]:
        for suffix in ["svg", "png", "pdf"]:
            source = roots[row["disease"].lower()] / "figures" / f'{row["figure"]}.{suffix}'
            copied = (out / row["pdf"]).with_suffix(f".{suffix}")
            assert sha256(source) == sha256(copied), f"Figure changed: {copied}"
    qa = out / "verification"
    qa.mkdir(exist_ok=True)
    node = shutil.which("node") or "/gpfs/share/apps/nodejs/22.9.0/bin/node"
    (qa / "check_interactions.cjs").write_text(NODE_CHECK)
    (qa / "interaction_inputs.json").write_text(json.dumps({"script": parser_html.scripts[0], "expected": expected}))
    checked = subprocess.run([node, str(qa / "check_interactions.cjs"), str(qa / "interaction_inputs.json")],
                             check=True, capture_output=True, text=True)
    interactions = json.loads(checked.stdout)
    assert interactions["unique_states"] == 72
    pdf = out / "OOF_ANALYSIS_FIGURES_EN.pdf"
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    assert re.search(r"Pages:\s+24\b", info)
    pdf_text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True).stdout
    pdf_pages = [page for page in pdf_text.split("\f") if page.strip()]
    assert len(pdf_pages) == 24 and all(len(page.strip()) > 100 for page in pdf_pages)
    assert all(label in pdf_text for label in ["BC", "CAD", "T2D"])
    for text_page, row in zip(pdf_pages, manifest["figures"], strict=True):
        source_text = subprocess.run(["pdftotext", "-layout", str(out / row["pdf"]), "-"],
                                     check=True, capture_output=True, text=True).stdout
        assert text_page.strip() == source_text.strip(), f"Merged PDF page mismatch: {row['page']}"
    original_integrity = verify_originals()
    assert original_integrity == manifest["source_analyses"]
    result = {
        "status": "PASS", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "html": {"language": "English", "no_untranslated_cjk": True, "embedded_figures": 24,
                 "unique_ids": len(parser_html.ids), "all_references_resolve": True,
                 "no_external_asset_dependencies": True, "browser_render_tested": False,
                 "interactive_javascript": interactions},
        "aggregate_tables_matched_to_originals": table_count,
        "numeric_checks": "OR90 independently reconstructed from counts; R2 and gain shares from aggregate likelihoods; all paired deltas checked",
        "figure_files_byte_identical_to_verified_originals": 72,
        "pdf": {"pages": 24, "selectable_text_on_all_pages": True, "each_page_matches_original_figure_text": True},
        "original_analyses_unchanged": original_integrity,
        "patient_level_data_exported": False,
        "verification_script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*"))
                            if path.is_file() and path.name != "VERIFIED.json" and path.suffix != ".zip"},
    }
    (out / "VERIFIED.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    main()
