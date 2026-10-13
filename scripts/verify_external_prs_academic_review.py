"""Check visible report values, downloads and desktop/mobile/PDF rendering."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/external_pgs/progress_review_20261012"
OUT = BASE / "html"
DATA = BASE / "academic_revision_20261012"
PRIMARY = BASE / "primary_disco_revision_20261012"
TARGET = OUT / "External_PRS_Review_20261012_CN.html"
SCREEN = OUT / "verification/primary_disco_revision"
GROUPS = ["AFR-like", "AMR-like", "EAS-like", "EUR-like", "SAS-like", "AMBIGUOUS"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    SCREEN.mkdir(exist_ok=True, parents=True)
    with (DATA / "development_metrics.csv").open() as f:
        metrics = list(csv.DictReader(f))
    def values(disease, method, group="ALL"):
        return next(r for r in metrics if r["disease"] == disease and r["method"] == method and r["ancestry_group"] == group)
    errors, requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=1)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url) if r.url.startswith(("http:", "https:")) else None)
        page.goto(TARGET.as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        assert page.locator("main>section").count() == 10
        assert page.locator("figure img").count() == 4
        assert page.locator("figure img").evaluate_all("xs=>xs.every(x=>x.complete && x.naturalWidth>0)")
        assert page.locator("[id]").evaluate_all("xs=>new Set(xs.map(x=>x.id)).size===xs.length")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        for link in page.locator("nav a").all():
            assert page.locator(link.get_attribute("href")).count() == 1
        for trait, count in [("t2d", 2), ("cad", 7), ("bc", 3)]:
            assert page.locator(f"#resources-{trait} tbody tr").count() == count
        for i, disease in enumerate(["T2D", "cad", "bc"]):
            row = page.locator("#primary-main tbody tr").nth(i).inner_text()
            for method in ["best_single_lr", "tabpfn_full"]:
                r = values(disease, method)
                for key in ["auc", "nagelkerke_r2"]:
                    assert f"{float(r[key]):.6f}" in row
            row = page.locator("#disco-main tbody tr").nth(i).inner_text()
            for method in ["selected_disco", "tabpfn_full"]:
                r = values(disease, method)
                for key in ["auc", "nagelkerke_r2"]:
                    assert f"{float(r[key]):.6f}" in row
        assert "完整 26" in page.locator("#disco-main tbody tr").first.inner_text()
        assert "完整 7" in page.locator("#disco-main tbody tr").nth(1).inner_text()
        assert "0.756701" not in page.locator("#disco-main tbody tr").first.inner_text()
        with (PRIMARY / "primary_paired_comparisons.csv").open() as f:
            pairs = list(csv.DictReader(f))
        for identity, comparison in [("primary-deltas", "full_vs_single_lr"), ("disco-deltas", "full_vs_disco")]:
            for i, disease in enumerate(["T2D", "cad", "bc"]):
                row = page.locator(f"#{identity} tbody tr").nth(i).inner_text()
                for metric in ["auc", "nagelkerke_r2"]:
                    r = next(r for r in pairs if r["disease"] == disease and r["comparison"] == comparison and r["metric"] == metric and r["ancestry_group"] == "ALL")
                    assert f"{float(r['candidate_minus_reference']):+.6f}" in row
                    for prefix in ["", "familywise_"]:
                        assert f"[{float(r[prefix+'ci_low']):+.4f}, {float(r[prefix+'ci_high']):+.4f}]" in row
        with (BASE / "historical_wrapper_metrics.csv").open() as f:
            historical = list(csv.DictReader(f))
        with (OUT / "paired_r2.csv").open() as f:
            ablation_r2 = list(csv.DictReader(f))
        assert page.locator("#history-main tbody tr").count() == 9
        for i, disease in enumerate(["T2D", "cad", "bc"]):
            for j, method in enumerate(["previous_wrapper", "external_pc_wrapper", "tabpfn_full"]):
                r = values(disease, method) if j == 2 else next(r for r in historical if r["disease"] == disease and r["model"] == method)
                row = page.locator("#history-main tbody tr").nth(i*3+j).inner_text()
                for metric in ["auc", "nagelkerke_r2"]:
                    assert f"{float(r[metric]):.6f}" in row
            rows = page.locator(f"#ablation-{disease} tbody tr")
            assert rows.count() == 6
            for j, comparison in enumerate(["full_vs_covariates", "full_vs_single_tabpfn", "continuous_vs_no_pc", "varying_vs_fixed", "hard_vs_continuous", "full_vs_pc"]):
                r = next(r for r in ablation_r2 if r["disease"] == disease and r["comparison"] == comparison)
                assert f"{float(r['candidate_minus_reference']):+.6f}" in rows.nth(j).inner_text()
        assert page.locator("#history-main").is_visible()
        assert page.locator("#ablation-T2D").is_visible()
        assert "3,308 / 602" in page.locator("#cohorts").inner_text()
        page.screenshot(path=str(SCREEN / "desktop_overview.png"))
        page.evaluate("scrollTo(0,document.querySelector('#single').offsetTop-18)")
        page.screenshot(path=str(SCREEN / "desktop_single.png"))
        page.locator("#resources-cad").screenshot(path=str(SCREEN / "desktop_cad_resources.png"))
        page.evaluate("scrollTo(0,document.querySelector('#disco').offsetTop-18)")
        page.screenshot(path=str(SCREEN / "desktop_disco.png"))
        page.locator("#history-main").screenshot(path=str(SCREEN / "desktop_history.png"))
        page.locator("#ablation-T2D").screenshot(path=str(SCREEN / "desktop_ablation_t2d.png"))
        page.locator("#expand").click()
        assert page.locator("details:not([open])").count() == 0
        for comparison, reference, candidate in [("full_vs_single_lr", "best_single_lr", "tabpfn_full"),
                                                 ("full_vs_disco", "selected_disco", "tabpfn_full")]:
            for disease in ["T2D", "cad", "bc"]:
                rows = page.locator(f"#sub-{comparison}-{disease} tbody tr")
                assert rows.count() == 6
                for i, group in enumerate(GROUPS):
                    text = rows.nth(i).inner_text()
                    assert group in text
                    for method in [reference, candidate]:
                        r = values(disease, method, group)
                        assert f"{int(r['n'])} / {int(r['cases'])} / {int(r['controls'])}" in text
                        for key in ["auc", "nagelkerke_r2"]:
                            assert f"{float(r[key]):.4f}" in text
        page.locator("#sub-full_vs_single_lr-bc").screenshot(path=str(SCREEN / "bc_subgroup_table.png"))
        for index, link in enumerate(page.locator("a.download").all()):
            name = link.get_attribute("download")
            payload = base64.b64decode(link.get_attribute("href").split(",", 1)[1])
            folders = [DATA, PRIMARY, BASE, OUT, ROOT / "outputs/external_pgs/cad_expansion_20261012/report"]
            source = next(folder / name for folder in folders if (folder / name).exists())
            assert hashlib.sha256(payload).hexdigest() == digest(source)
            if source.suffix == ".csv":
                with source.open() as f:
                    assert "sampleid" not in next(csv.reader(f))
            # One actual browser download plus exact payload checks for all links
            # avoids Chromium's repeated automatic-download throttling.
            if index == 0:
                with page.expect_download() as info:
                    link.click()
                item = info.value
                destination = SCREEN / item.suggested_filename
                item.save_as(str(destination))
                assert digest(destination) == digest(source)
                destination.unlink()
        for link in page.locator("a[href]").all():
            href = link.get_attribute("href")
            if not href.startswith(("#", "data:", "http:", "https:")):
                assert (OUT / unquote(href)).resolve().is_file(), href
        page.locator("#expand").click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("scrollTo(0,0)")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(SCREEN / "mobile_overview.png"))
        page.locator("#ancestry").scroll_into_view_if_needed()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(SCREEN / "mobile_ancestry.png"))
        page.set_viewport_size({"width": 1600, "height": 1100})
        page.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)")
        page.pdf(path=str(OUT / "External_PRS_Review_20261012_CN.pdf"), print_background=True, prefer_css_page_size=True)
        assert not errors and not requests, (errors, requests)
        verification = {"status": "verified", "edition": "primary_disco_history_ablation_revision_20261012", "browser": browser.version,
                        "primary_and_disco_tables_match_verified_csv": True, "all_36_subgroup_rows_match_verified_csv": True,
                        "main_tables_use_identical_full_models": True, "all_primary_intervals_match_csv": True,
                        "nine_historical_rows_match_csv": True, "eighteen_ablation_r2_rows_match_csv": True,
                        "historical_and_ablation_tables_visible_without_expansion": True,
                        "resource_table_model_families": {"T2D": 2, "CAD": 7, "BC": 3},
                        "embedded_figures": 4, "offline_download_hashes_verified": True,
                        "no_patient_rows_in_downloads": True, "desktop_and_mobile_no_page_overflow": True,
                        "javascript_errors": errors, "external_runtime_requests": requests,
                        "html_sha256": digest(TARGET), "pdf_sha256": digest(OUT / "External_PRS_Review_20261012_CN.pdf"),
                        "script_sha256": digest(Path(__file__))}
        (OUT / "HTML_VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
        (SCREEN / "BROWSER_VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
        print(json.dumps(verification))
        browser.close()


if __name__ == "__main__":
    main()
