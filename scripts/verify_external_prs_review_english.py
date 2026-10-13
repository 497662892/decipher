"""Verify the standalone English report against its frozen Chinese counterpart."""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/external_pgs/progress_review_20261012/html"
SOURCE = OUT / "External_PRS_Review_20261012_CN.html"
TARGET = OUT / "External_PRS_Review_20261012_EN.html"
SCREEN = OUT / "verification/english"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables(page):
    return page.locator("table").evaluate_all("""xs => Object.fromEntries(xs.map(t =>
        [t.id, Array.from(t.querySelectorAll('tbody tr')).map(r =>
        Array.from(r.querySelectorAll('td')).map(c => c.innerText.trim().replace(/\\s+/g, ' ')))]))""")


def main():
    SCREEN.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / "EN_BUILD_MANIFEST.json").read_text())
    assert digest(SOURCE) == manifest["chinese_source_sha256"]
    assert digest(TARGET) == manifest["html_sha256"]
    errors, requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.goto(SOURCE.as_uri(), wait_until="networkidle")
        page.evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)")
        original_tables = tables(page)
        original_urls = page.locator("a[href^='http']").evaluate_all("xs=>xs.map(x=>x.href)")
        original_data = page.locator("a.download").evaluate_all("xs=>Object.fromEntries(xs.map(x=>[x.download,x.href]))")
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: requests.append(r.url) if r.url.startswith(("http:", "https:")) else None)
        page.goto(TARGET.as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        assert page.locator("html").get_attribute("lang") == "en"
        assert page.locator("main>section").count() == 10
        assert page.locator("figure img").count() == 4
        assert page.locator("figure img").evaluate_all("xs=>xs.every(x=>x.complete&&x.naturalWidth>0)")
        assert not re.search(r"[\u4e00-\u9fff]", page.locator("body").inner_text())
        assert page.locator("[id]").evaluate_all("xs=>new Set(xs.map(x=>x.id)).size===xs.length")
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        assert page.locator("a[href^='http']").evaluate_all("xs=>xs.map(x=>x.href)") == original_urls
        page.screenshot(path=str(SCREEN / "desktop_overview.png"))
        page.locator("#disco-main").screenshot(path=str(SCREEN / "desktop_disco.png"))
        page.locator("#history-main").screenshot(path=str(SCREEN / "desktop_history.png"))
        page.locator("#ablation-T2D").screenshot(path=str(SCREEN / "desktop_ablation.png"))
        page.locator("figure").first.screenshot(path=str(SCREEN / "english_primary_figure.png"))
        page.locator("figure").nth(3).screenshot(path=str(SCREEN / "english_ablation_figure.png"))
        page.locator("#expand").click()
        assert page.locator("details:not([open])").count() == 0
        assert page.locator("#expand").inner_text() == "Collapse details"
        translated_tables = tables(page)
        assert original_tables.keys() == translated_tables.keys()
        numeric_cells = 0
        for identity, rows in original_tables.items():
            actual = translated_tables[identity]
            assert len(rows) == len(actual), identity
            for a, b in zip(rows, actual, strict=True):
                assert len(a) == len(b), identity
                for old, new in zip(a, b, strict=True):
                    if re.search(r"\d", old) and re.fullmatch(r"[\d\s.,+\-−/%\[\]()eE]+", old):
                        assert old == new, (identity, old, new)
                        numeric_cells += 1
                    if identity.startswith("resources-"):
                        # Population descriptions may reorder cohort names and counts.
                        assert sorted(re.findall(r"[0-9]+(?:[.,][0-9]+)*", old)) == sorted(re.findall(r"[0-9]+(?:[.,][0-9]+)*", new)), (identity, old, new)
        downloads = page.locator("a.download").evaluate_all("xs=>Object.fromEntries(xs.map(x=>[x.download,x.href]))")
        assert set(downloads) == set(original_data) - {"External_PRS_Figures.pdf"}
        for name, url in downloads.items():
            assert url == original_data[name], name
            data = base64.b64decode(url.split(",", 1)[1])
            if name.endswith(".csv"):
                assert "sampleid" not in data.decode().splitlines()[0]
        for link in page.locator("a[href]").all():
            href = link.get_attribute("href")
            if href.startswith("#"):
                assert page.locator(href).count() == 1
            elif not href.startswith(("data:", "http:", "https:")):
                assert (OUT / unquote(href)).resolve().is_file(), href
        for img in page.locator("figure img").all():
            svg = base64.b64decode(img.get_attribute("src").split(",", 1)[1]).decode()
            assert not re.search(r"[\u4e00-\u9fff]", svg)
        page.locator("#expand").click()
        assert page.locator("details[open]").count() == 0
        assert page.locator("#expand").inner_text() == "Expand all details"
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("scrollTo(0,0)")
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.screenshot(path=str(SCREEN / "mobile_overview.png"))
        page.locator("#disco").scroll_into_view_if_needed()
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.screenshot(path=str(SCREEN / "mobile_disco.png"))
        assert not errors and not requests, (errors, requests)
        assert digest(SOURCE) == manifest["chinese_source_sha256"]
        result = {"status": "verified", "browser": browser.version,
                  "table_count": len(original_tables), "numeric_cells_identical": numeric_cells,
                  "resource_table_numbers_identical": True, "aggregate_downloads_identical": len(downloads),
                  "chinese_source_unchanged": True, "english_figures": 4,
                  "no_chinese_in_visible_text_or_figures": True, "desktop_mobile_no_page_overflow": True,
                  "all_links_resolve": True, "javascript_errors": errors, "external_runtime_requests": requests,
                  "html_sha256": digest(TARGET), "script_sha256": digest(Path(__file__))}
        (OUT / "EN_HTML_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        browser.close()


if __name__ == "__main__":
    main()
