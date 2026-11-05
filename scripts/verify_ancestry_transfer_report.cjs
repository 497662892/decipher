/* Verify offline interactions, numerical readback, exports and responsive layout. */
const fs = require("fs"), path = require("path"), assert = require("assert"), crypto = require("crypto");
const {pathToFileURL} = require("url");
const root = path.resolve(__dirname, "..");
process.env.PLAYWRIGHT_BROWSERS_PATH ||= path.join(root, "outputs/report_vendor/browser_qa/browsers");
const {chromium} = require(path.join(root, "outputs/report_vendor/browser_qa/node_modules/playwright"));
const report = path.resolve(process.argv[2] || path.join(root, "outputs/ancestry_transfer_20261104/report"));
const html = path.join(report, "report_academic_zh.html");
const qa = path.join(report, "academic_assets/qa");
const sha = p => crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");

(async () => {
  fs.mkdirSync(qa, {recursive: true});
  const browser = await chromium.launch({headless: true});
  try {
    const context = await browser.newContext({offline: true, acceptDownloads: true});
    const errors = [], requests = [], downloads = [], layouts = [];
    context.on("request", request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(String(error)));
    page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
    await page.setViewportSize({width: 1280, height: 1000});
    await page.goto(pathToFileURL(html).href);
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator("h1").innerText(), "祖源坐标与 PRS 蒸馏");
    assert.equal(await page.locator("#summary-cards article").count(), 3);
    assert.equal(await page.locator("#teacher-cards article").count(), 3);
    assert(await page.locator("#teacher-results").isVisible());
    assert(await page.evaluate(() => document.getElementById("teacher-cards").getBoundingClientRect().top < document.getElementById("summary-cards").getBoundingClientRect().top));
    assert.equal(await page.locator("#selection-table tbody tr").count(), 36);
    assert.equal(await page.locator("#counts-table tbody tr").count(), 12);
    assert.equal(await page.evaluate(() => Object.keys(REPORT.figures).length), 48);
    let states = 0;
    for (const cohort of ["original_val", "nested_oof", "exclude_shared_group", "exclude_direct_0177"]) {
      await page.selectOption("#cohort", cohort);
      for (const metric of ["auroc", "log_loss", "brier", "nagelkerke_r2"]) {
        await page.selectOption("#metric", metric);
        for (const mode of ["official_pc", "reference_pc"]) {
          await page.selectOption("#baseline-mode", mode);
          await page.evaluate(() => Promise.all(Array.from(document.images).map(img => img.decode())));
          assert(await page.evaluate(({cohort, metric, mode}) => {
            const f = x => Number(x).toFixed(4);
            const s = x => (x < 0 ? "−" : "+") + Math.abs(x).toFixed(4);
            const ci = r => `${s(r.delta)} [${s(r.ci_low)}, ${s(r.ci_high)}]`;
            const traits = ["T2D", "cad", "bc"];
            for (const [id, arm] of [["summary-cards", "M0_KD"], ["teacher-cards", "TabPFN_v3"]]) for (const trait of traits) {
              const card = document.querySelector(`#${id} [data-trait="${trait}"]`);
              const a = REPORT.metrics.find(r => r.trait === trait && r.cohort === cohort && r.mode === "official_pc" && r.arm === arm);
              const b = REPORT.metrics.find(r => r.trait === trait && r.cohort === cohort && r.mode === "reference_pc" && r.arm === arm);
              const d = REPORT.paired_differences.find(r => r.trait === trait && r.cohort === cohort && r.metric === metric && r.arm === `reference_pc/${arm}` && r.reference === `official_pc/${arm}`);
              if (card.querySelector(".old").textContent !== f(a[metric]) || card.querySelector(".new").textContent !== f(b[metric])) return false;
              if (card.querySelector(".delta").textContent !== `Δ ${s(d.delta)}`) return false;
              if (card.querySelector(".ci-line").textContent !== `95% CI [${s(d.ci_low)}, ${s(d.ci_high)}]`) return false;
              if (!card.querySelector(".sample-count").textContent.startsWith(a.n.toLocaleString("en-US") + " 人")) return false;
            }
            for (const kind of ["transfer", "baseline"]) {
              const rows = Array.from(document.querySelectorAll(`#${kind}-table tbody tr`));
              if (rows.length !== (kind === "transfer" ? 18 : 9)) return false;
              for (const tr of rows) {
                const [trait, arm] = tr.dataset.key.split("/");
                const a = `${kind === "transfer" ? "reference_pc" : mode}/${arm}`;
                const b = kind === "transfer" ? `official_pc/${arm}` : `${mode}/LR_SINGLE_BEST`;
                const r = REPORT.paired_differences.find(r => r.trait === trait && r.cohort === cohort && r.metric === metric && r.arm === a && r.reference === b);
                if (tr.lastElementChild.textContent !== ci(r)) return false;
              }
              const key = [kind, cohort, metric, ...(kind === "baseline" ? [mode] : [])].join("__");
              const img = document.getElementById(`${kind}-plot`);
              if (img.src !== REPORT.figures[key].svg || img.naturalWidth === 0) return false;
              if (document.getElementById(`${kind}-pdf`).href !== REPORT.figures[key].pdf) return false;
            }
            return true;
          }, {cohort, metric, mode}), `Incorrect display: ${cohort}/${metric}/${mode}`);
          states++;
        }
      }
    }
    await page.selectOption("#cohort", "original_val");
    await page.selectOption("#metric", "auroc");
    await page.selectOption("#baseline-mode", "reference_pc");
    await page.locator("#subgroup-details > summary").click();
    for (const trait of ["T2D", "cad", "bc"]) {
      await page.selectOption("#subgroup-trait", trait);
      assert.equal(await page.locator("#subgroup-table tbody tr").count(), 6);
      assert(await page.evaluate(trait => {
        return Array.from(document.querySelectorAll("#subgroup-table tbody tr")).every(tr => {
          const rows = REPORT.subgroup_metrics.filter(r => r.trait === trait && r.group === tr.dataset.key && r.arm === "M0_KD");
          const cells = tr.querySelectorAll("td");
          return ["official_pc", "reference_pc"].every((mode, i) => {
            const value = rows.find(r => r.mode === mode).auroc;
            return cells[i + 2].textContent === (value == null ? "—" : value.toFixed(4));
          });
        });
      }, trait));
    }
    await page.selectOption("#subgroup-trait", "T2D");
    await page.locator("#subgroup-details > summary").click();
    for (const [selector, name] of [["#transfer-csv", "transfer"], ["#baseline-csv", "baseline"], ["#transfer-pdf", "transfer_pdf"], ["#baseline-pdf", "baseline_pdf"]]) {
      const pending = page.waitForEvent("download");
      await page.locator(selector).click();
      const file = await pending;
      const destination = path.join(qa, file.suggestedFilename());
      await file.saveAs(destination);
      const raw = fs.readFileSync(destination);
      if (name.endsWith("pdf")) assert.equal(raw.subarray(0, 5).toString(), "%PDF-");
      else assert.equal(raw.toString().trim().split("\n").length, name === "transfer" ? 19 : 10);
      downloads.push({name, filename: file.suggestedFilename(), sha256: sha(destination)});
    }
    await page.evaluate(() => document.querySelectorAll("details").forEach(d => d.open = true));
    for (const name of ["metrics", "paired_differences", "cohort_counts", "single_prs_selections", "subgroup_metrics"]) {
      const pending = page.waitForEvent("download");
      await page.locator(`[data-export="${name}"]`).click();
      const file = await pending;
      const destination = path.join(qa, file.suggestedFilename());
      await file.saveAs(destination);
      assert.equal(sha(destination), sha(path.join(report, `${name}.csv`)), "Export differs from sealed CSV");
      downloads.push({name, sha256: sha(destination)});
    }
    for (const width of [1280, 768, 390]) {
      await page.setViewportSize({width, height: 1000});
      for (const open of [false, true]) {
        await page.evaluate(open => document.querySelectorAll("details").forEach(d => d.open = open), open);
        const dimensions = await page.evaluate(() => ({width: innerWidth, bodyWidth: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight}));
        assert(dimensions.bodyWidth <= width + 1, `Horizontal page overflow: ${width}, expanded=${open}`);
        layouts.push({...dimensions, expanded: open});
      }
      await page.evaluate(() => { document.querySelectorAll("details").forEach(d => d.open = false); window.scrollTo({top: 0, behavior: "instant"}); });
      await page.screenshot({path: path.join(qa, `top_${width}.png`)});
      await page.locator("#teacher-results").screenshot({path: path.join(qa, `teacher_${width}.png`)});
      if (width === 1280) {
        await page.screenshot({path: path.join(qa, "full_desktop.png"), fullPage: true});
        for (const id of ["pc-effect", "baseline-effect"]) await page.locator(`#${id}`).screenshot({path: path.join(qa, `${id}.png`)});
      }
    }
    const links = await page.locator("a[href]").evaluateAll(nodes => nodes.map(n => n.getAttribute("href")));
    for (const link of links) {
      if (link.startsWith("#")) assert.equal(await page.locator(link).count(), 1);
      else if (!/^(https?:|data:|blob:)/.test(link)) assert(fs.existsSync(path.resolve(report, link)), `Missing local link ${link}`);
    }
    await page.setViewportSize({width: 1280, height: 1000});
    await page.emulateMedia({media: "print"});
    await page.pdf({path: path.join(qa, "print_preview.pdf"), printBackground: true, preferCSSPageSize: true});
    assert.deepEqual(errors, []);
    assert.deepEqual(requests, []);
    const receipt = {status: "passed", offline: true, browser: browser.version(), global_filter_combinations: states, summary_models_verified: ["TabPFN_v3", "M0_KD"], teacher_summary_visible_by_default: true, subgroup_combinations: 3, page_errors: errors, external_requests: requests, downloads, layouts, html_sha256: sha(html), verifier_sha256: sha(__filename)};
    fs.writeFileSync(path.join(qa, "BROWSER_VERIFIED.json"), JSON.stringify(receipt, null, 2) + "\n");
    console.log(JSON.stringify({status: "passed", combinations: states, subgroup_combinations: 3, downloads: downloads.length, page_errors: errors.length, external_requests: requests.length, layouts}));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
