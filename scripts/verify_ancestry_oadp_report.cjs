/* Exercise offline filters, numeric readback, exports and responsive layout. */
const fs = require("fs"), path = require("path"), assert = require("assert"), crypto = require("crypto");
const {pathToFileURL} = require("url");
const root = path.resolve(__dirname, "..");
process.env.PLAYWRIGHT_BROWSERS_PATH ||= path.join(root, "outputs/report_vendor/browser_qa/browsers");
const {chromium} = require(path.join(root, "outputs/report_vendor/browser_qa/node_modules/playwright"));
const report = path.resolve(process.argv[2] || path.join(root, "outputs/ancestry_oadp_20261105/report"));
const html = path.join(report, "report_zh.html"), qa = path.join(report, "assets/qa");
const sha = p => crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");

(async () => {
  fs.mkdirSync(qa, {recursive: true});
  const browser = await chromium.launch({headless: true});
  try {
    const context = await browser.newContext({offline: true, acceptDownloads: true});
    const errors = [], requests = [], downloads = [], layouts = [];
    context.on("request", r => { if (/^https?:/.test(r.url())) requests.push(r.url()); });
    const page = await context.newPage();
    page.on("pageerror", e => errors.push(String(e)));
    page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
    await page.setViewportSize({width: 1280, height: 1000});
    await page.goto(pathToFileURL(html).href);
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator("h1").innerText(), "PCA 投影收缩校正");
    assert.equal(await page.locator("#teacher-cards article").count(), 3);
    assert.equal(await page.locator("#student-cards article").count(), 3);
    assert(await page.evaluate(() => document.getElementById("teacher-cards").getBoundingClientRect().top < document.getElementById("student-cards").getBoundingClientRect().top));
    assert.equal(await page.evaluate(() => Object.keys(REPORT.figures).length), 16);
    let states = 0;
    for (const cohort of ["original_val", "nested_oof", "exclude_shared_group", "exclude_direct_0177"]) {
      await page.selectOption("#cohort", cohort);
      for (const metric of ["auroc", "log_loss", "brier", "nagelkerke_r2"]) {
        await page.selectOption("#metric", metric);
        await page.evaluate(() => Promise.all(Array.from(document.images).map(img => img.decode())));
        assert(await page.evaluate(({cohort, metric}) => {
          const f = x => x == null ? "—" : Number(x).toFixed(4);
          const s = x => (x < 0 ? "−" : "+") + Math.abs(x).toFixed(6);
          const point = (trait, mode, arm) => REPORT.metrics.find(r => r.trait === trait && r.cohort === cohort && r.mode === mode && r.arm === arm);
          const difference = (trait, arm) => REPORT.paired_differences.find(r => r.trait === trait && r.cohort === cohort && r.metric === metric && r.arm === `oadp_pc/${arm}` && r.reference === `reference_pc/${arm}`);
          for (const [id, arm] of [["teacher-cards", "TabPFN_v3"], ["student-cards", "M0_KD"]]) for (const trait of ["T2D", "cad", "bc"]) {
            const card = document.querySelector(`#${id} [data-trait="${trait}"]`);
            const a = point(trait, "reference_pc", arm), b = point(trait, "oadp_pc", arm), c = point(trait, "official_pc", arm), d = difference(trait, arm);
            if (card.querySelector(".old").textContent !== f(a[metric]) || card.querySelector(".new").textContent !== f(b[metric])) return false;
            if (card.querySelector(".official-value").textContent !== `官方 PC 对照：${f(c[metric])}`) return false;
            if (card.querySelector(".delta").textContent !== `Δ ${s(d.delta)}`) return false;
            if (card.querySelector(".ci-line").textContent !== `95% CI [${s(d.ci_low)}, ${s(d.ci_high)}]`) return false;
            if (!card.querySelector(".sample-count").textContent.startsWith(a.n.toLocaleString("en-US") + " 人")) return false;
          }
          for (const id of ["paired-table", "point-table"]) {
            const rows = Array.from(document.querySelectorAll(`#${id} tbody tr`));
            if (rows.length !== 18) return false;
            for (const row of rows) {
              const [trait, arm] = row.dataset.key.split("/");
              const cells = row.querySelectorAll("td");
              const modes = id === "paired-table" ? ["reference_pc", "oadp_pc"] : ["official_pc", "reference_pc", "oadp_pc"];
              if (!modes.every((mode, i) => cells[i + 2].textContent === f(point(trait, mode, arm)[metric]))) return false;
              const d = difference(trait, arm);
              if (id === "paired-table" && cells[4].textContent !== `${s(d.delta)} [${s(d.ci_low)}, ${s(d.ci_high)}]`) return false;
            }
          }
          const key = `${cohort}__${metric}`, img = document.getElementById("effect-plot");
          return img.src === REPORT.figures[key].svg && img.naturalWidth > 0 && document.getElementById("figure-pdf").href === REPORT.figures[key].pdf;
        }, {cohort, metric}), `Incorrect display: ${cohort}/${metric}`);
        states++;
      }
    }
    assert.equal(await page.locator("#geometry-table tbody tr").count(), 10);
    assert(await page.evaluate(() => Array.from(document.querySelectorAll("#geometry-table tbody tr")).every(row => {
      const r = REPORT.correction_by_pc.find(x => String(x.pc) === row.dataset.key), c = row.querySelectorAll("td");
      return c[1].textContent === r.sd_ratio.toFixed(3) && c[2].textContent === r.correlation.toFixed(6) && c[3].textContent === r.non_affine_rms_over_simple_sd.toFixed(4);
    })));
    await page.evaluate(() => document.querySelectorAll("details").forEach(d => d.open = true));
    for (const trait of ["T2D", "cad", "bc"]) {
      await page.selectOption("#subgroup-trait", trait);
      assert.equal(await page.locator("#subgroup-table tbody tr").count(), 6);
      assert(await page.evaluate(trait => Array.from(document.querySelectorAll("#subgroup-table tbody tr")).every(row => {
        const cells = row.querySelectorAll("td");
        return ["official_pc", "reference_pc", "oadp_pc"].every((mode, i) => {
          const r = REPORT.subgroup_metrics.find(x => x.trait === trait && x.group === row.dataset.key && x.arm === "TabPFN_v3" && x.mode === mode);
          return cells[i + 2].textContent === (r.auroc == null ? "—" : r.auroc.toFixed(4));
        });
      }), trait));
    }
    await page.selectOption("#subgroup-trait", "T2D");
    await page.selectOption("#cohort", "original_val");
    await page.selectOption("#metric", "auroc");
    for (const name of ["metrics", "paired_differences", "cohort_counts", "single_prs_selections", "subgroup_metrics", "correction_by_pc"]) {
      const pending = page.waitForEvent("download");
      await page.locator(`[data-export="${name}"]`).click();
      const file = await pending, destination = path.join(qa, file.suggestedFilename());
      await file.saveAs(destination);
      assert.equal(sha(destination), sha(path.join(report, `${name}.csv`)), "Downloaded CSV differs from source");
      downloads.push({name, sha256: sha(destination)});
    }
    const pending = page.waitForEvent("download");
    await page.locator("#figure-pdf").click();
    const file = await pending, destination = path.join(qa, file.suggestedFilename());
    await file.saveAs(destination);
    assert.equal(fs.readFileSync(destination).subarray(0, 5).toString(), "%PDF-");
    assert.equal(sha(destination), sha(path.join(report, "assets/original_val__auroc.pdf")));
    downloads.push({name: "figure_pdf", sha256: sha(destination)});
    for (const width of [1280, 768, 390]) {
      await page.setViewportSize({width, height: 1000});
      for (const open of [false, true]) {
        await page.evaluate(open => document.querySelectorAll("details").forEach(d => d.open = open), open);
        const dimensions = await page.evaluate(() => ({width: innerWidth, bodyWidth: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight}));
        assert(dimensions.bodyWidth <= width + 1, `Horizontal overflow: ${width}, details=${open}`);
        layouts.push({...dimensions, expanded: open});
      }
      await page.evaluate(() => { document.querySelectorAll("details").forEach(d => d.open = false); window.scrollTo({top: 0, behavior: "instant"}); });
      await page.screenshot({path: path.join(qa, `top_${width}.png`)});
      await page.locator("#teacher-results").screenshot({path: path.join(qa, `teacher_${width}.png`)});
      if (width === 1280) {
        await page.screenshot({path: path.join(qa, "full_desktop.png"), fullPage: true});
        for (const id of ["effect", "geometry"]) await page.locator(`#${id}`).screenshot({path: path.join(qa, `${id}.png`)});
      }
    }
    await page.setViewportSize({width: 1280, height: 1000});
    await page.selectOption("#cohort", "nested_oof");
    await page.locator("#teacher-results").screenshot({path: path.join(qa, "teacher_oof.png")});
    const links = await page.locator("a[href]").evaluateAll(nodes => nodes.map(n => n.getAttribute("href")));
    for (const link of links) {
      if (link.startsWith("#")) assert.equal(await page.locator(link).count(), 1);
      else if (!/^(https?:|data:|blob:)/.test(link)) assert(fs.existsSync(path.resolve(report, link)), `Missing local link ${link}`);
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(requests, []);
    const receipt = {status: "passed", offline: true, browser: browser.version(), global_filter_combinations: states, summary_models_verified: ["TabPFN_v3", "M0_KD"], teacher_first: true, geometry_rows: 10, subgroup_combinations: 3, page_errors: errors, external_requests: requests, downloads, layouts, html_sha256: sha(html), verifier_sha256: sha(__filename)};
    fs.writeFileSync(path.join(qa, "BROWSER_VERIFIED.json"), JSON.stringify(receipt, null, 2) + "\n");
    console.log(JSON.stringify({status: "passed", states, downloads: downloads.length, errors, requests, layouts}));
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
