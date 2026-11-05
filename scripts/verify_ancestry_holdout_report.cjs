/* Verify the offline report against its aggregate source tables. */
const fs=require("fs"), path=require("path"), assert=require("assert"), crypto=require("crypto");
const {pathToFileURL}=require("url");
const root=path.resolve(__dirname,"..");
process.env.PLAYWRIGHT_BROWSERS_PATH ||= path.join(root,"outputs/report_vendor/browser_qa/browsers");
const {chromium}=require(path.join(root,"outputs/report_vendor/browser_qa/node_modules/playwright"));
const report=path.resolve(process.argv[2] || path.join(root,"outputs/ancestry_holdout_agreement_20261105/report"));
const html=path.join(report,"report_zh.html"), qa=path.join(report,"assets/qa");
const sha=p=>crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
(async()=>{
  fs.mkdirSync(qa,{recursive:true});
  const browser=await chromium.launch({headless:true});
  try {
    const context=await browser.newContext({offline:true,acceptDownloads:true});
    const requests=[],errors=[],downloads=[],layouts=[];
    context.on("request",r=>{if(/^https?:/.test(r.url())) requests.push(r.url());});
    const page=await context.newPage();
    page.on("pageerror",e=>errors.push(String(e)));
    page.on("console",m=>{if(m.type()==="error") errors.push(m.text());});
    await page.setViewportSize({width:1280,height:1000});
    await page.goto(pathToFileURL(html).href);
    await page.evaluate(()=>document.fonts.ready);
    assert.equal(await page.locator("h1").innerText(),"Holdout 预测一致性");
    assert.equal(await page.evaluate(()=>REPORT.agreement.length),12);
    assert.equal(await page.evaluate(()=>Object.keys(REPORT.figures).length),8);
    let states=0;
    for(const scope of ["train_only","trainval_refit"]) for(const student of ["M0_KD","H_SELECTED"]){
      await page.selectOption("#scope",scope);await page.selectOption("#student",student);
      await page.evaluate(()=>Promise.all(Array.from(document.images).map(i=>i.decode())));
      assert(await page.evaluate(({scope,student})=>{
        const f=(x,n=4)=>Number(x).toFixed(n);
        for(const trait of ["T2D","cad","bc"]){
          const r=REPORT.agreement.find(r=>r.scope===scope&&r.student===student&&r.trait===trait);
          const c=document.querySelector(`#summary-cards [data-trait="${trait}"]`);
          if(c.querySelector(".pearson").textContent!==f(r.pearson_probability)) return false;
          if(c.querySelector(".spearman").textContent!==f(r.spearman)) return false;
          if(c.querySelector(".pearson-ci").textContent!==`95% CI [${f(r.pearson_probability_ci_low)}, ${f(r.pearson_probability_ci_high)}]`) return false;
          if(c.querySelector(".spearman-ci").textContent!==`95% CI [${f(r.spearman_ci_low)}, ${f(r.spearman_ci_high)}]`) return false;
          if(c.querySelector(".mae").textContent!==`${f(100*r.mae_probability,2)} pp`) return false;
          if(c.querySelector(".overlap").textContent!==`${f(100*r.top_decile_overlap,1)}%`) return false;
          if(c.querySelector(".sample-count").textContent!==`${r.n.toLocaleString("en-US")} 人 · ${r.student_model}`) return false;
          const t=document.querySelector(`#detail-table [data-trait="${trait}"]`).querySelectorAll("td");
          if(t[1].textContent!==f(r.mean_teacher)||t[2].textContent!==f(r.mean_student)) return false;
          if(t[3].textContent!==f(100*r.mean_student_minus_teacher,2)||t[4].textContent!==f(100*r.absolute_difference_p95,2)) return false;
          if(t[5].textContent!==`${r.top_decile_intersection} / ${r.top_decile_n} (${f(100*r.top_decile_overlap,1)}%)`) return false;
        }
        for(const kind of ["scatter","difference"]){
          const key=`${scope}__${student}__${kind}`,i=document.getElementById(`${kind}-plot`);
          if(i.src!==REPORT.figures[key].svg||i.naturalWidth===0||document.getElementById(`${kind}-pdf`).href!==REPORT.figures[key].pdf) return false;
        }
        return true;
      },{scope,student}),`Incorrect numerical display: ${scope}/${student}`);
      await page.locator("#analysis").screenshot({path:path.join(qa,`cards_${scope}_${student}.png`)});
      await page.locator("#scatter").screenshot({path:path.join(qa,`scatter_${scope}_${student}.png`)});
      states++;
    }
    assert.equal(await page.locator("#stability-table tbody tr").count(),9);
    assert(await page.evaluate(()=>Array.from(document.querySelectorAll("#stability-table tbody tr")).every(tr=>{
      const [trait,model]=tr.dataset.key.split("/"),r=REPORT.cross_scope_stability.find(r=>r.trait===trait&&r.model===model),c=tr.querySelectorAll("td");
      return c[2].textContent===r.pearson_probability.toFixed(4)&&c[3].textContent===r.spearman.toFixed(4)&&c[4].textContent===(100*r.mae_probability).toFixed(2);
    })));
    await page.selectOption("#scope","trainval_refit");await page.selectOption("#student","M0_KD");
    await page.evaluate(()=>document.querySelectorAll("details").forEach(d=>d.open=true));
    for(const name of ["agreement","availability","cross_scope_stability"]){
      const waiting=page.waitForEvent("download");await page.locator(`[data-export="${name}"]`).click();
      const d=await waiting,p=path.join(qa,d.suggestedFilename());await d.saveAs(p);
      assert.equal(sha(p),sha(path.join(report,`${name}.csv`)));
      downloads.push({name,sha256:sha(p)});
    }
    for(const kind of ["scatter","difference"]){
      const waiting=page.waitForEvent("download");await page.locator(`#${kind}-pdf`).click();
      const d=await waiting,p=path.join(qa,d.suggestedFilename());await d.saveAs(p);
      assert.equal(fs.readFileSync(p).subarray(0,5).toString(),"%PDF-");
      assert.equal(sha(p),sha(path.join(report,`assets/trainval_refit__M0_KD__${kind}.pdf`)));
      downloads.push({name:kind,sha256:sha(p)});
    }
    for(const width of [1280,768,390]){
      await page.setViewportSize({width,height:1000});
      for(const expanded of [false,true]){
        await page.evaluate(v=>document.querySelectorAll("details").forEach(d=>d.open=v),expanded);
        const dims=await page.evaluate(()=>({width:innerWidth,bodyWidth:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight}));
        assert(dims.bodyWidth<=width+1,`Horizontal overflow: ${width}/${expanded}`);layouts.push({...dims,expanded});
      }
      await page.evaluate(()=>{document.querySelectorAll("details").forEach(d=>d.open=false);window.scrollTo({top:0,behavior:"instant"});});
      await page.screenshot({path:path.join(qa,`top_${width}.png`)});
      if(width===1280){
        await page.screenshot({path:path.join(qa,"full_desktop.png"),fullPage:true});
        await page.locator("#difference").screenshot({path:path.join(qa,"difference_trainval_m0.png")});
      }
    }
    assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
    const receipt={status:"passed",offline:true,browser:browser.version(),filter_combinations:states,downloads,layouts,page_errors:errors,external_requests:requests,html_sha256:sha(html),verifier_sha256:sha(__filename)};
    fs.writeFileSync(path.join(qa,"BROWSER_VERIFIED.json"),JSON.stringify(receipt,null,2)+"\n");
    console.log(JSON.stringify({status:"passed",states,downloads:downloads.length,errors,requests,layouts}));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
