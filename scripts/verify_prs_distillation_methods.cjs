/* Actual offline browser checks for the standalone bilingual methods report. */
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const {pathToFileURL}=require('url');
const root=path.resolve(__dirname,'..');
const {chromium}=require(path.join(root,'outputs/report_vendor/browser_qa/node_modules/playwright'));
const out=path.join(root,'outputs/prs_distillation_bio/methods_20261027'),qa=path.join(out,'qa');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 fs.mkdirSync(qa,{recursive:true});
 const browser=await chromium.launch({headless:true});
 const context=await browser.newContext({offline:true});
 const errors=[],requests=[],results=[],htmlHashes={};let interactions=0;
 context.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
 for(const lang of ['zh','en']) for(const width of [1280,390]){
  const page=await context.newPage();page.on('pageerror',e=>errors.push(String(e)));
  await page.setViewportSize({width,height:960});
  const file=path.join(out,`methods_${lang}.html`);htmlHashes[path.basename(file)]=sha(file);
  await page.goto(pathToFileURL(file).href);await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('main > section').count(),7);
  assert.equal(await page.locator('.formula .katex').count(),11);
  assert.equal(await page.locator('#full-table tbody tr').count(),4);
  assert.equal(await page.locator('#filtered-table tbody tr').count(),4);
  assert.equal(await page.locator('[data-annotation]').count(),11);
  assert.equal(await page.locator('#symbol-guide dt').count(),9);
  assert.equal(await page.locator('#full-table tbody tr').last().locator('td').first().textContent(),'TabPFN 3');
  assert.equal(await page.locator('#filtered-table tbody tr').last().locator('td').first().textContent(),'TabPFN 3');
  for(const a of await page.locator('a').evaluateAll(elements=>elements.map(e=>e.getAttribute('href')))){
   if(a.startsWith('#'))assert.equal(await page.locator(a).count(),1);
   else if(!/^https?:/.test(a))assert(fs.existsSync(path.resolve(out,a)),a);
  }
  const layout=await page.evaluate(()=>({width:innerWidth,bodyWidth:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight}));
  assert(layout.bodyWidth<=width+1,'Page has horizontal overflow');
  await page.screenshot({path:path.join(qa,`${lang}_${width}_top.png`)});
  for(const section of ['model','distillation','validation']){
   await page.locator('#'+section).screenshot({path:path.join(qa,`${lang}_${width}_${section}.png`)});
  }
  if(width===1280){
   for(const metric of ['auroc','log_loss','nagelkerke_r2','brier','average_precision']){
    await page.selectOption('#metric',metric);
    assert(await page.evaluate(metric=>['full-table','filtered-table'].every((id,j)=>{
     const rows=Array.from(document.querySelectorAll('#'+id+' tbody tr'));
     return CFG.arms.every((arm,i)=>CFG.traits.every((trait,k)=>{
      const value=REPORT.metrics.find(r=>r.arm===arm&&r.trait===trait&&r.cohort===CFG.cohorts[j])[metric];
      return rows[i].querySelectorAll('td')[k+1].textContent===value.toFixed(4);
     }));
    }),metric));interactions++;
   }
   await page.selectOption('#metric','auroc');
   await page.locator('#paired-details > summary').click();
   for(const cohort of ['original_val','exclude_direct_0177']) for(const metric of ['auroc','log_loss','nagelkerke_r2','brier']){
    await page.selectOption('#cohort',cohort);await page.selectOption('#contrast-metric',metric);
    assert.equal(await page.locator('#paired-plot circle').count(),9);
    assert.equal(await page.locator('#paired-table tbody tr').count(),9);
    assert(await page.evaluate(({cohort,metric})=>{
     const expected=CFG.traits.flatMap(trait=>['LR_UNPENALIZED','M0_SUP','TabPFN_v3'].map(reference=>REPORT.gaps.find(r=>r.trait===trait&&r.cohort===cohort&&r.reference===reference&&r.metric===metric)));
     return Array.from(document.querySelectorAll('#paired-table tbody tr')).every((tr,i)=>tr.lastElementChild.textContent===signed(expected[i].delta)+' ['+signed(expected[i].ci_low)+', '+signed(expected[i].ci_high)+']');
    },{cohort,metric}));interactions++;
   }
   await page.selectOption('#cohort','exclude_direct_0177');await page.selectOption('#contrast-metric','auroc');
   await page.locator('#paired-details').screenshot({path:path.join(qa,`${lang}_${width}_paired.png`)});
  }
  await page.evaluate(()=>document.querySelectorAll('details').forEach(d=>d.open=true));
  await page.evaluate(()=>document.fonts.ready);
  const visibleText=await page.locator('body').innerText();
  assert(!visibleText.includes('Teacher V3')&&!visibleText.includes('TabPFN V3'));
  assert.equal(await page.locator('[data-annotation]:visible').count(),11);
  await page.locator('#symbol-guide').screenshot({path:path.join(qa,`${lang}_${width}_symbols.png`)});
  assert.equal(await page.locator('.formula:visible').count(),11);
  assert(await page.evaluate(()=>Array.from(document.fonts).some(f=>f.family==='KaTeX_Main'&&f.status==='loaded')));
  const formulaLayout=await page.locator('.formula').evaluateAll(nodes=>nodes.map(n=>({name:n.dataset.equation,width:n.clientWidth,scrollWidth:n.scrollWidth,height:n.clientHeight})));
  assert(formulaLayout.every(f=>f.height>30));
  if(width===1280)for(const name of ['gate','model','basis','loss','penalty','filter','r2']){
   await page.locator('[data-equation="'+name+'"]').screenshot({path:path.join(qa,`${lang}_${name}.png`)});
  }
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  results.push({language:lang,...layout,formulaLayout});await page.close();
 }
 assert.equal(interactions,26);assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
 const receipt={status:'passed',browser:browser.version(),offline:true,languages:2,sections_per_page:7,formulas_per_page:11,
  interactive_combinations:interactions,external_requests:0,page_errors:0,viewports:results,tested_html_sha256:htmlHashes,
  verifier_sha256:sha(__filename)};
 fs.writeFileSync(path.join(qa,'BROWSER_VERIFIED.json'),JSON.stringify(receipt,null,2)+'\n');
 console.log(JSON.stringify({status:receipt.status,interactive_combinations:interactions,page_errors:0,external_requests:0,
  viewports:results.map(({formulaLayout,...r})=>r),desktop_formula_overflow:results.filter(r=>r.width===1280).flatMap(r=>r.formulaLayout.filter(f=>f.scrollWidth>f.width+1).map(f=>({language:r.language,...f})))},null,2));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
