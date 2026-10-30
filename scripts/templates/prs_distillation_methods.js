"use strict";
const REPORT = JSON.parse(document.getElementById('report-data').textContent);
const CFG = REPORT.config;
const COLORS = ['#758697', '#ad8b55', '#087f83', '#766397'];
function renderTables() {
  const metric = document.getElementById('metric').value;
  const lower = ['log_loss', 'brier'].includes(metric);
  CFG.cohorts.forEach((cohort, index) => {
    const rows = REPORT.metrics.filter(r => r.cohort === cohort);
    const best = CFG.traits.map(trait => (lower ? Math.min : Math.max)(...rows.filter(r => r.trait === trait).map(r => r[metric])));
    const header = '<thead><tr><th>'+CFG.model_label+'</th>'+CFG.trait_names.map(x=>`<th>${x}</th>`).join('')+'</tr></thead>';
    const body = CFG.arms.map((arm, i) => '<tr class="'+(arm==='M0_KD'?'student':arm==='TabPFN_v3'?'teacher':'')+'"><td>'+CFG.names[i]+'</td>'+CFG.traits.map((trait,j)=>{
      const r=rows.find(x=>x.arm===arm&&x.trait===trait);
      return `<td class="${r[metric]===best[j]?'winner':''}">${r[metric].toFixed(4)}</td>`;
    }).join('')+'</tr>').join('');
    document.getElementById(index===0?'full-table':'filtered-table').innerHTML='<table>'+header+'<tbody>'+body+'</tbody></table>';
  });
  document.getElementById('metric-help').textContent=CFG.help[metric];
}
function signed(x) { return (x>=0?'+':'')+x.toFixed(4); }
function renderPairs() {
  const cohort=document.getElementById('cohort').value;
  const metric=document.getElementById('contrast-metric').value;
  const refs=['LR_UNPENALIZED','M0_SUP','TabPFN_v3'];
  const rows=CFG.traits.flatMap(trait=>refs.map(reference=>REPORT.gaps.find(r=>r.trait===trait&&r.cohort===cohort&&r.reference===reference&&r.metric===metric)));
  const extent=Math.max(...rows.flatMap(r=>[Math.abs(r.ci_low),Math.abs(r.ci_high)]))*1.2;
  const x=v=>465+v/extent*275;
  let svg='<svg viewBox="0 0 820 408" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Paired M0_KD differences and 95 percent intervals">';
  svg+='<text x="10" y="20" font-size="13" fill="#425f72">'+(['log_loss','brier'].includes(metric)?CFG.direction_down:CFG.direction_up)+'</text>';
  svg+=`<line x1="465" x2="465" y1="36" y2="350" stroke="#a9bcc7" stroke-dasharray="4 4"/>`;
  for (const tick of [-extent, -extent/2, 0, extent/2, extent]) {
    svg+=`<text x="${x(tick)}" y="381" text-anchor="middle" font-size="12" fill="#526b7b">${signed(tick)}</text>`;
  }
  rows.forEach((r,i)=>{
    const y=50+i*35, ti=CFG.traits.indexOf(r.trait), mi=CFG.arms.indexOf(r.reference), color=COLORS[mi];
    if(i%3===0) svg+=`<text x="10" y="${y+4}" font-size="13" font-weight="600" fill="#183144">${CFG.trait_names[ti]}</text>`;
    svg+=`<text x="85" y="${y+4}" font-size="12" fill="#526b7b">${CFG.names[mi]}</text>`;
    svg+=`<line x1="${x(r.ci_low)}" x2="${x(r.ci_high)}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="2"/><circle cx="${x(r.delta)}" cy="${y}" r="4" fill="${color}"/>`;
  });
  document.getElementById('paired-plot').innerHTML=svg+'</svg>';
  document.getElementById('paired-table').innerHTML='<table><thead><tr><th>'+CFG.endpoint_label+'</th><th>'+CFG.comparison_label+'</th><th>'+CFG.delta_label+'</th></tr></thead><tbody>'+rows.map(r=>'<tr><td>'+CFG.trait_names[CFG.traits.indexOf(r.trait)]+'</td><td>'+CFG.names[CFG.arms.indexOf(r.reference)]+'</td><td>'+signed(r.delta)+' ['+signed(r.ci_low)+', '+signed(r.ci_high)+']</td></tr>').join('')+'</tbody></table>';
}
document.getElementById('metric').addEventListener('change',renderTables);
document.getElementById('cohort').addEventListener('change',renderPairs);
document.getElementById('contrast-metric').addEventListener('change',renderPairs);
renderTables();
renderPairs();
