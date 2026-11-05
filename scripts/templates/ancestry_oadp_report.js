"use strict";
const REPORT = JSON.parse(document.getElementById("report-data").textContent);
const TRAITS = {T2D: ["T2D", "2 型糖尿病"], cad: ["CAD", "冠心病"], bc: ["BC", "女性乳腺癌"]};
const ARMS = {TabPFN_v3: "TabPFN v3 教师", M0_KD: "M0 蒸馏学生", H_SELECTED: "训练内选定学生", M0_SUP: "M0 监督学生", LR_ALL: "All-PRS LR", LR_SINGLE_BEST: "Single-best LR"};
const METRICS = {auroc: ["AUROC", true], log_loss: ["Log loss", false], brier: ["Brier score", false], nagelkerke_r2: ["Nagelkerke R²", true]};
const COHORTS = {
  original_val: ["原开发验证集", "原验证集已参与开发，本次为事后敏感性分析；尚未进行外部验证。"],
  nested_oof: ["训练集 · 嵌套 OOF", "共享外层五折；PRS 选择及教师软标签在外层训练部分内生成，外层患者仅预测。"],
  exclude_shared_group: ["验证集 · 排除共享亲缘组", "仅过滤与训练集共享已知基因型推断亲缘组的验证患者；训练集不变，无法判定者保留。"],
  exclude_direct_0177: ["验证集 · 排除直接近亲", "仅过滤与训练患者存在直接 KING ≥ 0.177 关系的验证患者；训练集不变，无法判定者保留。"]
};
const byId = id => document.getElementById(id);
const fmt = x => x == null ? "—" : Number(x).toFixed(4);
const signed = x => x == null ? "—" : (x < 0 ? "−" : "+") + Math.abs(x).toFixed(6);
const comma = x => Number(x).toLocaleString("en-US");
const esc = x => String(x).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const measured = (trait, cohort, mode, arm) => REPORT.metrics.find(r => r.trait === trait && r.cohort === cohort && r.mode === mode && r.arm === arm);
const gap = (trait, cohort, metric, arm) => REPORT.paired_differences.find(r => r.trait === trait && r.cohort === cohort && r.metric === metric && r.arm === `oadp_pc/${arm}` && r.reference === `reference_pc/${arm}`);
const interval = r => `${signed(r.delta)} [${signed(r.ci_low)}, ${signed(r.ci_high)}]`;
function rows(id, records) {
  byId(id).querySelector("tbody").innerHTML = records.map(r => `<tr data-key="${esc(r.key || "")}" class="${r.primary ? "primary-row" : ""}">${r.cells.map(x => `<td>${esc(x)}</td>`).join("")}</tr>`).join("");
}
function cards(id, arm, cohort, metric) {
  byId(id).innerHTML = Object.entries(TRAITS).map(([trait, [code, name]]) => {
    const a = measured(trait, cohort, "reference_pc", arm), b = measured(trait, cohort, "oadp_pc", arm), c = measured(trait, cohort, "official_pc", arm);
    const d = gap(trait, cohort, metric, arm), higher = METRICS[metric][1];
    const crossing = d.ci_low <= 0 && d.ci_high >= 0;
    const note = crossing ? "区间跨 0" : ((higher ? d.ci_low > 0 : d.ci_high < 0) ? "区间位于改善侧" : "区间位于下降侧");
    return `<article class="disease-card" data-trait="${trait}"><div class="disease-top"><h3 class="disease-code">${code}</h3><span class="disease-name">${name}</span></div><p class="sample-count">${comma(a.n)} 人 · ${comma(a.events)} 病例 · ${METRICS[metric][0]}</p><div class="value-pair"><div><strong class="old">${fmt(a[metric])}</strong><div class="value-label">未校正参考 PC</div></div><span class="arrow" aria-hidden="true">→</span><div><strong class="new">${fmt(b[metric])}</strong><div class="value-label">OADP 校正 PC</div></div></div><p class="reading-note official-value">官方 PC 对照：${fmt(c[metric])}</p><div class="delta-line"><strong class="delta">Δ ${signed(d.delta)}</strong><span>${note}</span></div><p class="ci-line">95% CI [${signed(d.ci_low)}, ${signed(d.ci_high)}]</p></article>`;
  }).join("");
}
function update() {
  const cohort = byId("cohort").value, metric = byId("metric").value;
  const [label, higher] = METRICS[metric];
  byId("cohort-note").textContent = COHORTS[cohort][1];
  byId("metric-help").textContent = `${label} ${higher ? "越高越好；正" : "越低越好；负"}差值表示改善。${metric === "nagelkerke_r2" ? "伪 R² 受病例构成影响，不宜直接比较不同人群。" : "差值为 OADP − 未校正参考 PC。"}`;
  cards("teacher-cards", "TabPFN_v3", cohort, metric);
  cards("student-cards", "M0_KD", cohort, metric);
  const key = `${cohort}__${metric}`, figure = REPORT.figures[key];
  byId("effect-plot").src = figure.svg;
  byId("effect-plot").dataset.figure = key;
  byId("effect-plot").alt = `${COHORTS[cohort][0]}，Δ ${label}，三疾病六模型的 OADP 减未校正参考 PC，配对差值和 95% CI。精确数值见图后展开表。`;
  byId("figure-pdf").href = figure.pdf;
  byId("figure-pdf").download = `oadp__${key}.pdf`;
  byId("figure-context").textContent = `${COHORTS[cohort][0]} · Δ ${label} · ${higher ? "正值" : "负值"}表示改善`;
  byId("paired-caption").textContent = byId("figure-context").textContent;
  byId("point-caption").textContent = `${COHORTS[cohort][0]} · ${label}`;
  const paired = [], points = [];
  for (const trait of Object.keys(TRAITS)) for (const arm of Object.keys(ARMS)) {
    const a = measured(trait, cohort, "reference_pc", arm)[metric], b = measured(trait, cohort, "oadp_pc", arm)[metric], c = measured(trait, cohort, "official_pc", arm)[metric];
    const meta = {key: `${trait}/${arm}`, primary: arm === "TabPFN_v3"};
    paired.push({...meta, cells: [TRAITS[trait][0], ARMS[arm], fmt(a), fmt(b), interval(gap(trait, cohort, metric, arm))]});
    points.push({...meta, cells: [TRAITS[trait][0], ARMS[arm], fmt(c), fmt(a), fmt(b)]});
  }
  rows("paired-table", paired);
  rows("point-table", points);
}
function subgroups() {
  const trait = byId("subgroup-trait").value;
  const source = REPORT.subgroup_metrics.filter(r => r.trait === trait && r.mode === "official_pc" && r.arm === "TabPFN_v3");
  rows("subgroup-table", source.map(r => ({key: r.group, cells: [r.group, `${comma(r.n)} / ${comma(r.events)}`, ...["official_pc", "reference_pc", "oadp_pc"].map(mode => fmt(REPORT.subgroup_metrics.find(x => x.trait === trait && x.group === r.group && x.mode === mode && x.arm === "TabPFN_v3").auroc)), r.unstable ? "小样本" : "描述性估计"]})));
}
rows("geometry-table", REPORT.correction_by_pc.map(r => ({key: r.pc, cells: [r.pc, r.sd_ratio.toFixed(3), r.correlation.toFixed(6), r.non_affine_rms_over_simple_sd.toFixed(4)]})));
for (const id of ["cohort", "metric"]) byId(id).addEventListener("change", update);
byId("subgroup-trait").addEventListener("change", subgroups);
document.querySelectorAll("[data-export]").forEach(button => button.addEventListener("click", () => {
  const name = button.dataset.export;
  const url = URL.createObjectURL(new Blob([REPORT.csv[name]], {type: "text/csv;charset=utf-8"}));
  const a = document.createElement("a"); a.href = url; a.download = `${name}.csv`; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}));
update(); subgroups();
