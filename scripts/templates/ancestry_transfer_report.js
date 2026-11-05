"use strict";
const REPORT = JSON.parse(document.getElementById("report-data").textContent);
const TRAITS = {T2D: ["T2D", "2 型糖尿病"], cad: ["CAD", "冠心病"], bc: ["BC", "女性乳腺癌"]};
const ARMS = {
  M0_KD: "M0 蒸馏学生", H_SELECTED: "训练内选定学生", TabPFN_v3: "TabPFN v3 教师",
  M0_SUP: "M0 监督学生", LR_ALL: "All-PRS LR", LR_SINGLE_BEST: "Single-best LR"
};
const MODES = {official_pc: "官方 PC", reference_pc: "统一参考 PC"};
const COHORTS = {
  original_val: ["原开发验证集", "该验证集此前已参与开发；此处为开发比较，不能当作从未使用的外部测试集。"],
  nested_oof: ["训练集 · 嵌套 OOF", "训练集的外层五折预测；PRS 选择及学生软标签在各外层训练折内产生，外层患者仅用于预测评估。"],
  exclude_shared_group: ["验证集 · 排除共享亲缘组", "仅排除与训练集共享已知基因型推断亲缘组的验证患者；训练集不变，无法判定者保留。属于事后敏感性分析。"],
  exclude_direct_0177: ["验证集 · 排除直接近亲", "仅排除与训练患者存在直接 KING ≥ 0.177 关系的验证患者；训练集不变，无法判定者保留。属于事后敏感性分析。"]
};
const METRICS = {
  auroc: ["AUROC", true, "区分病例与对照的能力。越高越好；正差值表示模型区分能力更高。"],
  log_loss: ["Log loss", false, "概率预测的对数损失。越低越好；负差值表示预测损失更小。"],
  brier: ["Brier score", false, "预测概率与结局之间的均方误差。越低越好；负差值表示误差更小。"],
  nagelkerke_r2: ["Nagelkerke R²", true, "相对空模型的伪 R²。越高越好；受病例构成影响，不宜直接比较不同人群。"]
};
const byId = id => document.getElementById(id);
const fmt = value => value == null ? "—" : Number(value).toFixed(4);
const signed = value => value == null ? "—" : (value < 0 ? "−" : "+") + Math.abs(value).toFixed(4);
const comma = value => Number(value).toLocaleString("en-US");
const escape = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const ciText = row => `${signed(row.delta)} [${signed(row.ci_low)}, ${signed(row.ci_high)}]`;
const metricRow = (trait, cohort, mode, arm) => REPORT.metrics.find(r => r.trait === trait && r.cohort === cohort && r.mode === mode && r.arm === arm);
const gapRow = (trait, cohort, metric, arm, reference) => REPORT.paired_differences.find(r => r.trait === trait && r.cohort === cohort && r.metric === metric && r.arm === arm && r.reference === reference);
const state = () => ({cohort: byId("cohort").value, metric: byId("metric").value, mode: byId("baseline-mode").value});
let shownTransfer = [], shownBaseline = [];

function tableRows(id, rows) {
  byId(id).querySelector("tbody").innerHTML = rows.map(({cells, primary = false, start = false, key = ""}) =>
    `<tr class="${primary ? "primary-row " : ""}${start ? "trait-start" : ""}" data-key="${escape(key)}">${cells.map(c => `<td>${escape(c)}</td>`).join("")}</tr>`).join("");
}

function showFigure(kind, cohort, metric, mode) {
  const key = [kind, cohort, metric, ...(kind === "baseline" ? [mode] : [])].join("__");
  const figure = REPORT.figures[key];
  const label = METRICS[metric][0];
  byId(`${kind}-plot`).src = figure.svg;
  byId(`${kind}-plot`).alt = `${COHORTS[cohort][0]}，${label}，${kind === "transfer" ? "参考 PC 减官方 PC" : MODES[mode] + "，模型减 single-best LR"}。三疾病配对差值及 95% 置信区间；精确数值见紧接图后的展开表。`;
  byId(`${kind}-plot`).dataset.figure = key;
  byId(`${kind}-pdf`).href = figure.pdf;
  byId(`${kind}-pdf`).download = `${key}.pdf`;
}

function updateCards(id, arm, cohort, metric) {
  const [label, higher] = METRICS[metric];
  byId(id).innerHTML = Object.entries(TRAITS).map(([trait, [code, name]]) => {
    const a = metricRow(trait, cohort, "official_pc", arm);
    const b = metricRow(trait, cohort, "reference_pc", arm);
    const gap = gapRow(trait, cohort, metric, `reference_pc/${arm}`, `official_pc/${arm}`);
    const straddles = gap.ci_low <= 0 && gap.ci_high >= 0;
    const interpretation = straddles ? "区间跨 0" : ((higher ? gap.ci_low > 0 : gap.ci_high < 0) ? "区间位于改善侧" : "区间位于下降侧");
    return `<article class="disease-card" data-trait="${trait}"><div class="disease-top"><h3 class="disease-code">${code}</h3><span class="disease-name">${name}</span></div><p class="sample-count">${comma(a.n)} 人 · ${comma(a.events)} 病例 · ${label}</p><div class="value-pair"><div><strong class="old">${fmt(a[metric])}</strong><div class="value-label">官方 PC</div></div><span class="arrow" aria-hidden="true">→</span><div><strong class="new">${fmt(b[metric])}</strong><div class="value-label">统一参考 PC</div></div></div><div class="delta-line"><strong class="delta">Δ ${signed(gap.delta)}</strong><span>${interpretation}</span></div><p class="ci-line">95% CI [${signed(gap.ci_low)}, ${signed(gap.ci_high)}]</p></article>`;
  }).join("");
}

function update() {
  const {cohort, metric, mode} = state();
  const [label, higher, help] = METRICS[metric];
  byId("metric-help").textContent = help;
  byId("cohort-note").textContent = COHORTS[cohort][1];
  const direction = higher ? "正值表示改善" : "负值表示改善";
  byId("transfer-context").textContent = `${COHORTS[cohort][0]} · Δ ${label} · ${direction}`;
  byId("baseline-context").textContent = `${COHORTS[cohort][0]} · ${MODES[mode]} · Δ ${label} · ${direction}`;
  updateCards("teacher-cards", "TabPFN_v3", cohort, metric);
  updateCards("summary-cards", "M0_KD", cohort, metric);
  const transferRows = [], baselineRows = [], points = [];
  shownTransfer = [];
  shownBaseline = [];
  for (const trait of Object.keys(TRAITS)) {
    for (const [i, arm] of Object.keys(ARMS).entries()) {
      const a = metricRow(trait, cohort, "official_pc", arm)[metric];
      const b = metricRow(trait, cohort, "reference_pc", arm)[metric];
      const gap = gapRow(trait, cohort, metric, `reference_pc/${arm}`, `official_pc/${arm}`);
      const meta = {primary: arm === "M0_KD", start: i === 0, key: `${trait}/${arm}`};
      transferRows.push({...meta, cells: [TRAITS[trait][0], ARMS[arm], fmt(a), fmt(b), ciText(gap)]});
      points.push({...meta, cells: [TRAITS[trait][0], ARMS[arm], fmt(a), fmt(b), signed(b - a)]});
      shownTransfer.push(gap);
      if (i < 3) {
        const row = gapRow(trait, cohort, metric, `${mode}/${arm}`, `${mode}/LR_SINGLE_BEST`);
        baselineRows.push({...meta, cells: [TRAITS[trait][0], ARMS[arm], fmt(metricRow(trait, cohort, mode, arm)[metric]), fmt(metricRow(trait, cohort, mode, "LR_SINGLE_BEST")[metric]), ciText(row)]});
        shownBaseline.push(row);
      }
    }
  }
  tableRows("transfer-table", transferRows);
  tableRows("baseline-table", baselineRows);
  tableRows("point-table", points);
  byId("point-caption").textContent = `${COHORTS[cohort][0]} · ${label} · ${higher ? "越高越好" : "越低越好"}`;
  showFigure("transfer", cohort, metric, mode);
  showFigure("baseline", cohort, metric, mode);
}

function updateSubgroups() {
  const trait = byId("subgroup-trait").value;
  const rows = REPORT.subgroup_metrics.filter(r => r.trait === trait && r.arm === "M0_KD" && r.mode === "official_pc");
  tableRows("subgroup-table", rows.map(a => {
    const b = REPORT.subgroup_metrics.find(r => r.trait === trait && r.group === a.group && r.arm === "M0_KD" && r.mode === "reference_pc");
    return {key: a.group, cells: [a.group, `${comma(a.n)} / ${comma(a.events)}`, fmt(a.auroc), fmt(b.auroc), a.unstable ? "小样本，估计不稳定" : "描述性估计"]};
  }));
  byId("subgroup-table").querySelectorAll("tbody tr").forEach((tr, i) => {
    if (rows[i].unstable) tr.lastElementChild.className = "small-sample";
  });
}

function download(content, name) {
  const url = URL.createObjectURL(new Blob([content], {type: "text/csv;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadRows(rows, kind) {
  const fields = Object.keys(rows[0]);
  const quote = x => '"' + String(x == null ? "" : x).replaceAll('"', '""') + '"';
  const csv = [fields.join(","), ...rows.map(row => fields.map(k => quote(row[k])).join(","))].join("\n") + "\n";
  const {cohort, metric, mode} = state();
  download(csv, [kind, cohort, metric, ...(kind === "baseline" ? [mode] : [])].join("__") + ".csv");
}

tableRows("selection-table", REPORT.single_prs_selections.map(r => ({cells: [TRAITS[r.trait][0], MODES[r.mode], r.outer_fold === -1 ? "全训练 → 验证集" : `外层折 ${r.outer_fold + 1} 的训练部分`, r.selected_prs, fmt(r.selection_log_loss)]})));
tableRows("counts-table", REPORT.cohort_counts.map(r => ({cells: [TRAITS[r.trait][0], COHORTS[r.cohort][0], comma(r.n), comma(r.events), comma(r.n_kinship_groups)]})));
byId("sources").innerHTML = Object.entries(REPORT.sources).map(([title, url]) => `<li><a href="${escape(url)}" rel="noreferrer">${escape(title)}</a></li>`).join("");
for (const id of ["cohort", "metric", "baseline-mode"]) byId(id).addEventListener("change", update);
byId("subgroup-trait").addEventListener("change", updateSubgroups);
byId("transfer-csv").addEventListener("click", () => downloadRows(shownTransfer, "transfer"));
byId("baseline-csv").addEventListener("click", () => downloadRows(shownBaseline, "baseline"));
document.querySelectorAll("[data-export]").forEach(button => button.addEventListener("click", () => download(REPORT.csv[button.dataset.export], `${button.dataset.export}.csv`)));
update();
updateSubgroups();
