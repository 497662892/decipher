"use strict";
const REPORT = JSON.parse(document.getElementById("report-data").textContent);
const TRAITS = {T2D: ["T2D", "2 型糖尿病"], cad: ["CAD", "冠心病"], bc: ["BC", "女性乳腺癌"]};
const SCOPES = {train_only: "仅 train · 已验证模型", trainval_refit: "train + validation · 重拟合"};
const ARMS = {TabPFN_v3: "TabPFN v3", M0_KD: "M0 蒸馏学生", H_SELECTED: "既定学生方案"};
const byId = id => document.getElementById(id);
const f = (x, n=4) => Number(x).toFixed(n);
const comma = x => Number(x).toLocaleString("en-US");
const ci = (r, key) => `[${f(r[key+"_ci_low"])}, ${f(r[key+"_ci_high"])}]`;
const row = (trait, scope, student) => REPORT.agreement.find(r => r.trait === trait && r.scope === scope && r.student === student);
function update() {
  const scope = byId("scope").value, student = byId("student").value;
  byId("scope-note").textContent = scope === "train_only" ? "沿用仅用 train 拟合的学生和教师上下文，holdout 只预测。" : "模型使用 train + validation 标签重拟合；软标签在合并训练范围内交叉生成，holdout 只预测。";
  byId("summary-cards").innerHTML = Object.entries(TRAITS).map(([trait,[label,name]]) => {
    const r = row(trait,scope,student);
    return `<article class="disease-card" data-trait="${trait}"><div class="disease-top"><h3 class="disease-code">${label}</h3><span class="disease-name">${name}</span></div><p class="sample-count">${comma(r.n)} 人 · ${r.student_model}</p><div class="value-pair"><div><strong class="new pearson">${f(r.pearson_probability)}</strong><div class="value-label">Pearson · 概率相关性</div></div></div><p class="ci-line pearson-ci">95% CI ${ci(r,"pearson_probability")}</p><div class="delta-line"><strong>Spearman</strong><span class="spearman">${f(r.spearman)}</span></div><p class="ci-line spearman-ci">95% CI ${ci(r,"spearman")}</p><p class="reading-note">平均绝对概率差：<strong class="mae">${f(100*r.mae_probability,2)} pp</strong><br>最高 10% 重叠：<strong class="overlap">${f(100*r.top_decile_overlap,1)}%</strong></p></article>`;
  }).join("");
  const title = `${SCOPES[scope]} · ${ARMS[student]}`;
  byId("figure-context").textContent = title;
  byId("table-caption").textContent = title;
  for (const kind of ["scatter", "difference"]) {
    const key = `${scope}__${student}__${kind}`, fig = REPORT.figures[key];
    const img = byId(`${kind}-plot`); img.src = fig.svg; img.dataset.figure = key;
    img.alt = `${title}，三个疾病的${kind === "scatter" ? "教师与学生概率对照" : "学生减教师概率差值"}。`;
    const a = byId(`${kind}-pdf`); a.href = fig.pdf; a.download = `${key}.pdf`;
  }
  byId("detail-table").querySelector("tbody").innerHTML = Object.keys(TRAITS).map(trait => {
    const r = row(trait,scope,student);
    return `<tr data-trait="${trait}"><td>${TRAITS[trait][0]}</td><td>${f(r.mean_teacher)}</td><td>${f(r.mean_student)}</td><td>${f(100*r.mean_student_minus_teacher,2)}</td><td>${f(100*r.absolute_difference_p95,2)}</td><td>${r.top_decile_intersection} / ${r.top_decile_n} (${f(100*r.top_decile_overlap,1)}%)</td></tr>`;
  }).join("");
}
byId("stability-table").querySelector("tbody").innerHTML = REPORT.cross_scope_stability.map(r => `<tr data-key="${r.trait}/${r.model}"><td>${TRAITS[r.trait][0]}</td><td>${ARMS[r.model]}</td><td>${f(r.pearson_probability)}</td><td>${f(r.spearman)}</td><td>${f(100*r.mae_probability,2)}</td></tr>`).join("");
for (const id of ["scope", "student"]) byId(id).addEventListener("change", update);
document.querySelectorAll("[data-export]").forEach(button => button.addEventListener("click", () => {
  const name = button.dataset.export, url = URL.createObjectURL(new Blob([REPORT.csv[name]], {type:"text/csv;charset=utf-8"}));
  const a = document.createElement("a"); a.href=url; a.download=`${name}.csv`; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}));
update();
