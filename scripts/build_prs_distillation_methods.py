"""Build a bilingual, offline methods report for the frozen BIO M0_KD student."""
from __future__ import annotations

import json
import os
import re
import zipfile
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from prs_report_math import check_latex, render_latex

from decipehr.external_pgs import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/prs_distillation_bio/methods_20261027"
BIO = ROOT / "outputs/prs_distillation_bio/exploration_20261026_3569648f99a8"
LR = ROOT / "outputs/prs_distillation_bio/plain_logistic_20261027"
KIN = ROOT / "outputs/prs_distillation_bio/validation_kinship_filters_20261027"
OLD = ROOT / "outputs/tabpfn_metadata_score_ablations/summary"
TEMPLATE = ROOT / "scripts/templates"
TRAITS = ["T2D", "cad", "bc"]
ARMS = ["LR_UNPENALIZED", "M0_SUP", "M0_KD", "TabPFN_v3"]
COHORTS = ["original_val", "exclude_direct_0177"]
METRICS = ["auroc", "log_loss", "nagelkerke_r2", "brier", "average_precision"]
EQUATIONS = {
    "linear": r"\eta_i^{\mathrm{LR}}=\alpha+\gamma^\top A_i+\delta^\top C_i+\theta_a a_i+\sum_{k=1}^{K}\theta_k Z_{ik},\qquad p_i=\sigma(\eta_i).",
    "gate": r"\begin{aligned}\pi_k(A_i)&=\frac{\exp(\bar W_k A_i+\bar c_k)}{\sum_{\ell=1}^{K}\exp(\bar W_\ell A_i+\bar c_\ell)},\\ \beta_k(A_i)&=\pi_k(A_i)\,r_k,\qquad r_k=\exp(\rho_k)>0,\qquad \sum_k\pi_k(A_i)=1.\end{aligned}",
    "model": r"\begin{aligned}\eta_i&=\underbrace{\alpha+\gamma^\top A_i+\delta^\top C_i}_{\text{non-age baseline}}+\underbrace{f(a_i)}_{\text{age adjustment}}+\underbrace{\sum_{k=1}^{K}\beta_k(A_i)Z_{ik}}_{\text{genetic contribution}},\\p_{S,i}&=\sigma(\eta_i)=\frac{1}{1+\exp(-\eta_i)}.\end{aligned}",
    "age": r"f(a)=\sum_{j=1}^{3}b_j B_j(a),\qquad b_j\geq0,\qquad f(a_0)=0.",
    "basis": r"\begin{aligned}\bar a&=\operatorname{clip}(a;q_0,q_3),\quad(q_0,q_1,q_2,q_3)=Q_{\mathcal D}(0,\tfrac13,\tfrac23,1),\quad a_0=Q_{\mathcal D}(\tfrac12),\\B_j(a)&=\operatorname{clip}(\bar a-q_{j-1};0,q_j-q_{j-1})-\operatorname{clip}(a_0-q_{j-1};0,q_j-q_{j-1}).\end{aligned}",
    "odds": r"\frac{\operatorname{odds}_S(a,A,C,Z)}{\operatorname{odds}_S(a_0,A,C,Z)}=\exp\{f(a)\},\qquad \operatorname{odds}(p)=\frac{p}{1-p}.",
    "oof": r"\begin{aligned}q_i^{\mathrm{OOF}}&=T_{\mathcal D\setminus F(i)}(x_i),&&i\in\mathcal D,\\q_j^{\mathrm{val}}&=T_{\mathcal D}(x_j),&&j\in\mathcal E.\end{aligned}",
    "loss": r"\begin{aligned}\widetilde y_i&=(1-\lambda)y_i+\lambda q_i^{\mathrm{OOF}},\\\mathcal L(\Theta)&=\frac1N\sum_i\left[(1-\lambda)\ell(y_i,p_{S,i})+\lambda\ell(q_i^{\mathrm{OOF}},p_{S,i})\right]+\mathcal R(\Theta)\\&=\frac1N\sum_i\ell(\widetilde y_i,p_{S,i})+\mathcal R(\Theta),\\\ell(t,p)&=-t\log p-(1-t)\log(1-p).\end{aligned}",
    "penalty": r"\begin{aligned}\mathcal R&=\lambda_b\left(\lVert\gamma\rVert_2^2+\lVert\delta\rVert_2^2+\lVert b\rVert_2^2\right)+\lambda_g\left(\lVert\bar W\rVert_F^2+\lVert\bar c\rVert_2^2\right)+\frac{\kappa_r}{K}\sum_{k=1}^{K}\rho_k^2,\\\bar W_k&=W_k-\frac1K\sum_{\ell=1}^{K}W_\ell,\qquad\bar c_k=c_k-\frac1K\sum_{\ell=1}^{K}c_\ell,\qquad\lambda_b=10^{-4}.\end{aligned}",
    "filter": r"\mathcal E_{\mathrm{filtered}}=\mathcal E\setminus\left\{i\in\mathcal E:\exists j\in\mathcal D,\ \widehat\phi^{\mathrm{KING}}_{ij}\geq0.177\right\}.",
    "r2": r"R^2_{\mathrm N}=\frac{1-\exp\{2(\ell_0-\ell_p)/n\}}{1-\exp(2\ell_0/n)},\qquad\ell_p=\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].",
}


def read(path):
    return json.loads(Path(path).read_text())


def formula_audit():
    """Numerically check the displayed M0 equations against the implementation."""
    import torch
    from scipy.special import expit, softmax

    from decipehr.prs_distillation_bio import BioStudent
    from decipehr.prs_distillation_v3 import SignAwarePreprocessor

    rng = np.random.default_rng(20261027)
    n, k = 51, 5
    a = np.linspace(-3, 3, n)
    knots, a0 = np.array([-2, -.4, .65, 2]), .1
    age = SignAwarePreprocessor(None, a0, 1., knots)
    t, implementation_basis = age.age_transform(a)
    clipped = np.clip(a, knots[0], knots[-1])
    basis = np.column_stack([np.clip(clipped-knots[j], 0, knots[j+1]-knots[j])
                             - np.clip(a0-knots[j], 0, knots[j+1]-knots[j]) for j in range(3)])
    np.testing.assert_allclose(basis, implementation_basis, atol=1e-14, rtol=0)
    z, ancestry = rng.normal(size=(n, k)), rng.normal(size=(n, 10))
    nonage = np.column_stack([np.ones(n), ancestry, np.eye(3)[np.arange(n) % 3]])
    model = BioStudent(k, 10, 14, 3, variant="M0", shape=False, seed=17)
    with torch.no_grad():
        for name, p in model.named_parameters():
            values = rng.normal(0, .2, tuple(p.shape))
            if name == "age_slopes":
                values = np.abs(values)
            p.copy_(torch.as_tensor(values, dtype=torch.float64))
    x = {"z": z, "a": ancestry, "nonage": nonage, "t": t, "age_basis": basis,
         "hinges": np.zeros((n, k, 4))}
    output = model({name: torch.as_tensor(v, dtype=torch.float64) for name, v in x.items()})
    state = {name: v.detach().numpy() for name, v in model.state_dict().items()}
    w, c = state["w"]-state["w"].mean(0), state["c"]-state["c"].mean()
    pi = softmax(ancestry @ w.T+c, axis=1)
    genetic = (pi*np.exp(state["log_r"])*z).sum(1)
    f_age = basis @ state["age_slopes"]
    eta0 = nonage @ state["base"]+genetic
    eta = eta0+f_age
    np.testing.assert_allclose(output["logit"].detach(), eta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(output["probability"].detach(), expit(eta), atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.exp(eta-eta0), np.exp(f_age), atol=1e-12, rtol=0)
    assert np.diff(f_age).min() >= -1e-12
    penalty = .0001*(np.square(state["base"][1:]).sum()+np.square(state["age_slopes"]).sum())
    penalty += .001*(np.square(w).sum()+np.square(c).sum())+.01*np.square(state["log_r"]).mean()
    actual_penalty = float(model.penalty(.001, .01, 0., .001, .0001).detach())
    np.testing.assert_allclose(actual_penalty, penalty, atol=1e-12, rtol=0)
    y, q, lam = rng.integers(2, size=n), rng.uniform(size=n), .75
    mixed = (1-lam)*y+lam*q
    direct = torch.nn.functional.binary_cross_entropy_with_logits(output["logit"], torch.tensor(mixed)).item()
    weighted = ((1-lam)*(np.logaddexp(0, eta)-y*eta)+lam*(np.logaddexp(0, eta)-q*eta)).mean()
    np.testing.assert_allclose(direct, weighted, atol=1e-12, rtol=0)
    return {"status": "passed", "synthetic_profiles": n, "new_fits": 0,
            "checked": ["clipped_centered_age_basis", "centered_softmax_gate", "positive_prs_scales", "M0_logit",
                        "sigmoid_probability", "age_odds_ratio", "age_monotonicity", "regularization", "mixed_BCE_identity"],
            "max_logit_error": float(np.abs(output["logit"].detach().numpy()-eta).max())}


def collect():
    used = {}

    def record(path, expected=None):
        h = sha256_file(path)
        if expected is not None:
            assert h == expected, path
        used[str(path.relative_to(ROOT))] = h

    def table(base, relative, receipt="VERIFIED.json", key="output_sha256"):
        seal = base / receipt
        record(seal)
        path = base / relative
        record(path, read(seal)[key][relative])
        return pd.read_csv(path)

    m = table(LR, "report/metrics.csv")
    m = m.loc[m.cohort.isin(COHORTS) & m.arm.isin(ARMS)].copy()
    assert len(m) == 24 and not m.duplicated(["trait", "cohort", "arm"]).any()
    for _, part in m.groupby(["trait", "cohort"]):
        assert part.n.nunique() == part.events.nunique() == 1 and set(part.arm) == set(ARMS)
    gaps = table(LR, "report/paired_differences.csv")
    gaps = gaps.loc[gaps.cohort.isin(COHORTS) & gaps.reference.isin(ARMS) & gaps.arm.eq("M0_KD")].copy()
    assert len(gaps) == 72
    lookup = m.set_index(["trait", "cohort", "arm"])
    for row in gaps.itertuples():
        delta = lookup.loc[(row.trait, row.cohort, row.arm), row.metric]-lookup.loc[(row.trait, row.cohort, row.reference), row.metric]
        np.testing.assert_allclose(delta, row.delta, atol=1e-12, rtol=0)
    km = table(KIN, "report/metrics.csv")
    for row in km.loc[km.cohort.isin(COHORTS) & km.arm.isin(ARMS)].itertuples():
        np.testing.assert_allclose([getattr(row, c) for c in METRICS], lookup.loc[(row.trait, row.cohort, row.arm), METRICS].to_numpy(float), atol=1e-12, rtol=0)
    counts = table(KIN, "report/cohort_counts.csv")
    counts = counts.loc[counts.cohort.isin(COHORTS)].copy()
    for row in counts.itertuples():
        assert row.n == lookup.loc[(row.trait, row.cohort, "M0_KD"), "n"]
        assert row.events == lookup.loc[(row.trait, row.cohort, "M0_KD"), "events"]
    seal = read(BIO / "COMPLETE.json")
    record(BIO / "COMPLETE.json")
    record(BIO / "PROTOCOL.json", seal["upstream_receipt_sha256"][str((BIO / "PROTOCOL.json").relative_to(ROOT))])
    protocol = read(BIO / "PROTOCOL.json")
    for name, h in protocol["identity"]["code_sha256"].items():
        record(ROOT / name, h)
    source = ROOT / protocol["identity"]["config"]["source_run"]
    source_seal = read(source / "COMPLETE.json")
    record(source / "COMPLETE.json")
    for name in ["scope_audit.csv"]:
        record(BIO / name)
    scopes = pd.read_csv(BIO / "scope_audit.csv").set_index("trait")
    settings = []
    for trait in TRAITS:
        core = BIO / "core" / trait
        receipt = core / "PREDICTIONS_COMPLETE.json"
        record(receipt, seal["upstream_receipt_sha256"][str(receipt.relative_to(ROOT))])
        locked = read(receipt)
        assert not locked["smoke"] and not locked["evaluation_labels_read"]
        for name in ["SELECTION_LOCK.json", "models/M0_KD/seed17/metadata.json", "tuning_results.csv"]:
            record(core / name, locked["output_sha256"][name])
        selection = read(core / "SELECTION_LOCK.json")
        assert not selection["selection_uses_val"]
        choice = selection["choices"]["M0_KD"]
        meta = read(core / "models/M0_KD/seed17/metadata.json")
        assert meta["variant"] == "M0" and not meta["shape"]
        assert meta["training"]["best_step"] == choice["steps"]
        assert meta["training"]["selection_scope"] == "locked_steps"
        tuning = pd.read_csv(core / "tuning_results.csv")
        tuning = tuning.loc[tuning.candidate.str.startswith("M0_l")]
        assert len(tuning) == 45 and tuning.candidate.nunique() == 15
        selected = tuning.loc[tuning.candidate.eq(choice["candidate"])]
        np.testing.assert_allclose(selected.loss.mean(), choice["H_loss"], atol=1e-12)
        assert selected.steps.median() == choice["steps"]
        prep = meta["preprocessing"]
        settings.append({"trait": trait, "prs_n": meta["dimensions"]["k"], "train_n": int(scopes.loc[trait, "D_n"]),
                         "U_n": int(scopes.loc[trait, "U_n"]), "H_n": int(scopes.loc[trait, "H_n"]),
                         "lambda": choice["settings"]["lam"], "gate_ridge": protocol["priors"][trait]["common_gate_ridge"],
                         "kappa_r": choice["settings"]["kappa_r"], "steps": choice["steps"],
                         "reference_age_sd": prep["reference_age"], "age_boundaries_sd": prep["age_boundaries"]})
        assert prep["fixed_preprocessing"]["fit_n"] == scopes.loc[trait, "D_n"]
    contract_path = source / "core/T2D/teacher_contract.json"
    record(contract_path, source_seal["output_sha256"][str(contract_path.relative_to(source))])
    teacher = read(contract_path)
    assert teacher["model_version"] == "V3" and teacher["parameters"]["n_estimators"] == 16
    for name in ["pooled_metrics.csv", "paired_bootstrap_deltas.csv", "RESULTS.md", "run_record.json"]:
        record(OLD / name)
    history = pd.read_csv(OLD / "paired_bootstrap_deltas.csv")
    history = history.loc[history.metric.eq("auc") & history.challenger.isin(["no_age", "no_pcs"])].copy()
    assert len(history) == 6 and history.delta.lt(0).all()
    old_metrics = pd.read_csv(OLD / "pooled_metrics.csv").set_index(["disease", "variant"])
    for row in history.itertuples():
        np.testing.assert_allclose(old_metrics.loc[(row.disease, row.challenger), "auc"]-old_metrics.loc[(row.disease, row.reference), "auc"], row.delta, atol=1e-12)
    for path in [Path(__file__), ROOT / "src/decipehr/prs_distillation_v3.py", ROOT / "src/decipehr/prs_distillation/student.py",
                 ROOT / "src/decipehr/prs_distillation/preprocessing.py", ROOT / "scripts/report_prs_validation_kinship_filters.py"]:
        record(path)
    return m, gaps, counts, pd.DataFrame(settings), history, teacher, used


def build_page(lang, formulas, metrics, gaps, counts, settings, history):
    def t(zh, en):
        return zh if lang == "zh" else en

    def eq(key):
        return formulas[key].replace('class="formula"', f'class="formula" data-equation="{key}"', 1)

    def table(frame):
        return '<div class="scroll">'+frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")+'</div>'

    def annotation(key, title_zh, title_en, zh, en):
        return '<aside class="annotation" data-annotation="'+key+'"><strong>'+t(title_zh, title_en)+'</strong><p>'+t(zh, en)+'</p></aside>'

    labels = [t("动机", "Motivation"), t("核心思想", "Core idea"), t("模型设计", "Model"),
              t("蒸馏与损失", "Distillation"), t("训练设置", "Training"), t("验证结果", "Validation"), t("总结", "Summary")]
    ids = ["motivation", "idea", "model", "distillation", "training", "validation", "summary"]
    title = t("可解释的 PRS 蒸馏模型", "Interpretable PRS distillation")
    blocks = [f'<header><div class="eyebrow">M0_KD · METHOD NOTE · 2026-10-27</div><a class="language" href="methods_{"en" if lang == "zh" else "zh"}.html">'+t("English", "Chinese version")+'</a>',
              '<h1>'+title+'</h1><p class="lead">'+t("用祖源调节 PRS 的组合权重，用年龄调节整体风险；从 TabPFN 学习一个可逐项解释的预测模型。", "Ancestry adjusts the PRS mixture, age adjusts overall risk, and TabPFN supplies soft targets for an explicitly decomposable predictor.")+'</p>',
              '<div class="badges"><span>TabPFN 3 → M0_KD</span><span>'+t("固定 PRS · 可变系数 Logistic", "Fixed PRSs · varying-coefficient logistic model")+'</span><span>'+t("train 内开发 · val 评估", "Develop on train · assess on val")+'</span></div></header>',
              '<nav aria-label="Sections">'+''.join(f'<a href="#{i}">{n+1}. {s}</a>' for n, (i, s) in enumerate(zip(ids, labels, strict=True)))+'</nav>']

    blocks += ['<section id="motivation"><h2>01 · '+labels[0]+'</h2><p>'+t(
        "TabPFN 的预测性能较强，但其预测过程难以直接写成少量、可检查的风险项。我们希望保留有用的预测信息，同时明确回答：哪些 PRS 在起作用、祖源如何改变它们的权重、年龄如何改变整体风险。",
        "TabPFN predicts well, but its prediction process is difficult to express as a few inspectable risk terms. We seek to retain useful predictive information while exposing which PRSs contribute, how ancestry changes their weights, and how age changes overall risk.")+'</p><p>'+t(
        "历史消融支持 Age 与 Ancestry 信息是性能的重要来源：从 TabPFN score 的输入中移除年龄或 PC1–PC10 后，三个疾病的 AUROC 均下降。由此提出，以 Logistic 回归为骨架，将祖源调权与年龄风险调整显式化，再通过蒸馏学习参数。",
        "Earlier ablations support age and ancestry as important sources of predictive information: removing age or PC1–PC10 from the TabPFN score inputs reduced AUROC for all three endpoints. This motivates a logistic backbone with explicit ancestry-dependent weights and age-related risk adjustment, learned through distillation.")+'</p>',
        '<details><summary>'+t("查看动机的历史证据", "Historical evidence behind the motivation")+'</summary><p class="note">'+t(
        "这是一项历史 train+val 合并人群的嵌套五折 OOF 分析，使用旧 PRS 面板；最终 Logistic wrapper 始终保留年龄、性别和 PCs，只改变中间 TabPFN score 的输入。它提供设计线索，不能说明 Age/Ancestry 解释了教师的全部优势，也不与第 6 节的直接 val 分数混用。",
        "This earlier nested five-fold OOF analysis used the combined train+val population and an older PRS panel. The final logistic wrapper always retained age, sex and PCs; only the intermediate TabPFN score inputs changed. It motivates the design without attributing the teacher's entire advantage to these inputs. Its scores are separate from the direct validation scores in Section 6.")+'</p>']
    h = history.pivot(index="disease", columns="challenger", values="delta").loc[TRAITS].reset_index()
    h.columns = [t("疾病", "Endpoint"), t("移除 Age：ΔAUROC", "Remove age: ΔAUROC"), t("移除 PCs：ΔAUROC", "Remove PCs: ΔAUROC")]
    h.iloc[:, 0] = ["T2D", "CAD", t("女性乳腺癌", "Female BC")]
    blocks += [table(h), '<p class="note">'+t("差值 = 消融版本 − 完整版本；历史样本量：T2D/CAD 8,663，女性乳腺癌 4,936。", "Difference = ablated minus full model; historical N: 8,663 for T2D/CAD and 4,936 for female BC.")+' <a href="motivation_evidence.csv">CSV</a></p></details></section>']

    blocks += ['<section id="idea"><h2>02 · '+labels[1]+'</h2><div class="architecture" role="img" aria-label="'+t("祖源门控、PRS 组合、年龄基线和概率预测", "Ancestry gate, PRS mixture, age baseline and probability output")+'">',
        '<div class="flow-card ancestry"><b>PC1–PC10</b><span>'+t("连续祖源坐标 → Softmax gate", "Continuous ancestry → softmax gate")+'</span></div><div class="flow-arrow turn">→</div>',
        '<div class="flow-card genetic"><b>'+t("固定 PRS × 祖源权重 × 正尺度", "Fixed PRSs × ancestry weights × positive scales")+'</b><span>'+t("各项贡献相加，得到遗传项", "Sum the contributions into a genetic term")+'</span></div><div class="flow-arrow">+</div>',
        '<div class="flow-card baseline"><b>'+t("年龄单调项 + PCs/性别基线", "Monotone age term + PC/sex baseline")+'</b><span>'+t("在 log-odds 上相加，再经 sigmoid 得到概率", "Add on the log-odds scale, then apply sigmoid")+'</span></div></div>',
        '<p>'+t("普通多变量 Logistic 使用固定 PRS 系数。M0 将它替换为随祖源变化的系数，并以单调分段线性函数表示年龄。给定祖源时，遗传项对 PRS 仍是线性的；整体属于可变系数 Logistic 模型。",
            "Ordinary multivariable logistic regression uses fixed PRS coefficients. M0 replaces them with ancestry-dependent coefficients and represents age by a monotone piecewise-linear function. Conditional on ancestry, the genetic term remains linear in the PRSs; the full predictor is a varying-coefficient logistic model.")+'</p>'+eq("linear"),
        annotation("linear", "读公式：从固定系数到个体化系数", "Reading the equation: fixed versus individualized coefficients",
            "普通 Logistic 的 θₖ 对所有患者相同；M0 的 βₖ(Aᵢ) 随患者的祖源坐标改变。PRS 本身已经预先算好，训练学习的是组合方式。η 是 log-odds，sigmoid 将它转换成 0–1 之间的预测概率。",
            "Ordinary logistic regression uses the same θₖ for every patient. M0 uses βₖ(Aᵢ), which varies with the patient's ancestry coordinates. The PRSs are already computed; training learns how to combine them. η is log-odds, and sigmoid maps it to a probability between zero and one.")+'</section>']

    blocks += ['<section id="model"><h2>03 · '+labels[2]+'</h2><p>'+t(
        "每个疾病独立拟合。Z 为标准化后的固定 PRS 向量（T2D 26 个、CAD 7 个、女性乳腺癌 5 个）；A 为 PC1–PC10；a 为提供的 age_sd；C 为性别指示变量，女性乳腺癌中移除恒定性别项。PRS 和 PCs 的均值/标准差只从当前训练范围估计，SNP 权重保持固定。",
        "Fit each endpoint separately. Z contains standardized fixed PRSs (26 for T2D, 7 for CAD, 5 for female BC); A contains PC1–PC10; a is the provided age_sd; C contains sex indicators, omitted when constant in female BC. PRS and PC scaling uses only the current training context. SNP weights stay fixed.")+'</p><h3>'+t("祖源决定 PRS 的组合系数", "Ancestry determines the PRS coefficients")+'</h3>'+eq("gate"),
        '<p>'+t("π 是和为 1 的门控比例；r 是每个 PRS 学习到的正尺度，允许不同分数具有不同强度。因此实际系数 β=πr 为正，但其总和不要求等于 1。祖源同时进入基础风险项和 gate；年龄及 PRS 本身不作为 gate 的输入。",
        "The gate proportions π sum to one. Each learned positive scale r permits a different PRS effect magnitude. Effective coefficients β=πr are positive and need not sum to one. Ancestry enters both the baseline and the gate; age and the PRSs themselves are not gate inputs.")+'</p>',
        annotation("gate", "读门控：比例 × 尺度 = 实际系数", "Reading the gate: proportion × scale = effective coefficient",
            "πₖ(Aᵢ) 表示相对分配比例；rₖ 是该 PRS 跨患者共享的正尺度；二者相乘才是 βₖ(Aᵢ)。示意：π=0.4、r=2、Z=1 时，该项贡献为 +0.8 log-odds；Z=−1 时为 −0.8。正系数允许低于训练均值的 PRS 降低预测风险。数值仅用于解释。",
            "πₖ(Aᵢ) is the relative allocation, while rₖ is a positive scale shared across patients for that PRS. Their product is βₖ(Aᵢ). Illustration: π=0.4, r=2 and Z=1 contribute +0.8 log-odds; Z=−1 contributes −0.8. Positive coefficients therefore allow below-mean PRSs to lower predicted risk. These numbers are illustrative.")+eq("model"),
        '<details id="symbol-guide"><summary>'+t("符号速查：输入、门控与模型参数", "Symbol guide: inputs, gating and model parameters")+'</summary><dl class="symbol-guide">'+''.join(
            '<dt>'+escape(symbol)+'</dt><dd>'+t(zh, en)+'</dd>' for symbol, zh, en in [
                ("i / k / K", "i 为患者编号，k 为 PRS 编号，K 为该疾病使用的 PRS 数量。", "i indexes patients, k indexes PRSs, and K is the number of PRSs for the endpoint."),
                ("Zᵢₖ / Aᵢ", "Zᵢₖ 是第 k 个 PRS 的训练范围标准化值；Aᵢ 是 10 维连续 ancestry PCs。", "Zᵢₖ is the kth PRS standardized in the training context; Aᵢ contains 10 continuous ancestry PCs."),
                ("aᵢ / Cᵢ", "aᵢ 是 age_sd；Cᵢ 是性别指示变量。年龄和性别不进入祖源 gate。", "aᵢ is age_sd; Cᵢ contains sex indicators. Neither age nor sex enters the ancestry gate."),
                ("α / γ / δ", "α 为截距；γ 和 δ 分别为 PCs 与性别的基线系数。", "α is the intercept; γ and δ are baseline coefficients for PCs and sex, respectively."),
                ("W̄ / c̄", "祖源 gate 的系数矩阵与偏置，分别为 K×10 和 K 维；横线表示在 PRS 通道间居中。", "Gate coefficient matrix and offsets, of size K×10 and K; bars denote centering across PRS channels."),
                ("πₖ / rₖ / ρₖ", "πₖ 是门控比例；rₖ 是正尺度；ρₖ=log(rₖ) 是直接优化的无约束参数。", "πₖ is a gate proportion; rₖ is a positive scale; ρₖ=log(rₖ) is the unconstrained parameter optimized directly."),
                ("βₖ(Aᵢ)", "有效 PRS 系数。固定祖源和其他输入时，该 PRS 增加 1 个训练标准差，log-odds 增加 βₖ。", "Effective PRS coefficient. With ancestry and other inputs fixed, increasing that PRS by one training standard deviation increases log-odds by βₖ."),
                ("bⱼ / Bⱼ(a)", "bⱼ 是第 j 段学习到的非负年龄斜率；Bⱼ(a) 是由训练年龄边界定义、以中位年龄居中的固定基函数。", "bⱼ is the learned nonnegative slope for age segment j; Bⱼ(a) is a fixed basis defined by training boundaries and centered at the training median age."),
                ("ηᵢ / pS,i", "ηᵢ 是三部分相加的预测 log-odds；pS,i 是 Student 最终输出的疾病概率。", "ηᵢ is the predicted log-odds after summing the three components; pS,i is the student's output disease probability."),
            ])+'</dl></details>',
        '<h3>'+t("年龄调节整体风险", "Age adjusts overall risk")+'</h3>'+eq("age")+eq("odds"),
        '<p>'+t("a₀ 是训练人群的年龄中位数。固定其他输入后，高于 a₀ 的年龄不会降低预测风险，低于 a₀ 的年龄不会提高预测风险；允许局部平坦。年龄在 log-odds 上相加，等价于对总 odds 乘以 exp[f(a)]。当前 M0 没有年龄×PRS 交互项。",
        "a₀ is the training median age. Holding other inputs fixed, ages above a₀ cannot lower predicted risk and ages below a₀ cannot raise it; flat segments are allowed. Adding age on the log-odds scale multiplies overall odds by exp[f(a)]. Current M0 has no age-by-PRS interaction.")+'</p>',
        annotation("age", "读年龄乘子：作用在 odds 上", "Reading the age multiplier: it acts on odds",
            "odds=p/(1−p)。示意：参考年龄下 p=20%，对应 odds=0.25；若某年龄的 exp[f(a)]=2，则 odds 变为 0.5，概率为 0.5/(1+0.5)≈33.3%。这说明 odds 翻倍时，概率通过 sigmoid 换算。该数值示例不是拟合结果。",
            "odds=p/(1−p). Illustration: p=20% at the reference age gives odds=0.25. If exp[f(a)]=2 at another age, odds become 0.5 and probability becomes 0.5/(1+0.5)≈33.3%. Doubling odds translates to probability through sigmoid. This numerical example is not a fitted result."),
        '<details><summary>'+t("三段年龄基函数的精确定义", "Exact three-segment age basis")+'</summary>'+eq("basis")+'<p>'+t(
        "clip(x;l,u)=min(max(x,l),u)。分段边界来自训练年龄的最小值、1/3 分位数、2/3 分位数和最大值。每段斜率 bⱼ≥0；训练范围外使用恒定外推。这里的年龄单位始终是 age_sd，没有换算成年龄（岁）。调参阶段以 U 估计这些量，最终拟合时以完整 train 重新估计。",
        "clip(x;l,u)=min(max(x,l),u). Segment boundaries are the training minimum, 1/3 and 2/3 quantiles, and maximum. Slopes bⱼ are nonnegative, with constant extrapolation beyond training support. Age remains in age_sd units, not years. These quantities are fitted on U during tuning and on all train for the final model.")+'</p></details>',
        '<p class="note">'+t("解释对象是每项 βₖ(A)Zₖ 对预测 log-odds 的贡献。门控比例不是祖源混合比例，PRS 贡献也不是因果效应；相关 PRS 的单项权重需要结合稳定性理解。",
        "Interpret βₖ(A)Zₖ as a contribution to predicted log-odds. Gate proportions are not ancestry fractions, and PRS contributions are not causal effects; correlated PRS weights require stability checks.")+'</p></section>']

    blocks += ['<section id="distillation"><h2>04 · '+labels[3]+'</h2><div class="steps">',
        '<div><b>1</b><strong>'+t("生成训练软标签", "Generate training soft targets")+'</strong><p>'+t("在训练范围内按患者/已知亲缘组五折交叉拟合 Teacher。每个患者由未见过该折标签的 Teacher 产生概率。", "Cross-fit the teacher within the training context using five patient/known-kinship-group folds. Each probability comes from a context excluding that fold.")+'</p></div>',
        '<div><b>2</b><strong>'+t("混合真实标签与软标签", "Mix hard and soft labels")+'</strong><p>'+t("以 λ 控制 Teacher 权重；用训练内部 H 的真实结局选择 λ、正则化和训练步数。", "Use λ to weight teacher targets. Select λ, regularization and steps using true outcomes in the train-internal H split.")+'</p></div>',
        '<div><b>3</b><strong>'+t("全 train 拟合并冻结", "Refit on all train and freeze")+'</strong><p>'+t("按锁定设置训练 Student，再预测 validation。Teacher 的 validation 预测使用完整 train 作为 context。", "Fit the student with locked settings and predict validation. The teacher's validation context is the complete training set.")+'</p></div></div>'+eq("oof"),
        '<p class="note">'+t("D 是当前训练集、E 是 validation、F(i) 是患者 i 所属的训练折。调参时把公式中的 D 换成 U，U 的 Teacher 软标签也只在 U 内生成，H 标签不进入其 context。训练 OOF 软标签与正式 validation 评估用途不同。",
        "D denotes train, E validation, and F(i) the training fold of patient i. During tuning, replace D with U: U soft targets are generated exclusively within U, without H labels in context. Training OOF targets and official validation evaluation serve different purposes.")+'</p>',
        annotation("oof", "读流程：context、OOF、U 与 H", "Reading the workflow: context, OOF, U and H",
            "context 是提供给 TabPFN 3 的带标签参考患者。OOF（out-of-fold）表示当前患者所在折不在参考 context 中。U 用于拟合候选 Student，H 用于选择超参数；最终将 U 与 H 合并为完整 train 重拟合，validation 的标签只用于计算结果。",
            "Context is the labeled reference set supplied to TabPFN 3. OOF (out-of-fold) means the patient's fold is excluded from that context. U fits candidate students and H selects hyperparameters. The final refit combines U and H into all train; validation labels are used only to compute evaluation results."),
        '<h3>'+t("损失：混合目标的二元交叉熵", "Loss: binary cross-entropy with mixed targets")+'</h3>'+eq("loss"),
        '<p>'+t("实现使用数值稳定的 BCEWithLogits，并对整个训练批次取均值。λ=0 即同结构的 M0_SUP；λ>0 为 M0_KD，λ=1 为纯软标签训练。直接使用 Teacher 缓存概率，不再额外施加蒸馏温度变换。",
        "The implementation uses numerically stable BCEWithLogits averaged over the full training batch. λ=0 gives the same-architecture M0_SUP; λ>0 gives M0_KD, and λ=1 uses only soft targets. Cached teacher probabilities are used directly, without an additional distillation-temperature transform.")+'</p>',
        annotation("loss", "读损失：每个符号代表什么", "Reading the loss: what the symbols mean",
            "yᵢ∈{0,1} 是真实标签，qᵢᴼᴼᶠ 是 TabPFN 3 的折外概率，ỹᵢ 是混合训练目标；N 为当前拟合患者数，Θ 为全部可学习参数，ℛ 为正则化。示意：y=1、q=0.8、λ=0.75 时，ỹ=0.25×1+0.75×0.8=0.85；λ 越大，目标越接近 Teacher 的软概率。",
            "yᵢ∈{0,1} is the true label, qᵢᴼᴼᶠ the TabPFN 3 out-of-fold probability, and ỹᵢ the mixed target. N is the fitting sample size, Θ all learnable parameters, and ℛ the regularizer. Illustration: y=1, q=0.8 and λ=0.75 give ỹ=0.25×1+0.75×0.8=0.85. Larger λ moves the target closer to the teacher probability."),
        '<details><summary>'+t("完整正则项与优化约束", "Full regularization and optimization constraints")+'</summary>'+eq("penalty")+'<p>'+t(
        "λᵦ 控制基础协变量和年龄斜率；λg 控制居中后的 gate 参数；κᵣ 控制 log(r) 的平均平方。截距 α 不惩罚。Adam 每步更新后，将年龄斜率投影到非负区间。M0 的 PRS 变换为恒等映射，年龄×PRS 修饰固定为 0。",
        "λᵦ penalizes baseline covariates and age slopes; λg penalizes centered gate parameters; κᵣ penalizes the mean squared log(r). The intercept α is unpenalized. After each Adam update, age slopes are projected onto the nonnegative domain. M0 uses identity PRS transforms and fixes age-by-PRS modifiers to zero.")+'</p>'+annotation(
            "regularization", "正则化的直观含义", "What the regularization does",
            "较大的 λg 更强地收缩 gate，使各 PRS 的门控分配接近均匀；各 PRS 的实际系数仍取决于 r。较大的 κᵣ 将 log(r) 收缩到 0，也就是将 r 拉向 1。‖·‖² 表示参数平方和，F 表示矩阵的 Frobenius 范数。蒸馏权重 λ 与这些正则化强度各有独立作用。",
            "A larger λg shrinks the gate more strongly toward uniform allocations; effective PRS coefficients still depend on r. A larger κᵣ shrinks log(r) toward zero, hence r toward one. Squared norms sum squared entries; F denotes the matrix Frobenius norm. The distillation weight λ and these penalty strengths play separate roles.")+'</details></section>']

    blocks += ['<section id="training"><h2>05 · '+labels[4]+'</h2><p>'+t(
        "全部选择在 train 内完成。先按冻结分组五折取一折作为 H、其余为 U：T2D/CAD 为 U=4,649、H=1,162；女性乳腺癌为 U=2,658、H=650。预处理只在拟合范围内估计。",
        "All tuning occurs inside train. One fold of the frozen grouped five-fold split forms H and the remainder forms U: U=4,649/H=1,162 for T2D/CAD and U=2,658/H=650 for female BC. Preprocessing is fitted only within its training context.")+'</p>',
        '<ul><li>'+t("搜索 λ∈{0,0.25,0.5,0.75,1}，κᵣ∈{10⁻⁴,10⁻³,10⁻²}；每个组合使用 seeds 17/29/43。M0 每个疾病共 15 个组合、45 次调参拟合。",
        "Search λ∈{0,0.25,0.5,0.75,1} and κᵣ∈{10⁻⁴,10⁻³,10⁻²}, each with seeds 17/29/43: 15 M0 combinations and 45 tuning fits per endpoint.")+'</li><li>'+t(
        "按三个种子的 H 真实标签 log loss 均值选择；差异在 10⁻⁴ 内时优先较小 λ、较强 κᵣ。SUP 与 KD 分别选择，最终步数取选中组合三个种子最佳步数的中位数。",
        "Select by mean H true-label log loss across the three seeds. Within 10⁻⁴ of the best loss, prefer smaller λ, then stronger κᵣ. SUP and KD are selected separately; final steps are the median of the chosen candidate's three best step counts.")+'</li><li>'+t(
        "Adam，全批次，学习率 0.01，梯度范数上限 10。调参最多 2,000 steps，每 10 steps 评估 H，连续 200 steps 无改善则停止；改善阈值为 10⁻⁶。最终拟合使用下表锁定步数。",
        "Adam with full batches, learning rate 0.01 and gradient-norm clipping at 10. Tuning runs up to 2,000 steps, checks H every 10 steps, and stops after 200 steps without improvement exceeding 10⁻⁶. Final refits use the locked step counts below.")+'</li></ul>']
    b = settings[["trait", "train_n", "lambda", "gate_ridge", "kappa_r", "steps"]].copy()
    b.trait = ["T2D", "CAD", t("女性乳腺癌", "Female BC")]
    b.columns = [t("疾病", "Endpoint"), t("最终 batch size", "Final batch size"), "λ", "λg", "κᵣ", "Steps"]
    blocks += [table(b), annotation("training", "读训练设置：batch、step 与 seed", "Reading training settings: batch, step and seed",
        "这里采用 full batch：一个 step 会用全部当前拟合患者计算一次损失并执行一次 Adam 更新。因此 T2D 的最终 batch size=5,811、230 steps，表示 230 次全训练集参数更新。seed 控制随机初始化；主结果的 seed 17 事先固定，三种 seed 的平均 H loss 用于调参。",
        "Training uses full batches: one step computes the loss over all current fitting patients and performs one Adam update. T2D's final batch size of 5,811 and 230 steps therefore mean 230 full-training-set updates. A seed controls random initialization; primary seed 17 is fixed, while mean H loss over three seeds selects settings."), '<p class="note">'+t(
        "λg 沿用先前 v1 在 train 内从 {10⁻⁴,10⁻³,10⁻²} 选出的值，本轮 M0 没有重新联合搜索它。基础正则 λᵦ=10⁻⁴。主结果固定 seed 17，另两个种子检查稳定性。",
        "λg is inherited from the earlier v1 train-only search over {10⁻⁴,10⁻³,10⁻²}; it is not jointly re-searched for M0. Baseline regularization λᵦ=10⁻⁴. Primary results use the fixed seed 17; two additional seeds assess stability.")+'</p>',
        '<details><summary>'+t("初始化、Teacher 与普通 Logistic 设置", "Initialization, teacher and plain-logistic settings")+'</summary><p>'+t(
        "M0 gate 的 W 以 N(0,0.01²) 初始化，c=0；ρ=0（r=1），年龄斜率和其余基线系数从 0 开始，截距从混合目标均值的 logit 开始。当前主模型使用随机初始化。",
        "Initialize gate W from N(0,0.01²), c=0, ρ=0 (r=1), and age slopes and other baseline coefficients at zero. The intercept starts at the logit of the mean mixed target. The primary model uses random initialization.")+'</p><p>'+t(
        "Teacher：TabPFN 3，包版本 8.0.1，16 estimators，random_state=42；沿用固定 checkpoint 与历史特征处理。Teacher 自身 softmax_temperature=0.9，与上面的 Student 蒸馏目标不额外温度缩放是不同设置。Student 以 float64 优化。",
        "Teacher: TabPFN 3, package 8.0.1, 16 estimators and random_state=42, with the frozen checkpoint and historical feature pipeline. The teacher's native softmax_temperature=0.9 is separate from the absence of additional student-target temperature scaling. Student optimization uses float64.")+'</p><p>'+t(
        "普通 Logistic：同一组 PRS、age_sd、性别和 PCs，固定系数、线性年龄项、无交互、无惩罚的最大似然估计。使用 sklearn lbfgs、C=∞、max_iter=20,000、tol=10⁻⁹；全部拟合收敛。M0_SUP 与 KD 同结构，但各自按训练内部规则选择正则化与步数。",
        "Plain logistic: the same PRSs, age_sd, sex and PCs, with fixed coefficients, a linear age term, no interactions and unpenalized maximum likelihood. sklearn lbfgs uses C=∞, max_iter=20,000 and tol=10⁻⁹; all fits converged. M0_SUP shares the KD architecture but independently selects penalties and steps within train.")+'</p></details></section>']

    blocks += ['<section id="validation"><h2>06 · '+labels[5]+'</h2><p>'+t(
        "以下使用冻结的全 train 模型直接预测 validation，Student 主种子为 17。四组输入包含同一组 PRS、Age、Ancestry PCs 和适用的性别变量。M0_SUP 是必要对照：它帮助判断增益是否来自蒸馏，而非仅来自结构变化。",
        "Frozen full-train models predict validation directly; student primary seed is 17. All four models receive the same PRSs, age, ancestry PCs and applicable sex variables. M0_SUP is the essential control for separating the distillation contrast from architectural changes.")+'</p>',
        annotation("comparators", "四组对照分别回答什么", "What the four comparators show",
            "普通 Logistic：固定系数的直接回归基线。M0_SUP：相同 Student 结构，只使用真实标签（SUP=supervised）。M0_KD：相同结构加入软标签（KD=knowledge distillation）。TabPFN 3：生成软标签的 Teacher，也是直接预测的性能参照。SUP/KD 各自选择正则化和步数，因此对比的是两种训练流程。",
            "Plain logistic is the fixed-coefficient regression baseline. M0_SUP uses the student architecture with hard labels only (SUP=supervised). M0_KD adds soft targets (KD=knowledge distillation). TabPFN 3 supplies those targets and serves as the direct-prediction reference. SUP and KD independently select penalties and steps, so the contrast compares their training procedures."),
        '<h3>'+t("直接近亲的排除定义", "Definition of the direct-relative filter")+'</h3>'+eq("filter"),
        '<p>'+t("若 validation 中一人与该疾病任一 train 患者存在直接 KING kinship≥0.177 的配对，则仅移除这名 validation 患者。这个阈值通常筛出一级亲属或更近关系；它是基因型推断，不是已核实的家谱。训练患者、模型和阈值均保持不变，亲缘状态未确定者保留。",
        "Remove a validation patient if a direct pair with any disease-specific training patient has KING kinship≥0.177. This convention screens for first-degree or closer relationships; it is genotype-based inference, not a verified pedigree. Training patients, fitted models and the cutoff stay fixed, and unresolved patients remain included.")+' <a href="https://www.cog-genomics.org/plink/2.0/distance#make_king" target="_blank" rel="noopener">'+t("PLINK/KING 定义", "PLINK/KING definition")+'</a></p>']
    rows = []
    for trait in TRAITS:
        original = counts.loc[counts.trait.eq(trait) & counts.cohort.eq("original_val")].iloc[0]
        clean = counts.loc[counts.trait.eq(trait) & counts.cohort.eq("exclude_direct_0177")].iloc[0]
        rows.append(["T2D" if trait == "T2D" else "CAD" if trait == "cad" else t("女性乳腺癌", "Female BC"),
                     f'{original.n:,} / {original.events:,}', int(clean.removed), f'{clean.n:,} / {clean.events:,}', int(clean.unresolved_retained)])
    blocks += [table(pd.DataFrame(rows, columns=[t("疾病", "Endpoint"), t("Full val：人数/病例", "Full val: N/cases"), t("排除人数", "Removed"), t("筛选后：人数/病例", "Filtered: N/cases"), t("保留亲缘未知者", "Unresolved retained")])),
        '<p class="note" data-annotation="counts">'+t("表中 N/人数包含病例与非病例；病例数指 y=1。亲缘筛选同时作用于两类患者，且四个模型使用完全相同的保留名单。φ̂ᴷᴵᴺᴳ 表示两人的亲缘系数估计。", "N includes cases and noncases; cases are patients with y=1. Relatedness filtering applies to both, and all four models use the identical retained cohort. φ̂ᴷᴵᴺᴳ denotes the estimated kinship coefficient for a pair.")+'</p>',
        '<p class="note">'+t("本规则检查 train–val 的直接配对，不做共享亲缘组的传递性排除；保留下来的人群仍可能存在较远或未知亲缘。", "This rule checks direct train–val pairs, without transitive shared-group removal. More distant or unknown relatedness may remain.")+'</p>',
        '<div class="controls"><label for="metric">'+t("结果指标", "Result metric")+'</label><select id="metric"><option value="auroc">AUROC ↑</option><option value="log_loss">Log loss ↓</option><option value="nagelkerke_r2">Nagelkerke R² ↑</option><option value="brier">Brier ↓</option><option value="average_precision">Average precision ↑</option></select></div>',
        '<div class="result-grid"><article><h3>'+t("完整 validation", "Full validation")+'</h3><div id="full-table" class="metric-table"></div></article><article><h3>'+t("排除直接近亲后", "After removing direct relatives")+'</h3><div id="filtered-table" class="metric-table"></div></article></div>',
        '<p id="metric-help" class="note" aria-live="polite"></p>',
        '<p class="note" data-annotation="table-key">'+t("读表：绿色底色定位 M0_KD；细下划线标记该列最优点估计，不表示统计显著。↑ 表示越高越好，↓ 表示越低越好。左右两表使用不同患者人群，应在各表内部横向比较模型。", "Table key: the green row highlights M0_KD; a thin underline marks the best point estimate in a column, without implying statistical significance. ↑ means higher is better; ↓ means lower is better. The two tables contain different cohorts, so compare models within each table.")+'</p>',
        '<div class="takeaway">'+t("Full val 中 TabPFN 3 的 AUROC 点估计最高。排除直接近亲后，M0_KD 与 TabPFN 3 的 AUROC 差值为 T2D +0.0011、CAD −0.0015、女性乳腺癌 +0.0007，三者的配对区间均跨 0。人群变化后的接近表现不等于已证明等效。",
        "TabPFN 3 has the highest AUROC point estimate on full val. After filtering direct relatives, M0_KD minus TabPFN 3 AUROC is +0.0011 for T2D, −0.0015 for CAD and +0.0007 for female BC; all three paired intervals cross zero. Similar performance in the filtered population does not establish equivalence.")+'</div>',
        '<details id="paired-details"><summary>'+t("查看 M0_KD 相对各对照的差值与 95% 区间", "M0_KD versus controls: differences and 95% intervals")+'</summary>',
        '<div class="controls"><label for="cohort">'+t("人群", "Cohort")+'</label><select id="cohort"><option value="original_val">'+t("完整 validation", "Full validation")+'</option><option value="exclude_direct_0177">'+t("排除直接近亲后", "Direct relatives removed")+'</option></select><label for="contrast-metric">'+t("差值指标", "Difference metric")+'</label><select id="contrast-metric"><option value="auroc">ΔAUROC</option><option value="log_loss">Δlog loss</option><option value="nagelkerke_r2">ΔNagelkerke R²</option><option value="brier">ΔBrier</option></select></div>',
        '<p class="note" data-annotation="forest-key">'+t("读图：差值始终为 M0_KD − 对照。圆点是差值点估计，横线是配对 95% 置信区间，竖虚线表示差值 0；颜色对应不同对照。AUROC/R² 看正方向，log loss/Brier 看负方向。区间跨 0 表示当前估计尚不足以明确区分两者，不等于证明相同。", "Plot key: differences are always M0_KD minus the comparator. A dot is the point estimate, a horizontal line the paired 95% confidence interval, and the vertical dashed line zero; colors identify comparators. Positive AUROC/R² and negative log-loss/Brier differences favor M0_KD. An interval crossing zero does not clearly distinguish the models or establish equality.")+'</p>',
        '<div id="paired-plot" class="paired-plot"></div><div id="paired-table" class="scroll"></div><p class="note">'+t(
        "区间来自 1,000 次可用亲缘组配对 bootstrap，条件于固定预测，没有重新拟合或多重比较校正。亲缘未知者以患者级占位组处理。当前 val 已被多次查看，这些属于开发性、事后结果。",
        "Intervals use 1,000 paired available-kinship-group bootstrap resamples conditional on fixed predictions, without refitting or multiplicity adjustment. Unresolved patients use individual placeholder groups. This repeatedly examined val provides developmental, post-hoc evidence.")+'</p></details>',
        '<details><summary>'+t("指标含义与 R² 定义", "Metric interpretation and R² definition")+'</summary><p>'+t(
        "AUROC 衡量排序；log loss 与 Brier 衡量概率预测质量；average precision 辅助观察类别不平衡，但受病例比例影响。R² 使用 Nagelkerke 伪 R²：ℓₚ 是预测的总对数似然，ℓ₀ 是同一评估人群中仅用病例比例预测的总对数似然。",
        "AUROC measures ranking; log loss and Brier assess probabilistic prediction quality. Average precision complements them under class imbalance but depends on prevalence. R² denotes Nagelkerke pseudo-R²: ℓₚ is the total prediction log likelihood and ℓ₀ is the prevalence-only log likelihood on the same assessment cohort.")+'</p>'+eq("r2")+'<p class="note">'+t("n 为当前评估人群的患者数，ℓ₀ 的病例比例基准只用于计算评价指标。这里的 R² 不是遗传力、解释方差百分比，也不是 Student 对 Teacher 的拟合 R²。这里预测的是疾病状态，未定义未来固定年限的发病风险。", "n is the number of patients in the current assessment cohort; the prevalence baseline ℓ₀ is used only to calculate evaluation metrics. This R² is neither heritability, percentage variance explained, nor student–teacher fidelity R². These models predict disease status, not incident risk over a defined future time horizon.")+'</p></details></section>']

    blocks += ['<section id="summary"><h2>07 · '+labels[6]+'</h2><ul><li>'+t(
        "方法上：固定 PRS、祖源门控、正 PRS 尺度、单调年龄项组成一个可分解的 Logistic Student；蒸馏后独立预测，不需要 Teacher 或训练集 context。",
        "Method: fixed PRSs, ancestry gating, positive PRS scales and a monotone age term form a decomposable logistic student. After distillation, prediction needs neither the teacher nor training-set context.")+'</li><li>'+t(
        "性能上：相对普通 Logistic，T2D 和女性乳腺癌在两种 val 人群中均有更高 AUROC，配对名义区间不跨 0；CAD 的 AUROC 接近，主要改善概率损失。",
        "Performance: versus plain logistic, T2D and female BC have higher AUROC in both validation cohorts, with nominal paired intervals excluding zero. CAD AUROC is similar, with its main improvement in probability loss.")+'</li><li>'+t(
        "蒸馏证据上：与同结构 M0_SUP 比，T2D 的 AUROC 增益较明确；CAD 很小，女性乳腺癌的增益仍有不确定性。当前结果支持把 M0_KD 作为简洁的主要 Student，后续仍需独立人群验证。",
        "Distillation evidence: against same-architecture M0_SUP, the T2D AUROC gain is clearer; CAD gains are small and the female-BC gain remains uncertain. M0_KD is a concise primary student, with independent-cohort validation still needed.")+'</li></ul></section>',
        '<footer><p>'+t("资料下载：", "Downloads: ")+'<a href="metrics.csv">'+t("指标", "Metrics")+'</a> · <a href="paired_differences.csv">'+t("配对区间", "Paired intervals")+'</a> · <a href="training_settings.csv">'+t("训练设置", "Training settings")+'</a> · <a href="METHODS.json">'+t("方法与来源", "Methods and sources")+'</a></p><p>'+t("公式使用 LaTeX/KaTeX 本地渲染，字体内嵌，离线可读。页面仅含汇总结果。", "Equations use locally rendered LaTeX/KaTeX with embedded fonts for offline viewing. The page contains aggregate results only.")+'</p></footer>']
    config = {"traits": TRAITS, "arms": ARMS, "cohorts": COHORTS,
              "names": [t("普通 Logistic", "Plain logistic"), "M0_SUP", "M0_KD", "TabPFN 3"],
              "model_label": t("模型", "Model"), "endpoint_label": t("疾病", "Endpoint"),
              "trait_names": ["T2D", "CAD", t("女性乳腺癌", "Female BC")],
              "comparison_label": t("对照", "Comparator"), "delta_label": t("差值及 95% CI", "Difference and 95% CI"),
              "direction_up": t("正差值有利于 M0_KD", "Positive differences favor M0_KD"),
              "direction_down": t("负差值有利于 M0_KD", "Negative differences favor M0_KD"),
              "help": {"auroc": t("AUROC 越高越好。表内为点估计；配对区间见下方展开内容。", "Higher AUROC is better. Tables show point estimates; expand below for paired intervals."),
                       "log_loss": t("Log loss 越低越好；它也是训练内部选择使用的指标。", "Lower log loss is better; this is also the train-internal selection metric."),
                       "nagelkerke_r2": t("Nagelkerke 伪 R² 越高越好，衡量相对病例比例基线的似然改善。", "Higher Nagelkerke pseudo-R² means better likelihood relative to the prevalence-only baseline."),
                       "brier": t("Brier 越低越好，为概率预测与真实二元结局的均方差。", "Lower Brier is better: the mean squared difference between probabilities and binary outcomes."),
                       "average_precision": t("Average precision 越高越好；跨人群比较时同时考虑病例比例。", "Higher average precision is better; account for prevalence when comparing cohorts.")}}
    data = {"config": config, "metrics": json.loads(metrics.to_json(orient="records", double_precision=15)),
            "gaps": json.loads(gaps.to_json(orient="records", double_precision=15))}
    page = '<!doctype html><html lang="'+lang+'"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title>@@MATH@@<style>@@CSS@@</style></head><body><main>'+''.join(blocks)+'</main>'
    page += '<script id="report-data" type="application/json">'+json.dumps(data, ensure_ascii=False).replace("</", "<\\/")+'</script><script>@@JS@@</script></body></html>'
    return page


def main():
    os.umask(0o077)
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, gaps, counts, settings, history, teacher, used = collect()
    audit = formula_audit()
    math_css, rendered, math_meta = render_latex(list(EQUATIONS.values()))
    formulas = dict(zip(EQUATIONS, rendered, strict=True))
    used.update(math_meta["source_sha256"])
    css_path, js_path = TEMPLATE / "prs_distillation_methods.css", TEMPLATE / "prs_distillation_methods.js"
    css, js = css_path.read_text(), js_path.read_text()
    for path in [css_path, js_path]:
        used[str(path.relative_to(ROOT))] = sha256_file(path)
    for name, frame in [("metrics", metrics), ("paired_differences", gaps), ("cohort_counts", counts),
                        ("training_settings", settings), ("motivation_evidence", history)]:
        frame.to_csv(OUT / f"{name}.csv", index=False)
    write_json(OUT / "FORMULA_VERIFIED.json", audit)
    for lang in ["zh", "en"]:
        page = build_page(lang, formulas, metrics, gaps, counts, settings, history)
        page = page.replace("@@MATH@@", math_css).replace("@@CSS@@", css).replace("@@JS@@", js)
        check_latex(page, len(EQUATIONS))
        assert page.count('<section id=') == 7
        if lang == "en":
            assert not re.search(r"[\u4e00-\u9fff]", page)
        assert not re.search(r'<(?:script|link)[^>]+(?:src|href)="https?://', page)
        (OUT / f"methods_{lang}.html").write_text(page)
    write_json(OUT / "METHODS.json", {"model": "BIO M0_KD", "new_training": False,
        "displayed_models": ARMS, "assessments": COHORTS, "primary_seed": 17,
        "teacher": teacher, "teacher_display_name": "TabPFN 3", "equations_latex": EQUATIONS, "math_renderer": math_meta,
        "annotation_topics": ["symbols", "gate_and_scales", "age_odds", "OOF_and_context", "mixed_targets", "regularization",
                              "training_units", "comparator_roles", "cohort_counts", "table_styling", "paired_intervals"],
        "same_ancestry_inputs_in_all_models": True, "age_is_standardized_not_years": True,
        "age_action": "additive monotone log-odds term; multiplicative overall odds relative to training median age",
        "age_prs_interaction": False, "tuning": "U/H within train; grouped five-fold teacher targets within each training scope",
        "close_relative_filter": "Remove validation only for direct QC-valid KING>=0.177 pair with disease-specific train; retain unresolved",
        "motivation_evidence": "Historical combined train+val nested OOF with earlier selected PRS panel and full-covariate logistic wrapper; not the current direct-val benchmark",
        "bootstrap": "1000 paired available-kinship-group draws, fixed predictions, no multiplicity correction",
        "scientific_status": "developmental; repeatedly inspected val; no independent-cohort confirmation",
        "sources": {"KING": "https://www.cog-genomics.org/plink/2.0/distance#make_king", "TabPFN": "https://github.com/PriorLabs/TabPFN"},
        "input_sha256": used})
    (OUT / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(OUT / "VERIFIED.json", {"status": "content_and_formula_checks_passed", "new_fits": 0,
        "models": 4, "cohorts": 2, "metrics_rows": len(metrics), "paired_rows": len(gaps),
        "formulas_per_page": len(EQUATIONS), "browser_render_tested": False, "input_sha256": used,
        "output_sha256": {p.name: sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name != "VERIFIED.json" and p.suffix != ".zip"}})
    with zipfile.ZipFile(OUT / "methods_bilingual.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.is_file() and p.suffix in {".html", ".csv", ".json"}:
                z.write(p, p.name)
    print(f"Built {OUT}: 2 languages, 7 sections, {len(EQUATIONS)} equations, 4 models, 2 validation cohorts", flush=True)


if __name__ == "__main__":
    main()
