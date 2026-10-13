# External PRS 原定队列分析协议

本阶段使用**已冻结的 External PRS，加年龄、性别和连续 PC，训练下游整合器**。恢复既定 official training → development 患者划分，补齐同队列的强对照。External 指公开 PRS 权重来源；目前没有新增 external validation cohort。

本协议修正上一版将新增高杂合筛选、计算亲缘簇和训练 OOF 设为主要评估的偏离。恢复依据是原定研究范围和用户纠正，不依据哪套结果的 AUC 更高。旧分析及其预测完整保留为敏感性分析，不据此替换原定队列的结论。

执行契约为 `outputs/external_pgs/correction_20261011/CONTRACT.json`；冻结输入、患者集合、列顺序、代码、模型和运行配置。结果见[修正后的分析报告](NYU_Internal_Analysis_Results_20261011.md)。

## 队列与评估

| 疾病 | Training 人数 | Training 病例 | Development 人数 | Development 病例 |
|---|---:|---:|---:|---:|
| T2D | 5,811 | 1,701 | 2,852 | 847 |
| CAD | 5,811 | 2,313 | 2,852 | 1,072 |
| 女性乳腺癌 | 3,308 | 602 | 1,628 | 320 |

女性资格严格使用 `configs/disease_eligibility.yaml` 和 `--eligibility-config`。原始表型、标签、分数和患者纳入不因新模型表现而改变。原定患者级边界保持不重叠；亲缘关系尚未可靠核验，不能声称完成了家系隔离。

所有方法使用相同的疾病合格患者。标准化、样条、参数、单 PRS 和基线选择均只在 training 内拟合；使用同一套疾病内三折划分完成内层选择。Development 只接收预测，不参与本次选择，但它曾参与历史开发，因此全部结果仍属于内部开发证据。

本轮主要结果是 official development 指标。训练 OOF 如另行执行，须在原定 training 上使用方法共享的患者级五折，并与 development 分表报告。之前 QC/亲缘版本的 OOF 不能替代这套原定队列的 OOF，更不能把二者 AUC 差异解释成模型退步。

## 冻结的 External PRS

唯一完整分数输入为 `outputs/external_pgs/run_20261010/scoring_v2/scores.long.csv`，SHA256 为 `15baa761a9cb5f59c2361c902ea1c2c33ecb588bade57581327dd1a9b9452f0f`。权重锁为同目录 `weights.scored.lock.json`。本轮不重估 SNP 权重，不重做 GWAS，不重新筛选 SNP，也不改变 PRS 的计分方向。

| 疾病 | 完整面板及顺序 | 数量 |
|---|---|---:|
| T2D | PGS005353–PGS005377，随后 PGS002308 | 26 |
| CAD | PGS003725、PGS004696、PGS012558 | 3 |
| 女性乳腺癌 | PGS005378、PGS004869、PGS018538 | 3 |

完整教师保留原始分数列，协变量为提供的 `age_sd`、`sex`、`PC1_sd`–`PC10_sd`；不添加 soft ancestry，不对完整面板引入 PC 残差化。缩放只使用当前训练上下文。编码后的特征数为 40、17、14；前两项包含 sex 的多列分类编码，女性 BC 的恒定 sex 被移除。

TabPFN 使用原来的 v3 checkpoint、16 estimators、seed 42，以及 L40S / 8 线程运行配置。正式对照前，须逐患者复现原完整教师的保存概率；仅 AUC 接近不算精确复现。

## 对照与祖源调权

| 对照 | 输入与选择规则 |
|---|---|
| 完整多 PRS TabPFN | 原定 26／3／3 个分数加年龄、性别和连续 PC |
| 最佳单 PRS TabPFN | 在 training 内选分数；保留完全相同的协变量 |
| 协变量 TabPFN | 年龄、性别、PC；用于量化遗传增量 |
| Logistic 与 CatBoost | 完整同输入；C 或深度等仅按 training 内层表现选择 |
| 灵活基础项加固定 PRS 系数 | 年龄和 PC 的加性样条，PRS 线性整合 |
| 灵活基础项加可变 PRS 系数 | 同一基础项，加 PRS×PC 并对交互项收缩 |
| 内层选定强基线 | 最佳单 PRS Logistic、完整 Logistic、灵活固定系数、CatBoost 中选择 |
| 无 PC 与 hard label | 完整 PRS 不变，单独改变祖源表示 |

T2D 的单 PRS 搜索同时包含 26 个原始分数和 5 个按公开系数组合的 D-PRISM metascores；后者的缩放在每个训练上下文内重新计算。CAD 和 BC 各有 3 个候选。完整多 PRS 教师仍保持原面板。

DiscoDivas 适配为独立的匹配面板：T2D 使用 5 个 metascores；CAD 使用 PGS003725、PGS012558；BC 使用全部 3 个分数。匹配 TabPFN、固定整合和 Disco 共同使用冻结的 1000G refPC 与同一训练拟合的分数协调流程。该面板与完整教师分别报告。

Disco 的几何插值系数沿用既有已核验产物；基础项允许非线性，线性或样条风险链接在 training 内选择。这个配置与旧报告的 Disco 对照存在方法差异，应使用本轮预测重新评估，不能沿用旧差值。

## 统计与交付

主要比较为三疾病 × 两项：完整 TabPFN 对内层选定强基线，匹配 TabPFN 对内层选定 Disco 风险链接。使用 2,000 次患者配对 bootstrap，六项 AUC 检验统一 Holm 校正。区间条件于已保存的 development 预测，不覆盖历史选择、重训练变异和未核验亲缘关系带来的全部不确定性。

同时报告最佳单 PRS、协变量、固定／可变系数、无 PC／hard label 的配对差异，以及 log loss、Brier、校准和祖源组人数。显著性与遗传增量、祖源调权增量、蒸馏增量分别对应，不能互相替代。

代码入口为 `scripts/correct_external_prs_analysis.py`：先 `--phase audit`，再 `--phase anchor` 验证原教师，最后 `--phase fit`。所有阶段都要求 eligibility 配置。报告入口为 `scripts/report_external_prs_correction.py`。新运行目录独立保存预测、训练内选择、预处理、完成凭证和输入输出哈希；旧结果不覆盖。

蒸馏以这些 External PRS 为输入，由 ancestry 调节组合权重，方案见[祖源调权蒸馏设计](Ancestry_Gated_PRS_Distillation_Design_20261011.md)。软标签学生训练不属于本轮已完成内容。
