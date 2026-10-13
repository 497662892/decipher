# T2D 相同 D-PRISM 来源的整合方法比较

本分析比较三条使用相同 25 个外部 D-PRISM 成分的预测流程，区分原始分数来源与中间特征数量。D-PRISM→五个 metascores→Disco 已使用全部 25 个成分；这项补充不需要重新学习五套祖源模型或修改官方 Disco 插值算法。

状态：五折、development 预测及核验均已完成。在原定 development 的 2,852 人中，直接整合 25 成分的 TabPFN AUC 为 0.763921，Disco 为 0.752842，差值 +0.011080。该总体开发增益不代表每个祖源组均获益，SAS-like 小组的方向相反。

运行前方案已冻结在 `prepared/PROTOCOL.md`。既有 26 分数主教师及已完成的修正报告继续按各自原配置解释。

## 比较对象

| 流程 | 原始分数来源 | 整合方式 | PRS 特征数量 |
|---|---|---|---:|
| 直接 TabPFN | PGS005353–PGS005377 | 25 个成分共同进入 TabPFN | 25 |
| metascore TabPFN | 同样 25 个成分 | 按 D-PRISM 公开系数形成五个 metascores，再进入 TabPFN | 5 |
| D-PRISM 加 Disco | 同样 25 个成分 | 五个 metascores 经官方 Disco 几何插值，再拟合疾病概率链接 | 插值前 5，链接时 1 |

三条流程均使用年龄、性别和相同的 1000G refPC1–refPC10。PGS002308 不属于 D-PRISM 的 25 个成分，本项分析在三条流程中均排除该分数。

三个配对问题分别为：直接 25 成分 TabPFN 对 Disco 的同来源比较；直接 25 成分对五个 metascores 的 TabPFN 表示比较；五个 metascores 上 TabPFN 对 Disco 的相同中间特征比较。各自的差异不能直接解释为祖源调权的独立机制贡献。

## Development 结果

全部方法使用相同的 2,852 人，其中 847 个 T2D 病例。Disco 的风险链接按 training 内层 OOF 选中样条版本。

| 流程 | AUC | Log loss | Brier | 校准斜率 |
|---|---:|---:|---:|---:|
| 直接 25 成分 TabPFN | 0.763921 | 0.513367 | 0.169905 | 0.983051 |
| 五 metascores TabPFN | 0.756701 | 0.522231 | 0.172584 | 0.945882 |
| D-PRISM 加 Disco | 0.752842 | 0.524190 | 0.173503 | 1.070557 |

| 配对比较 | ΔAUC | 普通 95% 区间 | 三项 Bonferroni 同时区间 |
|---|---:|---|---|
| 直接 25 成分 TabPFN − Disco | +0.011080 | [0.004485, 0.018260] | [0.003153, 0.019798] |
| 直接 25 成分 TabPFN − 五 metascores TabPFN | +0.007220 | [0.004048, 0.010366] | [0.003298, 0.011038] |
| 五 metascores TabPFN − Disco | +0.003859 | [-0.002047, 0.009962] | [-0.003163, 0.011335] |

本队列支持让 TabPFN 直接整合 25 个成分；在相同五个 metascores 上，TabPFN 对 Disco 的差异仍不确定。结果支持的是这套整合流程的开发表现，不足以把优势归因于“祖源调权已被证明有效”，也不能称为超过原论文全部资产构成的 Disco 模型。

原 26 分数主教师的 AUC 仍为 0.764329。它使用提供的临床 PC 且不做本次 PRS 的参考 PC 残差化；与本项 0.763921 的差异同时涉及分数、PC 和预处理，不能当作 PGS002308 的单因素贡献。

## 祖源分组结果

下表为直接 25 成分 TabPFN 相对 Disco 的探索性分组比较。遗传相似性标签不等同于精确祖源比例。

| 分组 | 人数 | 病例 | ΔAUC | 普通 95% 区间 |
|---|---:|---:|---:|---|
| AFR-like | 344 | 139 | +0.008107 | [-0.016187, 0.031669] |
| AMBIGUOUS | 342 | 119 | +0.025813 | [0.001319, 0.054264] |
| AMR-like | 253 | 85 | +0.000210 | [-0.015968, 0.019328] |
| EAS-like | 161 | 56 | +0.037075 | [-0.011399, 0.089800] |
| EUR-like | 1,650 | 393 | +0.012208 | [0.003858, 0.020901] |
| SAS-like | 102 | 55 | -0.046809 | [-0.087041, -0.008511] |

SAS-like 的 TabPFN AUC 为 0.704836，Disco 为 0.751644。这一反向结果需保留并在新数据中检验；不能用总体优势宣称全部祖源组均受益。分组区间未控制全部探索比较。

## Training 五折结果

共享五折 pooled OOF 使用原定 5,811 人、1,701 个病例；这是 training 内的交叉验证，与上一节完整 training 拟合后预测 development 的定义不同。

| 流程 | Pooled OOF AUC | Log loss | Brier |
|---|---:|---:|---:|
| 直接 25 成分 TabPFN | 0.743794 | 0.521503 | 0.174790 |
| 五 metascores TabPFN | 0.739297 | 0.526538 | 0.176569 |
| D-PRISM 加 Disco | 0.734632 | 0.529959 | 0.177868 |

直接 25 成分 TabPFN 对 Disco 的 OOF ΔAUC 为 +0.009162，普通 95% 区间 [0.004699, 0.013785]，三项同时区间 [0.003632, 0.014755]。区间条件于保存的 OOF 预测，不涵盖外层训练集重叠带来的全部重训练不确定性。

## D-PRISM 与二阶段整合

D-PRISM 是 Diabetes Polygenic Risk Scores in Multiple ancestries 国际合作项目。本研究使用其公开的 T2D 分数和组合系数。[作者说明](https://github.com/ahuertach/D-PRISM-T2D-PRS)

令 k 为五个目标祖源之一，j 为该套模型中的五个来源成分之一，Z 为训练内标准化的分数。公开组合为 M_ik = Σ_j c_kj Z_ikj。忽略后续线性预处理时，Disco 的 D_i = Σ_k a_k(PC_i) M_ik = Σ_k Σ_j a_k(PC_i)c_kj Z_ikj。因此，25→5→1 是对全部 25 个成分的受约束整合，五维并不意味着只使用五个原始成分。

本次复用冻结的公开 c 系数和官方 R 计算的 a 系数。当前五个参考中心是共同 PC 空间中的 1000G 代理中心，属于本项目适配，不称为原论文完整模型复现。[Disco 官方实现](https://github.com/YunfengRuan/DiscoDivas)

## 队列与预处理

使用原定 T2D training 5,811 人、1,701 个病例；development 2,852 人、847 个病例。患者集合、表型、外部 SNP 权重和原评分文件与 `correction_20261011` 一致。

五个 metascores 及 Disco 沿用已有训练内标准化、PC 残差化与缩放。直接 25 成分分支对每个成分执行同样的训练内 PC 线性投影去除，并以训练残差标准差缩放。所有插补、缩放、PC 回归、模型拟合和链接选择只接触当前训练范围；查询标签在预处理入口移除。

线性投影与公开线性组合可交换。本次逐患者核验：从 25 个校正后的成分重新构造五个 metascores，并展开官方 Disco 组合，须与已有实现一致，数值容差为 1e-10。直接分支保留 25 个 PRS 特征；两条 TabPFN 分支的协变量设计矩阵须一致。

## 训练和评价

使用原定 training 患者生成一份共享的患者级五折划分，各方法在相同外层折上产生预测，保存逐患者 OOF 并计算 pooled OOF 指标。划分沿用现有共同疾病与性别分层函数，患者不得跨折。该五折分析单独记录，不替换原 training→development 的性能定义。

Disco 的线性或样条风险链接与正则化参数沿用既有候选范围，在每个拟合范围内通过三折内层 OOF 选择；外层查询折和 development 均只预测。两条 TabPFN 分支固定 v3 checkpoint、16 estimators、seed 42、L40S 和 8 线程。

最后在全部 5,811 名 training 患者上拟合，在原 2,852 名 development 患者上评价。已有的 metascore TabPFN、Disco 线性链接、样条链接及训练内选定链接，必须逐概率重现已有修正结果，才接纳新增 25 成分结果。

报告 AUC、log loss、Brier、校准与祖源组表现。三个 AUC 配对比较使用 2,000 次按结局分层的患者 bootstrap，报告普通 95% 区间及三项 Bonferroni 同时区间。训练 OOF 与 development 分开报告；祖源分组为探索性分析。

development 此前已经使用，本次属于开发证据；患者 bootstrap 条件于保存的预测，不覆盖重复开发、重训练或未核验的家系相关性。外部队列验证与软标签蒸馏不属于本项结果。

## 执行与产物

入口：`scripts/evaluate_dprism_source_comparison.py`，依次执行 `--phase audit`、六个 `--phase fit --partition`、`--phase report`，均传入 `--eligibility-config configs/disease_eligibility.yaml`。

输出根目录：`outputs/external_pgs/dprism25_comparison_20261011/`。`prepared/manifest.json` 冻结输入、代码和五折；`partitions/` 保存五折与 development 预测；`report/` 保存指标、配对区间、代数等价核验、旧结果复现和拟合边界审计。原有结果文件保持其冻结版本。

六个拟合任务 `28365480_0`–`28365480_5` 和报告任务 `28365639` 均以退出码 0 完成。已有四项 development 配置（五 metascores TabPFN、Disco 线性/样条及选定链接）均逐患者、逐概率精确复现。

覆盖所有外层折及 development 的 metascore 重构最大绝对误差为 1.24×10⁻¹⁴，Disco 展开最大误差为 4.53×10⁻¹⁵。直接分支编码后为 39 列，即 25 个 PRS、年龄、3 列性别编码和 10 个 refPC。原始外部评分、权重和历史交付哈希均通过核验。

验证：`198 passed, 2 skipped`，Ruff 与 sbatch shell 语法检查通过。跳过的是默认关闭的真实 PLINK/PGS Catalog 与官方 R 集成测试；本项复用已冻结的分数和官方插值产物。五折划分中的罕见联合疾病/性别层仅一人，五个 T2D 查询折均含病例及对照，未因该提示删除患者。
