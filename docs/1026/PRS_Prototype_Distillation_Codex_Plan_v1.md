# 固定 PRS 原型的可解释 TabPFN 蒸馏：Codex 执行方案 v1

日期：2026-10-26  
交付性质：研究与实现规格；本文件不代表代码已在真实数据上运行，也不含新实验结果。

## 0. 给 Codex 的任务摘要

在现有 PRS 项目中实现一个小型、结构固定的可解释学生模型。每个已有 PRS 就是一个原型，不学习新的原型，不改变 SNP/GWAS 权重。

唯一主模型为：

\[
\eta_i=b(a_i,A_i,C_i)+s(a_i)\sum_{k=1}^{K}\pi_k(A_i)Z_{ik},\qquad p_i=\sigma(\eta_i),
\]

\[
\pi(A)=\operatorname{softmax}(WA+c),\qquad s(a)=\exp\{u(a)\}.
\]

其中，\(Z\) 是固定来源且经过已声明校正的 PRS；\(A\) 是连续 ancestry 坐标；\(a\) 是年龄；\(C\) 只包含项目原有的基础协变量，如 sex。蒸馏通过 TabPFN 的折外预测概率完成，不使用 SPEX、SHAP、隐藏层或 attention。

先完成 **Phase A：核心蒸馏试验**。只有需要与项目历史正式指标比较时，才运行 **Phase B：既有 Stage-2 协议兼容评估**。Phase B 增加的是评估工作，不改变学生模型。

### 0.1 不可改变的研究约束

- 每个已有 PRS 是固定原型；不加入 \(r_k\)、\(v_{kj}\)、新 prototype、MLP gating、可学习 ancestry center 或距离度量。
- Gating 只接收 ancestry；不接收 age、sex、PRS 数值或结局。
- 所有 PRS 共享一个年龄缩放函数；不增加 PRS-specific 或 ancestry-specific 的年龄曲线。
- 保留基础年龄、ancestry 和已存在协变量效应；年龄不能只通过 PRS 起作用。
- 不重建 PRS、不重训 GWAS、不做 SNP selection；不进行新的特征交互发现。
- 主实验中 Teacher 与 Student 使用相同的信息集合和同一 PRS panel。
- 不使用测试/official-validation 标签进行预处理、Teacher context、训练、调参、早停或校准。
- 不为提高 AUC 自动改成更复杂模型；失败也是有效结果，先诊断再提出扩展。
- 不把 ancestry 当作 race/ethnicity 的同义词；不把模型权重解释为 ancestry proportion 或遗传风险的因果分摊。
- 不上传患者级遗传/临床数据到外部推理服务。使用项目已有、合规的本地 TabPFN 环境。

### 0.2 要回答的三个问题

1. 同一个受约束学生结构，用 Teacher 软标签训练，是否优于只用真实标签训练？
2. 学生能否保留 Teacher 相对于 metadata-only 模型的 PRS 增量预测价值？
3. \(\pi_k(A)\) 与 \(s(a)\) 是否足够稳定，支持描述性的模型解释？

不预设蒸馏必然提高性能，不预设不同 ancestry 必须选择不同 PRS，也不预设年龄缩放一定递增或递减。

---

## 1. 执行顺序与仓库勘察

先读取仓库中的 `AGENTS.md`、README、环境文件、现有数据加载器、Teacher 配置、split manifest 和评估代码。尽量增量实现，不重写现有 pipeline，不覆盖已有结果。

在运行真实训练前输出 `discovery_report.md`，至少记录：

| 项目 | 必须确定的内容 |
|---|---|
| 数据与结局 | 当前 cohort、trait、label 定义、索引时点、纳入/排除规则 |
| 输入 | PRS 列及来源；age 单位；ancestry 编码；sex 等既有协变量 |
| 预处理 | PRS 校正/标准化来源、拟合人群、是否使用结局或 holdout |
| 划分 | official_train / official_validation / blind_holdout 的真实路径与 ID |
| 关联性 | patient ID、重复记录、family/kinship group 是否可用 |
| Teacher | package 版本、模型版本、checkpoint、estimators、概率平衡设置 |
| 评估 | 当前是否必须经过 Stage 2；score 列实际是 probability、logit 还是其他值 |
| 可复现性 | git commit、环境、硬件、现有缓存 provenance |

当前项目资料提示已有 decipEHR 工作流，涉及 CAD、T2D、女性 BC、age/sex 和 PC1–PC10，以及 official-train 内五折 OOF 和固定 Stage-2 评估。这些是需要沿用/核对的背景，不允许把历史样本量、PRS 个数或路径硬编码为新数据的事实。[P1]

特别检查：历史 PRS panel 中也存在按不同构建方法区分的分数。**不能把 LDpred2、PRS-CS 等方法名自动重命名成某种 ancestry 的 PRS。** 本次已有 PRS 的实际来源以 manifest 为准。固定原型可以是 ancestry-derived PRS，也可以是其他固定来源的 PRS，但报告必须如实命名。

若当前数据是新 cohort：保留新项目已有划分；只有确实没有既有划分时，才建立一次固定的开发/验证/测试划分，并记录为新的评估协议。不能偷偷合并历史 official_train 和 official_validation。

若缺少真实路径、列映射、checkpoint 权限或 PRS 校正来源：继续完成代码、测试、配置校验和 blocker 报告；不要生成虚构患者数据充当真实实验结果，也不要无限搜索不相关目录。

---

## 2. 数据接口与科学含义

### 2.1 一行一个患者、一个结局

建议用 CSV 或 Parquet，或适配现有加载器。多疾病分别训练，不在 v1 做 multi-task。

| 字段 | 要求 |
|---|---|
| `participant_id` | 不为空；同一 trait 中唯一；不能作为模型输入 |
| `split` | 既有 official_train / official_validation / blind_holdout |
| `y` | 明确的 0/1；未知不是 0；blind 标签不要求存在 |
| `age` | 原始年龄或可逆的已标准化年龄；记录单位与测量时点 |
| `PC1...PCd` | 现有连续 ancestry coordinates，默认复用已有 PC1–PC10 |
| `PRS_1...PRS_K` | manifest 明确列出的同一 trait 的固定 PRS |
| `sex` / 其他 C | 只沿用既有有效协变量；常量列移除并记录 |
| `family_id` | 可选，但已有时必须用于分组划分和重采样 |
| `ancestry_group` | 可用于分层汇报，不自动作为 gating 输入 |

不要重新定义 BC 的 eligible-female cohort；使用仓库已有规则。不要把缺失的结局、eligibility 或 ancestry 填成正常类别。

如果结局实际是有删失的 incident event / time-to-event，不得把本方案直接包装成生存预测。输出任务类型不匹配的 blocker；本规格只针对已经定义好的二分类结局。

若 cohort 为病例对照或经过富集采样，本方案的概率与校准只对应当前评价分布；没有适当的目标人群发生率与抽样校正，不能把它称为一般人群的绝对发病风险。

### 2.2 PRS manifest

每个 PRS 至少有以下元数据：

```text
trait, column, display_name, source_gwas, source_ancestry,
construction_method, risk_direction, correction_status,
correction_fit_scope, correction_artifact, scale_description
```

风险方向须来自分数定义或仅在允许的训练范围内确认；不得看测试集后翻转某个 PRS。不能因为训练样本相关性偶然为负，就自动倒转已定义的遗传风险方向。

每个 trait 的 PRS 列表在正式运行前固定。不要依据 official-validation 表现删除 PRS。

### 2.3 预处理原则

1. 已有经外部参考或合规训练人群校正的 PRS：复用现有定义与实现，不擅自追加第二套 ancestry residualization。
2. 若校正需在目标数据上估计：每个 fit scope 单独拟合，应用于对应 query scope；不能利用评估队列的结局或分布来重估尺度。
3. 已有标签无关、预先冻结的遗传 PCA/PRS 生成过程可沿用；必须记录是否使用过 unlabeled holdout genotype，避免把 transductive 特征构建误写成完全 inductive。
4. age/PC 的新中心化、标准化、spline knots 和缺失值参数，只使用当前拟合范围；所有参数保存。
5. 输入 PRS 尺度不可随试验臂任意变化，否则 softmax 权重不可比较。
6. 第一版不增加 rank-normalization、winsorization 或 data augmentation。若现有流程已经有这些操作，保留并记录。

**PRS 校正尚不明确时，不自动发明一种“ancestry 校正”作为正式方案。** 可以完成代码与纯软件测试，但真实训练应标记 `blocked_missing_score_provenance`，直到找到现有校正产物或经明确指定的替代流程。

### 2.4 缺失、常量和重复

复用项目已批准的缺失处理。没有既有规则时，默认严格校验 age/ancestry/PRS 完整性并报告，不能静默逐模型删除不同患者。

临时软件试跑允许显式开启训练集 median imputation，但必须标为 sensitivity，不自动成为正式主分析。对校正后 PRS 用 0 填充只在“0 确实代表参考均值”时才有意义。

在每个训练范围检查常量 PRS 与完全重复 PRS。完全重复列不应被解释成两个可辨识专家：记录并按预先固定的列顺序保留一个，或在报告中明确将其视为不可区分组。不能根据验证表现选择保留哪一个。近似重复列不自动删除，报告相关矩阵与权重不稳定性。

---

## 3. 精确学生结构

### 3.1 基础项

\[
b(a,A,C)=\alpha+\theta_b^\top\phi(a)+\gamma^\top A+\delta^\top C.
\]

age 使用低维 spline；ancestry 与协变量先采用线性主效应。不同试验臂的基础项完全相同。

### 3.2 年龄函数

\[
u(a)=u_0+\theta_s^\top\phi(a),\qquad s(a)=e^{u(a)}.
\]

默认实现：

```python
SplineTransformer(
    n_knots=3,
    degree=2,
    knots="quantile",
    include_bias=False,
    extrapolation="constant",
)
```

这产生 3 个年龄基函数；在当前 fit scope 上对输出列中心化，并保存中心。`include_bias=False` 避免 spline 隐含常量与显式截距重复。[S3]

这是 v1 预先设定的低复杂度参数化，不代表最佳函数族。若当前年龄取值过少导致 knots 重复，使用显式记录的线性年龄基函数；不能静默改变。

`u0` 必须可学习；**不得固定 s(reference_age)=1 后又不给其他全局效应尺度。** 展示时另算：

\[
s_{rel}(a)=s(a)/s(a_0),
\]

其中 \(a_0\) 为当前训练范围的中位年龄，保存后不根据评估人群改变。

超出训练年龄范围时采用已固定的 constant extrapolation，并输出 out-of-support 标记，不声称该区间效应可靠。

### 3.3 Ancestry softmax

\[
g_k(A)=w_k^\top A+c_k,\qquad
\pi_k(A)=\frac{e^{g_k(A)}}{\sum_l e^{g_l(A)}}.
\]

温度固定为 1。不要加 entropy、one-hot、top-k 或必须“多样化”的约束。

softmax 对所有 logits 的共同平移不变。实现可训练完整 W/c，但在 forward、正则化和导出时使用去除跨 k 均值的 W/c，避免把共同平移误解成有效参数。对于 K=1，显式返回权重 1，不构造无效 gating 参数。

gating input 只能是 A；原始 ancestry PC 的轴不是祖源百分比，不可输出“60% EUR ancestry”这样的解释。

### 3.4 输出与结构约束

\[
G_i=\sum_k\pi_{ik}Z_{ik},\quad
B_{ik}=s_i\pi_{ik},\quad
C_{ik}=B_{ik}Z_{ik},\quad
\eta_i=b_i+\sum_k C_{ik}.
\]

模型必须返回 `logit, probability, baseline_logit, genetic_score, age_scale, weights, effective_coefficients, contributions`。

必须通过以下数值断言：

\[
\sum_k\pi_{ik}=1,\qquad s_i>0,\qquad
\sum_k B_{ik}=s_i,\qquad \eta_i=b_i+\sum_k C_{ik}.
\]

这些性质也明确了限制：同一年龄下，ancestry 改变配比，但不另行改变系数之和；给定 ancestry 后，各 PRS 的相对配比不随年龄改变；在已定向的 Z 坐标中，各 PRS 的条件效应非负。

这些是研究假设，不是已知事实。整体预测失败时，不要在后台解除这些约束。

---

## 4. 蒸馏目标与优化

Teacher 返回疾病阳性的概率 q，不能使用 hard predicted class，也不能默认 `predict_proba[:, 1]` 就对应阳性；必须检查 `classes_`。

\[
\tilde y_i=(1-\lambda)y_i+\lambda q_i,
\]

\[
\mathcal L=\frac1N\sum_i\operatorname{BCEWithLogits}(\eta_i,\tilde y_i)+\Omega.
\]

这是输出蒸馏。真实标签与软标签的联合监督来自经典知识蒸馏思路。[S1] `BCEWithLogitsLoss` 接受 [0,1] 目标并直接作用于 logits。[S2]

默认不使用 `pos_weight`、class weighting、重采样或温度；不进行额外 Teacher 概率校准。既有 Teacher 如开启 probability balancing，记录其配置与输出含义，而不是默默修改。

### 4.1 要比较的 λ

```text
0, 0.25, 0.50, 0.75, 1.00
```

- λ=0：完全相同结构的直接监督对照，必须保留。
- λ=1：纯软标签蒸馏，必须单独报告。
- 中间值：混合蒸馏；在内部调参集选择最佳中间值。
- 最终“自动选择方案”允许选中 λ=0；这意味着本轮没有证据支持蒸馏更好，不能只汇报蒸馏赢的结果。

### 4.2 轻量正则化

建议显式实现：

\[
\Omega=\rho_g(\|W_c\|_F^2+\|c_c\|_2^2)
+10^{-4}(\|\theta_b\|^2+\|\gamma\|^2+\|\delta\|^2)
+10^{-3}\|\theta_s\|^2.
\]

采用平方和而不是未声明的 mean；不惩罚基础截距 alpha 或整体尺度截距 u0。

只搜索 `rho_g ∈ {1e-4, 1e-3, 1e-2}`。低维年龄基函数加系数 ridge 是 v1 的固定复杂度控制；不再添加新的 smoothness 搜索。

### 4.3 训练默认值

- Student：PyTorch，CPU、float64、full batch；大样本时可以明确改成 minibatch，但不改变目标。
- Adam，学习率 0.01，显式正则下 optimizer weight_decay=0。
- 最多 2,000 epochs；每 10 epochs 评估一次；patience 200 epochs；min_delta 1e-6。
- 梯度范数上限 10；记录 loss、梯度、s(a) 范围以及 NaN/Inf。
- 初始化：基础截距为当前训练软标签均值的 logit；其余基础项为 0；u0=log(0.25)，theta_s=0；W 为 seed 固定的微小扰动（SD 0.01），c=0。
- 随机种子 17、29、43；Teacher 的 fold/seed 固定并缓存共享，不能每个 λ 重新随机生成 Teacher。
- 数值溢出时失败并保存诊断。允许一次预先声明的学习率 0.003 重试；不默默 clamp s(a) 变成另一个模型。

不需要冻结 baseline 再单独训练 gate。所有学生参数联合优化，保持训练流程简单。

### 4.4 内部选择与最终单模型

混合与直接监督候选用内部验证的真实标签 log loss 选择 checkpoint/超参数。纯蒸馏版本用内部验证 Teacher 概率的交叉熵选择 checkpoint/正则，保留其“以 Teacher fidelity 为目标”的性质。

同一配置先汇总 3 个初始化的内部指标，再选配置，避免挑选最幸运初始化。最终主模型固定使用 seed=17；29/43 用于稳定性检查，不将三个模型的概率平均作为主模型，否则部署模型就不再是一个单一显式公式。

纯蒸馏候选内部按 Teacher fidelity 选择完毕后，若要在不同 λ 的模型之间选择最终预测方案，统一比较 H 上的真实标签 log loss。不能将软标签交叉熵与真实标签交叉熵直接混在一列排序。

最终 refit 的 epoch 数使用对应选中配置在内部调参阶段的 best_epoch 中位数，并写入 lockfile。不得根据 official-validation 表现修改。

---

## 5. Teacher：固定版本、本地推理和缓存

先复用项目已有 Teacher 设置与 checkpoint；当前项目背景为 TabPFN-3，历史不同 trait 的 estimators/checkpoint 可能不同。[P1]

**不要直接调用未指定版本的 `TabPFNClassifier()` 然后假定还是 v3。** 官方当前仓库的默认模型已变化，支持显式 `create_default_for_version(ModelVersion.V3)`。[S4]

示意接口，Codex 必须与本地安装版本核对：

```python
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3)
# 通过现有 adapter / 已验证的 get_params + set_params 设置
# n_estimators、device、random_state、checkpoint 与 probability balancing。
# 不支持的字段应明确报错，不能静默忽略。
clf.fit(X_context, y_context)
q = clf.predict_proba(X_query)[:, positive_class_index]
```

记录软件包版本、checkpoint 的绝对路径/名称和 SHA256、所有显式参数、实际默认参数、输入列顺序、随机种子、推理设备与 context/query IDs。没有可靠 provenance 的旧预测不能直接作为正式 OOF 缓存。

Teacher 各折独立初始化，context/query 不混合。一次只运行一个 GPU Teacher 任务；按需缓存，不同时启动大量模型抢占显存。不自动使用远程 TabPFN API。已存在的认证/许可状态可复用；缺少授权时报告，不替用户接受新的许可。

### 5.1 相同输入原则

Teacher 输入为相同的 Z、A、age 和 C；学生额外做 spline 只是已知变量的确定性变换，不是特权信息。

若旧 Teacher 使用 raw PRS、新 Student 使用 corrected PRS，不能将旧预测冒充 matched-input 主实验。应重新生成 matched-input Teacher 预测；旧 Teacher 仅作为单独标记的历史/敏感性对照，不混入主表。

### 5.2 缓存键

缓存至少包括：trait、数据 fingerprint、PRS manifest hash、fit-scope ID hash、fold ID hash、query ID hash、预处理 hash、Teacher 包/模型/checkpoint/config hash、seed、positive-class mapping。

文件中每行必须有 participant_id、fold_id、q_teacher。context ID 清单或其可验证索引另存于受控本地目录；报告不公开患者 ID。

没有 hash 匹配就重算或报错，不能靠相似文件名猜测对应关系。

---

## 6. Phase A：无泄漏的核心蒸馏试验

### 6.1 保留官方边界

记：

- D = official_train。
- E = official_validation：本轮不用于任何调参，只在方案锁定后评估。若历史已经反复使用，报告其 post-hoc/机制性地位，不能称为全新独立确认集。
- B = blind_holdout：第一轮不读取标签、不申请解盲、不自动提交预测。

先在 D 内做一次固定的约 80%/20% 内部分组分层划分：

\[
D=U\cup H,
\]

U 用于生成训练软标签和拟合学生，H 仅用于早停/选择。存在 family/kinship 时 group 必须保持完整；没有时至少按 participant 分组。可使用 StratifiedGroupKFold 取固定一折为 H；该工具尝试保留类别比例且不让 group 跨折。[S5]

尽量在可行时平衡已有 ancestry group 与结局，但小亚组优先保证分组独立性和每折两类都有，而不是不断重抽直到结果好看。固定 seed 和最终 ID manifest。

### 6.2 必须先划 U/H，再生成 U 的软标签

在 U 内做五折 OOF。对每折 f：

\[
q_i=T\{x_i;U\setminus U_f\},\qquad i\in U_f.
\]

该折的预处理只在 `U \\ U_f` 拟合；校正参数若为真正外部冻结的产物可以复用。聚合后每个 U 患者恰好得到一个 q。

再用全部 U 作 context，生成 H 的 Teacher 概率：

\[
q_H=T(X_H;U).
\]

学生在 U 上拟合；Student 自己的预处理在 U 拟合，应用于 H。

**禁止：先在整个 D 生成 OOF q，再把 D 随机拆成 U/H 进行学生调参。** 那样 U 中某些 q 的 Teacher context 会包含 H 的标签。即使每个 q 都没有看到患者自己的标签，内部 H 也不再是干净的选择集。

### 6.3 调参范围

五个 λ × 三个 gating ridge × 三个初始化，共 45 个轻量学生拟合。Teacher 预测在这些候选间共享，不重复生成。

λ=0 的拟合不依赖 q；代码应允许没有 Teacher 时独立完成直接监督 baseline。λ=1 的优化目标中不得不小心混入 y；它仍然可以用于最终与真实标签的评估。

内部候选指标仅用于调参，不能当作对外正式性能结果。

冻结并输出 `selection_lock.json`：

```text
trait / feature_manifest_hash / input_scaling_hash /
split_hash / student_structure / lambda / regularization /
refit_epochs / final_seed / teacher_config_hash /
age_basis_settings / subgroup_definitions / code_commit
```

纯蒸馏、直接监督、最佳混合三个报告臂分别保存配置。若 log loss 差距小于 1e-4，则采用预定 tie-break：优先较小 λ，然后较强 ridge；不能查看 E 后决定 tie-break。

### 6.4 锁定后，在完整 D 上 refit

1. 在 D 内重新生成五折 OOF Teacher 概率 q_D。不得把 U 的 q 当作覆盖 D 的最终缓存。
2. Student 预处理在 D 拟合。
3. 各固定配置在 D 上训练已锁定 epoch 数，不再以 E 早停。
4. Teacher 以完整 D 为 context，生成 q_E。
5. Student 在 E 上生成 p_E、权重、年龄尺度与分项贡献。
6. 最后才由独立评价函数读取 E 的 y，生成报告。

每个 trait 的核心 Teacher 成本约为两组“5 折 OOF + 1 个完整 context 对 query 预测”；缓存匹配时可复用。这里是拟合次数描述，不是运行时间承诺。

### 6.5 伪代码：划分边界比具体函数名更重要

```python
D, E, B = load_existing_splits()  # B labels never loaded
U, H = fixed_internal_split(D)

q_U = teacher_oof(scope=U, n_folds=5)   # no H/E/B labels
q_H = teacher_predict(context=U, query=H)

selected = select_student_configs(
    train=U, q_train=q_U,
    tune=H, q_tune=q_H,
    lambdas=[0, .25, .5, .75, 1],
    gate_ridges=[1e-4, 1e-3, 1e-2],
    seeds=[17, 29, 43],
)
write_selection_lock(selected)

q_D = teacher_oof(scope=D, n_folds=5)
q_E = teacher_predict(context=D, query=E)
students = refit_locked_models(D, q_D, selected)
predictions_E = predict_with_decomposition(students, E)
# metrics function is the first consumer of E.y in this phase
report_direct_diagnostics(E.y, q_E, predictions_E)
```

不要根据 q_D 的训练内评价重新挑选最佳 λ；选择已经结束。

---

## 7. 第一轮实验矩阵

### 7.1 必须完成的五类模型

| ID | 结构 / 训练 | 作用 |
|---|---|---|
| META | 只有同一 b(age,A,C)，真实标签训练 | 基础信息参照 |
| S_SUP | 完整学生，λ=0 | 判断不蒸馏时同一结构的能力 |
| S_KD_PURE | 完整学生，λ=1 | Teacher 保真度取向 |
| S_KD_MIX | 完整学生，内部选 λ∈{0.25,0.5,0.75} | 预测取向的混合蒸馏 |
| T_MATCHED | 相同输入信息的 TabPFN | Teacher 参照，而非理论性能上限 |

另输出 `SELECTED_POLICY` 指向基于内部选择的一个学生臂，允许最终指向 S_SUP。不要为了命名为“蒸馏”强行禁止 λ=0。

所有臂必须使用完全相同的评估患者，不得因某个模型缺失而静默更换分母。不同疾病分别汇报，不把性能最好的疾病当作唯一主结果。

### 7.2 核心流程通过后运行的两个轻量消融

使用已固定的最佳混合 λ 与同一基础项；不看 E 重新调参。

- **S_STATIC_GATE**：\(\pi=softmax(c)\)，删除 W；仍学习最优固定配比，不是把权重强行设为均匀。
- **S_CONSTANT_SCALE**：\(s=exp(u_0)\)，删除年龄变化部分 theta_s；保留基础年龄函数。不能把“无年龄缩放”错误实现为 s=1。

它们回答 ancestry gating 和 age scaling 是否各自必要。若主实验尚未完成，不先扩展其他模型。

### 7.3 PRS 增量解释的额外要求

相对于同一简单 META baseline 的改善，是实用的增量预测比较，但**不能单独证明 Teacher 的优势都是 PRS interaction**，因为 Teacher 也可能更好地拟合 metadata。

若要在论文中声称“保留了 Teacher 的 PRS 增量价值”，增加相同 context、相同设置的 `T_METADATA`，用 Teacher 自身的 `T_MATCHED vs T_METADATA` 作参照。该检查可作为核心跑通后的诊断，不将简单 META 的差值包装成纯遗传贡献。

不在 v1 加入 synthetic patients、PRS permutation 蒸馏、残差 Teacher 或新的模型家族。

---

## 8. 评价：预测、保真度和解释分别报告

### 8.1 直接模型诊断

均在锁定后的 E 上计算，报告命名为 `direct_distillation_diagnostics`，不是历史 Stage-2 正式分数。

真实标签性能：log loss（内部选择的主指标）、AUROC、Brier score、calibration intercept/slope。

校准诊断可通过评价集拟合截距/斜率进行描述，但该拟合**不允许回写模型或修正当前评估概率**。无法稳定估计时报告失败或区间很宽，不用截断值伪装正常。

Teacher 保真度至少报告：

\[
MAE_p=\operatorname{mean}|p_S-q_T|,
\]

\[
RMSE_p=\sqrt{\operatorname{mean}(p_S-q_T)^2},
\]

\[
KL=\operatorname{mean}\left[q_T\log\frac{q_T}{p_S}
+(1-q_T)\log\frac{1-q_T}{1-p_S}\right].
\]

计算 log/logit 指标时统一用 epsilon=1e-6；不将这种数值 clipping 混入 Student 的训练公式。可附 logit R²；方差近零时报告 NA，负值不得强行截成 0。

OOF Teacher 与最终完整 D context 的 Teacher 不是完全相同的函数：测试集 fidelity 是针对最终 T_MATCHED，不能称为某个唯一固定 Teacher 在所有训练点上的精确投影。

### 8.2 成对比较与不确定性

主比较：S_KD_MIX − S_SUP；其次 S_KD_PURE − S_SUP、各 Student − T_MATCHED。

至少报告 paired Δlog loss、ΔAUROC、ΔBrier，及固定模型预测的 paired bootstrap 95% 区间（默认 1,000 次）。同一 bootstrap 重采样索引必须用于所有模型；有 kinship group 时以 group 为单位。

这类区间只反映当前训练模型在评价样本上的重采样不确定性，不包含完整重训/Teacher 生成的不确定性。不能把它们误称为所有建模过程的总不确定性。

### 8.3 ancestry 与 age 分层

沿用已有 ancestry group 仅用于汇报；age 使用训练范围上预先固定的四分位区间。内部支持不足时少做分层，不能在 E 上挑选“显著”的年龄切点。

每个亚组报告 N、events、non-events、AUC、log loss、Brier。默认 N<100 或 events/non-events 任一<20 时，显示计数和“不稳定/仅探索”，不作可靠亚组优劣结论；这些阈值是预先设定的工程汇报规则，不是统计定律。

没有离散 ancestry group 时，不依据 PRS 或结局自动发明族群；只展示已有连续 PC 空间的覆盖与预测诊断。

### 8.4 增量保留比例仅作辅助

可选报告：

\[
R_{gain}=\frac{LL_{META}-LL_{Student}}{LL_{META}-LL_{Teacher}}.
\]

分母≤1e-4 时报告 NA；不强制限制在 [0,1]。它只是共享 baseline 下的操作性比值，不是“百分之多少生物学信息被保留”，也不是非劣效检验。

### 8.5 结论标准

不预先编造“必须保留 95% 性能”之类成功阈值。

- 蒸馏点估计与区间支持改善：报告改善及效应量。
- 两者接近且区间跨零：报告未见明确收益，不能称等效。
- λ=0 在内部选择胜出：明确记录目前没有选用蒸馏的证据。
- Student 与 Teacher 预测接近但 gate 不稳定：只支持预测压缩，不支持唯一权重解释。
- Teacher 明显更好：记录受约束模型的近似能力不足，不自动引入 r/v/MLP 来“修复”主实验。

---

## 9. 必须导出的解释与稳定性检查

### 9.1 个体级分解

受控本地输出每个 E 患者的：

```text
participant_id, p_teacher, p_student, baseline_logit,
age_scale, genetic_score, pi_<prs>, beta_<prs>, contribution_<prs>,
input_support_flags
```

其中 beta_k=s*pi_k，contribution_k=beta_k*Z_k。ID 只保留在受控结果目录，不放入公开图、报告或 Git。

### 9.2 图与表

输出以下 4 类核心图即可，不做大而全的解释工具箱：

1. `age_scaling`：s(age) 与 s(age)/s(reference_age)，标记训练年龄分布。
2. `gate_by_ancestry`：在真实患者 PC1/PC2 上展示每个 PRS 权重，或按已有组汇总；不将未观测的 PC 网格当真实患者。
3. `teacher_student_fidelity`：Teacher 与 Student 概率散点及差异分布。
4. `calibration`：同一 E 上 Teacher、S_SUP、S_KD_MIX 的预测校准。

个体 waterfall/贡献表可保存少量去标识示例，注明是模型 log-odds 分解，不是风险百分比分摊。

### 9.3 权重可辨识性

必须输出训练数据的 PRS 相关矩阵。两个 PRS 若相同，则 w*Z1+(1-w)*Z2 始终相同；可预测不等于可唯一解释。

报告 17/29/43 三次初始化的 E 上 pi、s 和预测差异。这个差异是优化稳定性，不是 95% confidence interval。

后续可进行固定 Teacher 软标签和冻结预处理条件下的学生重采样（如 50 次），但必须标记 `conditional_on_teacher_and_preprocessing`。第一版不需要完整重训 Teacher 的参数 bootstrap，也不能用便宜的条件 bootstrap 冒充完整不确定性。

---

## 10. Phase B：与项目既有 Stage-2 协议兼容

**本节在需要正式对比历史结果时执行；不让它阻塞 Phase A 的模型可行性试跑。**

已检索的项目规范要求：Stage 1 在 official_train 内生成五折 OOF score，official-validation score 使用完整 official_train context；Stage 2 对每个 arm 使用相同的非加权 logistic，输入 Stage-1 score + sex + age_sd + PC1_sd–PC10_sd，正式结果只在 official-validation 报告。[P1]

### 10.1 绝对不能做的比较

- Student 直接概率 AUC vs 历史经过 Stage-2 的 Teacher AUC。
- Student 训练内拟合概率作为 Stage-2 训练分数，而 Teacher 使用 OOF 分数。
- 修改 Stage-2 covariates、score representation 或 regularization，却仍沿用历史协议名称。
- 将 Stage-2 之后的预测误标成原始 Student 公式的直接 logit 分解。

### 10.2 Student 的 Stage-1 OOF 需要嵌套隔离

对 D 的每个外层折 V_f：

1. 定义 U_f=D\\V_f。
2. 若超参数需要数据驱动选择，必须只在 U_f 内选择，不能复用曾受 V_f 标签影响的全 D 选择结果来宣称严格嵌套 OOF。
3. 在 U_f 内生成 Teacher 的 OOF 训练软标签；这些 Teacher 的 context 从未包含 V_f 的标签。
4. 在 U_f 上训练对应 Student，预测 V_f，得到真正的 pipeline-level OOF 学生 score。
5. 对所有 V_f 聚合，再拟合既有 Stage-2 logistic。
6. 完整 D 上运行同样的选择/OOF 蒸馏训练流程，得到完整 Student 对 E 的 score；由 D 上已拟合 Stage 2 产生最终 E 预测。

**关键反例：** 不能把全 D 的 Teacher OOF q 简单切出 U_f，再训练 Student→V_f。尽管每个 q 没有看到自身 y，U_f 中部分 q 的 Teacher context 仍可能包含 V_f 的 y，形成间接泄漏。

Teacher 的 Stage-1 OOF 使用相同外层 V_f，且其完整 D 模型只用于预测 E。所有正式臂复用同一 split、cohort、正类定义和 Stage-2 evaluator。

### 10.3 官方指标与解释分开

从现有 evaluator 复用 AUROC、OR90、Nagelkerke R² 和 paired bootstrap 定义。记录 score 是 probability 还是 logit，不能自作主张切换；同名 R² 不能混用不同实现。

如果 evaluator 不存在或定义无法核实，输出 `blocked_official_evaluator`，保留 Phase A 诊断结果，不捏造兼容结果。

输出两个明确分离的报告：

- `direct_distillation_diagnostics.md`：显式学生公式的预测、fidelity 和解释。
- `official_stage2_comparison.md`：完全匹配旧协议的正式比较；解释曲线仍来自 Stage-1 学生，不能称为 Stage-2 最终模型的完全相同分解。

如果 score 用的是 Student probability，额外 Stage 2 的非线性组合通常不再等于原学生那条 logit 加性公式。必须在报告中写明。

official-validation 已被历史模型选择使用过时，两个报告都应注明这是 post-hoc/机制性评价；新的确认性结论需要真正未参与选择的 blind 或外部数据，但本次不得自动触碰这些标签。

---

## 11. 建议代码组织与命令接口

优先适配现有包结构；以下是拟实现接口，不代表这些脚本已经存在。

```text
prs_distill/
  data.py               # 数据、manifest、eligibility、正类映射
  splits.py             # 固定分组划分和作用域校验
  preprocessing.py      # 原有校正适配 + scope 内统计量/spline
  teacher.py            # 本地 TabPFN adapter、OOF 与 provenance
  student.py            # 唯一主模型及两个删除式消融
  train.py              # 软标签 BCE、选择、锁定与 refit
  metrics.py            # 直接指标、paired bootstrap、亚组
  explanations.py       # 精确贡献与权重/年龄曲线
  stage2_adapter.py     # 原有 evaluator 适配及嵌套 OOF
  cli.py
configs/
  prs_distillation_v1.yaml
tests/
  test_student_algebra.py
  test_soft_targets.py
  test_scope_isolation.py
  test_serialization.py
  test_stage2_isolation.py
```

建议命令：

```bash
# 1. 仅检查环境、列映射、来源与划分；不触发模型下载/训练
python -m prs_distill.cli audit --config configs/prs_distillation_v1.yaml

# 2. 软件测试。可以使用明确标注的合成 fixture，不产生真实效能结论
pytest tests/ -q

# 3. 同一流程的小规模接口试跑，不做正式选模
python -m prs_distill.cli smoke --config configs/prs_distillation_v1.yaml

# 4. Phase A：内部选择、锁定、完整 D refit、E 直接诊断
python -m prs_distill.cli run-core --config configs/prs_distillation_v1.yaml

# 5. 已锁定设置下的两个轻量消融
python -m prs_distill.cli ablate --config configs/prs_distillation_v1.yaml

# 6. 需要历史正式指标比较时才执行 Phase B
python -m prs_distill.cli evaluate-official --config configs/prs_distillation_v1.yaml

# 7. 统一报告，不修改任何训练/选择结果
python -m prs_distill.cli report --config configs/prs_distillation_v1.yaml
```

`smoke` 的子样本、fold 数、estimators 若减少，必须使用独立 run_id/cache namespace。smoke 缓存和性能不能进入正式 run-core；不因为 smoke 表现差就选择另一个更容易的 trait。

所有命令具备 `--dry-run`、清晰错误信息和断点缓存能力。已有缓存不匹配时明确解释，不静默覆盖。训练和评估解耦，重新绘图不能触发再次调参。

---

## 12. 预期产物

```text
outputs/<run_id>/<trait>/
  discovery_report.md
  resolved_config.yaml
  environment.json
  feature_manifest.tsv
  split_manifest.parquet          # 受控本地，含 ID
  audit_report.json
  selection_lock.json
  teacher_cache_manifest.json
  tuning_results.csv
  training_logs/
  models/
    S_SUP/
    S_KD_PURE/
    S_KD_MIX/
      student_state.pt
      preprocessing.joblib
      model_metadata.json
  direct_predictions.parquet      # 受控本地，含 ID 与精确贡献
  direct_metrics.csv
  paired_differences.csv
  subgroup_metrics.csv
  gate_stability.csv
  prs_correlation.csv
  figures/
  direct_distillation_diagnostics.md
  official_stage2/                 # Phase B 才出现
    nested_scope_manifest.json
    oof_stage1_scores.parquet
    metrics.csv
    official_stage2_comparison.md
```

最终学生部署 bundle 必须包含 PRS 列顺序、方向、尺度/校正引用、PC/age preprocessing、spline、模型参数和版本，不仅是 `.pt`。

提供本地推理接口：

```python
bundle = load_student_bundle(path)
result = bundle.predict(patient_table, return_components=True)
```

推理不依赖 TabPFN、不读取标签、不重新拟合任何预处理。列缺失/顺序错乱/不支持的类别必须明确报错。

真实数据、Teacher context、ID、缓存和患者预测加入 `.gitignore`。仅提交代码、合成测试 fixture、模板配置与去标识汇总；不得自动推送包含患者信息的内容到远程。

---

## 13. 必须通过的自动化测试

### 13.1 数学与结构

1. softmax 非负且各行和为 1。
2. s(a)>0；sum(beta)=s；baseline+sum(contribution)=logit。
3. Z 全为 0 时预测退化到 baseline，而不是删除年龄效应。
4. 固定 ancestry，改变 age 不改变 pi；固定 age/ancestry，改变 PRS 不改变 pi。
5. 固定输入坐标，增加单个 Z_k 不应降低预测（数值容差内）。
6. `S_STATIC_GATE` 对不同 ancestry 的 pi 相同；其 baseline 仍可随 ancestry 改变。
7. `S_CONSTANT_SCALE` 的 s 为可学习正标量，不是硬编码 1。
8. K=1 可正常运行，weight=1；没有新的 r/v 或隐藏专家参数。
9. 调换 PRS 列及对应 gate 参数后，预测保持一致。
10. softmax 共同平移不改变预测；导出参数使用已说明的 gauge。

### 13.2 损失

11. `BCE(logit, (1-lambda)*y+lambda*q)` 与两项 BCE 的加权和数值一致。
12. λ=0 时更改 q 不改变目标；λ=1 时更改 y 不改变目标（保持固定初始化/训练作用域）。
13. 梯度能到达 baseline、W/c、u0/theta_s；Teacher 概率已 detach。
14. 所有训练目标在 [0,1]；不对 probability 再调用 `BCEWithLogitsLoss`。
15. 正类 mapping 在标签顺序交换时仍正确；不盲取第二列。

### 13.3 作用域与泄漏

先固定 split manifest，再做下列 mutation tests，避免混淆“重新分层导致变化”和“标签泄漏”：

16. 每个 query ID 与 Teacher context ID 不相交；family group 同样不相交。
17. 修改 E 标签，训练目标、预处理、选中配置与预测保持不变，仅评价指标变化。
18. 修改 H 标签，不改变 U 的 Teacher 训练软标签；允许 H 上的选模结果变化。
19. Teacher OOF 每个患者恰有一条，且没有遗漏、重复或错序。
20. scope-local 缓存不能与另一个 scope 的缓存互换，即使列名相同。
21. Phase B：修改外层 V_f 的标签，U_f 内的 Teacher 软标签与学生训练过程不受影响；外层划分事先固定。
22. 预处理中的样本数、ID hash 和拟合 scope 一致；evaluation rows 不进入 scaler/spline fit。
23. 禁止从无法审计的历史全数据 OOF 文件静默提取正式训练目标。

### 13.4 可复现与输出

24. 保存/加载学生 bundle 后，相同原始输入的预测及每一项贡献一致。
25. 主模型 seed 固定；report-only 不触发训练或读取盲测标签。
26. 不同模型的评价 cohort ID 完全一致。
27. 未完成 Phase B 时，不能生成声称“official comparable”的表格。
28. 单类别或小亚组的指标安全返回 NA/诊断，不崩溃或填 0。
29. q=0/1 或极端 logits 的 loss/metrics 数值稳定；不悄悄改变模型定义。
30. YAML 的未知字段必须报错，不静默忽略配置拼写错误。

先通过 1–20、22–30 中适用于 Phase A 的检查，再运行真实完整核心实验。第 21 项及其他需要 Stage-2 adapter 的检查在 Phase B 前完成；不得让尚未启用的 Phase B 阻塞核心软件试跑。

---

## 14. 失败诊断与停止规则

| 现象 | 先做什么 | 不要做什么 |
|---|---|---|
| λ=0 最好 | 如实报告、检查 Teacher 校准与匹配输入 | 删除直接监督对照 |
| gate 接近恒定 | 比较 S_STATIC_GATE 与完整模型 | 强迫 gate 稀疏/分离 |
| s(age) 接近常数 | 比较 S_CONSTANT_SCALE | 强制年龄曲线非线性 |
| pi 随初始化大幅变化、预测稳定 | 查看 PRS 相关性，报告不可唯一分解 | 把权重解释成真实生物机制 |
| Teacher 好、Student 差 | 看哪个 ancestry/age 区间残差大及基础项不足 | 自动加 r_k/v_kj/MLP |
| 所有模型差 | 检查标签定义、score 方向、输入对齐、cohort eligibility | 直接归因于理论不成立 |
| q 几乎全 0 或 1 | 核实正类、context、Teacher 设置及标签泄漏 | 先调 softmax 温度掩盖问题 |
| 训练 NaN | 检查输入、log-scale、学习率、梯度 | 删除异常患者而不记录 |
| 新 Teacher 与历史结果不同 | 核实版本、校正、estimators、Stage 2 | 宣称新方法退步/进步而不对齐 |
| 缺少真实数据/权限 | 输出 blocker，并完成可独立完成的软件任务 | 捏造数据或自动外传 |

v1 结束后提交结果和诊断，不继续自动探索几十种模型。下一版结构调整必须基于已完成结果另行提出，不在此次执行中未经说明改变研究问题。

---

## 15. Codex 最终回报格式

最终回复应明确给出：

1. 实际修改的文件、命令及 git diff 摘要；没有执行的命令明确标注。
2. 实际采用的 trait、患者数量、PRS 列、ancestry 编码、Teacher checkpoint 与配置。
3. 测试通过/失败情况，以及数据隔离审计是否通过。
4. 核心模型表：META、S_SUP、S_KD_PURE、S_KD_MIX、T_MATCHED；内部选择是否落到 λ=0。
5. 三个研究问题分别得到什么证据；区别点估计、区间、优化稳定性与因果解释。
6. direct 与 official-stage2 的结果分开；Phase B 未跑就明确说未跑。
7. 产物路径、未解决 blocker、可复现命令；不假装已产生不存在的图/表/指标。

主结论应聚焦：**固定 PRS 原型的软加权结构，能否在保持解释形式的前提下，从 TabPFN 的软预测中获得超过同结构直接训练的收益。**

---

## 16. 方法与工程来源

以下来源用于核对既有方法/API；本文件中的具体模型约束、超参数和执行策略是本项目的研究设计，不是这些来源已经证明最优的结论。网络文档核查日期：2026-10-26。

[S1] Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network* (2015). 软标签知识蒸馏的基础来源。`https://arxiv.org/abs/1503.02531`

[S2] PyTorch 官方 `BCEWithLogitsLoss` 文档。核对 logits 输入、软标签范围与稳定实现。`https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html`

[S3] scikit-learn 官方 `SplineTransformer` 文档。核对基函数数量、include_bias 与外推行为。`https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html`

[S4] Prior Labs 官方 TabPFN 仓库和模型页。核对本地调用、显式版本选择和模型版本差异；运行时仍应以本地已锁定环境为准。`https://github.com/PriorLabs/TabPFN`；`https://docs.priorlabs.ai/models`

[S5] scikit-learn 官方 `StratifiedGroupKFold` 文档。核对 group 不跨折和类别分层的约束。`https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html`

[P1] 用户历史项目文件 `tabpfn_official_mechanism_ablations.html`，章节 “Canonical evaluation contract”，以及其中的 “Interpretation guardrail”。本次从用户文件 Library 检索核对。其历史定义为 official-train 五折 OOF、official-validation 使用完整训练 context、统一 Stage-2 logistic、AUROC/OR90/Nagelkerke R²，以及 post-hoc 解释性结果限制。本文件不把历史结果当作此次蒸馏实验的已知表现。
