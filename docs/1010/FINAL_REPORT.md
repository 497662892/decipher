# 2026-10-10 最终阶段总结：外部 PRS × ancestry-aware TabPFN

更新与核验时间：2026-10-10 20:27 EDT。本文件汇总截至此时已完成的实验；本次新增新版外部多 PRS 的离散 ancestry 三疾病评估，并复核既有存储预测。PRS 的外部 SNP 权重保持冻结。

当前主方案为 **冻结外部 PRS＋age＋sex＋PC1–PC10 → TabPFN-3**，不加入 soft ancestry。主方案 raw AUC 为 T2D **0.764329**、CAD **0.812933**、女性乳腺癌 **0.715345**。在另行完成的同输入对照中，TabPFN 对三个疾病的整体 AUC 均高于当前 DiscoDivas 适配版，六个主要比较的多重比较调整区间均高于零。这支持当前配置的预测优势，尚不能归因于祖源交互机制，也不能表述为超过 DiscoDivas 原文模型。

**当前方案与评估队列**

```mermaid
flowchart LR
    A[外部冻结 SNP 权重] --> B[PLINK2 计算每人的多个 PRS]
    B --> C[PRS + age + sex + PC1–PC10]
    C --> D[TabPFN-3]
    T[官方 train 的特征与标签作为 context] --> D
    D --> V[官方 val 风险预测与评估]
```

每个疾病单独构建预测器。TabPFN 使用 v3 default checkpoint、16 个内部集成估计器；PRS 权重保持作者发布版本，不在 decipEHR 内重训 GWAS 或 SNP 效应。每个人均计算该疾病的全部入选 PRS，然后由 TabPFN 整合，不先按祖源挑选一个分数。

age 和 sex **直接参与 TabPFN 预测**。这里的 PC-only 指祖源特征只保留连续 PC，并非只输入 PC；T2D/CAD 同时使用年龄和性别。乳腺癌按 `configs/disease_eligibility.yaml` 限定女性，恒定 sex 列被移除，age 保留。

| 疾病 | Train/context N | Train 病例 | Val N | Val 病例 |
| --- | --- | --- | --- | --- |
| T2D | 5811 | 1701 | 2852 | 847 |
| CAD | 5811 | 2313 | 2852 | 1072 |
| 乳腺癌（女性） | 3308 | 602 | 1628 | 320 |

官方 train 全部作最终 context，官方 val 只供预测和评估。缺失值处理、标准化、下游参数选择和 wrapper 拟合均限于 train；需要模型选择和校正训练时采用 train 内嵌套三折。此协议遵从用户对本实验的更新，**不使用 train＋val 合并五折作为本次主结果**。发现/调优队列不重叠按项目负责人确认记录，未进行独立参与者级连接审计。

**实际使用的外部 PRS**

| 疾病 | 模型 | 底层方法或来源 | 主方案分数数目及 PGS ID |
| --- | --- | --- | --- |
| T2D | D-PRISM | PRS-CSx；五祖源调优目标与来源成分 | 25；PGS005353–PGS005377，[资源示例](https://www.pgscatalog.org/score/PGS005353/) |
| T2D | PRScsx_T2D | 已发表多祖源综合分数 | 1；[PGS002308](https://www.pgscatalog.org/score/PGS002308/) |
| CAD | GPS_Mult | LDpred2，多祖源/多性状综合模型；EUR 调优 | 1；[PGS003725](https://www.pgscatalog.org/score/PGS003725/) |
| CAD | multi_anc_hg37CSx | PRS-CSx；混合祖源调优 | 1；[PGS004696](https://www.pgscatalog.org/score/PGS004696/) |
| CAD | CAD_EAS_MIXPRSplus | 面向 EAS 的 MIXPRS+ 综合模型 | 1；[PGS012558](https://www.pgscatalog.org/score/PGS012558/) |
| 乳腺癌 | Overall_Prop_M1 | AFR 来源；FSS、PRSice-2、elastic net | 1；[PGS005378](https://www.pgscatalog.org/score/PGS005378/) |
| 乳腺癌 | INTERVENE_MegaPRS | EUR 来源；MegaPRS | 1；[PGS004869](https://www.pgscatalog.org/score/PGS004869/) |
| 乳腺癌 | TPMI PRS-CS | 台湾汉族来源；PRS-CS | 1；[PGS018538](https://www.pgscatalog.org/score/PGS018538/) |

主方案输入分数数目为 **T2D 26、CAD 3、乳腺癌 3**。D-PRISM 的 25 列表示 5 个目标祖源 × 5 个来源祖源（AFR/AMR/EAS/EUR/SAS），共享研究与 GWAS 信息，不是 25 个独立发现研究。当前主方案直接输入这 25 列，不先合并成 5 列。

全部资源统一到 GRCh38，使用 PGS Catalog 匹配工具与 PLINK2 计分；缺失基因型所需频率来自固定 1000G 参考。34 个候选中，32 个通过预设的 75% SNP 匹配覆盖门槛。以下两个乳腺癌候选未入选：

| PGS | 匹配 SNP | 发表 SNP | 覆盖率 |
| --- | --- | --- | --- |
| PGS000004 | 212 | 313 | 67.732% |
| PGS005379 | 131341 | 175173 | 74.978% |

MegaPRS 和 TPMI PRS-CS 在读取本次结局表现前加入，记录于 outcome-blind 的 protocol amendment，覆盖门槛未变。TPMI 原模型采用较宽的 ICD 定义并包含男性对照；本项目目标队列仍限定女性。实际分数只覆盖目标数据中保留的匹配 SNP，不能称为原队列完整 SNP 集的精确复现。

**主方案的最新结果**

| 疾病 | 预测版本 | AUC ↑ | OR90 ↑ | Nagelkerke R² ↑ | Brier ↓ |
| --- | --- | --- | --- | --- | --- |
| T2D | TabPFN 直接概率 | 0.764329 | 6.6460 | 0.247488 | 0.169630 |
| T2D | train-OOF wrapper | 0.764410 | 7.1621 | 0.248623 | 0.169689 |
| CAD | TabPFN 直接概率 | 0.812933 | 8.0204 | 0.372462 | 0.169302 |
| CAD | train-OOF wrapper | 0.813016 | 7.8361 | 0.370372 | 0.169581 |
| 乳腺癌（女性） | TabPFN 直接概率 | 0.715345 | 4.6375 | 0.140703 | 0.143491 |
| 乳腺癌（女性） | train-OOF wrapper | 0.713365 | 4.5025 | 0.151991 | 0.142279 |

`raw` 是 TabPFN 在完整 train context 下直接输出的概率，已经利用 age、sex、PC；不是 PRS 单独预测。`official_wrapper` 将 train 内 OOF 概率的 logit 与 age/sex/PC 再送入 train 拟合的 logistic 模型，并应用于 val，概率裁剪阈值为 0.001。两种口径分别报告，不跨版本拼接 AUC、OR90 或 R²。

OR90 表示预测风险最高 10% 相对其余 90% 的疾病 odds ratio；Nagelkerke R² 为 held-out 概率似然相对截距模型的指标；Brier 越低越好。本项目的 R² 包含临床协变量贡献，与 DiscoDivas 原文的 covariate-adjusted full/null R² 不同。

**与之前官方 train→val 结果的对照**

以下使用相同合格 val 患者及相同标签，双方均展示包含 age/sex/PC 的 wrapper 结果。历史行是既有报告中回顾性选出的 TabPFN-3 full 配置；不是 train＋val 合并五折 OOF 指标。

| 疾病 | 方案 | AUC | OR90 | Nagelkerke R² |
| --- | --- | --- | --- | --- |
| T2D | 历史 TabPFN-3 full | 0.756605 | 7.4402 | 0.236815 |
| T2D | 当前外部 PGS＋PC | 0.764410 | 7.1621 | 0.248623 |
| CAD | 历史 TabPFN-3 full | 0.806535 | 7.8361 | 0.356246 |
| CAD | 当前外部 PGS＋PC | 0.813016 | 7.8361 | 0.370372 |
| 乳腺癌（女性） | 历史 TabPFN-3 full | 0.668896 | 2.8718 | 0.103809 |
| 乳腺癌（女性） | 当前外部 PGS＋PC | 0.713365 | 4.5025 | 0.151991 |

| 疾病 | 当前 − 历史 ΔAUC | 95% 配对 CI | 原六项比较族调整 CI |
| --- | --- | --- | --- |
| T2D | +0.007805 | [+0.001365, +0.014533] | [-0.000670, +0.017361] |
| CAD | +0.006482 | [+0.002740, +0.010304] | [+0.001342, +0.011329] |
| 乳腺癌（女性） | +0.044469 | [+0.019207, +0.068871] | [+0.010990, +0.076313] |

当前 AUC 三项均上升，乳腺癌差值最大；但 **T2D OR90 从 7.4402 降到 7.1621，CAD OR90 保持 7.8361**，不能描述为所有指标均改善。T2D 的历史对照 AUC 区间在多重比较调整后跨零。历史配置经过 val 回顾性选择；新旧 PRS 库、OOF 折数不同，乳腺癌的 TabPFN checkpoint/估计器设置也不同。这些是探索性的整体预测流程对照，不能单独证明外部 PRS 或 ancestry-aware 机制的因果贡献。

**最新补充：新版离散 ancestry 替换实验**

保持新版 T2D 26、CAD 3、女性乳腺癌 3 个外部分数、age/sex、TabPFN checkpoint/16 estimators/seed 42 及官方 train→val 队列一致，仅用已有 1000G hard labels 的 one-hot 替换连续 PC 输入。五个参考类及 AMBIGUOUS 均保留，所有编码和标准化仅拟合 train/context。连续 PC 和无显式祖源输入两组复用已核验的同配置预测。

| 疾病 | 连续 PC AUC | 离散 ancestry AUC | 无显式祖源 AUC | ΔAUC（离散 − PC） | 95% 配对 CI | 三项比较调整 CI |
| --- | --- | --- | --- | --- | --- | --- |
| T2D | 0.764329 | 0.759363 | 0.758745 | -0.004966 | [-0.008562, -0.001234] | [-0.009426, -0.000510] |
| CAD | 0.812933 | 0.803136 | 0.802871 | -0.009797 | [-0.013539, -0.006403] | [-0.014477, -0.005592] |
| 乳腺癌（女性） | 0.715345 | 0.702015 | 0.700475 | -0.013330 | [-0.021898, -0.005426] | [-0.023522, -0.003469] |

三个疾病的离散组 AUC 均降低，绝对降幅约为 T2D 0.0050、CAD 0.0098、乳腺癌 0.0133；离散组点估计接近无显式祖源输入组。主要三项比较采用 2,000 次配对、结局分层 bootstrap，调整区间单项覆盖率 98.3333%，三个调整区间均低于零。这是当前冻结模型的条件不确定性，未预设“大幅下降”的非劣效界值。

主表使用 **TabPFN 直接概率**；官方 wrapper 会重新加入 PC，不能作为完整替换的证据。两套实际祖源表示的来源不同，本实验不是对同一组 PC 数值逐列分箱，也不能将性能差值全部归因于离散化。当前结果支持保留连续 PC；祖源亚组和无祖源输入对照仍为探索性。完整 OR90、R²、Brier、亚组区间及 wrapper 结果见 [ANCESTRY_REPRESENTATION_ABLATION.md](/gpfs/data/shenlab/Jiajian/HW/decipehr/docs/1010/ANCESTRY_REPRESENTATION_ABLATION.md)。

**为何主方案去掉 soft ancestry**

| 疾病 | PRS＋age/sex＋PC AUC | 额外加 soft AUC | ΔAUC（soft − PC） | 95% 配对 CI |
| --- | --- | --- | --- | --- |
| T2D | 0.764329 | 0.763905 | -0.000424 | [-0.001685, +0.000801] |
| CAD | 0.812933 | 0.811810 | -0.001124 | [-0.003216, +0.000879] |
| 乳腺癌（女性） | 0.715345 | 0.715506 | +0.000160 | [-0.004893, +0.005101] |

这三个差值均很小，95% 区间均跨零。按用户选择，主方案保留连续 PC，将 soft 输入作为已完成敏感性结果。soft ancestry 是五个参考祖源的连续相似度权重，不是祖源比例；“soft AUC”只是该输入版本的普通 AUROC，并非另一种 AUC 指标。去掉 soft 的主方案选择发生在检查 val 结果之后，属于开发决定，不应写成独立盲测前的预注册选择。

**与 DiscoDivas 的同输入对照**

| 疾病 | 对照双方共享的 PRS 输入 | 与主方案的区别 |
| --- | --- | --- |
| T2D | 25 个 D-PRISM 成分按发表系数组成 5 个祖源目标 metascore | 主方案为 26 个原始外部分数；对照中去掉 PGS002308，25→5 |
| CAD | GPS_Mult、MIXPRS+ 共 2 个分数 | 主方案另外保留 PGS004696；对照缺少该混合调优队列的可识别 PC 中心而排除它 |
| 乳腺癌 | AFR Overall_Prop_M1、EUR MegaPRS、EAS PRS-CS 共 3 个分数 | 分数清单相同，但对照增加统一 PRS–PC 校正 |

三种对照方法共享 train 拟合的 PRS–PC1–PC10 残差化与标准化、相同患者、age/sex/PC 和固定 split。DiscoDivas 用官方 R 插值组合得到一个分数，再接 ridge logistic；线性整合直接用共享成分分数；TabPFN 用同样的成分分数和协变量。D-PRISM 合并所需成分标准化也仅在对应 train/context 内拟合，发表组合系数固定。

| 疾病 | 方法（raw） | AUC | OR90 | Nagelkerke R² | Brier |
| --- | --- | --- | --- | --- | --- |
| T2D | DiscoDivas＋LR | 0.747072 | 6.0671 | 0.213098 | 0.175568 |
| T2D | 多 PRS 线性整合 | 0.747498 | 5.9593 | 0.212285 | 0.175553 |
| T2D | 同输入 TabPFN | 0.765747 | 7.7330 | 0.246752 | 0.169294 |
| CAD | DiscoDivas＋LR | 0.799862 | 6.1642 | 0.336306 | 0.175241 |
| CAD | 多 PRS 线性整合 | 0.800209 | 5.7983 | 0.336658 | 0.175133 |
| CAD | 同输入 TabPFN | 0.811875 | 7.8361 | 0.369667 | 0.169846 |
| 乳腺癌（女性） | DiscoDivas＋LR | 0.677535 | 2.7847 | 0.096930 | 0.148290 |
| 乳腺癌（女性） | 多 PRS 线性整合 | 0.685464 | 3.0530 | 0.107120 | 0.147111 |
| 乳腺癌（女性） | 同输入 TabPFN | 0.716607 | 4.3714 | 0.140865 | 0.143893 |

| 疾病 | 比较方向 | ΔAUC | 95% 配对 CI | 六项比较族调整 CI |
| --- | --- | --- | --- | --- |
| T2D | TabPFN − DiscoDivas | +0.018676 | [+0.011437, +0.026260] | [+0.009100, +0.029642] |
| T2D | TabPFN − 线性整合 | +0.018249 | [+0.011493, +0.025565] | [+0.009236, +0.028078] |
| CAD | TabPFN − DiscoDivas | +0.012014 | [+0.007208, +0.016723] | [+0.005887, +0.018311] |
| CAD | TabPFN − 线性整合 | +0.011666 | [+0.006885, +0.016559] | [+0.005721, +0.017852] |
| 乳腺癌（女性） | TabPFN − DiscoDivas | +0.039072 | [+0.020070, +0.058271] | [+0.013869, +0.065158] |
| 乳腺癌（女性） | TabPFN − 线性整合 | +0.031142 | [+0.015216, +0.047507] | [+0.009360, +0.054340] |

配对 bootstrap 使用 2,000 次按结局分层的患者重采样；三疾病 × 两个对照构成六项比较族，调整区间为 99.1667% Bonferroni percentile CI。区间条件于已存储的预测，不包含重新拟合或先前模型选择的不确定性。

同输入 TabPFN 对整体 AUC 的优势在三个疾病均成立。祖源亚组仍属探索性：T2D 的 SAS-like 差值为 −0.019342（95% CI −0.059584 至 +0.017805）；乳腺癌 SAS-like 为 −0.007353（−0.077206 至 +0.062500，42 人、8 病例）。不能推断每个祖源都获益；AMBIGUOUS 也不等同于已确认混合祖源。

**DiscoDivas 实现核验与原文配置进展**

官方插值源码保持不变；运行器修复了 `regress.PCA=FALSE` 分支对 data.frame 索引语义的兼容问题，并保持患者排序对应。独立 R 检查对同一输入、同一拟合范围重算校正与组合，误差如下：

| 疾病 | train 校正最大误差 | val 校正最大误差 | 原始 R TRUE 分支 train 组合误差 |
| --- | --- | --- | --- |
| T2D | 2.731e-14 | 2.931e-14 | 1.199e-14 |
| CAD | 7.550e-15 | 7.994e-15 | 7.994e-15 |
| 乳腺癌（女性） | 1.155e-14 | 1.199e-14 | 9.770e-15 |

D-PRISM 官方 R 函数的五个 metascore 最大误差为 **5.773e-15**。这验证了相同输入下的运算一致性，不等于作者原文模型或不同队列标准化方案完全相同。

现有对照使用 1000G 人群中心代理：作者载荷的共同 SNP 覆盖为 53,571/59,529（89.9914%），未通过原先 90% 门槛，因此在疾病评估前仅用 2,503 名外部 1000G 样本和共同标记重拟合 PCA，再将目标患者投影到同一坐标系；目标患者未参与 PCA 拟合。这个坐标系和作者发布载荷的坐标系不同。

正式论文的真实性状使用 LDpred2-grid（102 组/GWAS），CAD 7 组、T2D 8 组发现 GWAS，All of Us 内按 AFR/EAS/EUR/SAS 调优后插值；没有乳腺癌原文实验。**本次已从正式补充数据找到并提取 All of Us 四祖源 PC 中心**，更新此前“未找到实际调优中心”的状态。但尚未在已检查的补充材料、作者仓库、release/issue 或 PGS Catalog 中找到 CAD/T2D 共 **8 套冻结成品模型**，或足以恢复它们的最终超参数、组合系数和标准化参数。因此原文模型评估状态是 **NOT RUN**，没有新原文配置 AUC。[正式论文](https://doi.org/10.1016/j.ajhg.2026.05.006)

已找到的 All of Us 中心不能直接套到当前重拟合 PC 坐标系，也不能任意用于来自其他调优队列的替代模型。官方默认 `--regress.PCA TRUE` 在传入目标队列做不使用标签的校正；本项目采用 train 拟合后应用于 val，是明确的部署适配。取得作者冻结权重后，仍需对齐作者 PC 投影，并分别说明冻结模型部署与目标队列校正的评估范围。

**验证状态与可追溯文件**

本次重新核对了输入/输出 SHA256、主模型选择记录、合格患者和标签一致性，并从已保存预测重算既有 **72 个整体“方法 × 疾病 × 概率版本”结果行**的 AUC、OR90、R²、Brier。既有配对区间沿用存储结果并复核点差；新增离散 ancestry 分析另行核验 126 个整体/亚组结果行并重新计算配对 bootstrap。实时 Slurm 记录确认主评估数组 `28316248`、DiscoDivas 数组 `28319084` 和新版离散祖源数组 `28327660` 的九个任务均为 COMPLETED、ExitCode 0:0。

本次代码验证为 **189 passed、2 skipped**，全项目 Ruff 通过。两个跳过项分别是需显式启用的真实 PLINK2/PGS Catalog 和官方 DiscoDivas R 集成测试；本次没有改动相应计分或 R 实现。新增测试验证离散输入编码仅来自 train，并且 TabPFN 直接预测不受被排除的 PC、soft 或 val 标签影响。

| 内容 | 文件 |
| --- | --- |
| 当前主方案选择 | [selection.json](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/pc_primary/selection.json) |
| 完整外部模型清单和元数据 | [weights.scored.lock.json](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/run_20261010/scoring_v2/weights.scored.lock.json) |
| 主方案全部方法/亚组指标 | [summary_metrics.csv](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/run_20261010/report/summary_metrics.csv) |
| 新版连续/离散/无祖源对照 | [ANCESTRY_REPRESENTATION_ABLATION.md](/gpfs/data/shenlab/Jiajian/HW/decipehr/docs/1010/ANCESTRY_REPRESENTATION_ABLATION.md)；[VERIFIED.json](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/hard_ancestry_20261010/report/VERIFIED.json) |
| 历史同队列比较 | [matched_metrics.csv](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/run_20261010/comparison_with_previous_official_validation/matched_metrics.csv)；[paired_auc_comparisons.csv](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/run_20261010/comparison_with_previous_official_validation/paired_auc_comparisons.csv) |
| DiscoDivas 对照完整报告（含 wrapper 与亚组） | [REPORT.md](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/discodivas/report_20261010/REPORT.md) |
| 官方 R 数值核验 | [NUMERICAL_AUDIT.json](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/discodivas/implementation_audit_20261010/NUMERICAL_AUDIT.json)；[DPRISM_AUDIT.json](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/discodivas/implementation_audit_20261010/DPRISM_AUDIT.json) |
| 原文资源最新状态 | [REPORT.md](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/discodivas/original_configuration_20261010/REPORT.md)；[PREFLIGHT.json](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/discodivas/original_configuration_20261010/PREFLIGHT.json) |
| 原文配置及所缺模型 | [paper_configuration.json](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/discodivas/original_configuration_20261010/paper_configuration.json)；[required_author_models.csv](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/discodivas/original_configuration_20261010/required_author_models.csv) |
| 本报告核验与实时任务状态 | [VERIFIED.json](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/final_summary_20261010/VERIFIED.json)；[live_jobs.psv](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/final_summary_20261010/live_jobs.psv) |
| 本报告重建脚本 | [build_report.py](/gpfs/data/shenlab/Jiajian/HW/decipehr/outputs/external_pgs/final_summary_20261010/build_report.py) |

**结论的适用范围与后续工作**

目前可以表述为：“在 decipEHR 的既有官方开发验证集上，冻结外部 PRS 与 age/sex/PC 联合输入的 TabPFN 显示良好的预测表现；在统一分数输入和预处理的比较中，整体 AUC 高于当前 DiscoDivas＋logistic 适配方案。”

官方 val 已用于此前开发与方案选择，因此不是新获得的、完全未触碰的独立测试集。不能据此声称超过 DiscoDivas 原文模型、所有祖源均改善、或增益全部来自祖源感知机制。主要待办是取得作者冻结模型后完成原文模型部署对照，并在新的独立测试队列确认泛化；进一步模型或特征选择继续限于 train。SPLENDID 暂未纳入实际运行。
