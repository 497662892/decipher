# External PRS 整合两日进展与下一阶段方案

整理日期：2026 年 10 月 12 日。覆盖 10 月 10 日的 External PRS 主流程及祖源表示实验、10 月 11 日原定人群的修正比较，以及随后完成的 T2D D-PRISM25 补充比较。

版本说明：本文保留当时的 26/3/3 面板快照。后续 CAD 扩展、Ho2022 BC 扩展及 26/7/5 hold-out wrapper 流程见[最新执行说明](Ho2022_BC_and_External_PRS_Holdout_20261012_CN.md)。

**目前建立的是一个有开发集支持的 External PRS 整合系统。**已经观察到 PRS 相对协变量的增量、连续 PC 相对当前离散祖源表示的优势，以及完整 TabPFN 相对训练内选定常规基线的整体优势。尚未证明完整多 PRS 必然优于一个强 PRS，也未证明显式祖源调权带来额外收益。软标签蒸馏和独立外部队列验证尚未完成。

这里的 External 有两个含义，必须区分：**外部 PRS 权重已经具备并在使用；独立外部验证队列目前没有。**我们的 development 是此前已经用于开发的官方 validation，不能把它重新命名成独立 external test。

**1 选择了哪些 External PRS 以及为什么**

这次采用作者公开的成品 SNP 权重，以多祖源覆盖、不同来源及成熟方法的互补性为主要选取理由，并要求能下载、核对表型和 genome build、对齐等位基因且通过既定 SNP 匹配覆盖门槛。这是一套有理由的候选面板，不是穷尽所有已发表 PRS 后证明的最优集合。发现 GWAS 祖源、模型调优祖源、LD 参考祖源和我们评估的人群分别记录。本文沿用资源中的简称：AFR 非洲祖源、AMR 美洲混合祖源、EAS 东亚祖源、EUR 欧洲祖源、SAS 南亚祖源。

| 疾病与模型 | 实际输入数 | 方法与发现数据 | 作者的调优或参考人群 | 在本面板中的作用 |
|---|---:|---|---|---|
| T2D D-PRISM，PGS005353–PGS005377 | 25 个成分 | PRS-CSx；AFR、AMR、EAS、EUR、SAS 五类 GWAS | 按五个目标祖源分别调优，见下表 | 提供多祖源来源与目标调优的成分，允许下游直接整合 |
| T2D PRScsx_T2D，[PGS002308](https://www.pgscatalog.org/score/PGS002308/) | 1 | PRS-CSx auto/meta；EUR 898,130、EAS 日本人群 177,415、AFR 23,827 | Catalog 未单列独立表型调优队列 | 补充另一套已发表的多祖源综合分数 |
| CAD GPS_Mult，[PGS003725](https://www.pgscatalog.org/score/PGS003725/) | 1 | LDpred2 后整合多祖源、多性状信息；Catalog 登记发现样本 1,443,386，含 AFR/EAS/EUR/AMR/SAS | UK Biobank 欧洲祖源 116,649 人，4,412 病例 | 多祖源发现数据、欧洲祖源调优的强综合模型 |
| CAD multi_anc_hg37CSx，[PGS004696](https://www.pgscatalog.org/score/PGS004696/) | 1 | PRS-CSx；BBJ、CARDIoGRAMplusC4D、MVP、UKB 等 EUR/AFR/AMR/EAS GWAS | MVP 87,724 人；含 EUR 56,359、AFR 16,987、AMR 9,992、EAS/SAS 合计 4,386 | 补充混合祖源调优的综合分数 |
| CAD_EAS_MIXPRSplus，[PGS012558](https://www.pgscatalog.org/score/PGS012558/) | 1 | MIXPRS+ 综合模型；Catalog 方法项含 JointPRS、SDPRX、SBayesRC、ClinicalPRS；EUR 86,995 与 BBJ EAS 212,453 | EAS 目标；Catalog 未单列独立调优队列，不能将 EAS 评估集写成调优集 | 提供面向东亚的综合模型；不是一个可拆分的独立 EAS 成分 |
| BC Overall_Prop_M1，[PGS005378](https://www.pgscatalog.org/score/PGS005378/) | 1 | FSS、PRSice-2、elastic net；AFR 女性发现样本 25,334，12,174 病例 | AFR 女性 28,953 人，13,913 病例；AABC、AABCG、AMBER、BCAC、GBHS、ROOT | 增加非洲祖源女性的专门模型 |
| BC INTERVENE_MegaPRS，[PGS004869](https://www.pgscatalog.org/score/PGS004869/) | 1 | MegaPRS auto；EUR 乳腺癌 GWAS，Catalog 登记 139,274 人 | Catalog development 字段的 404 名非芬兰欧洲祖源 1000G 样本是遗传参考，不应写成 404 名乳腺癌病例训练集 | 提供欧洲祖源的 genome-wide 预测模型 |
| BC TPMI PRS-CS，[PGS018538](https://www.pgscatalog.org/score/PGS018538/) | 1 | PRS-CS，phi=1e-5；台湾汉族 TPMI GWAS 326,227 人，12,566 病例 | TPMI 汉族调优样本 73,949 人，4,385 病例 | 增加东亚来源的模型 |

各模型的人群和样本数来自实际评分时冻结的 PGS Catalog metadata；不是我们训练集的人数，也不能跨模型相加为独立发现样本量。[冻结资源人群明细](../../outputs/external_pgs/progress_review_20261012/external_resource_populations.json)保留了完整队列名称、发现与调优字段。MegaPRS 的算法与研究流程见[原文方法](https://www.nature.com/articles/s41467-024-48938-2)；TPMI 的汉族发现及独立 PRS 调优设计见[原文方法](https://www.nature.com/articles/s41586-025-09350-y)。TPMI 原始表型较宽，Catalog 记录中包含男性对照；本项目的乳腺癌训练和评估始终限于合格女性。

D-PRISM 的 25 个成分表示 **5 个调优目标 × 每个目标的 5 个来源成分**，并非 25 个独立 GWAS 或 25 个最终成品模型。五个来源 GWAS 的 Catalog 样本量分别为 AFR 128,695、AMR 82,013、EAS 400,683、EUR 1,486,934、SAS 87,223；涉及 MVP、BioMe、SIGMA、BBJ、UKB、FinnGen、LOLIPOP 等不同队列。每套目标模型均利用五种来源的信息。来源字段及队列见 [PGS005353](https://www.pgscatalog.org/score/PGS005353/)、[PGS005354](https://www.pgscatalog.org/score/PGS005354/)、[PGS005355](https://www.pgscatalog.org/score/PGS005355/)、[PGS005356](https://www.pgscatalog.org/score/PGS005356/)、[PGS005357](https://www.pgscatalog.org/score/PGS005357/)。

| 调优目标 | 五个成分的 PGS ID | 外部调优队列 | 人数 | 病例 |
|---|---|---|---:|---:|
| AFR | PGS005353–PGS005357 | BioVU，非洲裔美国人 | 6,779 | 1,409 |
| AMR | PGS005358–PGS005362 | SIGMA，Hispanic American/Mexican | 5,747 | 2,964 |
| EAS | PGS005363–PGS005367 | BBJ，日本人群 | 2,837 | 906 |
| EUR | PGS005368–PGS005372 | BioVU，欧洲裔美国人 | 21,706 | 3,694 |
| SAS | PGS005373–PGS005377 | LOLIPOP，British Indian | 5,715 | 2,019 |

例如，`trainedforAFR_AMRweights` 表示面向 AFR 调优模型里的 AMR 来源成分；不能把两个标签合并成“这是 AFR 的 GWAS”。作者也发布了把每套五个成分组合成一个 metascore 的固定系数。我们主模型直接输入全部 25 个成分；只有相应基线才使用五个发表组合。调优队列字段可分别查询 [AFR](https://www.pgscatalog.org/score/PGS005353/)、[AMR](https://www.pgscatalog.org/score/PGS005358/)、[EAS](https://www.pgscatalog.org/score/PGS005363/)、[EUR](https://www.pgscatalog.org/score/PGS005368/)、[SAS](https://www.pgscatalog.org/score/PGS005373/) 条目。[D-PRISM 官方说明](https://github.com/ahuertach/D-PRISM-T2D-PRS)

最终输入为 **T2D 26、CAD 3、女性乳腺癌 3**。共计 34 个候选、32 个通过既定 75% SNP 覆盖门槛：

| 模型 | 实际匹配 SNP 占发表 SNP |
|---|---:|
| D-PRISM 25 成分 | 94.83%–98.86% |
| T2D PGS002308 | 97.58% |
| CAD GPS_Mult / multi_anc_hg37CSx / MIXPRS+ | 97.33% / 97.56% / 98.49% |
| BC Overall_Prop_M1 / MegaPRS / TPMI | 84.14% / 96.95% / 98.07% |
| 排除的 BC PRS313，PGS000004 | 67.73% |
| 排除的 BC PGS005379 | 74.98% |

两个 BC 候选因覆盖不足排除后，MegaPRS 与 TPMI 在查看本轮结局表现前作为 amendment 加入，75% 门槛没有更改。计分统一到 GRCh38，用 Catalog 匹配工具与 PLINK2 完成，缺失基因型使用固定外部参考频率。权重冻结，但目标数据未覆盖所有发表 SNP，故不等于完整 SNP 集的原队列复现。覆盖率是可计分性检查，不是性能合格证。[实际覆盖记录](../../outputs/external_pgs/progress_review_20261012/external_resource_inventory.csv)

**2 当前整合方式及其相对历史方案的表现**

已完成的主模型是 **External PRS + age + sex + 连续 PC1–PC10 → TabPFN-3 → 疾病状态概率**。每个人均输入该疾病全部入选分数，不按祖源先选一个分数。TabPFN 使用原定 v3 default checkpoint、16 estimators、seed 42；女性 BC 的恒定性别特征移除。当前模型允许风险映射随 PC 改变，但没有输出已经可解释、已验证有效的显式祖源权重函数。这样的权重函数是下一阶段学生模型要检验的内容。

```mermaid
flowchart LR
    W[作者发布的冻结 SNP 权重] --> S[PLINK2 为每人计算全部入选 PRS]
    S --> X[PRS 加年龄 性别 连续 PC]
    T[官方 training 特征及标签] --> F[TabPFN 整合器]
    X --> F
    F --> P[原定 development 概率及评价]
```

| 疾病 | Training 人数 / 病例 | Development 人数 / 病例 |
|---|---:|---:|
| T2D | 5,811 / 1,701 | 2,852 / 847 |
| CAD | 5,811 / 2,313 | 2,852 / 1,072 |
| 女性乳腺癌 | 3,308 / 602 | 1,628 / 320 |

PRS 权重来自外部；插补、标准化、下游模型及参数选择使用 training。development 只进入预测和性能计算，但它已参与整个研究的历史开发决策，因此这里报告开发证据。乳腺癌按 `configs/disease_eligibility.yaml` 限定资格。下表为主模型的**直接概率**，不是 PRS 单独预测的性能：

| 疾病 | AUC | OR90 | Nagelkerke R² | Brier | Log loss |
|---|---|---|---|---|---|
| T2D | 0.764329 | 6.6460 | 0.247488 | 0.169630 | 0.512604 |
| CAD | 0.812933 | 8.0204 | 0.372462 | 0.169302 | 0.502345 |
| 女性乳腺癌 | 0.715345 | 4.6375 | 0.140703 | 0.143491 | 0.449274 |

OR90 是风险最高 10% 相对其余 90% 的 odds ratio；这里的 Nagelkerke R² 来自预测概率相对截距模型的似然，包括协变量贡献。它不是 PRS 解释的遗传率，也不能与 Disco 原文 covariate-adjusted full/null R² 直接比数值。当前概率反映本数据的疾病状态，不是十年发病风险。

“之前自己训练”应准确称为**基于官方提供的外部 GWAS 汇总统计自行构建 PRS**。它使用 P+T、LDpred2、lassosum2、PRS-CS、SDPR、MegaPRS、SBayesRC 等工具，随后进行下游整合；不是用我们几千名患者重新做 GWAS。原始 GWAS 分别登记为 CAD GCST003116、BC GCST90479801、T2D GCST90479885。[官方汇总统计配置](../../configs/official_sumstats.yaml)

为比较新旧系统，下表双方均使用原定同一 development 患者与协变量 logistic wrapper，不能拿这里的 wrapper 数字与上表直接概率交叉拼接。历史 TabPFN 最佳观测面板分别为 T2D 8 个、CAD 4 个、BC 5 个自行构建的 PRS；具体方法和 checkpoint 见[历史比较表](../../outputs/official_validation_tables/FINAL_METHOD_COMPARISON.md)。

| 疾病 | 完整流程及 wrapper | AUC | OR90 | Nagelkerke R² |
|---|---|---|---|---|
| T2D | 此前自行估计 PRS 后整合 | 0.756605 | 7.4402 | 0.236815 |
| T2D | 当前作者冻结 PRS 后整合 | 0.764410 | 7.1621 | 0.248623 |
| CAD | 此前自行估计 PRS 后整合 | 0.806535 | 7.8361 | 0.356246 |
| CAD | 当前作者冻结 PRS 后整合 | 0.813016 | 7.8361 | 0.370372 |
| 女性乳腺癌 | 此前自行估计 PRS 后整合 | 0.668896 | 2.8718 | 0.103809 |
| 女性乳腺癌 | 当前作者冻结 PRS 后整合 | 0.713365 | 4.5025 | 0.151991 |

| 疾病 | 当前 − 历史 ΔAUC | 95% 配对区间 | 原六项比较族同时区间 |
|---|---|---|---|
| T2D | +0.007805 | [+0.001365, +0.014533] | [-0.000670, +0.017361] |
| CAD | +0.006482 | [+0.002740, +0.010304] | [+0.001342, +0.011329] |
| 女性乳腺癌 | +0.044469 | [+0.019207, +0.068871] | [+0.010990, +0.076313] |

三病 AUC 点估计都提高，女性 BC 的差值最大。T2D OR90 则由 7.4402 降至 7.1621，CAD OR90 保持 7.8361；不能说所有指标都提高。T2D 的新旧 AUC 差异在原六项比较族调整后跨零。历史配置由过去 development 表现回顾性选出；新旧 PRS 面板、wrapper OOF 折数不同，BC 的 TabPFN checkpoint 与估计器设置也不同。因此这是**整套预测流程的历史比较**，不能隔离“换用作者成品权重”的单独因果贡献。[新旧配对证据](../../outputs/external_pgs/progress_review_20261012/historical_wrapper_auc_comparisons.csv)

在当前冻结 External PRS 面板内，还完成了以下同队列、直接概率对照：

| 当前 External PRS 下游模型 | T2D AUC | CAD AUC | 女性 BC AUC |
|---|---|---|---|
| TabPFN 仅协变量 | 0.727311 | 0.801358 | 0.652468 |
| TabPFN 最佳单 PRS | 0.764198 | 0.812371 | 0.712538 |
| TabPFN 完整面板 | 0.764329 | 0.812933 | 0.715345 |
| 完整面板 Logistic | 0.746165 | 0.801329 | 0.685964 |
| 灵活基础项加固定 PRS 系数 | 0.753873 | 0.803683 | 0.695054 |
| CatBoost | 0.746058 | 0.805209 | 0.699331 |
| Training 内选定常规基线 | 0.746058 | 0.800093 | 0.695054 |

Training 内选定的常规基线为 T2D CatBoost、CAD 最佳单 PRS Logistic、BC 灵活固定系数模型。它们由 training 内层结果决定，不能在看到 development 后改成该列最高的常规算法。完整 TabPFN 相对这三个选定基线的 ΔAUC 分别为 +0.018271、+0.012841、+0.020291；均通过与 Disco 比较合计六项主要检验的 Holm 校正。表中也展示了其他常规方法，避免只报告一个选择后较弱的对照。

**3 与 DiscoDivas AJHG 2026 的比较**

DiscoDivas 按个体与 PRS 调优队列在共同 PC 空间中的距离，连续组合多个 PRS。AJHG 2026 原文的 CAD/T2D 实验使用外部 GWAS、LDpred2-grid 和 All of Us 祖源分层调优后的模型；其真实数据实验没有女性乳腺癌。本文比较的是**该整合框架在我们已有外部分数上的适配**。[正式论文](https://doi.org/10.1016/j.ajhg.2026.05.006)、[官方代码](https://github.com/YunfengRuan/DiscoDivas)

**T2D 主比较使用直接输入全部 25 个 D-PRISM 成分的 TabPFN。**现有 Disco 适配使用同一批 25 个成分按作者发表系数组成的五个 metascores；因此两者共享原始 PRS 来源，但中间表示不同。五 metascores TabPFN 仅作为下方的辅助压缩对照。CAD 使用 GPS_Mult 与 EAS MIXPRS+ 两个分数，因当前缺少 PGS004696 混合调优队列可识别的 PC 中心而不将其放入这组匹配比较；BC 使用原来的三个分数。对照双方共享患者、1000G refPC、PRS 的训练内 PC 残差化和缩放、年龄与性别。Disco 的线性或样条风险链接及正则化由 training 内选择，三个疾病均选择样条链接。

| 疾病 | TabPFN 输入 | Disco 适配输入 | Disco 适配 AUC | TabPFN AUC | TabPFN − Disco | 95% 配对区间 |
|---|---|---|---|---|---|---|
| T2D | 直接输入 25 个 D-PRISM 成分 | 同 25 成分按发表系数组成 5 个 metascores | 0.752842 | 0.763921 | +0.011080 | [+0.004485, +0.018260] |
| CAD | 2 个分数 | 同 2 个分数 | 0.801932 | 0.808205 | +0.006274 | [+0.002590, +0.010251] |
| 女性乳腺癌 | 3 个分数 | 同 3 个分数 | 0.690341 | 0.707446 | +0.017105 | [+0.002248, +0.032476] |

**T2D 直接 25 成分的 ΔAUC 为 +0.011080，普通 95% 配对区间及本项三比较调整后的同时区间均高于零**，后者为 [+0.003153, +0.019798]。CAD、BC 仍是相同中间 PRS 输入的比较，在原六项主要检验中 Holm p 分别为 0.0100、0.0470。本表汇集已完成的冻结比较；T2D 补充实验与 CAD/BC 的检验族不同，不把它们写成一次共同校正。“差异不明确”仅适用于下方五 metascores TabPFN 对 Disco 的辅助对照，不能再作为 T2D 主比较的结论。主模型 T2D 26 个、CAD 3 个分数的 AUC，也不能直接代替本表控制输入来源和预处理的数字。

为解释 T2D 主比较的增益，下面保留同一批 25 个 D-PRISM 来源的三流程分解。第一行是主比较使用的直接 25 成分版本，第二行是辅助的五 metascores 压缩对照。三者均使用相同参考 PC 和训练内预处理；只有表示及整合方法不同：

| 同一 25 成分来源的流程 | Development AUC | Development Brier | Training 五折 OOF AUC |
|---|---|---|---|
| 25 成分直接进入 TabPFN | 0.763921 | 0.169905 | 0.743794 |
| 5 个发表 metascores 进入 TabPFN | 0.756701 | 0.172584 | 0.739297 |
| 5 个发表 metascores 进入 Disco | 0.752842 | 0.173503 | 0.734632 |

| Development 比较 | ΔAUC | 95% 配对区间 | 本项三比较同时区间 |
|---|---|---|---|
| 直接 25 成分 TabPFN − Disco | +0.011080 | [+0.004485, +0.018260] | [+0.003153, +0.019798] |
| 直接 25 成分 TabPFN − 5 metascores TabPFN | +0.007220 | [+0.004048, +0.010366] | [+0.003298, +0.011038] |
| 5 metascores TabPFN − Disco | +0.003859 | [-0.002047, +0.009962] | [-0.003163, +0.011335] |

**五 metascores 是复用作者完整模型的适配选择，不是 Disco 的普遍输入上限。**25→5 保留了每个原始成分的贡献，但固定了组内相对系数，压缩了可独立利用的信息。直接 25 成分比五个发表组合更适合当前 TabPFN 的结果，有普通和三项调整区间支持；直接 25 成分对 Disco 的结果则是整个流程的比较，不能单独归因于 Disco 的祖源插值机制。允许 25 个成分独立学习系数、同时利用 Disco 几何权重的扩展尚未运行，按目前决定暂缓。

上表的 training 五折 pooled OOF 与完整 training 拟合后预测 development 的样本、训练上下文和评估定义不同，单列供内部稳定性检查，不用较低的 OOF 数字替换主性能。三流程共享患者级五折；query 标签不进入预处理、拟合或选择。所有已有 metascore/Disco development 概率均在补充实验中精确复现。[补充实验详报](../1011/T2D_DPRISM25_Comparison_20261011.md)

总体优势并非每个祖源组都成立：T2D 直接 25 成分相对 Disco 在 SAS-like 的 AUC 为 0.704836 对 0.751644，ΔAUC −0.046809，名义 95% CI [−0.087041, −0.008511]；该组只有 102 人、55 病例，分组比较未控制全部探索检验。需要保留这一反向结果，而不是以整体 AUC 声称祖源间普遍改善。

当前比较还采用 1000G 代理中心，不是作者各模型实际调优人群中心。10 月 10 日资源审计已找到原文 All of Us 四祖源 PC 中心，但在所检查公开资源中未找到 CAD/T2D 共八套冻结成品模型及完整重建参数；原文资产版本在本项目的状态仍为 **NOT RUN**。公开 AoU 中心也不能直接套进我们另行拟合的 PC 坐标系。因此没有“超过 Disco 原文全部模型”的结果。[原文资产审计](../../outputs/external_pgs/discodivas/original_configuration_20261010/REPORT.md)

**4 已完成的消融分别说明什么**

以下数值均为左侧变更相对参照的 development ΔAUC，括号为普通 95% 配对区间。前四行来自修正后的同队列比较；最后两行来自各自冻结的祖源表示实验，bootstrap 随机种子/比较族不同，不把区间混成同一次检验。

| 已完成问题及差值方向 | T2D ΔAUC [95% CI] | CAD ΔAUC [95% CI] | 女性 BC ΔAUC [95% CI] |
|---|---|---|---|
| 加入 External PRS，相对仅协变量 | +0.037018 [+0.024915, +0.049375] | +0.011576 [+0.006812, +0.016415] | +0.062877 [+0.035418, +0.087438] |
| 完整面板，相对最佳单 PRS | +0.000131 [-0.003083, +0.003695] | +0.000562 [-0.001454, +0.002706] | +0.002807 [-0.007442, +0.013659] |
| 连续 PC，相对无显式祖源输入 | +0.005583 [+0.001502, +0.009366] | +0.010063 [+0.006338, +0.013919] | +0.014870 [+0.003713, +0.026193] |
| 祖源可变 PRS 系数，相对固定系数 | -0.000437 [-0.001712, +0.000888] | +0.000097 [-0.000223, +0.000393] | +0.000189 [-0.000446, +0.000814] |
| Hard labels 替换连续 PC | -0.004966 [-0.008562, -0.001234] | -0.009797 [-0.013539, -0.006403] | -0.013330 [-0.021898, -0.005426] |
| 在连续 PC 之外加入 soft ancestry | -0.000424 [-0.001685, +0.000801] | -0.001124 [-0.003216, +0.000879] | +0.000160 [-0.004893, +0.005101] |

逐项解释：

1. **PRS 是否提供额外信号：有支持。**完整 TabPFN 相对仅协变量 TabPFN，三个疾病的差值区间均高于零。这说明外部分数提供了当前系统可利用的预测信息，不能进一步称为遗传因果效应。
2. **是否必须很多个 PRS：未证明。**完整面板相对最佳单 PRS TabPFN 的差值很小、三个区间都跨零。最佳单分数由 training 选择：T2D 是 D-PRISM 的 EUR 目标 metascore，CAD 是 PGS003725，BC 是 PGS004869。T2D 的“单分数”本身融合五个来源，不是单一祖源 GWAS。未检出优势也不等于已证明两种模型等效。
3. **是否保留连续 PC：有支持。**相对不输入显式祖源特征，连续 PC 有增益。去掉 PC 后 PRS 本身仍可能带有祖源相关信息，不能把无 PC 版本称为完全无祖源信号。
4. **离散标签能否替代 PC：当前结果不支持。**Hard labels 替换后，三个疾病的 AUC 下降，三项 Bonferroni 同时区间分别为 [−0.009426, −0.000510]、[−0.014477, −0.005592]、[−0.023522, −0.003469]。标签来自 1000G 相似性判断，与原定临床 PC 来源不同；这不是对完全相同 PC 数值的纯分箱实验，不能把全部差值归因于离散化。
5. **PC 之外再加 soft ancestry：未发现额外收益。**三个区间均跨零。soft 是参考祖源相似度，不是祖源比例。去掉 soft 的主方案是在看过 development 后决定的开发选择，不能改写成盲测前预注册。
6. **PRS 权重随祖源变化是否有帮助：当前显式模型未显示。**比较的是共享灵活基础项的固定系数与 PRS×PC 可变系数模型；三个差值均接近零。这不能证明 TabPFN 不存在交互，也不能用“PC 有用”反过来证明“PRS 祖源调权有用”。
7. **25 成分能否无损压成五个发表组合：当前结果不支持。**T2D 同预处理 TabPFN 的直接 25 成分 AUC 更高。它回答的是这个特定固定压缩的预测代价，不能推出所有降维都无效；原 26 输入主模型与补充 25 输入模型同时改变了 PC/预处理，所以两者的微小差值不能解释为 PGS002308 的独立贡献。

前述结果能支持的主张是：**冻结外部 PRS 配合连续 PC 的 TabPFN 整合，在当前 development 有预测价值；它超过若干常规整合基线，并在部分匹配面板上超过当前 Disco 适配。**当前仍不能将“多个 PRS 的必要性”“祖源自适应权重的机制收益”“各祖源均改善”“跨队列迁移成功”写成已证实结论。

已完成的附加工作还包括输入和运行配置审计、原模型逐概率复现、概率质量和祖源亚组评价、评分冗余及 QC/亲缘敏感性检查。此前额外 QC/亲缘版本改变了人群和 context，其 T2D/CAD/BC training OOF 0.7369/0.7863/0.6827 不属于上面原定 development 主性能。敏感性结果保留原始范围，不用于决定主线成败，也不复用为当前学生软标签。

概率校准同样需要保留：完整教师校准斜率约为 T2D 0.978、CAD 1.000、BC 1.067，但 BC 平均预测概率 0.1560 低于本 development 观察比例 0.1966。斜率接近 1 不代表概率整体水平已准确，下一步学生评价不能只比较 AUC。

**5 下一步按什么顺序推进**

**第一步，固定主线和参照。**教师继续使用当前 26/3/3 个外部分数、原定连续 PC 和资格人群；保留最佳单 PRS TabPFN、固定系数整合、当前 Disco 适配三个有不同作用的参照。五 metascores 放在对应适配/压缩分析内，不要求所有主模型先压成五维。现有每疾病 18 个评估配置已经覆盖主要硬标签问题，不再无目的扩大算法清单。

**第二步，先验证小模型蒸馏，而不是先假定祖源 gate 必须成功。**首轮学生保留完整原始 PRS 输入。小模型的“压缩”指把 TabPFN 的预测映射变成可独立推理的小模型，不必等于减少 PRS 数目。

| 首轮学生 | 遗传项 | 训练目标 | 回答的问题 |
|---|---|---|---|
| 固定系数学生 | Σ βj Zj | 真实标签；真实/教师软标签混合 | 固定整合能否保留教师能力；软标签有没有帮助 |
| 祖源可变系数学生 | Σ (βj + γjᵀPC) Zj | 与固定系数学生相同 | 在相同基础风险项和损失下，祖源调权能否增加收益 |

两类学生共享年龄/PC 灵活基础项和性别处理。对 γ 加较强收缩，使数据不支持调权时可以退回固定系数。先使用成熟的正则化 Logistic 求解，有符号系数允许负值；暂不增加复杂神经网络或强制 softmax 比例解释。

软标签起点为教师交叉拟合概率 q，混合目标为 `(1−λ)y + λq`，只预设 λ∈{0, 0.5, 1}。λ=0 是不可缺少的同结构真实标签对照。学生参数预先按内层 log loss 选择，AUC、OR90、R²、Brier 和校准统一报告；避免根据 development 挑一个漂亮指标。

**第三步，用共享五折和严格的教师交叉拟合评价。**同一患者只属于一个外层查询折，各方法共享该划分。为学生内层训练样本生成软标签的教师，context 也必须完全限制在该内层训练集，不能看到学生内层验证标签。一次全 training 教师 OOF 缓存不能自动用于所有学生内层选择。按现有五折外层、三折学生内层和三折教师交叉拟合设计，每疾病约 65 次教师拟合；可在相同训练范围、分数面板、checkpoint 和预处理哈希下复用缓存。正式计算前先做一个上下文的运行时间与内存试测。这是下一阶段计划，本轮未启动学生训练。[详细蒸馏设计](../1011/Ancestry_Gated_PRS_Distillation_Design_20261011.md)

**第四步，以三个独立问题决定是否继续。**

| 问题 | 主要判据 | 可能的决定 |
|---|---|---|
| 学生能否替代教师 | 预先约定的可接受性能损失、成对区间、推理时间、内存、模型大小 | 即使没有提高 AUC，也可在保留足够性能且降低成本时成为部署方案 |
| 蒸馏本身有没有价值 | 同结构 λ>0 对 λ=0 的结局指标与教师保真度 | 软标签没有帮助时，保留硬标签小模型，不强作蒸馏增益主张 |
| 祖源调权是否必要 | 同损失、同基础项下可变系数对固定系数 | 无稳定增量时采用固定系数；有增量再研究非负 gate 或更高容量 |

保真度可用概率 MAE、Bernoulli KL 与排序一致性描述，但不能替代真实结局评价。进一步保存多 seed 重训练的预测稳定性、各祖源样本/病例数与校准；由于 PRS 高相关，单个系数不直接解释为生物学贡献。SAS-like 等小组的反向结果需要优先核查稳定性，不能靠反复查看同一 development 挑到正结果。

独立 external cohort 暂缺时，可以完成上述训练内开发、跨折复现和部署成本评价。重复划分无法使已经使用过的 development 恢复盲测属性；真正的迁移性和临床应用主张仍留给未来独立队列。当前最值得推进的成果是**一个保留教师预测能力、可独立推理且明确说明祖源调权是否有用的学生模型**。

**证据与完成状态**

| 工作 | 状态 | 主要证据 |
|---|---|---|
| External PRS 筛选、对齐与评分 | 已完成，32 个有效分数 | [资源清单](../../outputs/external_pgs/progress_review_20261012/external_resource_inventory.csv) |
| 原定人群的三疾病比较 | 已完成，每疾病 18 个评估配置 | [54 行性能](../../outputs/external_pgs/progress_review_20261012/current_development_metrics.csv)、[配对结果](../../outputs/external_pgs/progress_review_20261012/current_auc_comparisons.csv) |
| 相同 D-PRISM25 来源的三流程及共享五折 | 已完成 | [Development 与 OOF](../../outputs/external_pgs/progress_review_20261012/dprism25_metrics.csv)、[配对结果](../../outputs/external_pgs/progress_review_20261012/dprism25_auc_comparisons.csv) |
| 祖源表示、最佳单分数、显式可变系数等消融 | 已完成，按第 4 部分解释 | [Hard ancestry](../../outputs/external_pgs/progress_review_20261012/hard_ancestry_comparisons.csv)、[Soft ancestry](../../outputs/external_pgs/progress_review_20261012/soft_ancestry_comparisons.csv) |
| 原文冻结 Disco 成品模型评估 | 尚未运行，已审计资源不足 | [资源审计](../../outputs/external_pgs/discodivas/original_configuration_20261010/REPORT.md) |
| 无预设五维压缩的 Disco 扩展 | 尚未运行，暂缓 | 属于独立扩展方法，不计入当前性能 |
| 教师软标签缓存与学生蒸馏 | 设计已完成，学生尚未训练 | [蒸馏设计](../1011/Ancestry_Gated_PRS_Distillation_Design_20261011.md) |
| 独立外部队列验证 | 尚未完成 | 目前没有可用独立队列 |

本次整理从保存预测重算核对三疾病 54 个配置的 AUC、Brier 和 log loss，以及 D-PRISM25 的 10 个整体 AUC 结果行；同时验证 35 个既有交付/报告文件哈希，并核对配对点差、患者集合和病例数。区间沿用已保存的 2,000 次配对 bootstrap，本轮未重做 bootstrap 或拟合模型。各区间均条件于保存预测，不包含整个历史开发选择与重新训练的变异；既有患者级划分也不等于经核验的家系隔离。外部发现及调优队列与本项目的独立性按项目确认记录，尚未进行跨库参与者级连接审计。

[本轮核验凭证](../../outputs/external_pgs/progress_review_20261012/EVIDENCE_VERIFICATION.json)和[复算表](../../outputs/external_pgs/progress_review_20261012/prediction_recalculation.csv)记录来源哈希和检查结果。此前最新实现验证为 198 passed、2 个默认关闭的集成测试 skipped，Ruff 通过；本轮只新增文档及汇总核验产物，没有重跑该代码测试套件，也没有提交新训练任务。原始预测、冻结实验文档和既有性能均保持不变。
