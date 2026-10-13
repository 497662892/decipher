# Ho2022 乳腺癌扩展与 External PRS hold-out 执行说明

本次对应三项工作：在原 BC 面板加入 Ho2022 ASN/EUR 分量、运行现行 External PRS＋TabPFN＋wrapper 的 hold-out 预测、保留可审计的公开代码。All of Us 对接评估另见[可行性文档](All_of_Us_HPC_Feasibility_CN_20261012.md)。所有原始数据、权重、逐人预测及模型相关运行文件保留在 Git 忽略目录。

## 公开权重与计分

新增的是 [Ho et al., 2022](https://doi.org/10.1016/j.gim.2021.11.008) Supplementary Tables S3/S4 的两个 PRS-CSx 后验分量。ASN/EUR 是其祖源来源分量，不代表只给 ASN/EUR 患者计算；本次对所有符合条件的女性同时输入两项。它们是联合估计的成分，不能视为彼此独立的 GWAS 证据。公开文章所述发现及调优队列不包含 All of Us；这一结论基于报告的队列来源，并非逐人跨库比对。

| 分量 | 发表 SNP 数 | 本项目实际匹配 SNP 数 | 覆盖率 |
|---|---:|---:|---:|
| Ho2022 ASN | 888,765 | 878,661 | 98.8631% |
| Ho2022 EUR | 947,621 | 936,804 | 98.8585% |

原始工作簿为 GRCh37，目标基因型为 GRCh38。转换使用冻结校验和的 UCSC chain 与 pyliftover；负链转换对两种等位基因同时取互补，beta 保持不变。未映射或歧义位点保留在发表 SNP 总数这个分母中。随后使用官方 PGS Catalog formatter/matcher 和 PLINK2，沿用预先固定的 75% 覆盖率门槛。未在 NYU 或 hold-out 上重新估计 SNP beta。

hold-out 直接复用训练计分已经形成的 SNP／效应等位基因／系数矩阵。两边 25,722,569 个变异的 CHROM、POS、ID、REF、ALT 按顺序规范化后校验和完全相同；INFO 中的队列频率不参与该轴比较，也不用于本次缺失基因型填补。缺失剂量沿用冻结的 1000G 外部频率，PLINK 使用 `--error-on-freq-calc` 阻止临时估计目标队列频率。每项 PRS 汇总 `*_SUM`，而不是 `*_AVG`，并检查实际计分变异集合与训练一致。

## BC 比较与选择规则

旧面板为 PGS005378、PGS004869、PGS018538；扩展面板增加两个 Ho 分量。两边共用 3,308 名训练女性（602 病例）及原有五折分配，并在独立于该次拟合的 1,628 名 development 女性（320 病例）上预测。development 此前已经被反复用于研究决策，因此不是新的独立外部验证。

| 评价范围／输出 | 旧 3 分数 AUC | 新 5 分数 AUC | 旧／新 Nagelkerke R² |
|---|---:|---:|---:|
| Development，TabPFN 原始概率 | 0.715345 | 0.718834 | 0.140703 / 0.146464 |
| Development，wrapper | 0.713365 | 0.718081 | 0.151991 / 0.158971 |
| Training 五折 pooled OOF，原始概率 | 0.689989 | 0.697615 | 0.109302 / 0.116586 |
| Training 五折 pooled OOF，wrapper | 0.686339 | 0.694010 | 0.111595 / 0.118863 |

配对区间采用 2,000 次结局分层患者 bootstrap、seed 42。原始概率保留预先定义的八项检验族（两种整合器 × AUC/R² × development/OOF），wrapper 另列八项补充检验族。Bonferroni 同时区间与普通 95% 区间均保存；祖源相似性亚组仅报告描述性点估计。区间条件于已保存预测，不覆盖重训练、开发选择和家系依赖的不确定性。

TabPFN＋wrapper 的 development ΔAUC 为 +0.004716，普通 95% CI 为 [−0.006100, +0.016507]；training OOF ΔAUC 为 +0.007670，95% CI 为 [−0.000003, +0.015682]。八项同时区间分别为 [−0.011079, +0.021162] 和 [−0.003118, +0.018449]。因此，新增分量有一致方向的点估计收益，尚未确立显著增益。PC-adjusted Logistic 对照的 OOF 点估计也改善，但 development 增量很小；不能仅凭训练 OOF 将 5 分数判为普遍优越。

本次 hold-out 的选择规则在 BC 结果出来前记录于 `holdout_wrapper_20261012/PLAN.json`：固定当前主方法 `tabpfn_prs_pc`，在现有完整面板与扩展完整面板之间按 development wrapper AUC 选择，精确并列时比较 log loss。结果选定 T2D 26、CAD 7、BC 5。这里的“当前最优”限于这个固定流程下的已比较完整面板，不代表所有文献模型或全部历史消融中的全局最优，也不表示差异均已显著。

## Hold-out 的实际模型步骤

1. **冻结输入。** 保存所选 score ID、权重与计分文件校验和、模型 checkpoint、代码、seed 42 和预处理设置。全程拒绝将 hold-out 结局列传入预测 API。
2. **合并有标签训练 context。** 将原 train 与 development 合并，并逐项核对其与最终训练文件一致。T2D/CAD 各使用 8,663 人；BC 根据 `configs/disease_eligibility.yaml` 仅使用 4,936 名女性。hold-out 与这些患者 ID 无交集。
3. **生成 wrapper 的训练输入。** 在上述 context 内做三折分层交叉预测。每折仅用另外两折拟合 PRS/协变量转换和 TabPFN context，为留下的一折生成概率。每名患者恰有一个 OOF 概率，其自身标签不参与该次预测。这里的三折是最终 wrapper 的内部训练步骤，与用于方法比较的外层五折不同。
4. **拟合 wrapper。** 将 OOF 概率裁剪至 `[0.001, 0.999]` 后取 logit，以该 logit、年龄、性别、PC1–PC10 为输入，使用现有 Logistic 回归实现。数值缺失值填补、数值标准化及性别编码均仅在 context 拟合；`class_weight=None`、`C=1`、`lbfgs`、`max_iter=2000`。wrapper 的训练特征来自 OOF，而非 TabPFN 对自身训练患者的拟合预测。
5. **预测 hold-out。** 用全部有标签 context 构建同一 TabPFN v3 模型（固定 checkpoint，16 estimators）；输入冻结 PRS、年龄、性别及连续 PC1–PC10，得到 hold-out 概率。沿用相同裁剪/logit 规则，再应用第 4 步已拟合的 wrapper，得到最终概率。TabPFN 使用有标签 context 条件化；本次没有微调通用 checkpoint 的神经网络参数。
6. **复核。** 保存逐人原始概率、最终 wrapper 概率、训练 OOF 概率、预处理参数和每次拟合的患者集合哈希。从保存的 OOF 概率及 hold-out 原始概率重新拟合 Logistic wrapper，独立核对最终概率，同时检查完整覆盖、概率范围、女性限制及所有冻结输入未改变。

用户已确认没有 hold-out 标签，因此本次交付是 3,922 名 T2D、3,922 名 CAD、2,187 名女性 BC 的预测与核查，不计算 hold-out AUC、R² 或校准表现。最终预测是本研究模型输出，不应直接解释为经过临床人群校准的绝对发病风险。

最终核查已完成：共 3,922 名不同患者、10,031 条患者－疾病预测；三病 wrapper 的独立重算最大绝对误差均为 0。无训练／hold-out 患者重叠，输入哈希全部一致。宽表对非女性或性别未知患者的 BC 概率留空；没有为不符合条件者构造预测。

最终文件为 `holdout_predictions.long.csv`、`holdout_wrapper_probabilities.csv`、`cohort_audit.csv` 和 `FINAL_VERIFIED.json`，位于上述 hold-out 运行目录。逐人输出权限为 0600。三病完整运行回执同时保留训练 OOF 和预处理记录。

作业证据：Ho 计分 `28394658`、四项比较 `28394659`、hold-out 冻结计分 `28394769`、配对报告 `28394901` 均完成。首次 hold-out 阵列 `28394844` 的 BC 成功，T2D/CAD 因同一节点 CUDA 设备不可用失败；仅这两项经 `28395024` 重跑完成，失败记录独立保留。报告首轮 `28394845` 主动取消，改为预定整体检验族并保留亚组点估计，避免额外亚组 bootstrap 重复计算。没有改变计分、模型或选择参数。

工程验证：使用缓存依赖离线运行完整 pytest，210 passed、2 skipped；Ruff 和全部 Slurm launcher 的 Bash 语法检查通过。新增测试验证未标注 hold-out API 与原 validation wrapper 数值一致、结局隔离、女性限制、缺失分数拒绝，以及变异轴对 INFO 的不敏感性和对等位基因改变的敏感性。

## 复现入口与证据

| 阶段 | 代码／本地运行目录 |
|---|---|
| 导入公开工作簿 | `scripts/prepare_ho2022_bc.py`；`configs/external_pgs_ho2022_bc.yaml` |
| 计分与组装 BC 面板 | `slurm/ho2022_bc_score.sbatch`；`scripts/assemble_ho2022_bc_expansion.py` |
| BC development／五折比较 | `slurm/ho2022_bc_evaluate.sbatch` |
| 配对统计 | `scripts/report_ho2022_bc.py`；`slurm/ho2022_bc_report.sbatch` |
| Hold-out 冻结计分 | `scripts/score_external_pgs_holdout.py`；`slurm/external_pgs_holdout_score.sbatch` |
| 面板冻结 | `scripts/select_external_pgs_holdout.py` |
| Hold-out wrapper | `scripts/predict_external_pgs_holdout.py`；`slurm/external_pgs_holdout_predict.sbatch` |
| 独立复核 | `scripts/verify_external_pgs_holdout.py` |
| BC 结果与校验 | `outputs/external_pgs/bc_ho2022_preparation_20261012/report/` |
| Hold-out 完整运行 | `outputs/external_pgs/holdout_wrapper_20261012/` |

每个运行入口默认拒绝覆盖已有结果。重做实验需使用新的输出目录，并相应调整依赖这些路径的 manifest/launch 参数；勿删除原封存运行再套用旧回执。研究用输出文件不在 GitHub 代码仓库中。
