# 外部队列运行手册

本手册对应 2026 年 11 月冻结的新标准参考 PC 模型。默认使用 **train + validation 合并后模型**，同时支持仅 train 模型。主要蒸馏模型是 M0_KD；按病种选择的学生作为可选项保留。

输入 GRCh38 基因型与患者协变量，程序计算训练时实际使用的 **26 个 T2D、7 个 CAD、5 个女性 BC 原始 PRS**，投影到固定 1000G PC 坐标，再运行 TabPFN v3 和蒸馏模型。权重、PC、模型与训练变换均不在外部队列重拟合。

[完整英文接口说明](EXTERNAL_INFERENCE.md) · [All of Us 操作流程](EXTERNAL_ALLOFUS.md) · [配置模板](../configs/external_cohort.example.yaml)

## 最短操作路径

1. 获取已导出的私有模型包。代码仓库不含患者数据、模型参数或 SNP 权重。
2. 准备 GRCh38 基因型和 sampleid/sex 协变量表。
3. 修改一个 YAML 配置。
4. 执行一条 run 命令。age 尚未解决时先完成遗传特征；补齐兼容 age 后复用这些结果继续推理。

~~~bash
cd /path/to/bundle/runtime
uv sync --frozen --extra tabpfn
cp configs/external_cohort.example.yaml cohort.yaml
# 修改 bundle、run_dir、genotypes、covariates、runtime.plink2
uv run --frozen --extra tabpfn python scripts/external_inference.py run --config cohort.yaml
~~~

所有相对路径均以 **YAML 所在目录** 为基准。移动到新环境后，配置中的绝对路径更容易核查。PLINK2 需单独安装或加载；本地验证版本是 2026-02-28 的 v2.0.0-a.6.33LM。软件依赖按 uv.lock 安装，模型使用本地文件，不调用在线推理服务。

CPU 可以运行全部遗传准备和蒸馏模型；TabPFN 通常分配 GPU。仅运行蒸馏模型时设 models: [student]、device: cpu。模型包依然包含 TabPFN 所需的私有训练上下文，请按整个包管理。

## 输入如何准备

基因型支持 PGEN（.pgen/.pvar/.psam）、BED（.bed/.bim/.fam）、VCF，以及带显式参考等位基因约定与 .sample 文件的 BGEN。PVAR 可压缩为 .pvar.zst。可输入一个完整 fileset，也可逐染色体列出相同患者集合的文件。

~~~yaml
genotypes:
  - format: pfile
    path: /private/genotypes/chr1
  - format: pfile
    path: /private/genotypes/chr2
~~~

基因型需已经完成适合该平台/数据发布版本的样本质控、变异质控和插补质量处理。程序从调用或插补后的基因型开始，不从 FASTQ/CRAM 开始。GRCh37 转换、模糊链方向、重复变异和重叠区段合并需要在上游解决。VCF 若需使用 DS，显式设置 dosage_field: DS，避免只读取 GT。

程序按位置和等位基因匹配冻结 SNP，允许命名不同与 REF/ALT 互换。同名但等位基因不兼容不会直接匹配。每个非空 shard 必须含所选全部患者，IID 不能在不同 FID 中重复；sampleid 的前导零会保留。当前支持不交错的染色体 shard；测试版本的 PLINK2 不支持任意交错 PGEN 合并。

协变量最少包含：

~~~text
sampleid,sex
example001,Female
example002,Male
example003,Unknown
~~~

sex_mapping 必须明确。BC 依据包内 eligibility.yaml 仅输出女性患者；男性和 Unknown 不输出 BC。协变量中不允许混入 T2D/cad/bc/y/y_true 结局列。原官方 PC 列即使存在也不使用。

## age 的边界

现有 age_sd 是原始资料已经变换后的数值，原始年龄与转换参数尚不明确。按您的决定，本次不推测转换方式。

~~~yaml
age:
  mode: deferred
~~~

此时 run 输出 preparation_complete_prediction_deferred_age，表示 **PRS 与 PC 完成，预测待 age**。不是模型已完成推理。

日后获得可核实的兼容数值后，在协变量增加 age_sd，并改为：

~~~yaml
age:
  mode: training_scale
  contract_id: decipehr_provided_age_sd_v1
  provenance: "写明真实数据来源或已核实的转换版本"
~~~

重新执行 run 即可复用遗传准备。不能直接输入岁数，也不能在外部队列重新算年龄 z-score。程序能核查声明与数值格式，无法替代对 age 转换本身的科学核实。

## 两个训练范围与两类模型

~~~yaml
inference:
  scope: trainval_refit
  models: [tabpfn, student]
  student: M0_KD
  traits: [T2D, cad, bc]
  device: cuda
  batch_size: 1024
  name: merged
~~~

比较仅 train 模型时，改 scope: train_only、name: train_only；需模型包导出时包含两个范围。遗传特征仍共用，不重复计算。

student 可选 M0_KD、selected、both。selected 对应 T2D=M0_KD、CAD=HGB_PAIR_KD、BC=G_A6_KD；both 不重复输出相同的 T2D M0。TabPFN 使用包内冻结训练特征与标签建立推理上下文，外部患者及其标签不参与拟合。

本地回放覆盖 3,922 名患者（BC 为 2,187 名女性）及两个训练范围。蒸馏模型与整批 TabPFN 均复现到 5e-16 以内；默认每批 1,024 人的 TabPFN 与整批推理最大概率差为 0.002629，Pearson/Spearman 均大于 0.9999998。冻结软件使用 auto precision；这里只确认批次大小会影响数值，未把原因完全归结于某个数值机制。若需严格复现该 holdout，设置 batch_size: 5000；大型外部队列则预先固定可承受的批次大小和患者顺序，并保留这项数值差异说明。

## PRS 与 PC 为什么要固定

这里的 PRS 是训练时实际使用的有效匹配 SNP 子集及原系数。不能随意替换为同名完整 PGS、新版本权重、外部队列标准化分数或 ancestry residualized 分数。原始分数采用 PLINK 的 SUM，并使用固定参考频率处理缺失，绝不估计外部队列 AF。

PC 使用固定 1000G 载荷、频率及原有 variance-standardize SUM 命令；不重新拟合 PCA，不做 OADP。特别保留原 PLINK 的缺失值约定：测试版本下，PC 缺失项的默认贡献与频率有关，并不等同于当前 PLINK 示例中 no-mean-imputation 的零贡献。程序不会悄悄切换定义；建议完整 PC 位点覆盖。

默认 missing_variants: error，缺少任何冻结位点即停止并留下覆盖报告。可显式选择 reference_mean：缺失位点由独立的全缺失 PLINK 数据流按冻结规则计分后合并，同时仍需通过覆盖和样本缺失率门槛。

报告同时给出每个 PRS 的非零系数项覆盖率与绝对系数权重覆盖率，PC 覆盖率，以及缺失 SNP 列表。默认 99% 覆盖、5% 样本 dosage 缺失率只是工程门槛，不代表已证明科学等价。严格模式即使覆盖超过 99%，仍拒绝任何缺失位点。不能为跑通而不经分析地放宽阈值。

## 结果和中断恢复

主要结果位于 run_dir：

- preparation/scores.long.csv：原始 PRS。
- preparation/reference_pcs.tsv：固定参考 PC。
- preparation/genetic_features.parquet：遗传特征，不含年龄或结局。
- preparation/coverage/、preparation/qc/：覆盖与缺失情况。
- preparation/PREPARED.json：准备完成证据。
- inference/NAME/predictions.csv：患者、病种、模型、训练范围、概率、logit 与超出训练支持范围标记。
- inference/NAME/cohort_counts.csv：各病种纳入/排除人数。
- inference/NAME/feature_distribution.csv：输入分布与训练上下文比较。
- inference/NAME/COMPLETE.json：预测完成及校验。
- RUN_STATUS.json：最近运行状态。

每个阶段有命令、日志、耗时与校验。重跑会复用已验证完成的阶段；未完成目录保留后重做该阶段。遗传输入、患者集合或准备代码改变时应使用新 run_dir；推理选项改变时应使用新 name。补齐 age 不需要重算 PRS/PC。

最终表有 SHA256；大于 128 MiB 的基因型使用文件大小与修改时间标识，不能表述为完整内容哈希。PGEN 快速路径引用只读原始基因型；完成前保留这些源文件。

## 有结局后独立评价

~~~bash
uv run --frozen --extra tabpfn python scripts/external_inference.py evaluate \
  --prediction-dir /private/run/inference/merged \
  --outcomes /private/outcomes.csv \
  --phenotype-note "病例/对照定义、观察时间窗及版本" \
  --out-dir /private/run/evaluation/merged
~~~

标签表使用 sampleid 和对应病种的 0/1/缺失列。未知标签不当作对照。输出 AUROC、AUPRC、Brier、log loss 及分母；不重拟合或自动校准，目前是点估计。

外部验证还需核查表型定义、随访、病例抽样、祖源分层及与训练/发现数据的重叠。原 decipEHR 的重叠审查不能直接推广到 All of Us。能生成预测、能复现本地 holdout，与证明外部预测性能，是不同的验证结果。

## 维护者导出

~~~bash
uv run --frozen --extra tabpfn python scripts/external_inference.py export \
  --sources configs/external_model_sources.yaml \
  --scope both --out-dir models/external_reference_pc_bundle
~~~

导出前核查已封存来源，目标必须为新目录。包内包含 **TabPFN 训练患者特征及标签**，不是可公开分享的纯 checkpoint。使用目的环境许可的方式处理私有包；本脚本不上传数据。

从模板到运行的完整参数、错误处理、文件含义和上游资料见英文手册；本地验收以 outputs/external_inference_20261105/VERIFIED.json 为准。All of Us 的真实数据接入与外部预测性能不由本地回放自动证明。

