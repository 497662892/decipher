# 从 PLINK 数据到各 PRS 模型输入的数据预处理流程

## 1. 文档范围

本文只描述下面这一段数据链：

```text
原始 PLINK2 genotype
        │
        ├── variant-level QC ──> 统一的 QC genotype
        │
        └── sample alignment ──> training / validation / holdout keep files

官方 GWAS summary statistics
        │
        ├── 字段标准化
        └── 与 QC genotype 做 variant 和 allele harmonization
                              │
                              └──> 各 PRS 方法可直接读取的输入文件
```

本文在“各方法的直接输入文件已经生成”处结束，不包括：

- PRS 模型参数拟合；
- 患者级 PRS score 的生成；
- linear 或 TabPFN ensemble；
- covariate preprocessing；
- official-validation 指标计算。

## 2. 原始输入

### 2.1 PLINK2 genotype

Training phase 的原始 genotype 是按染色体拆分的 PLINK2 pfile：

```text
data/raw/training_phase/genotype_files/plink_files/
├── Testing_Chr1.pgen
├── Testing_Chr1.pvar
├── Testing_Chr1.psam
├── ...
└── Testing_Chr22.{pgen,pvar,psam}
```

对应的 imputation information 文件为：

```text
data/raw/training_phase/supplementary_files/info_scores/
└── All_Samples_InfoScore_v02_chr{1..22}.txt.gz
```

Final holdout phase 使用：

```text
data/raw/holdout_phase/genotype_files/plink_files/
└── Holdout_chr{1..22}.{pgen,pvar,psam}
```

这些路径和 QC 阈值由
[`configs/qc_dream_training.yaml`](../configs/qc_dream_training.yaml)定义。

### 2.2 官方 GWAS summary statistics

三个疾病均使用 GWAS Catalog summary statistics。Canonical 文件和研究信息记录在
[`configs/official_sumstats.yaml`](../configs/official_sumstats.yaml)。

| Disease | Summary-statistics名称 | GWAS Catalog accession | PubMed | 原始文件名 | 缺失per-variant N时使用的default N |
|---|---|---|---:|---|---:|
| CAD | Coronary artery disease | GCST003116 | 26343387 | `26343387-GCST003116-EFO_0000378.h.tsv.gz` | 158,816 |
| BC | Breast cancer [female] (PheCode 174.1) | GCST90479801 | 39024449 | `GCST90479801.tsv.gz` | 11,777 |
| T2D | Type 2 diabetes (PheCode 250.2) | GCST90479885 | 39024449 | `GCST90479885.tsv.gz` | 572,189 |

这里的 default N 是各 ancestry case-control effective sample size
`4 / (1 / cases + 1 / controls)`之和，不是简单的总样本数。仅当原始文件缺少有效的
per-variant `N`或`N_eff`时才使用它。

## 3. PLINK genotype 的公共 QC 层

### 3.1 从 INFO 文件生成每条染色体的 variant list

对每条染色体读取`All_Samples_InfoScore_v02_chr*.txt.gz`，保留：

- `INFO >= 0.8`；
- `MAF >= 0.01`；
- variant ID非缺失。

当前配置将频率列`MAF`明确解释为 minor allele frequency，因此`max_af: 0.99`
不会形成额外的上界过滤。

每条染色体生成：

```text
outputs/genotype_qc/lists/chr{chr}.info_maf.snplist
```

实现：
[`scripts/filter_info_scores.py`](../scripts/filter_info_scores.py)和
[`src/decipehr/qc.py`](../src/decipehr/qc.py)。

### 3.2 PLINK2 variant-level QC

对每条染色体执行等价于：

```bash
plink2 \
  --pfile Testing_Chr{chr} \
  --extract chr{chr}.info_maf.snplist \
  --snps-only just-acgt \
  --max-alleles 2 \
  --geno 0.05 dosage \
  --make-pgen
```

因此保留的variant满足：

- imputation INFO至少0.8；
- MAF至少0.01；
- dosage missingness不超过5%；
- 仅保留A/C/G/T SNP；
- 仅保留biallelic variants。

当前默认流程：

- **不按HWE删除variant**，但会生成HWE报告；
- **不按sample missingness删除患者**；
- 不在genotype QC阶段读取phenotype；
- 不在这一阶段进行GWAS allele harmonization。

每条染色体的QC结果为：

```text
outputs/genotype_qc/qc_final/chr{chr}.qc.{pgen,pvar,psam}
```

### 3.3 合并1–22号染色体

PLINK2使用`--pmerge-list`合并所有染色体并执行`--sort-vars`，得到公共的
genome-wide target genotype：

```text
outputs/genotype_qc/qc_final/
├── all_chromosomes.qc.pgen
├── all_chromosomes.qc.pvar
└── all_chromosomes.qc.psam
```

同时生成全基因组missingness、allele frequency、HWE和pgen-info报告。

完整命令由
[`scripts/write_genotype_qc_commands.py`](../scripts/write_genotype_qc_commands.py)
生成。

## 4. 样本集合与 genotype variant 集合

### 4.1 Official-validation阶段

将`all_chromosomes.qc.psam`中的genotype sample IDs分别与：

- 已标注training phenotype；
- official-validation covariate IDs

取交集，生成：

```text
data/processed/sample_sets/
├── labeled_train.keep.psam
└── unlabelled_validation.keep.psam
```

这里不会复制或重新编码genotype。Training和official validation仍然使用同一个
`all_chromosomes.qc` pfile，只是在运行各PRS方法时使用不同的PLINK `--keep`
文件。

实现：
[`scripts/make_sample_sets.py`](../scripts/make_sample_sets.py)和
[`src/decipehr/sample_sets.py`](../src/decipehr/sample_sets.py)。

### 4.2 Final holdout阶段

Final holdout genotype来自另一套PLINK文件，因此不能假设其variant集合和allele
编码与training-QC genotype完全相同。项目按染色体执行：

1. 从`chr{chr}.qc.pvar`提取training-QC variant IDs。
2. 用这些IDs从`Holdout_chr{chr}`提取holdout variants。
3. 再从提取后的holdout `.pvar`取得实际共同variant IDs。
4. 使用共同IDs重新提取training genotype。
5. 审计两边的variant顺序、ID、REF和ALT；发现不一致时立即失败。
6. 分别合并1–22号染色体。

最终得到严格共享variant空间的两个pfile：

```text
outputs/final_genotype/qc_final/
├── train_val.common.{pgen,pvar,psam}
└── holdout_test.common.{pgen,pvar,psam}
```

实现：
[`scripts/write_final_genotype_projection_commands.py`](../scripts/write_final_genotype_projection_commands.py)
和
[`scripts/audit_pvar_compatibility.py`](../scripts/audit_pvar_compatibility.py)。

## 5. Summary statistics 的公共标准化层

每个疾病的原始GWAS文件首先被转换成统一schema：

```text
SNP  CHR  BP  A1  A2  P  BETA  [SE]  [N]
```

处理包括：

- 识别不同GWAS文件中的同义字段名；
- chromosome去除`chr`前缀并统一表示；
- `BP`、`P`、`BETA`、`SE`和`N`转换为数值；
- allele转换为大写；
- 如果只有`OR`，使用`BETA = log(OR)`；
- 删除关键字段缺失或非有限数值的记录；
- 删除重复variant ID；
- 当原始variant ID不能直接使用时，生成decipEHR规范化variant ID。

标准化输出为：

```text
data/processed/sumstats/standardized/
├── cad.GCST003116.standardized.tsv.gz
├── bc.GCST90479801.standardized.tsv.gz
└── T2D.GCST90479885.standardized.tsv.gz
```

实现：
[`scripts/prepare_sumstats_common.py`](../scripts/prepare_sumstats_common.py)和
[`src/decipehr/sumstats.py`](../src/decipehr/sumstats.py)。

## 6. Summary statistics 与 target genotype 的 harmonization

标准化sumstats与QC genotype的
`all_chromosomes.qc.pvar`进行匹配：

1. 优先按summary-statistic SNP ID和target `.pvar` ID匹配。
2. ID不能匹配时，按`CHR + BP + allele pair`匹配。
3. 要求GWAS A1/A2与target REF/ALT构成同一allele pair。
4. 删除非A/C/G/T alleles。
5. 删除A/T和C/G ambiguous SNP。
6. 删除重复、multi-match和allele mismatch记录。
7. 当前公共流程不允许strand flipping。

输出仍保留GWAS effect allele和effect size，同时增加：

```text
source_snp, source_chr, source_bp,
target_snp, target_ref, target_alt, matched_by
```

公共harmonized输出为：

```text
data/processed/sumstats/harmonized/
├── cad.GCST003116.harmonized.tsv.gz
├── bc.GCST90479801.harmonized.tsv.gz
└── T2D.GCST90479885.harmonized.tsv.gz
```

对应审计报告位于：

```text
data/processed/sumstats/reports/
└── {disease}.{accession}.harmonized.summary.json
```

实现：
[`src/decipehr/sumstats_harmonization.py`](../src/decipehr/sumstats_harmonization.py)。

至此形成两个所有PRS方法共享的公共层：

```text
QC target genotype:
    *.pgen + *.pvar + *.psam

Disease-specific GWAS:
    {disease}.{accession}.harmonized.tsv.gz
```

## 7. 从公共层生成各方法的直接输入

| 方法 | Genotype侧的直接输入 | Summary-statistics侧的直接输入 | 额外reference或annotation |
|---|---|---|---|
| P+T | QC PLINK2 pfile和sample keep file | Harmonized `SNP, A1, BETA, P`；training-only PLINK clumping后生成多P-value阈值score table | 不需要外部LD reference；LD由training genotype估计 |
| LDpred2 | QC pfile按样本/variant提取后转为`.bed/.bim/.fam`，再由`bigsnpr::snp_readBed`建立backing files | Harmonized表在R中转成`chr, pos, rsid, a1, a0, beta, beta_se, n_eff` | 支持target genotype、UKB HM3+或其他external LD reference |
| lassosum2 | Training、scoring和fit-scoring样本分别转换为PLINK1 BED输入 | 与LDpred2相同的harmonized基础字段，并补齐`beta_se`和`n_eff` | 当前流程要求UKB HM3+或其他external LD reference |
| PRS-CS | 用harmonized variants生成PRS-CS格式BIM；最终score仍回到target PGEN | `{disease}.prscs.sst`：`SNP, A1, A2, BETA, SE/P`；同时生成rsID到target ID映射 | 官方PRS-CS LD reference panel |
| SDPR | Target-reference模式使用training samples生成BED/BIM和LD reference；prebuilt模式生成与rsID兼容的valid BIM | `{disease}.sdpr.ss`：`SNP, A1, A2, BETA, P, N` | Target-derived或prebuilt SDPR reference |
| sBayesRC | QC PGEN；先仅用training samples计算PLINK2 `.afreq` | `{disease}.sbayesrc.ma`：`SNP, A1, A2, freq, b, se, p, N`，以及rsID-target ID map | GCTB LD matrix/eigen files和annotation file |
| MegaPRS | 仅用training samples生成allele frequency；再用入选predictors建立PLINK1 BED reference | `{disease}.megaprs.summaries`：`Predictor, A1, A2, BETA, SE, n, A1Freq`；另生成predictor list | LDAK reference；annotated版本额外使用annotation files |

### 7.1 P+T

P+T在training genotype上执行PLINK2 clumping。Clumped variants与harmonized
sumstats连接后，生成：

```text
SNP  A1  BETA_P1  BETA_P0.5  BETA_P0.05  ...
```

每个beta列对应一个P-value threshold。这个表和QC pfile就是PLINK2
`--score`前的直接输入。

实现：
[`scripts/write_pt_full_train_commands.py`](../scripts/write_pt_full_train_commands.py)
和[`scripts/build_pt_scorefile.py`](../scripts/build_pt_scorefile.py)。

### 7.2 LDpred2与lassosum2

两个方法都需要将PLINK2 PGEN转换成PLINK1
`.bed/.bim/.fam`，随后由`bigsnpr`读取。Harmonized sumstats会：

- 从`BETA`和`SE`取得`beta_se`；
- 若`SE`缺失，则由`BETA`和`P`近似推导；
- 优先使用per-variant `N/N_eff`；
- 缺失时使用manifest中的default N；
- 与target genotype和指定LD reference再次匹配。

实现：
[`scripts/run_ldpred2_task.R`](../scripts/run_ldpred2_task.R)和
[`scripts/run_lassosum2_task.R`](../scripts/run_lassosum2_task.R)。

### 7.3 PRS-CS

因为PRS-CS reference通常使用rsID，项目从原始GWAS和harmonized表建立：

```text
outputs/prscs_full_train/{disease}/inputs/
├── {disease}.prscs.sst
├── {disease}.prscs.bim
└── {disease}.prscs.target_id_map.tsv
```

`.sst`和`.bim`使用rsID进入PRS-CS；模型输出的rsID随后通过
`target_id_map.tsv`映射回competition genotype中的target variant ID。

实现：
[`scripts/prepare_prscs_inputs.py`](../scripts/prepare_prscs_inputs.py)。

### 7.4 SDPR

生成：

```text
outputs/sdpr_full_train/{disease}/inputs/
├── {disease}.sdpr.ss
├── {disease}.sdpr.target_id_map.tsv   # prebuilt-reference模式
└── {disease}.sdpr.valid.bim           # prebuilt-reference模式
```

极小P值会截断到`1e-300`，避免SDPR解析或数值计算失败。

实现：
[`scripts/prepare_sdpr_inputs.py`](../scripts/prepare_sdpr_inputs.py)。

### 7.5 sBayesRC

先在training samples上运行PLINK2 `--freq`。该频率与harmonized effect allele
方向对齐后，生成GCTB `.ma`文件：

```text
SNP  A1  A2  freq  b  se  p  N
```

如果`SE`缺失，则由`BETA`和`P`推导；`N`缺失时使用manifest default N。

实现：
[`scripts/prepare_sbayesrc_inputs.py`](../scripts/prepare_sbayesrc_inputs.py)。

### 7.6 MegaPRS

同样只使用training samples计算allele frequency，然后生成LDAK输入：

```text
Predictor  A1  A2  BETA  SE  n  A1Freq
```

以及只包含入选variant IDs的predictor list。该list用于从training QC genotype
建立MegaPRS所需的PLINK BED reference。Annotated MegaPRS在此基础上再接入
LDAK annotation files，但不会改变公共genotype或harmonized-sumstats层。

实现：
[`scripts/prepare_megaprs_inputs.py`](../scripts/prepare_megaprs_inputs.py)。

## 8. 防止数据泄漏的关键边界

- Genotype的INFO/MAF/missingness QC不读取疾病标签。
- P+T clumping仅使用training genotype。
- 需要target-derived LD或allele frequency的方法只使用training samples。
- Official-validation样本只通过单独的`--keep`文件用于scoring。
- Final holdout genotype只被投影到training-QC variant空间，不参与variant选择。
- Summary-statistics harmonization使用variant metadata和alleles，不使用competition标签。
- BC女性eligibility属于后续疾病建模的sample selection，不是公共genotype
  variant QC的一部分。

## 9. 本阶段的最终交付物

在进入各PRS模型前，应当能够同时确认：

1. Training/scoring genotype的`.pgen/.pvar/.psam`存在。
2. 对应的training、validation或holdout `keep.psam`存在。
3. 三个疾病的standardized和harmonized summary statistics存在。
4. Harmonization audit不是空输出，且记录了匹配和删除原因。
5. Final阶段的train-val与holdout-test `.pvar`通过完全兼容性审计。
6. 所选PRS方法对应的`.sst`、`.ss`、`.ma`、`.summaries`、BED或score table已经生成。

满足以上条件后，数据才进入P+T、LDpred2、lassosum2、PRS-CS、SDPR、
sBayesRC或MegaPRS本身。
