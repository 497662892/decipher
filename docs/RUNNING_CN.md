# PRS、TabPFN 与蒸馏模型运行手册

本手册对应 2026-10-30 的代码版本。所有命令从仓库根目录执行，使用 Bash。
只调用已训练的蒸馏模型，完成第 1、2 节即可。第 3–5 节分别说明 PRS 计分、
TabPFN 整合和重新训练。结果及其评价范围见[最新结果](1030/LATEST_RESULTS.md)，
接口定义见[推理接口说明](prs_inference.md)。

## 1. 获取代码与安装环境

新机器上先克隆；已有工作副本时，在确认没有待保留的本地修改后更新：

```bash
# 新机器
git clone https://github.com/497662892/decipher.git
cd decipher

# 已有工作副本：在其根目录执行
git status --short
git pull --ff-only origin main
```

本项目 HPC 根目录为 `/gpfs/data/shenlab/Jiajian/HW/decipehr`。
下面的 `UV_CACHE_DIR`、输出和日志均位于仓库忽略目录：

```bash
umask 077
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv sync --locked --extra dev

# 需要运行 TabPFN 时，安装同一个锁定环境中的 tabpfn extra
uv sync --locked --extra dev --extra tabpfn

uv run --no-sync --extra dev pytest -q
uv run --no-sync --extra dev ruff check .
```

`uv` 需预先安装。Linux 锁定环境使用 PyTorch 2.9.1 的 CUDA 12.9 wheel；
蒸馏推理仍可在 CPU 上运行。TabPFN GPU 任务需要兼容的驱动、已分配的 GPU 和
本地 checkpoint。现有 HPC 环境已安装依赖时可直接使用后续的 `--no-sync` 命令；
离线节点再加 `--offline`，不要为复现旧模型临时升级包。

**克隆仓库只获得代码、配置、测试和汇总文档。** 下列文件需要从项目的受控本地存储取得：

| 任务 | 必需的本地文件 |
|---|---|
| 蒸馏模型推理 | 保存的 Student bundle，以及每人的 PRS＋协变量表 |
| 从基因型计分 | QC 后的 PLINK2 fileset、冻结权重、外部等位基因频率及相应回执 |
| TabPFN 整合 | 上述 PRS、训练标签与协变量、query 协变量、祖源标签、TabPFN checkpoint |
| 重新蒸馏 | 冻结训练数据、共享折分、Teacher OOF 缓存、来源配置及校验回执 |

这些真实数据、模型权重和逐人输出不随 GitHub 发布。只加载可信来源的
`joblib` bundle；复制模型时同时保留其校验文件和目录结构。

## 2. 直接运行最新蒸馏模型（CPU）

### 2.1 模型目录与输入列

当前本地模型根目录：

```bash
export STUDENT_RUN=outputs/prs_distillation_bio/exploration_20261026_3569648f99a8
```

若迁移到其他机器，可将 `STUDENT_RUN` 改为实际路径。根目录下需要保留：

```text
core/<trait>/models/M0_KD/seed17/
    state.pt  preprocessing.joblib  metadata.json  HASHES.json
core/bc/models/G_A6_KD/seed17/
    state.pt  preprocessing.joblib  metadata.json  HASHES.json
low_order/cad/models/HGB_PAIR_KD/
    model.joblib  FIT.json
report/val/POLICY_LOCK.json
```

`<trait>` 为 `T2D`、`cad` 或 `bc`，大小写不能改变。只使用某个显式模型时，
仅需相应 bundle；使用 `BIO_POLICY` 时还需 `POLICY_LOCK.json`。

输入为 CSV、TSV 或 Parquet 宽表，每人一行：

| 列 | 要求 |
|---|---|
| `sampleid` | 非空、唯一，建议存为字符串 |
| `age_sd` | 与训练来源定义一致的标准化年龄，不是年龄年数 |
| `sex` | `Female`、`Male` 或已支持的 `Unknown` |
| `PC1_sd` … `PC10_sd` | 与训练来源采用相同投影及标准化定义 |
| 以下各病 PRS 列 | 原始冻结 PRS 分数，列名及计分定义均须一致 |

| 疾病 | PRS 列名 |
|---|---|
| T2D，26 列 | `PGS005353` 至 `PGS005377`，以及 `PGS002308` |
| CAD，7 列 | `PGS003725`, `PGS004696`, `PGS012558`, `PGS003727`, `PGS005143`, `PGS000296`, `PGS003866` |
| BC，5 列 | `PGS005378`, `PGS004869`, `PGS018538`, `LOCAL_HO2022_BC_ASN`, `LOCAL_HO2022_BC_EUR` |

不要在新患者上重新计算 PRS 的均值／标准差。接口会使用 bundle 中保存的训练变换，
并按保存的顺序排列列。也不能将另一个 PRS 仅改名后当作同一输入。
原始年龄或新算的 PCA 必须先按训练来源的定义完成转换。

无需提供结局列。BC 使用 `configs/disease_eligibility.yaml` 过滤，只为符合条件的
女性输出预测；非女性不会被填入零概率。缺列、重复 ID、非有限数值等会直接报错。

### 2.2 用现有验证特征运行六个示例

此示例复用本地无结局的 `evaluation_features.parquet`，每次创建新的输出目录。
不会重新训练，也不需要 TabPFN checkpoint：

```bash
export FEATURE_ROOT=outputs/prs_distillation_v1/phase_a_fulltrain_20261026_3d4334d2_0161ad36d950
mkdir -p outputs/student_inference
export PREDICTION_DIR
PREDICTION_DIR=$(mktemp -d outputs/student_inference/replay_XXXXXXXX)

for trait in T2D cad bc; do
  for arm in M0_KD BIO_POLICY; do
    uv run --offline --no-sync python scripts/predict_prs_student.py \
      --run-dir "$STUDENT_RUN" \
      --trait "$trait" --arm "$arm" --seed 17 \
      --input "$FEATURE_ROOT/$trait/evaluation_features.parquet" \
      --output "$PREDICTION_DIR/${trait}_${arm}.csv" \
      --eligibility-config configs/disease_eligibility.yaml
  done
done
```

对于这里指定的封存输入，T2D、CAD 各输出 2,852 人，BC 输出 1,628 名女性，
每种模式各一份 CSV。新队列的数量取决于实际输入及资格过滤。

| 请求的模式 | T2D | CAD | 女性 BC |
|---|---|---|---|
| `M0_KD` | M0_KD | M0_KD | M0_KD |
| `BIO_POLICY` | M0_KD | HGB_PAIR_KD | G_A6_KD |

`BIO_POLICY` 读取训练内部已经锁定的选择。它不会根据新患者结局重新选模型。
主结果使用 seed 17；参数模型保存的 seed 29/43 属于稳定性分析，HGB 仅支持 seed 17。

每个 CSV 同时生成 `<文件名>.csv.json`。只查看回执即可确认数量和执行状态：

```bash
uv run --offline --no-sync python - <<'PY'
import json
import os
from pathlib import Path

for path in sorted(Path(os.environ["PREDICTION_DIR"]).glob("*.csv.json")):
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "prediction_complete"
    assert receipt["new_fits"] == 0 and receipt["labels_used"] is False
    print(path.name, receipt["resolved_arm"], receipt["n_predicted"])
PY
```

CSV 包含 `sampleid, trait, arm, seed, probability, logit, age_out_of_support,
sex_out_of_support`。`probability` 是直接 Student 概率，`logit` 是对应原始 logit；
二者都没有额外添加 Logistic wrapper 或比赛提交所用的标准化变换。
支持范围标记用于提示超出训练年龄／性别支持范围的输入。

### 2.3 对新患者运行

按第 2.1 节准备例如 `data/processed/new_patients/T2D_features.parquet` 后：

```bash
mkdir -p outputs/student_inference
NEW_PREDICTION_DIR=$(mktemp -d outputs/student_inference/new_cohort_XXXXXXXX)
uv run --offline --no-sync python scripts/predict_prs_student.py \
  --run-dir "$STUDENT_RUN" \
  --trait T2D --arm M0_KD --seed 17 \
  --input data/processed/new_patients/T2D_features.parquet \
  --output "$NEW_PREDICTION_DIR/T2D.csv" \
  --eligibility-config configs/disease_eligibility.yaml
```

CAD／BC 分别换成 `--trait cad`／`--trait bc` 及对应特征文件。
若已有 `scores.long.csv`（`sampleid,disease,model,prs`），先检查每个患者－疾病－模型
组合唯一，再将 `model` 展开为上述 PRS 列，并按 `sampleid` 与协变量做一对一连接。
检查所有符合资格的患者具有完整分数，不要因 inner join 静默丢失患者。

## 3. 从基因型计算 PRS

需要 `plink2`、QC 后的 `.pgen/.pvar/.psam`、核实过的 genome build 和外部参考频率。
HPC 可用 `module avail plink` 查看软件；当前流程使用：

```bash
module load plink/2.0-20260228
plink2 --version
```

### 3.1 复用当前冻结权重，对相同变异轴的 holdout 计分

下面是现有比赛数据的完整计分命令，需在 CPU compute allocation 内执行。
三处 `--source` 必须保留匹配后的系数矩阵、变异列表和回执；不能只复制结果 CSV。

```bash
mkdir -p outputs/external_pgs
SCORING_WORK=$(mktemp -d outputs/external_pgs/score_replay_XXXXXXXX)
uv run --no-sync python scripts/score_external_pgs_holdout.py \
  --source outputs/external_pgs/run_20261010/scoring_v2 \
  --source outputs/external_pgs/cad_expansion_20261012/scoring_new \
  --source outputs/external_pgs/bc_ho2022_preparation_20261012/scoring_new \
  --pfile outputs/final_genotype/qc_final/holdout_test.common \
  --reference-freq outputs/external_pgs/reference_1000g_trainval/reference.target_ids.afreq \
  --plink2 plink2 --threads 8 --memory-mb 32000 \
  --out-dir "$SCORING_WORK/scoring"
```

成功输出 `scoring/scores.long.csv` 和 `scoring/VERIFIED.json`，以及 PLINK 日志和
各分块计分文件。它沿用训练时的 SNP、系数和外部频率，并检查患者不重叠、变异轴及
实际使用的变异集合。不同基因型变异轴的新队列需要先完成显式 harmonization；
不能直接跳过检查。该计分银行可含额外候选，模型仅使用已冻结的 26／7／5 面板。

### 3.2 从公开资源开始准备

完整资源审查、外部频率构建、PGS Catalog inventory/fetch/score 命令见
[External PGS 工作流](external_pgs_benchmark.md)。顺序为：
`inventory` → 完成生成的 `review.yaml` → `fetch` → `prepare-reference` → `score`。
`configs/external_pgs.yaml` 是基础候选清单，不能单独替代最新 CAD／BC 扩展后的冻结面板。

Ho2022 两个 BC 分量有独立导入入口，需先按配置准备原工作簿及 liftover chain：

```bash
mkdir -p outputs/external_pgs
HO_WORK=$(mktemp -d outputs/external_pgs/ho_import_XXXXXXXX)
uv run --no-sync --with openpyxl==3.1.5 --with pyliftover==0.4.1 \
  python scripts/prepare_ho2022_bc.py \
  --manifest configs/external_pgs_ho2022_bc.yaml \
  --chain data/resources/liftover/hg19ToHg38.over.chain.gz \
  --out-dir "$HO_WORK/weights"
```

这一步只导入权重。后续 PLINK 计分及 3／5 分数面板组装的依赖见
[Ho2022 执行说明](1012/Ho2022_BC_and_External_PRS_Holdout_20261012_CN.md)。
`assemble_ho2022_bc_expansion.py` 还需要准备好的 `PROTOCOL.json`、baseline 银行等文件，
不应直接对仅有 `weights/` 的空运行目录调用。

P+T、LDpred2、PRS-CS、lassosum2、SDPR、sBayesRC、MegaPRS 的内部 PRS 路线
继续使用 README 的[逐步运行命令](../README.md#step-by-step-use)及对应
`scripts/write_*_full_train_commands.py`／`slurm/*_full_train*.sbatch`。
这些内部模型与当前外部 26／7／5 面板有各自的输入定义。

## 4. TabPFN 对 PRS 的整合

### 4.1 原 train → 原 validation 的整合评价

当前三病已冻结的 context PRS 银行为：

| 疾病 | `BANK` |
|---|---|
| T2D | `outputs/external_pgs/run_20261010/scoring_v2` |
| CAD | `outputs/external_pgs/cad_expansion_20261012/banks/expanded` |
| BC | `outputs/external_pgs/bc_ho2022_preparation_20261012/banks/expanded` |

下面以 T2D 为例，先做输入核查，再在 GPU allocation 内执行正式评价。
CAD、BC 同时替换 `TRAIT` 和 `BANK`，每次使用新 `EVAL_WORK`。

```bash
TRAIT=T2D
BANK=outputs/external_pgs/run_20261010/scoring_v2
CHECKPOINT=outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt
mkdir -p outputs/external_pgs
EVAL_WORK=$(mktemp -d outputs/external_pgs/validation_replay_XXXXXXXX)
eval_args=(
  --weight-lock "$BANK/weights.scored.lock.json"
  --scores "$BANK/scores.long.csv" --scoring-run "$BANK/scoring_run.json"
  --train-phenotype data/processed/phenotype.csv
  --validation-phenotype data/processed/official_validation_phenotype.csv
  --ancestry-labels outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv
  --eligibility-config configs/disease_eligibility.yaml
  --diseases "$TRAIT" --methods tabpfn_prs_pc
  --model-path "$CHECKPOINT" --seed 42
)
uv run --offline --no-sync python scripts/evaluate_external_pgs_validation.py \
  "${eval_args[@]}" --preflight-only --out-dir "$EVAL_WORK/preflight"

# 以下正式计算需 GPU allocation
uv run --offline --no-sync python scripts/evaluate_external_pgs_validation.py \
  "${eval_args[@]}" --device cuda --out-dir "$EVAL_WORK/evaluation"
```

此流程用原 train 的标签拟合整合器及训练内部 wrapper；原 validation 不进入训练
context。成功结果包括 `run.json`、`summary_metrics.csv` 和逐人预测。
`--preflight-only` 通过只表示输入核查通过，不表示 GPU 预测已完成。

### 4.2 train＋development → 无标签 holdout

此入口依赖现有比赛目录。除 `SELECTION.json` 引用的冻结银行和评分回执外，还读取：

```text
data/processed/phenotype.csv
data/processed/official_validation_phenotype.csv
data/processed/final/train_val_phenotype.csv
data/processed/final/holdout_test_covariates.csv
outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv
outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.holdout_test.tsv
outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt
```

`holdout_test_covariates.csv` 不应含疾病结局列。以下复用封存的 holdout 分数，
创建新的预测目录，保留原有结果；软链接要求原计分目录在执行节点可读。

```bash
FROZEN_HOLDOUT=outputs/external_pgs/holdout_wrapper_20261012
mkdir -p outputs/external_pgs
export HOLDOUT_RUN
HOLDOUT_RUN=$(mktemp -d outputs/external_pgs/holdout_replay_XXXXXXXX)
cp "$FROZEN_HOLDOUT/SELECTION.json" "$HOLDOUT_RUN/SELECTION.json"
ln -s "$(realpath "$FROZEN_HOLDOUT/scoring")" "$HOLDOUT_RUN/scoring"

for trait in T2D cad bc; do
  uv run --offline --no-sync python scripts/predict_external_pgs_holdout.py \
    --run-dir "$HOLDOUT_RUN" --disease "$trait" --seed 42 \
    --model-path outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt \
    --eligibility-config configs/disease_eligibility.yaml --preflight-only
done
```

预检会创建 `preflight/<trait>/PREFLIGHT.json`。同一目录内重复预检会因目录已存在
而失败，应使用新的运行目录。正式计算可在已分配的 GPU 节点运行：

```bash
export HF_HUB_OFFLINE=1 TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1
for trait in T2D cad bc; do
  uv run --offline --no-sync python scripts/predict_external_pgs_holdout.py \
    --run-dir "$HOLDOUT_RUN" --disease "$trait" --seed 42 --device cuda \
    --model-path outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt \
    --eligibility-config configs/disease_eligibility.yaml
done
uv run --offline --no-sync python scripts/verify_external_pgs_holdout.py \
  --run-dir "$HOLDOUT_RUN"
```

每病生成 `predictions/<trait>/holdout_predictions.csv`、`wrapper_training_oof.csv`、
`fit_audit.csv`、`preprocessing.csv` 和 `VERIFIED.json`。独立复核后，运行根目录生成
`holdout_predictions.long.csv`、`holdout_wrapper_probabilities.csv`、`cohort_audit.csv`
及 `FINAL_VERIFIED.json`。现有 holdout 的预期数量为 T2D／CAD 各 3,922，女性 BC 2,187。
没有 holdout 标签，因此不计算 holdout AUC。

使用 Slurm 时可参考 `slurm/external_pgs_holdout_predict.sbatch` 的资源和环境设置。
该历史 launcher 的运行目录写死在默认值；重跑需把实际命令改为传入新的
`--run-dir "$HOLDOUT_RUN"`，不能仅设置 `RUN_DIR` 后直接提交旧 launcher。

需要比赛格式时，使用 `scripts/export_external_pgs_logit_submission.py`，具体命令及
所需冻结 OOF、历史提交参照见[标准化 logit 导出说明](1013/External_PRS_Standardized_Logit_Submission.md)。
`y_prob`、`wrapper_prob`、Student `probability` 和标准化提交 logit 是不同输出。

## 5. 重新训练蒸馏模型的入口

只做推理无需执行本节。训练需具备完整来源链，计算应在 Slurm allocation 中进行。
当前 BIO 配置引用已完成的 v1 full-train 和 v3 运行，先检查
`configs/prs_distillation_bio.yaml` 中的 `source_run`、`v3_run`。
`prepare` 会核对缓存、软件版本、代码及数据哈希，缺少来源文件时会失败。

若本地已有这两套封存来源，可创建独立配置，只修改输出根目录：

```bash
mkdir -p outputs/prs_distillation_bio
export BIO_WORK
BIO_WORK=$(mktemp -d outputs/prs_distillation_bio/retrain_XXXXXXXX)
uv run --offline --no-sync python - <<'PY'
import os
from pathlib import Path
import yaml

work = Path(os.environ["BIO_WORK"])
config = yaml.safe_load(Path("configs/prs_distillation_bio.yaml").read_text())
config["output_root"] = str(work / "runs")
(work / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
uv run --offline --no-sync python scripts/run_prs_distillation_bio.py prepare \
  --config "$BIO_WORK/config.yaml" \
  --eligibility-config configs/disease_eligibility.yaml
```

将终端最后打印的 `run_dir` 填入下面的 `RUN_DIR`。CPU 阵列依次对各病运行 smoke
和正式参数模型训练；随后仅运行三病全训练范围的 HGB 对照：

```bash
export RUN_DIR='替换为 prepare 打印的 run_dir'
mkdir -p logs/prs_distillation_bio
BIO_JOB=$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR" \
  slurm/prs_distillation_bio.sbatch)
BOOST_JOB=$(sbatch --parsable --dependency="afterok:$BIO_JOB" --array=0-2 \
  --export=ALL,RUN_DIR="$RUN_DIR" slurm/prs_monotone_boosting.sbatch)
squeue -j "$BIO_JOB,$BOOST_JOB"
```

全训练参数模型位于 `core/<trait>/models/`，对应
`core/<trait>/PREDICTIONS_COMPLETE.json`；低阶模型位于
`low_order/<trait>/models/`，对应 `low_order/<trait>/COMPLETE.json`。
作业 ID 只表示已提交，需检查 Slurm 成功状态及这些回执。
新训练的疾病特异选择需要重新锁定，不能复制旧 `POLICY_LOCK.json` 冒充新选择。

其余复现入口及依赖：

| 阶段 | 命令／launcher | 前置要求 |
|---|---|---|
| v1 Teacher＋Student | `run_prs_distillation.py audit`；`slurm/prs_distillation.sbatch` | 使用 `configs/prs_distillation_fulltrain.yaml`；详见 [Phase A 命令](1026/IMPLEMENTATION.md) |
| v3 缓存复用 | `run_prs_distillation_v3.py prepare`；`slurm/prs_distillation_v3.sbatch` | 完成并校验的 v1 来源及 Teacher 缓存 |
| BIO nested Teacher | `slurm/prs_bio_nested_teacher.sbatch` | `RUN_DIR` 为 BIO 根；五个外折 GPU 任务 |
| BIO nested Student | `slurm/prs_bio_nested_student.sbatch` | 五折 Teacher 准备全部成功；15 个 CPU 任务 |
| nested 低阶模型 | `slurm/prs_monotone_boosting.sbatch` | 用 `sbatch --array=3-17 ...` 覆盖阵列；nested Student 已完成 |
| 评价与选择锁 | `report_prs_distillation_bio.py --run-dir ... --scope val` 或 `--scope nested` | 相应模型、预测及脚本 `FM_ROOT` 下的 TabPFN 3.5／Kumo 冻结对照结果完整 |
| 完整实验封存 | `finalize_prs_bio_exploration.py --run-dir ...` | 所有主实验、nested 及规定的附加对照完成 |

上述 `.py` 入口均位于 `scripts/`，使用 `uv run --offline --no-sync python scripts/…`。
只训练 M0 或仅运行全训练范围，不满足完整研究报告／finalizer 的所有依赖。
结果总览与双语方法报告的脚本为 `build_prs_results_overview.py`、
`build_prs_distillation_methods.py`；它们使用已封存的本地源路径及固定输出路径，
运行前需检查脚本常量，避免改写已归档报告。

## 6. 常见问题及完成标准

| 现象 | 处理 |
|---|---|
| `No such file`，找不到 bundle／PRS／checkpoint | 克隆不含这些文件；按第 1 节恢复所需本地资源并核对路径 |
| 模型／缓存 checksum 不匹配 | 恢复匹配版本的文件及环境；不要删除或重写校验回执来绕过检查 |
| `FileExistsError` | 使用新的输出目录或文件名，保留原结果 |
| 缺 PRS 列或非有限输入 | 核对冻结面板、join 覆盖与分数定义；不要自动填零 |
| BC 输出少于输入 | 检查 `n_ineligible` 及女性资格规则；全部不合格时会报错 |
| `age_out_of_support=True` | 输入超出训练支持，模型使用保存的边界外常量年龄外推；核对年龄单位 |
| CUDA 不可用 | 在获得 GPU 的计算节点运行 TabPFN；Student 推理可使用 CPU |
| 离线安装失败 | 先在可联网节点完成锁定依赖安装，计算节点使用 `--offline --no-sync` |

推理完成需同时存在预测文件和成功回执；TabPFN holdout 全流程还需
`FINAL_VERIFIED.json`。方法比较使用共享患者折分并报告 pooled OOF，所有拟合和选择
限于相应训练范围。原 validation、训练 OOF 与无标签 holdout 的用途和结果须分别报告。

本手册发布时实际执行了第 2.2 节的六条推理命令：14,664 条预测与封存概率完全一致，
新增拟合数为 0。第 4 节的 T2D validation 预检及三病 holdout 预检通过；
14 个 Bash 代码块、13 个 CLI 帮助入口、5 个 Slurm launcher 语法和本地文档链接
检查通过。本次仅补充运行文档及验证示例，未重新进行基因型计分、GPU 预测或模型训练。
