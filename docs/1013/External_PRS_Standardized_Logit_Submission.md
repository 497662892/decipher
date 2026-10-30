# External PRS：与原 GoGene GitHub 一致的标准化 logit 提交

2026-10-13。本轮复用已经完成的预测，恢复历史提交的输出尺度、表头和患者顺序。

用户指定的 [GoGene GitHub output](https://github.com/nyu-shenlab/dreams-team-gogene/tree/main/output)
包含三份制表符分隔的 TXT。它们与本地
`submissions/final_phase/Go Gene!_audited_pipeline_with_trainval_5cv/`
三份疾病提交文件的 SHA256 完全一致，分值为标准化后的直接 TabPFN logit。

## 新版文件

本地目录：`submissions/final_phase/gogene_external_pgs_20261013/`。

| 文件 | 内容 |
|---|---|
| `output/gogene_T2D.txt` | `sampleid, T2D_score`；8,663 OOF + 3,922 holdout = 12,585 人 |
| `output/gogene_cad.txt` | `sampleid, cad_score`；8,663 OOF + 3,922 holdout = 12,585 人 |
| `output/gogene_bc.txt` | `sampleid, bc_score`；4,936 女性 OOF + 2,187 女性 holdout = 7,123 人 |
| `gogene_external_pgs_standardized_logits.zip` | 上述三份 TXT，置于压缩包根目录 |
| `holdout_standardized_logits.csv` | 3,922 人的 holdout 宽表，含三个疾病的标准化 logit |
| `scores.long.csv` | 保留 cohort、fold、direct_probability、direct_logit 和 score 的长表 |
| `standardization.csv` | 每疾病的 OOF 均值/标准差、截断数、分数范围及转换前后 OOF AUC |
| `run_record.json`、`VERIFIED.json` | 来源、哈希、模型配置及独立核查记录 |

TXT 实际以 tab 分隔，上表用逗号列举表头。宽表中不符合 BC 女性规则的 1,735 人 BC 留空；BC TXT 中不包含这些人。所有患者级导出文件权限为 0600；本轮仅生成本地文件。

## 分数定义及来源

每疾病使用 `logit(clip(p, 0.001, 0.999))`，再减去 train+development 五折 OOF logits 的均值，除以其总体标准差（ddof=0）。同一组参数应用于该疾病 OOF 和 holdout；holdout 不参与估计均值或标准差。这是最终提交的尺度转换。

`p` 来自直接 TabPFN 输出 `y_prob`；现有 Logistic wrapper 的 `wrapper_prob` 不用于此提交。旧 probability 文件保持原状。

- 五折 OOF：`outputs/external_pgs/frozen_internal_external_trainval_20261013/report/oof_predictions.csv` 的 external arm。
- Holdout：`outputs/external_pgs/holdout_wrapper_20261012/holdout_predictions.long.csv`，由完整 train+development 作为 context 预测。
- T2D/CAD/BC 分别使用 26/7/5 个冻结外部 PRS；BC 包括 Ho 2022 ASN/EUR。
- OOF 和 holdout 的 checkpoint、PRS 面板、训练队列、方法及 seed 已核对。Holdout CSV 原先以紧凑文本保存 float32 概率，读取时先恢复 float32，再提升至 float64，与原预测核查规则一致。

## 验证结果

172 个来源文件哈希复核通过；15 个外部面板 OOF 折及完整 train+development holdout context 通过患者哈希核查。三份 TXT 与历史文件的患者集合、顺序和表头一致。独立分数复算及逆变换的最大绝对误差为 `1.7763568394002505e-15`；ZIP 解压内容核查通过。

| 疾病 | 直接概率 pooled OOF AUC | 提交分数 pooled OOF AUC | OOF 截断人数 | Holdout 截断人数 |
|---|---:|---:|---:|---:|
| T2D | 0.7581874935 | 0.7581874935 | 55 | 0 |
| CAD | 0.8048324110 | 0.8048324110 | 26 | 0 |
| BC | 0.7112224892 | 0.7112224892 | 33 | 1 |

这次恢复提交格式和尺度，没有重新训练模型，AUC 也未因格式转换提高。两套 PRS 面板此前有 development 选择历史；这里的 OOF 条件于冻结面板。用户已确认没有 holdout 标签，故没有 holdout 性能指标。

工程验证：225 passed、2 skipped；两个跳过项为默认不启用的 PLINK/外部 PGS 和 DiscoDivas R 集成测试。Ruff 和 git diff --check 通过。

## 重现

在仓库根目录运行；导出器拒绝覆盖已有目录，重现时指定新的输出目录。

```bash
uv run --no-sync python scripts/export_external_pgs_logit_submission.py \
  --comparison-dir outputs/external_pgs/frozen_internal_external_trainval_20261013 \
  --holdout-dir outputs/external_pgs/holdout_wrapper_20261012 \
  --eligibility-config configs/disease_eligibility.yaml \
  --out-dir submissions/final_phase/gogene_external_pgs_reproduction
```

算法导出和患者/折号验证位于 `src/decipehr/submission.py`，合成测试位于 `tests/test_submission.py`。流程先验证来源与队列，再导出、读回和独立复算；不会提交训练任务或上传 GitHub。
