# decipEHR PRS Competition

This repository contains the workflow code used for the decipEHR DREAM
polygenic risk score (PRS) competition. It is a public reference for organizing
PRS preprocessing, candidate PRS methods, model validation, and PRS ensembles.

The repository is intentionally glue code. Mature genetics tools do the
domain-heavy work; this code handles validation, sample sets, command
generation, score collection, metric aggregation, ensemble inputs, and
submission formatting.

Private competition data, genotype files, phenotype files, summary statistics,
model outputs, and submissions are not included. Keep them in ignored local
folders such as `data/`, `outputs/`, `models/`, and `submissions/`.

## Key Idea

The main modeling strategy is disease-specific PRS integration. Instead of
choosing only one PRS score, the pipeline builds multiple candidate PRS families,
selects useful inputs per disease, and uses TabPFN-3 to combine selected PRS
scores with covariates (`sex`, `age_sd`, and ancestry PCs). The resulting
TabPFN score is then evaluated through the same official logistic validation
interface as the single-PRS baselines.

## Pipeline

The workflow has three layers:

1. **Preprocessing**
   - Build audited train, official validation, and final test sample sets.
   - Run genotype QC with PLINK2.
   - Harmonize allowed summary statistics to:
     `SNP, CHR, BP, A1, A2, P, BETA`, with optional `SE` and `N`.
   - Apply disease eligibility rules, especially female-only breast cancer
     analyses through `configs/disease_eligibility.yaml`.

2. **Candidate single PRS**
   - P+T via PLINK2.
   - LDpred2 via `bigsnpr`.
   - PRS-CS via the official implementation.
   - lassosum2, SDPR, sBayesRC, and MegaPRS when resources are available.
   - Every candidate is evaluated with the same official validation interface.

3. **Ensemble**
   - Select one validation-tuned candidate from each PRS family.
   - Build long-format PRS inputs.
   - Compare linear/ridge stacking and TabPFN-3 ensembles.
   - When using cross-validation, use one shared patient-level 5-fold split for
     fair method comparison.

## Validation Summary

Primary metric: `OR90`, the odds ratio comparing the top 10% of scores against
the lower 90%. AUC is included as a familiar diagnostic.

The main validation table uses the same official logistic validation interface:

- **Covariates only**: logistic model using covariates but no PRS score.
- **Model score + covariates**: logistic model using one PRS or ensemble score
  plus `sex`, `age_sd`, and `PC1_sd` to `PC10_sd`.
- Breast cancer metrics use the female-only eligibility rule in
  `configs/disease_eligibility.yaml`.

### Official Validation Baselines

**CAD**

| Method | AUC | OR90 |
|---|---:|---:|
| Covariates only | 0.791 | 5.05 |
| Best single PRS: MegaPRS annotated + covariates | 0.795 | 5.92 |
| TabPFN-3 PRS-only score + covariates | 0.791 | 5.05 |
| Final TabPFN-3 score + covariates | 0.809 | 7.49 |

**BC**

| Method | AUC | OR90 |
|---|---:|---:|
| Covariates only | 0.614 | 1.43 |
| Best single PRS: lassosum2 + covariates | 0.644 | 2.09 |
| TabPFN-3 PRS-only score + covariates | 0.628 | 1.77 |
| Final TabPFN-3 score + covariates | 0.662 | 2.46 |

**T2D**

| Method | AUC | OR90 |
|---|---:|---:|
| Covariates only | 0.705 | 4.30 |
| Best single PRS: MegaPRS annotated + covariates | 0.741 | 6.90 |
| TabPFN-3 PRS-only score + covariates | 0.732 | 5.75 |
| Final TabPFN-3 score + covariates | 0.763 | 8.04 |

Best single PRS settings in the table:

| Disease | Single PRS baseline | PRS hyperparameters |
|---|---|---|
| CAD | MegaPRS annotated | `model=megaprs_bayesr` |
| BC | lassosum2 | `delta=1`, `lambda=0.0125357276756835` |
| T2D | MegaPRS annotated | `model=megaprs_bayesr` |

All three rows use the same validation wrapper: PRS score plus `sex`, `age_sd`,
and `PC1_sd` to `PC10_sd`, fitted with
`LogisticRegression(solver="lbfgs", max_iter=2000, class_weight=None)`.

Summary statistics used for PRS construction:

| Disease | Source | Accession | Trait | PubMed ID | Default `N` used when needed |
|---|---|---|---|---|---:|
| CAD | GWAS Catalog | `GCST003116` | Coronary artery disease | `26343387` | 158,816 |
| BC | GWAS Catalog | `GCST90479801` | Breast cancer [female] / PheCode 174.1 | `39024449` | 11,777 |
| T2D | GWAS Catalog | `GCST90479885` | Type 2 diabetes / PheCode 250.2 | `39024449` | 572,189 |

These files are configured in `configs/official_sumstats.yaml`, standardized to
`SNP, CHR, BP, A1, A2, P, BETA` plus optional `SE` and `N`, then harmonized to
the target genotype variants before P+T, LDpred2, PRS-CS, lassosum2, SDPR,
sBayesRC, and MegaPRS scoring. When per-variant `N` was unavailable, the
default `N` above was used as the summed ancestry-specific effective sample size
recorded in the manifest.

Main takeaways:

- The PRS-only TabPFN-3 baseline was the key nonlinear-ensemble baseline: it
  used only PRS features inside TabPFN, then entered the same logistic
  validation interface as the other model scores.
- Best single PRS baselines are also shown through the same logistic interface,
  making them comparable to the ensemble scores.
- The final gain came from disease-specific PRS selection plus a TabPFN score
  trained with both selected PRS and covariate features.

Direct score-only ranking checks were also useful during development, but they
are not the main comparison because they do not use the official logistic
validation interface.

The final selected setting was not simply "use every PRS". Disease-specific
input dropping was used to choose the final PRS families, then the selected
input set was carried into the final scoring pipeline.

## Final Model

The final selected approach was a disease-specific TabPFN-3 ensemble using
selected PRS inputs plus covariates as model input features. The covariates are
`sex`, `age_sd`, and `PC1_sd` to `PC10_sd`.

| Disease | Kept PRS families | Dropped PRS families |
|---|---|---|
| CAD | LDpred2, PRS-CS, SDPR, MegaPRS no annot, MegaPRS annotated | SBayesRC 7M, lassosum2, P+T |
| BC | P+T, LDpred2, lassosum2, PRS-CS, MegaPRS annotated, SBayesRC 7M | SDPR, MegaPRS no annot |
| T2D | LDpred2, SDPR, MegaPRS no annot, MegaPRS annotated, SBayesRC 7M | P+T, PRS-CS, lassosum2 |

Final scoring:

- TabPFN-3 with `clip_eps=0.001` and `n_estimators=16`.
- TabPFN input features: selected PRS family scores plus `sex`, `age_sd`, and
  `PC1_sd` to `PC10_sd`.
- Official validation scores: evaluated through the logistic validation
  interface reported above.
- Final test scores: TabPFN logits using the available labeled samples as
  context.
- Final test samples were prediction-only and were not used for model
  selection.

## Repository Layout

- `src/decipehr/`: reusable Python package.
- `scripts/`: command-line pipeline steps.
- `slurm/`: HPC wrappers for the major workflows; see `slurm/README.md`.
- `configs/`: project and eligibility configuration.
- `tests/`: synthetic tests only.
- `docs/`: longer design notes.
- `jupyter-notebook/`: exploration notebooks.

## Environment

Python dependencies are managed with `uv`.

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev ruff check .
```

TabPFN is optional:

```bash
uv sync --extra dev --extra tabpfn
printf 'TABPFN_TOKEN=<your-priorlabs-api-key>\n' > .env.tabpfn
chmod 600 .env.tabpfn
bin/setup_tabpfn.sh
```

External tools are installed separately:

- `plink2`
- R plus `bigsnpr`
- official PRS-CS
- GCTB for sBayesRC
- LDAK for MegaPRS
- SDPR and lassosum2 dependencies

On HPC, use `module avail <tool>` and `module load <tool>` as needed.

## Local Inputs Needed

The repository does not include private competition inputs. To run the pipeline
end to end, provide local paths for:

- PLINK2-compatible genotype files for labeled, validation, and final test
  samples.
- Phenotype/covariate tables with sample IDs, labels where available, `sex`,
  `age_sd`, and `PC1_sd` to `PC10_sd`.
- Official or otherwise allowed GWAS summary statistics for each PRS method.
- Method-specific resources such as LD reference panels, external binaries,
  and ancestry reference files.

The example commands below assume these inputs are linked under `data/raw/` and
that local paths are configured in `configs/default.yaml`, `configs/qc.yaml`,
and `configs/official_sumstats.yaml`.

## Step-by-Step Use

Create local artifact folders:

```bash
mkdir -p data/raw data/processed data/resources outputs models submissions
```

Place or link competition data under `data/raw/`, then adapt
`configs/default.yaml`, `configs/qc.yaml`, and `configs/official_sumstats.yaml`
to your local paths.

### 1. Inventory and Harmonize Summary Statistics

```bash
uv run python scripts/inventory_sumstats.py \
  --root data/raw \
  --out data/processed/sumstats_inventory.csv

uv run python scripts/write_sumstats_common_commands.py \
  --manifest configs/official_sumstats.yaml \
  --sumstats-root data/raw/training_phase/supplementary_files/summary_statistics \
  --target-pvar outputs/genotype_qc/qc_final/all_chromosomes.qc.pvar \
  --out-dir data/processed/sumstats \
  --out outputs/sumstats_common/sumstats_common.commands.sh
```

Slurm wrappers: `slurm/sumstats_common_*.sbatch` and
`slurm/sumstats_final_common.sbatch`.

### 2. Preprocess Samples and Genotypes

```bash
uv run python scripts/write_genotype_qc_commands.py \
  --config configs/qc.yaml \
  --out outputs/qc/genotype_qc.commands.sh

uv run python scripts/make_sample_sets.py \
  --genotype-psam outputs/genotype_qc/qc_final/all_chromosomes.qc.psam \
  --phenotype data/raw/training_phase/phenotype_files/Training_Cov_file.txt \
  --validation-covariates data/raw/training_phase/phenotype_files/Testing_Cov_file.txt \
  --patient-col sampleid \
  --out-dir data/processed/sample_sets
```

Final-phase labeled/test setup:

```bash
uv run python scripts/prepare_final_trainval_test.py \
  --train-val-phenotype data/raw/holdout_phase/phenotype_files/Training_Phase_Cov_file_with_Phenotype.txt \
  --holdout-covariates data/raw/holdout_phase/phenotype_files/Holdout_Cov_file.txt \
  --train-qc-psam outputs/genotype_qc/qc_final/all_chromosomes.qc.psam \
  --holdout-psam data/raw/holdout_phase/genotype_files/plink_files/Holdout_chr1.psam \
  --out-dir data/processed/final
```

Slurm wrappers: `slurm/preprocess_data.sbatch`,
`slurm/final_genotype_*.sbatch`, and
`slurm/final_trainval_make_trainval_test_pfile.sbatch`.

### 3. Run Single PRS Candidates

Example for P+T:

```bash
uv run python scripts/write_pt_full_train_commands.py \
  --disease cad \
  --manifest configs/official_sumstats.yaml \
  --genotype-prefix outputs/genotype_qc/qc_final/all_chromosomes.qc \
  --harmonized-dir data/processed/sumstats/harmonized \
  --train-keep data/processed/sample_sets/labeled_train.keep.psam \
  --score-keep data/processed/sample_sets/unlabelled_validation.keep.psam \
  --out-dir outputs/pt_full_train

sbatch slurm/pt_full_train_cad.sbatch
```

Other command generators:

- `scripts/write_ldpred2_full_train_commands.py`
- `scripts/write_prscs_full_train_commands.py`
- `scripts/write_lassosum2_full_train_commands.py`
- `scripts/write_sdpr_full_train_commands.py`
- `scripts/write_sbayesrc_full_train_commands.py`
- `scripts/write_megaprs_full_train_commands.py`

Slurm wrappers are included for each family, for example
`slurm/ldpred2_full_train_*.sbatch`, `slurm/prscs_full_train_*.sbatch`,
`slurm/sdpr_full_train_*.sbatch`, `slurm/sbayesrc_full_train_*.sbatch`, and
`slurm/megaprs_full_train_*.sbatch`.

### 4. Validate Models

```bash
uv run python scripts/validate_official_model.py \
  --model-name pt \
  --disease cad \
  --train-prs-long-scores outputs/pt_full_train/cad/scores/cad.pt_full_train_training_scores_long.csv \
  --validation-prs-long-scores outputs/pt_full_train/cad/scores/cad.pt_full_train_scores_long.csv \
  --eligibility-config configs/disease_eligibility.yaml \
  --out-dir outputs/official_validation_pt/cad
```

The validator writes point metrics, bootstrap summaries, eligibility counts,
and provenance.

### 5. Build Ensembles

```bash
uv run python scripts/build_ensemble_inputs.py \
  --disease cad \
  --families pt,ldpred2,lassosum2,prscs,sdpr,sbayesrc,megaprs \
  --out-dir outputs/ensemble_inputs

uv run python scripts/build_ensemble_prs.py \
  --disease cad \
  --train-prs-long-scores outputs/ensemble_inputs/cad.train_long.csv \
  --validation-prs-long-scores outputs/ensemble_inputs/cad.validation_long.csv \
  --out-dir outputs/ensemble/cad
```

TabPFN and final-phase wrappers:

- `slurm/tabpfn_ensemble.sbatch`
- `slurm/final_trainval_tabpfn_cv.sbatch`
- `slurm/final_trainval_tabpfn_cv_inputs.sbatch`
- `slurm/final_trainval_tabpfn_final_inputs.sbatch`
- `slurm/final_trainval_tabpfn_final_predict.sbatch`

### 6. Summarize Performance

```bash
uv run python scripts/collect_model_performance.py \
  --manifest configs/performance_manifest.example.csv \
  --rank-metric top10_vs_lower90_or \
  --out outputs/performance/model_performance.csv \
  --wide-out outputs/performance/model_performance.wide.csv
```

### 7. Run the Final Prediction Pipeline

```bash
uv run python scripts/run_final_tabpfn_oof_holdout_pipeline.py \
  --train-phenotype data/processed/final/train_val_phenotype.csv \
  --splits data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv \
  --holdout-covariates data/processed/final/holdout_test_covariates.csv \
  --train-prs-long-scores 'outputs/tabpfn3_final_trainval_selected_inputs/{disease}.train_long.csv' \
  --holdout-prs-long-scores 'outputs/tabpfn3_final_trainval_selected_inputs/{disease}.test_long.csv' \
  --eligibility-config configs/disease_eligibility.yaml \
  --out-dir submissions/final_phase/tabpfn3_audited_pipeline
```

Slurm wrapper: `slurm/final_tabpfn_oof_holdout_pipeline.sbatch`.

## Validation Rules

- Use one shared patient-level 5-fold split for method comparison.
- Never put the same patient in more than one fold.
- Fit imputers, scalers, PRS parameters, model parameters, and ensemble weights
  using training folds only.
- Validation and final test samples are prediction-only.
- Use `configs/disease_eligibility.yaml` for disease eligibility.
- If summary statistics are derived from internal training labels, regenerate
  them inside each outer fold using only training-fold patients.

## Public Release Hygiene

Do not commit real genotype, phenotype, summary-statistics, submission, model,
cache, or output files unless explicitly approved for public release. Synthetic
fixtures in `tests/` are allowed.

Before sharing changes:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
git status --short
```
