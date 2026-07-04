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

## Pipeline

The workflow has three layers:

1. **Preprocessing**
   - Build audited train, validation, train-val, and holdout sample sets.
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
   - Use one shared patient-level 5-fold split for fair train-val comparison.

## Validation Summary

Primary metric: `OR90`, the odds ratio comparing the top 10% of predictions
against the lower 90%. AUC is included as a familiar diagnostic.

| Disease | Covariates only AUC / OR90 | Best single PRS AUC / OR90 | Earlier PRS-only TabPFN AUC / OR90 | Final TabPFN setting | Final official-val AUC / OR90 | Final direct-logit AUC / OR90 |
|---|---:|---:|---:|---|---:|---:|
| CAD | 0.791 / 5.05 | PRS-CS 0.796 / 5.36 | 0.791 / 5.05 | drop SBayesRC, lassosum2, P+T | 0.808 / 7.49 | 0.808 / 7.32 |
| BC | 0.614 / 1.43 | lassosum2 0.644 / 2.09 | 0.628 / 1.77 | drop SDPR, MegaPRS no annot | 0.665 / 2.09 | 0.669 / 2.54 |
| T2D | 0.705 / 4.30 | MegaPRS annotated 0.741 / 6.52 | 0.732 / 5.75 | drop P+T, PRS-CS, lassosum2 | 0.758 / 6.65 | 0.759 / 6.90 |

Notes:

- The PRS-only TabPFN baseline exists in earlier records under
  `outputs/tabpfn3_grid_nonannotated_a0p20/.../official_validation_prs_only/`.
  It was useful as a baseline but was not the final model.
- The final official-val column evaluates the TabPFN-derived score through the
  fixed official validation logistic interface.
- The final direct-logit column evaluates the raw TabPFN score, matching the
  final submission scoring style more closely.

## Final Model

The final selected approach was a disease-specific TabPFN-3 ensemble using
selected PRS inputs plus `sex`, `age_sd`, and `PC1_sd` to `PC10_sd`.

| Disease | Kept PRS families | Dropped PRS families |
|---|---|---|
| CAD | LDpred2, PRS-CS, SDPR, MegaPRS no annot, MegaPRS annotated | SBayesRC 7M, lassosum2, P+T |
| BC | P+T, LDpred2, lassosum2, PRS-CS, MegaPRS annotated, SBayesRC 7M | SDPR, MegaPRS no annot |
| T2D | LDpred2, SDPR, MegaPRS no annot, MegaPRS annotated, SBayesRC 7M | P+T, PRS-CS, lassosum2 |

Final scoring:

- TabPFN-3 with `clip_eps=0.001` and `n_estimators=16`.
- Train-val scores: shared 5-fold out-of-fold TabPFN logits.
- Holdout scores: TabPFN logits using all eligible train-val patients as
  context.
- Standardization: disease-specific mean/sd from train-val 5-fold OOF logits.
- Holdout samples were not used for model selection or standardization.

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

Final-phase train-val/holdout setup:

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

TabPFN and final train-val wrappers:

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

### 7. Run the Final Holdout Pipeline

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
- Validation folds and holdout samples are prediction-only.
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
