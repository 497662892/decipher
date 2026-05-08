# decipEHR PRS Competition

This repository organizes reproducible polygenic risk score (PRS) baselines for the decipEHR DREAM competition. The project code is intentionally glue code: use mature tools for genotype processing, PRS scoring, modeling, and metrics; keep this repository focused on configuration, schema checks, cross-validation orchestration, evaluation, and submission formatting.

## Directory Layout

- `bin/`: shell wrappers for common workflows.
- `jupyter-notebook/`: exploration and prototyping notebooks.
- `src/decipehr/`: reusable library code.
- `scripts/`: command-line pipeline entry points.
- `tests/`: unit and integration tests using synthetic data only.
- `configs/`: versioned configuration templates.
- `docs/`: analysis contracts, mature-tool policy, and GitHub private-repo guidance.

Private data and generated artifacts belong under `data/`, `outputs/`, `models/`, and `submissions/`; these paths are intentionally ignored by Git.

## Environment

This project uses `uv`.

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev ruff check .
```

External PRS tools are not installed by `uv` and should be managed separately: `plink2` for genotype QC/P+T scoring, `bigsnpr` for LDpred2, and PRS-CS for the PRS-CS track.

## Workflow

1. Extract the competition contract from `DREAM_data_V1.pdf`, `04_09 decipEHR project.pdf`, and existing notebooks.
2. Inventory and standardize official/provided summary statistics.
3. Use the newly released official validation labels for model comparison.
4. For each disease, repeat this validation procedure 1000 times: sample 1000 official validation patients for prediction, fit the full logistic regression model on the remaining validation patients, and score the sampled patients only.
5. Select the final model from the repeated-holdout validation summaries before generating the official submission.

```bash
uv run python scripts/inventory_sumstats.py --root data/raw --out data/processed/sumstats_inventory.csv
uv run python scripts/run_official_validation.py \
  --labelled-phenotype /gpfs/data/deciphEHRlab/deciphEHRdream/holdout_phase/phenotype_files/Training_Phase_Cov_file_with_Phenotype.txt \
  --validation-covariates /gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/phenotype_files/Testing_Cov_file.txt \
  --patient-col sampleid \
  --labels cad,bc,T2D \
  --covariates sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd \
  --eligibility-config configs/disease_eligibility.yaml \
  --eval-size 1000 \
  --n-repeats 1000 \
  --seed 42 \
  --validation-phenotype-out data/processed/official_validation_phenotype.csv \
  --out-dir outputs/official_validation_baseline
```

The official validation cohort is defined by
`training_phase/phenotype_files/Testing_Cov_file.txt`. The released labels are
read from `holdout_phase/phenotype_files/Training_Phase_Cov_file_with_Phenotype.txt`
and joined by `sampleid`. If a per-patient PRS table is available, pass it with
`--prs-scores`; columns named `prs_cad`, `prs_bc`, and `prs_T2D` are included in
the disease-specific full logistic model.

After the common summary-statistics layer has produced `harmonized/*.tsv.gz`, prepare
fold-aware P+T task scripts. The generated tasks use training-fold samples for LD
clumping, then score all disease-eligible labeled samples with one multi-threshold
score file per fold/r2 setting.

```bash
uv run python scripts/write_pt_cv_commands.py \
  --disease cad \
  --manifest configs/official_sumstats.yaml \
  --splits data/processed/splits/disease_specific_split_seed42.csv \
  --genotype-prefix outputs/genotype_qc/qc_final/all_chromosomes.qc \
  --harmonized-dir data/processed/sumstats/harmonized \
  --out-dir outputs/pt_cv

sbatch slurm/pt_cv_cad.sbatch

uv run python scripts/collect_pt_cv_scores.py \
  --task-manifest outputs/pt_cv/cad/commands/cad.pt_cv.tasks.csv \
  --out outputs/pt_cv/cad/scores/cad.pt_cv_scores_long.csv

uv run python scripts/evaluate_pt_cv.py \
  --disease cad \
  --scores outputs/pt_cv/cad/scores/cad.pt_cv_scores_long.csv \
  --covariates sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd \
  --out-dir outputs/pt_cv/cad/evaluation
```

LDpred2 uses the same patient-level folds, but delegates the method to `bigsnpr`
in R. The generated fold tasks first convert the disease-eligible scoring set
from PLINK2 pgen to bed, then run `scripts/run_ldpred2_task.R` for one LDpred2
grid per fold.

```bash
uv run python scripts/write_ldpred2_cv_commands.py \
  --disease cad \
  --manifest configs/official_sumstats.yaml \
  --splits data/processed/splits/disease_specific_split_seed42.csv \
  --genotype-prefix outputs/genotype_qc/qc_final/all_chromosomes.qc \
  --harmonized-dir data/processed/sumstats/harmonized \
  --out-dir outputs/ldpred2_cv

sbatch slurm/ldpred2_cv_cad.sbatch

uv run python scripts/collect_ldpred2_cv_scores.py \
  --task-manifest outputs/ldpred2_cv/cad/commands/cad.ldpred2_cv.tasks.csv \
  --out outputs/ldpred2_cv/cad/scores/cad.ldpred2_cv_scores_long.csv

uv run python scripts/evaluate_ldpred2_cv.py \
  --disease cad \
  --scores outputs/ldpred2_cv/cad/scores/cad.ldpred2_cv_scores_long.csv \
  --covariates sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd \
  --out-dir outputs/ldpred2_cv/cad/evaluation
```

For the official GWAS Catalog summary statistics, generate PHI-safe command templates first. These commands only use file paths, public summary statistics, INFO score files, and PLINK2 variant data; run them on an approved node with access to the private data.

```bash
uv run python scripts/write_genotype_qc_commands.py \
  --pfile-pattern '/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/genotype_files/plink_files/Testing_Chr{chr}' \
  --info-pattern '/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/supplementary_files/info_scores/All_Samples_InfoScore_v02_chr{chr}.txt.gz' \
  --out-dir outputs/qc/genotypes \
  --out outputs/qc/genotype_qc.commands.sh

uv run python scripts/write_official_prs_commands.py \
  --manifest configs/official_sumstats.yaml \
  --sumstats-root /gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/supplementary_files/summary_statistics \
  --pfile-prefix outputs/qc/genotypes/all_chromosomes.qc \
  --out outputs/pt/official_prs.commands.sh
```

## Baseline Tiers

- Tier 0: covariate-only logistic regression with scikit-learn.
- Tier 1: P+T using official/provided summary statistics and PLINK2.
- Tier 2: LDpred2 via `bigsnpr`.
- Tier 3: PRS-CS or Lassosum2 if time and compute allow.
- Tier 4: logistic/ridge ensemble using scikit-learn.

If summary statistics are official and independent of the competition training labels, they may be reused across folds. If summary statistics are derived from internal training labels, regenerate them inside each outer fold using only training-fold patients.

## Validation

Disease-specific eligibility rules are configured in `configs/disease_eligibility.yaml`. Breast cancer validation is restricted to eligible female patients. `scripts/run_official_validation.py` writes `eligibility_counts.csv` so the filtered analysis set is auditable.

The default validation strategy is repeated holdout on the official validation set: sample 1000 patients for prediction, fit on the remaining validation patients, and repeat 1000 times per disease. Sampling is label-stratified by default. Select models primarily by the mean challenge OR across repeats: the odds ratio comparing the top 10% of full-model predictions against the lower 90%. Use mean Nagelkerke's R2 as the secondary challenge metric, and keep AUC, AUPRC, Brier score, PRS per-standard-deviation odds ratio, and older risk-stratification odds ratios as supporting diagnostics.

The older disease-specific 5-fold scripts remain in the repository for sensitivity checks and for workflows that need fold-aware PRS generation, but they are no longer the primary validation path now that the official validation labels are available.

## GitHub Hygiene

Use a private repository. Commit code, configuration templates, documentation, and synthetic tests. Do not commit real genotype, phenotype, summary statistics, model weights, outputs, or submissions. See `docs/github_private_repo.md`.
