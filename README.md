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
3. Create disease-specific patient-level 5-fold splits for fair per-disease model comparison.
4. Run covariate-only, P+T, stronger PRS, and ensemble comparisons using out-of-fold predictions.
5. Retrain the selected final pipeline on all official training data before generating the official submission.

```bash
uv run python scripts/inventory_sumstats.py --root data/raw --out data/processed/sumstats_inventory.csv
uv run python scripts/prepare_data.py --phenotype data/raw/Training_Cov_file.txt --patient-col sampleid --labels cad,bc,T2D --covariates sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd --out data/processed/phenotype.csv
uv run python scripts/create_ancestry_clusters.py --phenotype data/processed/phenotype.csv --patient-col sampleid --labels cad,bc,T2D --age-col age_sd --sex-col sex --clusters-out data/processed/ancestry_clusters.csv --report-out outputs/ancestry/ancestry_cluster_report.csv --quality-out outputs/ancestry/ancestry_cluster_quality.csv --plot-out outputs/ancestry/pc1_pc2_clusters.svg
uv run python scripts/make_splits.py --phenotype data/processed/phenotype.csv --patient-col sampleid --labels cad,bc,T2D --disease-specific --ancestry-clusters data/processed/ancestry_clusters.csv --age-col age_sd --sex-col sex --sex-balance-labels cad,T2D --eligibility-config configs/disease_eligibility.yaml --diagnostics-out outputs/cv/disease_specific_split_summary.csv --out data/processed/splits/disease_specific_split_seed42.csv
uv run python scripts/run_cv.py --phenotype data/processed/phenotype.csv --splits data/processed/splits/disease_specific_split_seed42.csv --patient-col sampleid --labels cad,bc,T2D --covariates sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd --eligibility-config configs/disease_eligibility.yaml --out-dir outputs/cv_baseline
```

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

Disease-specific eligibility rules are configured in `configs/disease_eligibility.yaml`. Breast cancer CV is restricted to eligible female patients; sex is not used as an extra BC split target. `scripts/run_cv.py` writes `eligibility_counts.csv` so the filtered analysis set is auditable.

Use the same disease-specific 5 folds for every method compared within that disease. Select models primarily by the challenge OR: the odds ratio comparing the top 10% of full-model predictions against the lower 90%. Use Nagelkerke's R2 as the secondary challenge metric, and keep AUC, AUPRC, Brier score, PRS per-standard-deviation odds ratio, and older risk-stratification odds ratios as supporting diagnostics.

## GitHub Hygiene

Use a private repository. Commit code, configuration templates, documentation, and synthetic tests. Do not commit real genotype, phenotype, summary statistics, model weights, outputs, or submissions. See `docs/github_private_repo.md`.
