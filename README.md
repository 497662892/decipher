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
4. For each disease, fit the logistic model once on the original training phenotype, predict all official validation patients, and estimate OR/Nagelkerke R2 with 1000 bootstrap resamples of the prediction table.
5. For final holdout submission, retrain PRS generation and PRS-based logistic models on the combined training+validation labelled phenotype, then score the holdout set.

```bash
uv run python scripts/inventory_sumstats.py --root data/raw --out data/processed/sumstats_inventory.csv
uv run python scripts/validate_official_model.py \
  --model-name baseline \
  --out-dir outputs/official_validation_baseline
```

The official validation cohort is defined by
`training_phase/phenotype_files/Testing_Cov_file.txt`. The released labels are
read from `holdout_phase/phenotype_files/Training_Phase_Cov_file_with_Phenotype.txt`
and joined by `sampleid`. If selected per-patient PRS tables are available, pass
training scores with `--train-prs-scores` and validation scores with
`--prs-scores`; columns named `prs_cad`, `prs_bc`, and `prs_T2D` are included in
the disease-specific full logistic model. For PRS grids, pass training and
validation long tables with `--train-prs-long-scores` and `--prs-long-scores`.

For validation, PRS models should be built on the full original training set,
not on fold-specific CV training subsets. For P+T, use `labeled_train.keep.psam`
for LD clumping/model construction and score the official validation samples:

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

uv run python scripts/collect_pt_scores.py \
  --task-manifest outputs/pt_full_train/cad/commands/cad.pt_full_train.tasks.csv \
  --out outputs/pt_full_train/cad/scores/cad.pt_full_train_scores_long.csv

uv run python scripts/collect_pt_scores.py \
  --task-manifest outputs/pt_full_train/cad/commands/cad.pt_full_train.tasks.csv \
  --sscore-column fit_sscore \
  --out outputs/pt_full_train/cad/scores/cad.pt_full_train_training_scores_long.csv

uv run python scripts/validate_official_model.py \
  --model-name pt \
  --disease cad \
  --train-prs-long-scores outputs/pt_full_train/cad/scores/cad.pt_full_train_training_scores_long.csv \
  --validation-prs-long-scores outputs/pt_full_train/cad/scores/cad.pt_full_train_scores_long.csv \
  --out-dir outputs/official_validation_pt/cad
```

The PRS logistic evaluation needs the same PRS parameter grid scored on both the
model-training patients and the validation patients. Generate the training-score
long table with the same score files before running the command above.

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

LDpred2 delegates the method to `bigsnpr` in R. Prepare a HapMap3+
target-variant extract first so generated tasks convert only HapMap3+ overlap
variants, then compute LD with genetic-map positions. For official validation,
use the full original training samples to fit the LDpred2 grid, then score both
the model-training patients and official validation patients.

```bash
module load r/4.5.3
export R_LIBS_USER=$PWD/data/resources/R/4.5.3-library
Rscript scripts/install_ldpred2_r_deps.R

mkdir -p data/resources/ldpred2 data/resources/genetic_maps
curl -L --fail \
  -o data/resources/ldpred2/map_hm3_plus.rds \
  https://api.figshare.com/v2/file/download/37802721

Rscript scripts/prepare_ldpred2_hm3plus_extract.R \
  --hm3-plus-map data/resources/ldpred2/map_hm3_plus.rds \
  --target-pvar outputs/genotype_qc/qc_final/all_chromosomes.qc.pvar \
  --build hg38 \
  --out-prefix data/processed/ldpred2/hm3plus_target

# Grid exploration: h2 x p x sparse.
uv run python scripts/write_ldpred2_full_train_commands.py \
  --disease cad \
  --manifest configs/official_sumstats.yaml \
  --genotype-prefix outputs/genotype_qc/qc_final/all_chromosomes.qc \
  --harmonized-dir data/processed/sumstats/harmonized \
  --train-keep data/processed/sample_sets/labeled_train.keep.psam \
  --score-keep data/processed/sample_sets/unlabelled_validation.keep.psam \
  --variant-extract data/processed/ldpred2/hm3plus_target.snplist \
  --ld-cache-dir outputs/ldpred2_ld_cache \
  --genetic-map-dir data/resources/genetic_maps \
  --out-dir outputs/ldpred2_full_train

VARIANT_EXTRACT=data/processed/ldpred2/hm3plus_target.snplist \
LD_CACHE_DIR=outputs/ldpred2_ld_cache \
GENETIC_MAP_DIR=data/resources/genetic_maps \
REGENERATE_LDPRED2_TASKS=1 \
sbatch slurm/ldpred2_full_train_cad.sbatch

uv run python scripts/collect_ldpred2_scores.py \
  --task-manifest outputs/ldpred2_full_train/cad/commands/cad.ldpred2_full_train.tasks.csv \
  --out outputs/ldpred2_full_train/cad/scores/cad.ldpred2_full_train_scores_long.csv

uv run python scripts/collect_ldpred2_scores.py \
  --task-manifest outputs/ldpred2_full_train/cad/commands/cad.ldpred2_full_train.tasks.csv \
  --score-file-column fit_score_file \
  --out outputs/ldpred2_full_train/cad/scores/cad.ldpred2_full_train_training_scores_long.csv

uv run python scripts/validate_official_model.py \
  --model-name ldpred2 \
  --disease cad \
  --train-prs-long-scores outputs/ldpred2_full_train/cad/scores/cad.ldpred2_full_train_training_scores_long.csv \
  --validation-prs-long-scores outputs/ldpred2_full_train/cad/scores/cad.ldpred2_full_train_scores_long.csv \
  --out-dir outputs/official_validation_ldpred2/cad

# LDpred2-auto sensitivity run with the UKB HM3+ LD reference.
# Disease wrappers default to grid-informed auto starts from the UKB validation run:
# T2D h2=0.05 p=0.1,0.3,1; cad h2=0.2 p=0.03,0.1,0.3; bc h2=0.2 p=0.003,0.01,0.03.
VARIANT_EXTRACT=data/processed/ldpred2/hm3plus_target.snplist \
REGENERATE_LDPRED2_TASKS=1 \
sbatch slurm/ldpred2_auto_full_train_cad.sbatch

sbatch slurm/ldpred2_auto_full_train_eval_cad.sbatch
```

The older `scripts/write_ldpred2_cv_commands.py` and
`scripts/evaluate_ldpred2_cv.py` path is still useful for sensitivity checks and
OOF diagnostics, but the full-train path above is the primary route for official
validation and holdout scoring.

LDpred2 writes the expensive LD correlation matrix as a sparse SFBM cache under
`LD_CACHE_DIR` or `--ld-cache-dir`. Repeated runs reuse the cache when the bed
files, training samples, matched variants, genetic map, and LD window metadata
match. Set `REUSE_LDPRED2_LD=0` to force a rebuild after intentionally changing
those inputs. If task scripts already exist from an older run, set
`REGENERATE_LDPRED2_TASKS=1` once so the Slurm wrapper rewrites them with the
current cache settings.

If a harmonized summary-statistics file lacks valid per-variant `N`/`N_eff`,
set `default_n` for that trait in `configs/official_sumstats.yaml` or override
with `DEFAULT_N=<trusted sample size>` when submitting the LDpred2 job. This is
only a runtime LDpred2 fallback; standardized and harmonized sumstats keep
provider-supplied sample-size columns only. The CAD GCST003116 manifest entry
uses a study-level effective sample size because both the raw and harmonised
source files lack per-variant sample size. The LDpred2 task script derives
missing `SE` values from `BETA` and `P`, but sample size must come from the
source summary statistics or an explicit default.

sBayesRC delegates the method to official GCTB. Project code prepares GCTB
`.ma` inputs, generates `--impute-summary` and `--sbayes RC` commands, maps
GCTB `*.snpRes` posterior effects back to target variant IDs, and scores with
PLINK2. Before submitting jobs, install or expose GCTB and download a matching
sBayesRC LD eigen reference plus annotation file from the GCTB resources. The
LD eigen reference and annotation SNP IDs must match the rsIDs written to the
`.ma` file.

```bash
# Required local resources. Example paths; replace with the actual downloaded GCTB resources.
export GCTB_BIN=data/resources/gctb/gctb
export LDM_EIGEN_DIR=data/resources/sbayesrc/ukbEUR_HM3
export ANNOT_FILE=data/resources/sbayesrc/annotation/annot_baseline2.2.txt

# Generate and run the full workflow: training allele frequencies -> .ma ->
# GCTB --impute-summary -> GCTB --sbayes RC -> PLINK2 scores.
LDM_EIGEN_DIR="$LDM_EIGEN_DIR" \
ANNOT_FILE="$ANNOT_FILE" \
GCTB_BIN="$GCTB_BIN" \
VARIANT_EXTRACT=data/processed/ldpred2/hm3plus_target.snplist \
sbatch slurm/sbayesrc_full_train_cad.sbatch

sbatch slurm/sbayesrc_full_train_eval_cad.sbatch
```

The Slurm wrapper fits sBayesRC with the labelled training patients only. It
uses those same training samples to compute target allele frequencies for the
GCTB `.ma` file, then scores both training and official validation samples for
model selection and reporting. The downloaded default above uses the official
EUR HapMap3 LD eigen reference. If you intentionally use a 7M sBayesRC LD
reference, set `LDM_EIGEN_DIR` and `VARIANT_EXTRACT` to paths compatible with
that reference. To start with all QC target variants before GCTB summary
imputation, submit with `VARIANT_EXTRACT=""`; the default remains the HM3-plus
target extract.

MegaPRS delegates prediction to LDAK. Project code follows the official two-step
MegaPRS workflow: prepare LDAK-format summary statistics
(`Predictor, A1, A2, BETA, SE, n, A1Freq`), build a PLINK bed reference panel
from labelled training samples, calculate predictor correlations with
`--calc-cors`, fit effects with `--mega-prs`, convert `*.effects` to PLINK2
score files, and run the same official validation workflow.

For the official-like profile, first install or expose LDAK and download the
LDAK resources:

```bash
export LDAK_BIN=/path/to/ldak
sbatch slurm/download_megaprs_resources.sbatch
```

Then run one disease:

```bash
export LDAK_BIN=/path/to/ldak

LDAK_BIN="$LDAK_BIN" \
MEGAPRS_MODEL=bayesr \
sbatch slurm/megaprs_full_train_cad.sbatch

sbatch slurm/megaprs_full_train_eval_cad.sbatch
```

The default MegaPRS profile is now `MEGAPRS_PROFILE=target`: it uses Berisa
breakpoints (`data/resources/megaprs/berisa.txt`), drops ambiguous A/T and C/G
summary-stat rows, builds the LD reference from predictors that remain in the
LDAK summary-stat file, and passes `--check-MCMC NO` so LDAK does not abort
when the target LD correlations closely match the summary statistics.
BaselineLD v2.2 annotations are not enabled by default because those files use
rsID predictor names, while the decipEHR target reference uses internal
`c*_...` variant IDs. Only set `MEGAPRS_PROFILE=annotated` or provide
`ANNOTATION_NUMBER`/`ANNOTATION_PREFIX` after building an rsID-named MegaPRS
reference or after remapping the annotations to target IDs.
`MEGAPRS_MODEL=bayesr` keeps LDAK's recommended BayesR prior and built-in
pseudo cross-validation. For a quick HM3-style smoke test, set
`MEGAPRS_PROFILE=quick`.

To use BaselineLD annotations with the current target-ID MegaPRS reference,
remap the rsID annotation files through a disease-specific `target_id_map.tsv`
and filter to the joined MegaPRS `.cors.bim` predictors:

```bash
uv run python scripts/remap_megaprs_annotations.py \
  --annotation-prefix data/resources/megaprs/BaselineLD/BaselineLD \
  --annotation-number 86 \
  --target-id-map outputs/sbayesrc_7m_full_train/cad/inputs/cad.sbayesrc.target_id_map.tsv \
  --predictors outputs/megaprs_full_train/cad/reference/cad.megaprs.cors.cors.bim \
  --out-prefix outputs/megaprs_full_train/cad/annotations/BaselineLD_target/BaselineLD \
  --summary-out outputs/megaprs_full_train/cad/annotations/BaselineLD_target/remap_summary.json

MEGAPRS_PROFILE=target_annotated sbatch slurm/megaprs_full_train_cad.sbatch
```

This keeps the LDAK MegaPRS predictor IDs consistent across summary statistics,
LD correlations, and annotations, while still scoring effects back on the
decipEHR target genotype IDs.

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
- Tier 3: PRS-CS, sBayesRC, SDPR, MegaPRS, or Lassosum2 if time and compute allow.
- Tier 4: logistic/ridge ensemble using scikit-learn.

If summary statistics are official and independent of the competition training labels, they may be reused across folds. If summary statistics are derived from internal training labels, regenerate them inside each outer fold using only training-fold patients.

## Ensemble PRS

First select the best validation-tuned candidate from each non-annotated PRS
family and combine those selected predictions into one long ensemble input:

```bash
uv run python scripts/build_ensemble_inputs.py \
  --disease cad \
  --families pt,ldpred2,lassosum2,prscs,sdpr,sbayesrc,megaprs \
  --out-dir outputs/ensemble_inputs
```

By default this reads each family's official validation `summary_metrics.csv`
if available, otherwise `point_metrics.csv`, filters to `prs_covariate_<disease>`,
and selects the best row by `top10_vs_lower90_or_mean`. It then extracts the
matching train and validation PRS rows for that candidate and writes
`outputs/ensemble_inputs/<disease>.train_long.csv`,
`outputs/ensemble_inputs/<disease>.validation_long.csv`, and
`outputs/ensemble_inputs/<disease>.selected_single_models.csv`. Use
`--allow-missing` while PRS-CS, SDPR, or sBayesRC jobs are still incomplete.

Then build official-compatible ensemble PRS files with:

```bash
uv run python scripts/build_ensemble_prs.py \
  --disease cad \
  --train-prs-long-scores outputs/ensemble_inputs/cad.train_long.csv \
  --validation-prs-long-scores outputs/ensemble_inputs/cad.validation_long.csv \
  --out-dir outputs/ensemble/cad
```

The preferred long input format is `sampleid,disease,model,prs`, where `model`
can be any set of available PRS methods, for example `pt`, `ldpred2`,
`lassosum2`, `prscs`, `sdpr`, `sbayesrc`, and `megaprs`. The script fits two score-level stacking
strategies on labelled training patients:

- `prs_only`: `disease ~ PRS_1 + ... + PRS_k`.
- `covariate_adjusted`: `disease ~ PRS_1 + ... + PRS_k + age/sex/PCs`.

Both strategies export only a single genetic score column named `prs_<disease>`.
Covariates are used only to learn the adjusted ensemble weights; they are not
included in the exported PRS value. The outputs under each strategy directory
can be passed directly to the fixed official validation interface:

```bash
uv run python scripts/validate_official_model.py \
  --model-name ensemble_covariate_adjusted \
  --disease cad \
  --train-prs-scores outputs/ensemble/cad/covariate_adjusted/train_prs_scores.csv \
  --validation-prs-scores outputs/ensemble/cad/covariate_adjusted/validation_prs_scores.csv \
  --out-dir outputs/official_validation_ensemble/cad/covariate_adjusted
```

The ensemble builder also writes `validate_ensemble_commands.sh` with one
validation command per strategy.

To evaluate ridge settings comprehensively, generate a build-and-validation grid:

```bash
uv run python scripts/write_ensemble_grid_commands.py \
  --disease cad \
  --ensemble-input-dir outputs/ensemble_inputs \
  --out-root outputs/ensemble_grid \
  --c-grid 0.01,0.03,0.1,0.3,1,3,10 \
  --out outputs/ensemble_grid/cad.ensemble_grid.commands.sh
```

After those commands finish, rank the ensemble settings with:

```bash
uv run python scripts/collect_ensemble_grid_results.py \
  --grid-root outputs/ensemble_grid \
  --disease cad \
  --out outputs/ensemble_grid/cad.ensemble_grid_summary.csv
```

## TabPFN Ensemble Setup

TabPFN is kept behind a separate `uv` extra because it pulls a large PyTorch
stack. The project pins the Linux extra to `torch==2.9.1+cu129` via the PyTorch
CUDA 12.9 wheel index, which matches the current HPC environment better than
the latest PyPI torch wheel.

TabPFN-2.5/2.6/3 weights are gated by Prior Labs license acceptance. For headless
HPC jobs, get the API key from the [Prior Labs account page](https://ux.priorlabs.ai/account)
after accepting the license, then put it in a local ignored file:

```bash
printf 'TABPFN_TOKEN=<your-priorlabs-api-key>\n' > .env.tabpfn
chmod 600 .env.tabpfn
```

The Slurm wrapper sources `.env.tabpfn` automatically, disables browser login,
and stores generated caches under ignored paths:

- `outputs/uv-cache`
- `outputs/huggingface-cache`
- `outputs/tabpfn-model-cache`

Install and check the local setup with:

```bash
bin/setup_tabpfn.sh
```

For an online license check before submitting a GPU job:

```bash
set -a
source .env.tabpfn
set +a
.venv/bin/python scripts/check_tabpfn_setup.py \
  --require-auth \
  --verify-license
```

Then run the TabPFN ensemble job:

```bash
sbatch slurm/tabpfn_ensemble.sbatch
```

The TabPFN extra tracks the current public TabPFN major release; `tabpfn>=8,<9`
uses TabPFN-3 by default. To compare against the previous default, set
`TABPFN_MODEL_VERSION=v2.6`. To force an explicit checkpoint path, set
`TABPFN_MODEL_PATH`; for example:

```bash
TABPFN_MODEL_VERSION=default \
TABPFN_MODEL_PATH=outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt \
OUT_DIR=outputs/tabpfn_ensemble_nonannotated_a0p20_v3 \
sbatch slurm/tabpfn_ensemble.sbatch
```

## Validation

Disease-specific eligibility rules are configured in `configs/disease_eligibility.yaml`. Breast cancer validation is restricted to eligible female patients. `scripts/validate_official_model.py` is the stable entry point for official validation; it fixes the official paths, labels, covariates, eligibility rules, unweighted logistic regression, and bootstrap settings from `configs/default.yaml`. The lower-level `scripts/run_official_validation.py` writes `eligibility_counts.csv` so the filtered analysis set is auditable.

The default validation strategy fits once on the model-training phenotype, predicts
the complete official validation set, then estimates the challenge OR and
Nagelkerke's R2 using 1000 bootstrap resamples of those predictions. Select
models primarily by the mean bootstrap challenge OR: the odds ratio comparing the
top 10% of full-model predictions against the lower 90%. Use mean bootstrap
Nagelkerke's R2 as the secondary challenge metric, and keep AUC, AUPRC, Brier
score, PRS per-standard-deviation odds ratio, and older risk-stratification odds
ratios as supporting diagnostics.

The older disease-specific 5-fold scripts remain in the repository for sensitivity checks and for workflows that need fold-aware PRS generation, but they are no longer the primary validation path now that the official validation labels are available.

## GitHub Hygiene

Use a private repository. Commit code, configuration templates, documentation, and synthetic tests. Do not commit real genotype, phenotype, summary statistics, model weights, outputs, or submissions. See `docs/github_private_repo.md`.
