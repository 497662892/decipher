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
3. Create one shared patient-level 5-fold split for all methods.
4. Run covariate-only, P+T, stronger PRS, and ensemble comparisons using out-of-fold predictions.
5. Retrain the selected final pipeline on all official training data before generating the official submission.

```bash
uv run python scripts/inventory_sumstats.py --root data/raw --out data/processed/sumstats_inventory.csv
uv run python scripts/make_splits.py --phenotype data/raw/phenotype.csv --patient-col patient_id --labels disease1,disease2,disease3 --out data/processed/splits/split_seed42.csv
uv run python scripts/run_cv.py --phenotype data/processed/phenotype.csv --splits data/processed/splits/split_seed42.csv --labels disease1,disease2,disease3 --covariates age,sex,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10 --eligibility-config configs/disease_eligibility.yaml --out-dir outputs/cv_baseline
```

Generate an inspectable PLINK2 P+T command file after standardizing a disease summary-statistics file:

```bash
uv run python scripts/write_pt_commands.py --genotype-kind pfile --bfile-prefix data/processed/genotypes/train --sumstats data/processed/sumstats/disease1.standardized.tsv --score-prefix outputs/pt/disease1 --out outputs/pt/disease1.pt.commands.sh
```

## Baseline Tiers

- Tier 0: covariate-only logistic regression with scikit-learn.
- Tier 1: P+T using official/provided summary statistics and PLINK2.
- Tier 2: LDpred2 via `bigsnpr`.
- Tier 3: PRS-CS or Lassosum2 if time and compute allow.
- Tier 4: logistic/ridge ensemble using scikit-learn.

If summary statistics are official and independent of the competition training labels, they may be reused across folds. If summary statistics are derived from internal training labels, regenerate them inside each outer fold using only training-fold patients.

## Validation

Disease-specific eligibility rules are configured in `configs/disease_eligibility.yaml`. For breast cancer, update the disease key and female sex encodings to match the official phenotype table before running CV. `scripts/run_cv.py` writes `eligibility_counts.csv` so the filtered analysis set is auditable.

Use the same 5 folds for every method. Report per-disease and macro summaries for AUC, AUPRC, Brier score, PRS per-standard-deviation odds ratio, and risk-stratification odds ratios.

## GitHub Hygiene

Use a private repository. Commit code, configuration templates, documentation, and synthetic tests. Do not commit real genotype, phenotype, summary statistics, model weights, outputs, or submissions. See `docs/github_private_repo.md`.
