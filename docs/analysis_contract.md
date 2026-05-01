# Analysis Contract

This file records the project rules that should be confirmed from the official PDFs and then kept stable across notebooks, scripts, and submissions.

## Competition Entities

- Diseases: confirm the three official disease labels, case/control coding, and any exclusion rules from `DREAM_data_V1.pdf` and `04_09 decipEHR project.pdf`.
- Patient key: use one canonical patient identifier across phenotype, covariates, genotype, folds, PRS scores, predictions, and submission files.
- Outcome type: treat each disease as a binary prediction task unless the official documentation states otherwise.
- Official metric: use the competition metric for model selection; keep AUC, AUPRC, Brier score, OR per SD, and risk-stratification ORs as internal diagnostics.

## Mature Tool Policy

Use established tools for domain-heavy work:

- Genotype QC, clumping, thresholding, and scoring: `plink2`.
- LDpred2: `bigsnpr` / R implementation.
- PRS-CS: official PRS-CS implementation.
- Classical models and metrics: `scikit-learn`, `scipy`, and `statsmodels`.
- Dataframe/table handling: `pandas`, `pyarrow`, and optionally `polars` for large tables.

Project code should only do orchestration, schema validation, fold generation, command generation, metric aggregation, and submission formatting. Do not reimplement LD clumping, LDpred2, PRS-CS, allele scoring, imputation engines, or standard metrics.

## Summary Statistics Rules

- First inventory all files that look like official/provided summary statistics.
- Confirm for each disease whether summary statistics are official allowed resources.
- Standardize to `SNP, CHR, BP, A1, A2, P, BETA` plus optional `SE, N`.
- If only `OR` is available, use `BETA = log(OR)`.
- Confirm genome build and allele convention before scoring.
- If summary statistics were generated independently of the official training labels, they can be reused across all CV folds.
- If summary statistics are derived from internal training labels, regenerate them inside each outer fold using only training-fold patients.

## Ancestry And LD Reference

- Confirm whether the dataset includes ancestry PCs, reported ancestry, array/batch/site, or other population structure covariates.
- Use ancestry PCs in covariate and PRS+covariate models unless official rules prohibit them.
- Match LD reference ancestry to the competition cohort as closely as possible.
- Report ancestry mismatch as a caveat when interpreting PRS performance.

## Cross-Validation Rules

- Use one shared patient-level 5-fold split for all methods.
- The same patient must never appear in two folds.
- All methods must use the same fold assignments for fair comparison.
- Fit imputers, scalers, PCA-derived transforms, model parameters, and ensemble weights using only training folds.
- Validation folds are prediction-only.
- Store out-of-fold predictions and compute metrics from pooled OOF predictions.

## Interpretation Rules

- Treat PRS performance as predictive association, not causality.
- Report uncertainty and caveats for low case counts, class imbalance, ancestry mismatch, and phenotype ambiguity.
- Do not compare methods using a validation set that was also used for parameter selection without clearly labeling it as exploratory.

