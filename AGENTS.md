# AGENTS.md

Instructions for Codex and other coding agents working in this repository.

## Project Boundary

- The repository root is `/gpfs/data/shenlab/Jiajian/HW/decipehr`.
- Treat everything inside this folder as part of the decipEHR PRS competition project.
- Do not edit files outside this repository unless explicitly asked.

## Environment

- Use `uv` for Python environment management.
- Do not commit `.venv/` or local caches.
- Preferred commands:

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev ruff check .
```

## Engineering Style

- Follow `karpathy-guidelines.mdc`: keep changes simple, surgical, and verifiable.
- Prefer readable glue code over broad abstractions.
- Do not add speculative framework code.
- Keep notebooks for exploration; move reusable logic into `src/decipehr/` or `scripts/`.

## Mature Tool Policy

- Do not reimplement established PRS/genetics methods.
- Use mature tools for domain-heavy work:
  - `plink2` for genotype QC, clumping, thresholding, and scoring.
  - `bigsnpr` / LDpred2 for LDpred2 workflows.
  - official PRS-CS implementation for PRS-CS workflows.
  - `scikit-learn`, `scipy`, and `statsmodels` for standard modeling and metrics.
- Project Python code should handle validation, fold definitions, command generation, metric aggregation, and submission formatting.

## Data And Privacy

- Do not commit real genotype, phenotype, summary-statistics, submission, model, or output files unless the user explicitly confirms they are safe to publish.
- Keep generated artifacts under ignored directories such as `data/`, `outputs/`, `models/`, and `submissions/`.
- Synthetic fixtures in `tests/` are allowed.

## PRS Validation Rules

- Use one shared patient-level 5-fold split for method comparison.
- The same patient must never appear in more than one fold.
- All methods should use the same folds for fair comparison.
- Fit imputers, scalers, model parameters, PRS parameter choices, and ensemble weights using only training folds.
- Validation folds are prediction-only.
- Store out-of-fold predictions and compute pooled OOF metrics.

## Summary Statistics Rules

- First inventory official/provided summary statistics.
- Standardize summary statistics to `SNP, CHR, BP, A1, A2, P, BETA` plus optional `SE, N`.
- If only `OR` is available, use `BETA = log(OR)`.
- Confirm genome build, allele convention, and whether the summary statistics are official allowed resources before scoring.
- If summary statistics are derived from internal training labels, regenerate them inside each outer fold using only training-fold patients.

## Verification

- Before committing code changes, run:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

- If a check cannot be run, explain why in the final response.

## GitHub

- Remote: `origin` points to `https://github.com/497662892/decipher.git`.
- Use `main` as the primary branch.
- Keep commits scoped and descriptive.
- Do not push private data or bulky generated artifacts.

