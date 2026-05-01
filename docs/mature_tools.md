# Mature Tooling Map

The project should avoid rebuilding established methods. Use this map when deciding where logic belongs.

| Task | Preferred tool | Project responsibility |
| --- | --- | --- |
| Genotype QC | `plink2` | Generate and record commands; parse outputs only when needed |
| Clumping and thresholding | `plink2` | Generate parameter grid and score commands |
| Allele dosage scoring | `plink2 --score` | Standardize summary stats and collect score files |
| LDpred2 | `bigsnpr` in R | Provide inputs and record output PRS columns |
| PRS-CS | official PRS-CS | Provide inputs and record output PRS columns |
| Logistic/ridge baselines | `scikit-learn` | Shared folds, feature lists, output predictions |
| Metrics | `scikit-learn`, `statsmodels` | Aggregate per-fold and OOF summaries |
| Data validation | `pandas`, local schema checks | Fail fast on missing columns and invalid labels |

If a future implementation adds large-scale genotype or LD operations directly in Python, it should be treated as a design error unless there is a clear benchmarked reason.

