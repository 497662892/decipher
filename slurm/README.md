# Slurm Wrappers

This directory contains optional HPC wrappers for the main pipeline stages.
They assume the repository root is the parent of `slurm/` and write logs under
`logs/`. Create that directory before submitting jobs:

```bash
mkdir -p logs
```

Most scripts can be customized with environment variables at submit time:

```bash
CONFIG=configs/default.yaml sbatch slurm/preprocess_data.sbatch
DEVICE=cuda sbatch slurm/final_tabpfn_oof_holdout_pipeline.sbatch
```

Core groups:

- Preprocessing: `preprocess_data.sbatch`, `final_genotype_*.sbatch`.
- Summary statistics: `sumstats_common_*.sbatch`, `sumstats_final_common.sbatch`.
- Single PRS families: `pt_*`, `ldpred2_*`, `prscs_*`, `lassosum2_*`,
  `sdpr_*`, `sbayesrc_*`, and `megaprs_*`.
- TabPFN ensembles: `tabpfn_ensemble.sbatch`, `final_trainval_tabpfn_*.sbatch`.
- Final holdout pipeline: `final_tabpfn_oof_holdout_pipeline.sbatch`.

Cluster-specific settings such as partitions, modules, and tool paths may need
local edits. Keep private data paths in environment variables or local config
files rather than committing them.
