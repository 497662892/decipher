#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=0-1
#SBATCH --job-name=prs_constraint_report
#SBATCH --output=logs/prs_distillation_bio/%x-%A_%a.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache" MPLCONFIGDIR="$PWD/outputs/matplotlib-cache"
scopes=(val nested)
uv run --offline --no-sync python -u scripts/report_prs_constraint_ablation.py \
  --run-dir "${RUN_DIR:?}" --scope "${scopes[${SLURM_ARRAY_TASK_ID:?}]}"
