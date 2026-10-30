#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=prs_bio_report
#SBATCH --output=logs/prs_distillation_bio/%x-%j.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export MPLCONFIGDIR="$PWD/outputs/matplotlib-cache"
uv run --offline --no-sync python -u scripts/report_prs_distillation_bio.py \
  --run-dir "${RUN_DIR:?}" --scope "${REPORT_SCOPE:?}"
