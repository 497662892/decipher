#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:15:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=frozen_prs_report
#SBATCH --output=logs/external_pgs/frozen-prs-report-%j.log

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
uv run --offline --no-sync python -u scripts/report_frozen_prs_comparison.py \
  --run-dir "${RUN_DIR:-outputs/external_pgs/frozen_internal_external_trainval_20261013}"
