#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=0-14
#SBATCH --job-name=prs_bio_nested
#SBATCH --output=logs/prs_distillation_bio/%x-%A_%a.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
traits=(T2D cad bc)
task="${SLURM_ARRAY_TASK_ID:?}"
uv run --offline --no-sync python -u scripts/nested_prs_distillation_bio.py train \
  --parent-run "${RUN_DIR:?}" --fold "$((task / 3))" --trait "${traits[$((task % 3))]}"
