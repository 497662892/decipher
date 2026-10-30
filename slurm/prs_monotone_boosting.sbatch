#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=0-17
#SBATCH --job-name=prs_monotone_boost
#SBATCH --output=logs/prs_distillation_bio/%x-%A_%a.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
traits=(T2D cad bc)
task="${SLURM_ARRAY_TASK_ID:?}"
root="${RUN_DIR:?}"
if (( task >= 3 )); then
  root="$root/nested/outer$(((task - 3) / 3))"
fi
uv run --offline --no-sync python -u scripts/run_prs_monotone_boosting.py --run-dir "$root" \
  --trait "${traits[$((task % 3))]}" --eligibility-config configs/disease_eligibility.yaml
