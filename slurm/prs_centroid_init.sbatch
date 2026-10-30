#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=0-17%3
#SBATCH --job-name=prs_centroid
#SBATCH --output=logs/prs_centroid_init/%x-%A_%a.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
traits=(T2D cad bc)
scopes=(val outer0 outer1 outer2 outer3 outer4)
idx="${SLURM_ARRAY_TASK_ID:?}"
trait="${traits[$((idx % 3))]}"
scope="${scopes[$((idx / 3))]}"
uv run --offline --no-sync python -u scripts/run_prs_centroid_init.py train \
  --out-dir "${RUN_DIR:?}" --scope "$scope" --trait "$trait" \
  --eligibility-config configs/disease_eligibility.yaml
