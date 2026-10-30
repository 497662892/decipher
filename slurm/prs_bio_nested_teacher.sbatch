#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%2
#SBATCH --job-name=prs_bio_inner_teacher
#SBATCH --output=logs/prs_distillation_bio/%x-%A_%a.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --offline --no-sync python -u scripts/nested_prs_distillation_bio.py prepare \
  --parent-run "${RUN_DIR:?}" --fold "${SLURM_ARRAY_TASK_ID:?}"
