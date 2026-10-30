#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --array=0-29%2
#SBATCH --job-name=frozen_prs_cv
#SBATCH --output=logs/external_pgs/frozen-prs-cv-%A_%a.log

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TABPFN_MODEL_CACHE_DIR="$PWD/outputs/tabpfn-model-cache"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
run_dir="${RUN_DIR:-outputs/external_pgs/frozen_internal_external_trainval_20261013}"
diseases=(T2D cad bc)
arms=(internal external)
index="${SLURM_ARRAY_TASK_ID:?Array index required}"
disease="${diseases[$((index / 10))]}"
arm="${arms[$(((index % 10) / 5))]}"
fold="$((index % 5))"
uv run --offline --no-sync python -u scripts/run_frozen_prs_comparison.py \
  --run-dir "$run_dir" --disease "$disease" --arm "$arm" --fold "$fold" \
  --eligibility-config configs/disease_eligibility.yaml --device cuda
