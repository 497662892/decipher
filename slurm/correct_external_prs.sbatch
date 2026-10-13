#!/usr/bin/env bash
#SBATCH --job-name=external_prs_correction
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-2%3
#SBATCH --output=outputs/external_pgs/correction_20261011/logs/%x_%A_%a.out
#SBATCH --error=outputs/external_pgs/correction_20261011/logs/%x_%A_%a.err
set -euo pipefail
umask 077
cd /gpfs/data/shenlab/Jiajian/HW/decipehr
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TABPFN_MODEL_CACHE_DIR="$PWD/outputs/tabpfn-model-cache"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
diseases=(T2D cad bc)
uv run --no-sync python -u scripts/correct_external_prs_analysis.py \
  --phase "${CORRECTION_PHASE:-anchor}" --disease "${diseases[${SLURM_ARRAY_TASK_ID}]}" \
  --eligibility-config configs/disease_eligibility.yaml
