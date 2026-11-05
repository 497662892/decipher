#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gl40s_short
#SBATCH --array=0-2
#SBATCH --job-name=prs_holdout_predict
#SBATCH --output=logs/ancestry_holdout/%x-%A_%a.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
traits=(T2D cad bc)
/gpfs/data/shenlab/Jiajian/software/miniconda3/bin/uv run --offline --no-sync python -u scripts/run_ancestry_holdout_agreement.py predict \
  --out-dir "${RUN_DIR:?}" --trait "${traits[${SLURM_ARRAY_TASK_ID:?}]}" \
  --eligibility-config configs/disease_eligibility.yaml
