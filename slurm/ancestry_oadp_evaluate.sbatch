#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=0-2
#SBATCH --job-name=prs_oadp_eval
#SBATCH --output=logs/ancestry_oadp/%x-%A_%a.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export MPLCONFIGDIR="$PWD/outputs/report_vendor/matplotlib"
traits=(T2D cad bc)
/gpfs/data/shenlab/Jiajian/software/miniconda3/bin/uv run --offline --no-sync python -u scripts/evaluate_ancestry_oadp.py \
  --out-dir "${RUN_DIR:?}" --trait "${traits[${SLURM_ARRAY_TASK_ID:?}]}"
