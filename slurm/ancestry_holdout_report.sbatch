#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_dev
#SBATCH --job-name=prs_holdout_report
#SBATCH --output=logs/ancestry_holdout/%x-%j.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export MPLCONFIGDIR="$PWD/outputs/report_vendor/matplotlib"
/gpfs/data/shenlab/Jiajian/software/miniconda3/bin/uv run --offline --no-sync python -u scripts/report_ancestry_holdout_agreement.py \
  --out-dir "${RUN_DIR:?}"
