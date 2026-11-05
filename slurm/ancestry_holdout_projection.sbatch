#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=prs_holdout_pc
#SBATCH --output=logs/ancestry_holdout/%x-%j.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
module load plink/2.0-20260228
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
/gpfs/data/shenlab/Jiajian/software/miniconda3/bin/uv run --offline --no-sync python -u scripts/project_holdout_reference_pcs.py \
  --out-dir "${RUN_DIR:?}/projection" --threads "$SLURM_CPUS_PER_TASK"
