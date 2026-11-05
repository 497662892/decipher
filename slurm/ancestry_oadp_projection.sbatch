#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=prs_oadp_pc
#SBATCH --output=logs/ancestry_oadp/%x-%j.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
module load r/4.5.3 plink/2.0-20260228
export LD_LIBRARY_PATH="/gpfs/share/apps/git/2.49.0/lib64:${LD_LIBRARY_PATH:-}"
export R_LIBS_USER="$PWD/data/resources/R/4.5.3-library"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
/gpfs/data/shenlab/Jiajian/software/miniconda3/bin/uv run --offline --no-sync python -u scripts/prepare_ancestry_oadp.py \
  --out-dir "${RUN_DIR:?}/projection" --threads "$SLURM_CPUS_PER_TASK"
