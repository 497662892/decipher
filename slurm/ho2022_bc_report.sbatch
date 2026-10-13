#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=ho2022_bc_report
#SBATCH --output=logs/external_pgs/ho2022-bc-report-%J.log

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
uv run --no-sync python -u scripts/report_ho2022_bc.py
