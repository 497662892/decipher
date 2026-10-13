#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=external_pgs_report
#SBATCH --output=logs/external_pgs/report-%j.log

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
uv run --no-sync python -u scripts/summarize_external_pgs_validation.py \
  --runs-root "${RUN_DIR:-outputs/external_pgs/pc_primary/official_validation}" \
  --scoring-dir "${SCORE_DIR:-outputs/external_pgs/run_20261010/scoring_v2}" \
  --out-dir "${REPORT_DIR:-outputs/external_pgs/pc_primary/report}" \
  --eligibility-config configs/disease_eligibility.yaml \
  "$@"
