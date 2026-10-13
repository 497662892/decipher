#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=external_pgs_score
#SBATCH --output=logs/external_pgs/score-%J.log

set -eo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$REPO_DIR"
module load plink/2.0-20260228
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export POLARS_MAX_THREADS="$OMP_NUM_THREADS"

uv run --no-sync python -u scripts/external_pgs.py score \
  --weight-lock "${WEIGHT_LOCK:-outputs/external_pgs/run_20261010/weights.amended.lock.json}" \
  --pfile outputs/final_genotype/qc_final/train_val.common \
  --target-build GRCh38 \
  --reference-freq outputs/external_pgs/reference_1000g_trainval/reference.target_ids.afreq \
  --reference-provenance '1000 Genomes GRCh38 20190312, 2503 reference individuals; see reference_1000g_trainval/reference_run.json' \
  --threads "$OMP_NUM_THREADS" --memory-mb 32000 --min-overlap 0.75 \
  --out-dir "${OUT_DIR:-outputs/external_pgs/run_20261010/scoring_v2}" \
  "$@"
