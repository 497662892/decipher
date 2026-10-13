#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=discodivas-pca
#SBATCH --output=logs/discodivas/projection-%j.log

set -euo pipefail
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
module load plink/2.0-20260228
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
uv run --no-sync python -u scripts/prepare_discodivas_projection.py \
  --source-lock outputs/external_pgs/discodivas_resource_audit_20261010/source.lock.json \
  --weight-lock outputs/external_pgs/run_20261010/scoring_v2/weights.scored.lock.json \
  --reference-run outputs/external_pgs/reference_1000g_trainval/reference_run.json \
  --target-pfile outputs/final_genotype/qc_final/train_val.common \
  --panel data/resources/1000g/grch38/1000g_panel.tsv \
  --out-dir "${PROJECTION_DIR:-outputs/external_pgs/discodivas/projection_20261010}" \
  --threads 8 --memory-mb 48000 "$@"
