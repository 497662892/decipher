#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=external_holdout_score
#SBATCH --output=logs/external_pgs/holdout-score-%J.log

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
uv run --no-sync python -u scripts/score_external_pgs_holdout.py \
  --source outputs/external_pgs/run_20261010/scoring_v2 \
  --source outputs/external_pgs/cad_expansion_20261012/scoring_new \
  --source outputs/external_pgs/bc_ho2022_preparation_20261012/scoring_new \
  --pfile outputs/final_genotype/qc_final/holdout_test.common \
  --reference-freq outputs/external_pgs/reference_1000g_trainval/reference.target_ids.afreq \
  --plink2 /gpfs/share/apps/plink/2.0-20260228/plink2 \
  --out-dir outputs/external_pgs/holdout_wrapper_20261012/scoring
