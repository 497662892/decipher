#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=ho2022_bc_score
#SBATCH --output=logs/external_pgs/ho2022-bc-score-%J.log

set -eo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load plink/2.0-20260228
set -u
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 POLARS_MAX_THREADS=8
run_dir="${RUN_DIR:-outputs/external_pgs/bc_ho2022_preparation_20261012}"
uv run --no-sync python -u scripts/external_pgs.py score \
  --weight-lock "$run_dir/weights/weights.lock.json" \
  --pfile outputs/final_genotype/qc_final/train_val.common \
  --target-build GRCh38 \
  --reference-freq outputs/external_pgs/reference_1000g_trainval/reference.target_ids.afreq \
  --reference-provenance '1000 Genomes GRCh38 20190312, 2503 reference individuals; see reference_1000g_trainval/reference_run.json' \
  --threads 8 --memory-mb 32000 --min-overlap 0.75 \
  --out-dir "$run_dir/scoring_new"
uv run --no-sync python -u scripts/assemble_ho2022_bc_expansion.py --run-dir "$run_dir"
