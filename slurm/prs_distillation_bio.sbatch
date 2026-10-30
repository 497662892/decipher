#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=0-2
#SBATCH --job-name=prs_bio
#SBATCH --output=logs/prs_distillation_bio/%x-%A_%a.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
traits=(T2D cad bc)
trait="${traits[${SLURM_ARRAY_TASK_ID:?}]}"
args=(--run-dir "${RUN_DIR:?}" --trait "$trait" --eligibility-config configs/disease_eligibility.yaml)
uv run --offline --no-sync python -u scripts/run_prs_distillation_bio.py smoke "${args[@]}"
uv run --offline --no-sync python -u scripts/run_prs_distillation_bio.py train "${args[@]}"
