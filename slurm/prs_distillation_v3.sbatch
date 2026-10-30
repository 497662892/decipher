#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=0-2
#SBATCH --job-name=prs_distill_v3
#SBATCH --output=logs/prs_distillation_v3/%x-%A_%a.log

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
traits=(T2D cad bc)
trait="${traits[${SLURM_ARRAY_TASK_ID:?}]}"
run_dir="${RUN_DIR:?Prepare v3 before submission}"
args=(--run-dir "$run_dir" --trait "$trait" --eligibility-config configs/disease_eligibility.yaml)
uv run --offline --no-sync python -u scripts/run_prs_distillation_v3.py smoke "${args[@]}"
uv run --offline --no-sync python -u scripts/run_prs_distillation_v3.py train "${args[@]}"
