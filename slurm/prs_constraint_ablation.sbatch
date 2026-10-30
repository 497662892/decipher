#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=0-53%6
#SBATCH --job-name=prs_constraint_ablation
#SBATCH --output=logs/prs_distillation_bio/%x-%A_%a.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
task="${SLURM_ARRAY_TASK_ID:?}"
scope_index=$((task / 9))
trait_index=$(((task % 9) / 3))
prior_index=$((task % 3))
traits=(T2D cad bc)
priors=(AGE PRS FREE)
scope_root="${RUN_DIR:?}"
if ((scope_index > 0)); then
  scope_root="$scope_root/nested/outer$((scope_index - 1))"
fi
uv run --offline --no-sync python -u scripts/run_prs_constraint_ablation.py \
  --run-dir "$scope_root" --trait "${traits[trait_index]}" --constraint "${priors[prior_index]}" \
  --eligibility-config configs/disease_eligibility.yaml
