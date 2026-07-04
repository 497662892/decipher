#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-22%22
#SBATCH --job-name=sdpr_full_train
#SBATCH --output=logs/sdpr_full_train/%x-%A_%a.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DISEASE="${DISEASE:-}"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

if [[ -z "$DISEASE" ]]; then
  echo "DISEASE is required; use a disease wrapper or export DISEASE=cad/bc/T2D." >&2
  exit 1
fi

cd "$REPO_DIR"

SDPR_MODULE="${SDPR_MODULE:-}"
if command -v module >/dev/null 2>&1 && [[ -n "$SDPR_MODULE" ]]; then
  module load "$SDPR_MODULE" || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"
export REUSE_SDPR_EFFECTS="${REUSE_SDPR_EFFECTS:-1}"

OUT_DIR="${OUT_DIR:-outputs/sdpr_full_train}"
GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
SCORE_GENOTYPE_PREFIX="${SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"
FIT_SCORE_GENOTYPE_PREFIX="${FIT_SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"
TASKS_FILE="${TASKS_FILE:-$OUT_DIR/$DISEASE/commands/$DISEASE.sdpr_full_train.array_tasks.txt}"
SDPR_SUMSTATS="${SDPR_SUMSTATS:-$OUT_DIR/$DISEASE/inputs/$DISEASE.sdpr.ss}"
REF_DIR="${REF_DIR:-$OUT_DIR/$DISEASE/ref}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" logs/sdpr_full_train

test -s "$TASKS_FILE" || {
  echo "Missing SDPR task list: $TASKS_FILE" >&2
  echo "Run slurm/sdpr_prepare_reference_${DISEASE}.sbatch before this array job." >&2
  exit 1
}
test -s "$SDPR_SUMSTATS" || { echo "Missing SDPR sumstats: $SDPR_SUMSTATS" >&2; exit 1; }
test -d "$REF_DIR" || { echo "Missing SDPR reference directory: $REF_DIR" >&2; exit 1; }

TASK_SCRIPT="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")"
if [[ -z "$TASK_SCRIPT" ]]; then
  echo "No task script for array index $SLURM_ARRAY_TASK_ID in $TASKS_FILE" >&2
  exit 1
fi

bash "$TASK_SCRIPT"
