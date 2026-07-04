#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --partition=gl40s_short
#SBATCH --job-name=run_score_scripts
#SBATCH --output=logs/run_score_scripts/%x-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCORE_SCRIPT_LIST="${SCORE_SCRIPT_LIST:-}"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

if [[ -z "$SCORE_SCRIPT_LIST" ]]; then
  echo "SCORE_SCRIPT_LIST is required." >&2
  exit 1
fi

cd "$REPO_DIR"

PLINK_MODULE="${PLINK_MODULE:-plink/2.0-20260228}"
if command -v module >/dev/null 2>&1; then
  module load "$PLINK_MODULE" || true
fi
if ! command -v plink2 >/dev/null 2>&1 && [[ -n "${PLINK2_DIR:-}" && -x "$PLINK2_DIR/plink2" ]]; then
  export PATH="$PLINK2_DIR:$PATH"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

mkdir -p "$UV_CACHE_DIR" logs/run_score_scripts

test -s "$SCORE_SCRIPT_LIST" || { echo "Missing score script list: $SCORE_SCRIPT_LIST" >&2; exit 1; }

echo "Started score scripts from $SCORE_SCRIPT_LIST at $(date)"
while IFS= read -r SCORE_SCRIPT; do
  [[ -n "$SCORE_SCRIPT" ]] || continue
  bash "$SCORE_SCRIPT"
done < "$SCORE_SCRIPT_LIST"
echo "Finished score scripts from $SCORE_SCRIPT_LIST at $(date)"
