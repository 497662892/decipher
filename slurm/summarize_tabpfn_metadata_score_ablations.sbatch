#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_medium
#SBATCH --job-name=tabpfn_meta_summary
#SBATCH --output=logs/tabpfn_metadata_score_ablations/%x-%J.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache/tabpfn-metadata-score-ablations}"
export TMPDIR="${TMPDIR:-/tmp}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

CONFIG="${CONFIG:-configs/tabpfn_metadata_score_ablations.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

mkdir -p logs/tabpfn_metadata_score_ablations "$UV_CACHE_DIR"

echo "Starting corrected metadata-score ablation summary at $(date)"
"$PYTHON_BIN" scripts/summarize_tabpfn_metadata_score_ablations.py \
  --config "$CONFIG"
echo "Finished corrected metadata-score ablation summary at $(date)"
