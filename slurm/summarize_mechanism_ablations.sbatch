#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_medium
#SBATCH --job-name=prs_mech_summary
#SBATCH --output=logs/mechanism_ablations/%x-%J.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache/mechanism-ablations}"
export TMPDIR="${TMPDIR:-/tmp}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

CONFIG="${CONFIG:-configs/mechanism_ablations.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

mkdir -p logs/mechanism_ablations "$UV_CACHE_DIR"

echo "Starting mechanism-ablation summary at $(date)"
"$PYTHON_BIN" scripts/summarize_mechanism_ablations.py --config "$CONFIG"
echo "Finished mechanism-ablation summary at $(date)"
