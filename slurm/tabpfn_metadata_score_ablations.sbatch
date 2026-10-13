#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=deciphehr
#SBATCH --gres=gpu:1
#SBATCH --array=0-14%5
#SBATCH --job-name=tabpfn_meta_score
#SBATCH --output=logs/tabpfn_metadata_score_ablations/%x-%A_%a.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache/tabpfn-metadata-score-ablations}"
export HF_HOME="${HF_HOME:-$REPO_DIR/outputs/huggingface-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"
export TABPFN_NO_BROWSER=1
export TABPFN_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_TELEMETRY=1
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$REPO_DIR/outputs/tabpfn-model-cache}"

CONFIG="${CONFIG:-configs/tabpfn_metadata_score_ablations.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

mkdir -p logs/tabpfn_metadata_score_ablations "$UV_CACHE_DIR" "$HF_HOME"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing project Python environment: $PYTHON_BIN" >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import tabpfn' >/dev/null; then
  echo "Missing TabPFN dependency. Run: uv sync --extra tabpfn" >&2
  exit 2
fi

echo "Starting metadata-score ablation task ${SLURM_ARRAY_TASK_ID} at $(date)"
"$PYTHON_BIN" scripts/run_tabpfn_metadata_score_ablation.py \
  --config "$CONFIG" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --device cuda
echo "Finished metadata-score ablation task ${SLURM_ARRAY_TASK_ID} at $(date)"
