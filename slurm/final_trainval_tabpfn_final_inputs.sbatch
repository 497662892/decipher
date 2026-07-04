#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu4_dev,gl40s_dev
#SBATCH --job-name=tabpfn_final_inputs
#SBATCH --output=logs/tabpfn_final_holdout/%x-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache}"

LABELS="${LABELS:-cad,bc,T2D}"
FAMILIES="${FAMILIES:-pt,ldpred2,lassosum2,prscs,sdpr}"
SELECTION_FILE="${SELECTION_FILE:-outputs/tabpfn_cv_ensemble_inputs/selected_single_models.csv}"
OUT_DIR="${OUT_DIR:-outputs/tabpfn_final_ensemble_inputs}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR" logs/tabpfn_final_holdout

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
if [[ -x "$PYTHON_BIN" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  PYTHON_CMD=(uv run --extra dev python)
fi

ALLOW_ARGS=()
if [[ "$ALLOW_MISSING" == "1" ]]; then
  ALLOW_ARGS=(--allow-missing)
fi

echo "Started TabPFN final holdout input build at $(date)"
"${PYTHON_CMD[@]}" scripts/build_final_ensemble_inputs_from_cv_selection.py \
  --selection-file "$SELECTION_FILE" \
  --labels "$LABELS" \
  --families "$FAMILIES" \
  --out-dir "$OUT_DIR" \
  "${ALLOW_ARGS[@]}"

echo "Finished TabPFN final holdout input build at $(date)"
