#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=megaprs_eval
#SBATCH --output=logs/official_validation_megaprs/megaprs_eval-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DISEASE="${DISEASE:-}"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

if [[ -z "$DISEASE" ]]; then
  echo "DISEASE is required; export DISEASE=cad/bc/T2D." >&2
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

MEGAPRS_DIR="${MEGAPRS_DIR:-outputs/megaprs_full_train}"
VALIDATION_OUT_ROOT="${VALIDATION_OUT_ROOT:-outputs/official_validation_megaprs}"
TASK_MANIFEST="${TASK_MANIFEST:-$MEGAPRS_DIR/$DISEASE/commands/$DISEASE.megaprs_full_train.tasks.csv}"
SCORE_SCRIPT_LIST="${SCORE_SCRIPT_LIST:-$MEGAPRS_DIR/$DISEASE/commands/$DISEASE.megaprs_full_train.score_scripts.txt}"
VALIDATION_PRS_LONG_SCORES="${VALIDATION_PRS_LONG_SCORES:-$MEGAPRS_DIR/$DISEASE/scores/$DISEASE.megaprs_full_train_scores_long.csv}"
TRAIN_PRS_LONG_SCORES="${TRAIN_PRS_LONG_SCORES:-$MEGAPRS_DIR/$DISEASE/scores/$DISEASE.megaprs_full_train_training_scores_long.csv}"
OUT_DIR="${OUT_DIR:-$VALIDATION_OUT_ROOT/$DISEASE}"
MODEL_NAME="${MODEL_NAME:-megaprs}"
N_BOOTSTRAPS="${N_BOOTSTRAPS:-1000}"
SEED="${SEED:-42}"
PRS_PARAM_COLS="${PRS_PARAM_COLS:-model}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR" "$MEGAPRS_DIR/$DISEASE/scores" logs/official_validation_megaprs

if [[ -x "$PYTHON_BIN" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: neither PYTHON_BIN=$PYTHON_BIN nor uv is available." >&2
    exit 1
  fi
  PYTHON_CMD=(uv run --extra dev python)
fi

test -s "$TASK_MANIFEST" || { echo "Missing MegaPRS task manifest: $TASK_MANIFEST" >&2; exit 1; }
test -s "$SCORE_SCRIPT_LIST" || { echo "Missing MegaPRS score script list: $SCORE_SCRIPT_LIST" >&2; exit 1; }

echo "Started MegaPRS full-train official validation for $DISEASE at $(date)"

while IFS= read -r SCORE_SCRIPT; do
  [[ -n "$SCORE_SCRIPT" ]] || continue
  bash "$SCORE_SCRIPT"
done < "$SCORE_SCRIPT_LIST"

"${PYTHON_CMD[@]}" scripts/collect_megaprs_scores.py \
  --task-manifest "$TASK_MANIFEST" \
  --out "$VALIDATION_PRS_LONG_SCORES"

"${PYTHON_CMD[@]}" scripts/collect_megaprs_scores.py \
  --task-manifest "$TASK_MANIFEST" \
  --sscore-column fit_sscore \
  --out "$TRAIN_PRS_LONG_SCORES"

"${PYTHON_CMD[@]}" scripts/validate_official_model.py \
  --model-name "$MODEL_NAME" \
  --disease "$DISEASE" \
  --train-prs-long-scores "$TRAIN_PRS_LONG_SCORES" \
  --validation-prs-long-scores "$VALIDATION_PRS_LONG_SCORES" \
  --prs-param-cols "$PRS_PARAM_COLS" \
  --n-bootstraps "$N_BOOTSTRAPS" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"

echo "Finished MegaPRS full-train official validation for $DISEASE at $(date)"
