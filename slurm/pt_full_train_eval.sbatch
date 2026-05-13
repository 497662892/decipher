#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=pt_full_eval
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/official_validation_pt/pt_full_eval-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
DISEASE="${DISEASE:-}"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  source /gpfs/home/jm10850/.bashrc
fi

set -u

if [[ -z "$DISEASE" ]]; then
  echo "DISEASE is required; export DISEASE=cad/bc/T2D." >&2
  exit 1
fi

cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

PT_DIR="${PT_DIR:-outputs/pt_full_train}"
VALIDATION_OUT_ROOT="${VALIDATION_OUT_ROOT:-outputs/official_validation_pt}"
TASK_MANIFEST="${TASK_MANIFEST:-$PT_DIR/$DISEASE/commands/$DISEASE.pt_full_train.tasks.csv}"
VALIDATION_PRS_LONG_SCORES="${VALIDATION_PRS_LONG_SCORES:-$PT_DIR/$DISEASE/scores/$DISEASE.pt_full_train_scores_long.csv}"
TRAIN_PRS_LONG_SCORES="${TRAIN_PRS_LONG_SCORES:-$PT_DIR/$DISEASE/scores/$DISEASE.pt_full_train_training_scores_long.csv}"
OUT_DIR="${OUT_DIR:-$VALIDATION_OUT_ROOT/$DISEASE}"
MODEL_NAME="${MODEL_NAME:-pt}"
N_BOOTSTRAPS="${N_BOOTSTRAPS:-1000}"
SEED="${SEED:-42}"
PRS_PARAM_COLS="${PRS_PARAM_COLS:-clump_r2,p_threshold}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR" "$PT_DIR/$DISEASE/scores" logs/official_validation_pt

if [[ -x "$PYTHON_BIN" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: neither PYTHON_BIN=$PYTHON_BIN nor uv is available." >&2
    exit 1
  fi
  PYTHON_CMD=(uv run --extra dev python)
fi

test -s "$TASK_MANIFEST" || { echo "Missing P+T task manifest: $TASK_MANIFEST" >&2; exit 1; }

echo "Started P+T full-train official validation for $DISEASE at $(date)"
echo "Task manifest: $TASK_MANIFEST"
echo "Validation PRS long scores: $VALIDATION_PRS_LONG_SCORES"
echo "Training PRS long scores: $TRAIN_PRS_LONG_SCORES"
echo "Output: $OUT_DIR"

"${PYTHON_CMD[@]}" scripts/collect_pt_scores.py \
  --task-manifest "$TASK_MANIFEST" \
  --out "$VALIDATION_PRS_LONG_SCORES"

"${PYTHON_CMD[@]}" scripts/collect_pt_scores.py \
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

echo "Finished P+T full-train official validation for $DISEASE at $(date)"
