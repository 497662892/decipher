#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=deciphehr
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --job-name=tabpfn_single_bridge
#SBATCH --output=logs/tabpfn_single_bridge/%x-%A_%a.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_DIR"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

TABPFN_ENV_FILE="${TABPFN_ENV_FILE:-$REPO_DIR/.env.tabpfn}"
if [[ -f "$TABPFN_ENV_FILE" ]]; then
  set -a
  source "$TABPFN_ENV_FILE"
  set +a
fi

set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache}"
export HF_HOME="${HF_HOME:-$REPO_DIR/outputs/huggingface-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"
export TABPFN_NO_BROWSER="${TABPFN_NO_BROWSER:-1}"
export TABPFN_DISABLE_TELEMETRY="${TABPFN_DISABLE_TELEMETRY:-1}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$REPO_DIR/outputs/tabpfn-model-cache}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TABPFN_MODEL_CACHE_DIR" logs/tabpfn_single_bridge

DISEASES=(cad bc T2D)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if ((TASK_ID < 0 || TASK_ID >= ${#DISEASES[@]})); then
  echo "SLURM_ARRAY_TASK_ID must be 0, 1, or 2; got $TASK_ID" >&2
  exit 2
fi
DISEASE="${DISEASES[$TASK_ID]}"

DEFAULT_CHECKPOINT="$TABPFN_MODEL_CACHE_DIR/tabpfn-v3-classifier-v3_default.ckpt"
BINARY_CHECKPOINT="$TABPFN_MODEL_CACHE_DIR/tabpfn-v3-classifier-v3_20260417_binary.ckpt"
case "$DISEASE" in
  cad|T2D)
    N_ESTIMATORS=16
    MODEL_PATH="$DEFAULT_CHECKPOINT"
    BALANCE_PROBABILITIES=false
    ;;
  bc)
    N_ESTIMATORS=8
    MODEL_PATH="$BINARY_CHECKPOINT"
    BALANCE_PROBABILITIES=true
    ;;
esac

if [[ ! -f "$MODEL_PATH" ]] && command -v hf >/dev/null 2>&1; then
  hf download Prior-Labs/tabpfn_3 "$(basename "$MODEL_PATH")" \
    --local-dir "$TABPFN_MODEL_CACHE_DIR"
fi
if [[ ! -f "$MODEL_PATH" && -z "${TABPFN_TOKEN:-}" ]]; then
  echo "Missing $MODEL_PATH and no checkpoint download credential is available." >&2
  exit 2
fi

CONFIG="${CONFIG:-configs/official_validation_processed.yaml}"
ELIGIBILITY_CONFIG="${ELIGIBILITY_CONFIG:-configs/disease_eligibility.yaml}"
INPUT_ROOT="${INPUT_ROOT:-outputs/ancestry_stratified_best_single_vs_tabpfn3/tabpfn_best_single_bridge_inputs}"
BRIDGE_ROOT="${BRIDGE_ROOT:-outputs/ancestry_stratified_best_single_vs_tabpfn3/tabpfn_best_single_bridge}"
MODEL_OUT="$BRIDGE_ROOT/$DISEASE/model"
WRAPPER_OUT="$BRIDGE_ROOT/$DISEASE/standard_wrapper"
FULL_COVARIATES="sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
  uv sync --frozen --extra tabpfn
fi

echo "Started single-best-PRS TabPFN bridge for $DISEASE at $(date)"
echo "Checkpoint: $MODEL_PATH; n_estimators=$N_ESTIMATORS; balance=$BALANCE_PROBABILITIES"

"$PYTHON_BIN" scripts/prepare_tabpfn_single_best_bridge_inputs.py \
  --disease "$DISEASE" \
  --out-dir "$INPUT_ROOT"

TABPFN_KWARGS="{\"n_estimators\":$N_ESTIMATORS,\"model_path\":\"$MODEL_PATH\",\"balance_probabilities\":$BALANCE_PROBABILITIES}"
"$PYTHON_BIN" scripts/build_tabpfn_ensemble_prs.py \
  --config "$CONFIG" \
  --patient-col sampleid \
  --labels "$DISEASE" \
  --covariates "$FULL_COVARIATES" \
  --eligibility-config "$ELIGIBILITY_CONFIG" \
  --train-prs-long-scores "$INPUT_ROOT/{disease}.train_long.csv" \
  --validation-prs-long-scores "$INPUT_ROOT/{disease}.validation_long.csv" \
  --strategies covariate_adjusted \
  --n-folds 5 \
  --train-prediction-mode oof \
  --score-scale train_sd \
  --clip-eps 0.001 \
  --device cuda \
  --tabpfn-model-version v3 \
  --tabpfn-kwargs "$TABPFN_KWARGS" \
  --tabpfn-model-cache-dir "$TABPFN_MODEL_CACHE_DIR" \
  --no-validation-commands \
  --out-dir "$MODEL_OUT"

"$PYTHON_BIN" scripts/validate_official_model.py \
  --config "$CONFIG" \
  --model-name tabpfn3_best_single_bridge \
  --patient-col sampleid \
  --disease "$DISEASE" \
  --covariates "$FULL_COVARIATES" \
  --eligibility-config "$ELIGIBILITY_CONFIG" \
  --train-prs-scores "$MODEL_OUT/covariate_adjusted/train_prs_scores.csv" \
  --validation-prs-scores "$MODEL_OUT/covariate_adjusted/validation_prs_scores.csv" \
  --validation-phenotype-out "$WRAPPER_OUT/validation_phenotype.csv" \
  --out-dir "$WRAPPER_OUT" \
  --point-only \
  --write-predictions

echo "Finished single-best-PRS TabPFN bridge for $DISEASE at $(date)"
