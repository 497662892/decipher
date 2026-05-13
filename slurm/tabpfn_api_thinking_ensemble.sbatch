#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=tabpfn_api_think
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/tabpfn_api_thinking/%x-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
cd "$REPO_DIR"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  source /gpfs/home/jm10850/.bashrc
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
export TMPDIR="${TMPDIR:-/tmp}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

if [[ -z "${PRIORLABS_API_KEY:-}" && -n "${TABPFN_TOKEN:-}" ]]; then
  export PRIORLABS_API_KEY="$TABPFN_TOKEN"
fi

if [[ -z "${PRIORLABS_API_KEY:-}" ]]; then
  {
    echo "Missing Prior Labs API token for this non-interactive Slurm job."
    echo "Set PRIORLABS_API_KEY or TABPFN_TOKEN in $TABPFN_ENV_FILE."
  } >&2
  exit 2
fi

mkdir -p "$UV_CACHE_DIR" logs/tabpfn_api_thinking

CONFIG="${CONFIG:-configs/default.yaml}"
ELIGIBILITY_CONFIG="${ELIGIBILITY_CONFIG:-configs/disease_eligibility.yaml}"
LABELS="${LABELS:-cad,bc,T2D}"
ENSEMBLE_INPUT_DIR="${ENSEMBLE_INPUT_DIR:-outputs/ensemble_inputs_nonannotated_a0p20}"
OUT_DIR="${OUT_DIR:-outputs/tabpfn_api_thinking_nonannotated_a0p20_covariate_adjusted}"
STRATEGIES="${STRATEGIES:-covariate_adjusted}"
N_FOLDS="${N_FOLDS:-5}"
TRAIN_PREDICTION_MODE="${TRAIN_PREDICTION_MODE:-oof}"
SCORE_SCALE="${SCORE_SCALE:-train_sd}"
CLIP_EPS="${CLIP_EPS:-1e-5}"
THINKING_EFFORT="${THINKING_EFFORT:-high}"
THINKING_TIMEOUT_S="${THINKING_TIMEOUT_S:-300}"
THINKING_METRIC="${THINKING_METRIC:-roc_auc}"
TABPFN_SYSTEMS="${TABPFN_SYSTEMS:-preprocessing,thinking}"
HTTP_TIMEOUT_S="${HTTP_TIMEOUT_S:-1800}"
API_CONFLICT_RETRIES="${API_CONFLICT_RETRIES:-5}"
API_CONFLICT_SLEEP_S="${API_CONFLICT_SLEEP_S:-60}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

echo "Started Prior Labs TabPFN API thinking ensemble at $(date)"
echo "Using ensemble inputs under $ENSEMBLE_INPUT_DIR"
echo "Writing outputs under $OUT_DIR"
echo "Labels: $LABELS"
echo "Strategies: $STRATEGIES; folds: $N_FOLDS; train_mode: $TRAIN_PREDICTION_MODE"
echo "Thinking effort: $THINKING_EFFORT; metric: $THINKING_METRIC; timeout_s: $THINKING_TIMEOUT_S"
echo "TabPFN API systems: $TABPFN_SYSTEMS"
echo "API conflict retries: $API_CONFLICT_RETRIES; sleep_s: $API_CONFLICT_SLEEP_S"

if [[ "${SKIP_UV_SYNC:-1}" != "1" ]]; then
  uv sync --frozen --extra dev
fi

test -x "$PYTHON_BIN" || { echo "Missing Python executable: $PYTHON_BIN" >&2; exit 2; }

VALIDATION_ARGS=()
if [[ "$RUN_VALIDATION" != "1" ]]; then
  VALIDATION_ARGS=(--no-validation-commands)
fi

"$PYTHON_BIN" scripts/build_tabpfn_api_ensemble_prs.py \
  --config "$CONFIG" \
  --labels "$LABELS" \
  --eligibility-config "$ELIGIBILITY_CONFIG" \
  --train-prs-long-scores "$ENSEMBLE_INPUT_DIR/{disease}.train_long.csv" \
  --validation-prs-long-scores "$ENSEMBLE_INPUT_DIR/{disease}.validation_long.csv" \
  --strategies "$STRATEGIES" \
  --n-folds "$N_FOLDS" \
  --train-prediction-mode "$TRAIN_PREDICTION_MODE" \
  --score-scale "$SCORE_SCALE" \
  --clip-eps "$CLIP_EPS" \
  --tabpfn-systems "$TABPFN_SYSTEMS" \
  --thinking-effort "$THINKING_EFFORT" \
  --thinking-timeout-s "$THINKING_TIMEOUT_S" \
  --thinking-metric "$THINKING_METRIC" \
  --http-timeout-s "$HTTP_TIMEOUT_S" \
  --api-conflict-retries "$API_CONFLICT_RETRIES" \
  --api-conflict-sleep-s "$API_CONFLICT_SLEEP_S" \
  "${VALIDATION_ARGS[@]}" \
  --out-dir "$OUT_DIR"

if [[ "$RUN_VALIDATION" == "1" ]]; then
  bash "$OUT_DIR/validate_tabpfn_api_ensemble_commands.sh"
fi

echo "Finished Prior Labs TabPFN API thinking ensemble at $(date)"
