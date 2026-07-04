#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=deciphehr
#SBATCH --gres=gpu:1
#SBATCH --job-name=tabpfn_cv
#SBATCH --output=logs/tabpfn_final_trainval/%x-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" && -z "${PYTORCH_ALLOC_CONF:-}" ]]; then
  export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
fi
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"
export TABPFN_NO_BROWSER="${TABPFN_NO_BROWSER:-1}"
export TABPFN_DISABLE_TELEMETRY="${TABPFN_DISABLE_TELEMETRY:-1}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$REPO_DIR/outputs/tabpfn-model-cache}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TABPFN_MODEL_CACHE_DIR" logs/tabpfn_final_trainval

TABPFN_AUTH_READY=0
if [[ -n "${TABPFN_TOKEN:-}" ]]; then
  TABPFN_AUTH_READY=1
elif [[ -n "${HOME:-}" && -s "$HOME/.cache/tabpfn/auth_token" ]]; then
  TABPFN_AUTH_READY=1
elif [[ -n "${HOME:-}" && -s "$HOME/.tabpfn/token" ]]; then
  TABPFN_AUTH_READY=1
fi
if [[ "$TABPFN_AUTH_READY" != "1" ]]; then
  {
    echo "Missing TabPFN authentication for this non-interactive Slurm job."
    echo "Set TABPFN_TOKEN in $TABPFN_ENV_FILE, or run an interactive TabPFN login once."
  } >&2
  exit 2
fi

CONFIG="${CONFIG:-configs/default.yaml}"
ELIGIBILITY_CONFIG="${ELIGIBILITY_CONFIG:-configs/disease_eligibility.yaml}"
PHENOTYPE="${PHENOTYPE:-data/processed/final/train_val_phenotype.csv}"
SPLITS="${SPLITS:-data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv}"
LABELS="${LABELS:-cad,bc,T2D}"
ENSEMBLE_INPUT_DIR="${ENSEMBLE_INPUT_DIR:-outputs/tabpfn_cv_ensemble_inputs}"
OUT_DIR="${OUT_DIR:-outputs/tabpfn_cv_final_trainval}"
DEVICE="${DEVICE:-cuda}"
STRATEGIES="${STRATEGIES:-prs_only,covariate_adjusted}"
INNER_FOLDS="${INNER_FOLDS:-5}"
TRAIN_PREDICTION_MODE="${TRAIN_PREDICTION_MODE:-oof}"
CALIBRATION_METHODS="${CALIBRATION_METHODS:-prs_covariate}"
CLASS_WEIGHT="${CLASS_WEIGHT:-none}"
SCORE_SCALE="${SCORE_SCALE:-train_sd}"
CLIP_EPS="${CLIP_EPS:-1e-5}"
TABPFN_KWARGS="${TABPFN_KWARGS:-}"
TABPFN_MODEL_VERSION="${TABPFN_MODEL_VERSION:-default}"
TABPFN_MODEL_PATH="${TABPFN_MODEL_PATH:-}"
TABPFN_N_ESTIMATORS="${TABPFN_N_ESTIMATORS:-}"
TABPFN_RANDOM_STATE="${TABPFN_RANDOM_STATE:-}"
TABPFN_BALANCE_PROBABILITIES="${TABPFN_BALANCE_PROBABILITIES:-}"

if [[ -z "$TABPFN_KWARGS" ]]; then
  TABPFN_KWARGS_PARTS=()
  if [[ -n "$TABPFN_N_ESTIMATORS" ]]; then
    TABPFN_KWARGS_PARTS+=("\"n_estimators\":$TABPFN_N_ESTIMATORS")
  fi
  if [[ -n "$TABPFN_RANDOM_STATE" ]]; then
    TABPFN_KWARGS_PARTS+=("\"random_state\":$TABPFN_RANDOM_STATE")
  fi
  if [[ -n "$TABPFN_MODEL_PATH" ]]; then
    case "$TABPFN_MODEL_PATH" in
      *\"*|*\\*)
        echo "TABPFN_MODEL_PATH must not contain quotes or backslashes; got $TABPFN_MODEL_PATH"
        exit 2
        ;;
    esac
    TABPFN_KWARGS_PARTS+=("\"model_path\":\"$TABPFN_MODEL_PATH\"")
  fi
  if [[ -n "$TABPFN_BALANCE_PROBABILITIES" ]]; then
    case "$TABPFN_BALANCE_PROBABILITIES" in
      true|false) ;;
      *)
        echo "TABPFN_BALANCE_PROBABILITIES must be true or false; got $TABPFN_BALANCE_PROBABILITIES"
        exit 2
        ;;
    esac
    TABPFN_KWARGS_PARTS+=("\"balance_probabilities\":$TABPFN_BALANCE_PROBABILITIES")
  fi
  if ((${#TABPFN_KWARGS_PARTS[@]})); then
    printf -v TABPFN_KWARGS '{%s}' "$(IFS=,; echo "${TABPFN_KWARGS_PARTS[*]}")"
  else
    TABPFN_KWARGS="{}"
  fi
fi

TABPFN_CACHE_ARGS=()
if [[ -n "${TABPFN_MODEL_CACHE_DIR:-}" ]]; then
  TABPFN_CACHE_ARGS=(--tabpfn-model-cache-dir "$TABPFN_MODEL_CACHE_DIR")
fi

if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
  uv sync --frozen --extra tabpfn
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

echo "Started TabPFN final train-val 5-CV at $(date)"
echo "Using ensemble inputs under $ENSEMBLE_INPUT_DIR"
echo "Writing CV outputs under $OUT_DIR"
echo "Strategies: $STRATEGIES; inner_folds: $INNER_FOLDS; calibration: $CALIBRATION_METHODS"
"$PYTHON_BIN" scripts/evaluate_tabpfn_ensemble_cv.py \
  --config "$CONFIG" \
  --labels "$LABELS" \
  --phenotype "$PHENOTYPE" \
  --splits "$SPLITS" \
  --eligibility-config "$ELIGIBILITY_CONFIG" \
  --train-prs-long-scores "$ENSEMBLE_INPUT_DIR/{disease}.train_long.csv" \
  --strategies "$STRATEGIES" \
  --inner-folds "$INNER_FOLDS" \
  --train-prediction-mode "$TRAIN_PREDICTION_MODE" \
  --calibration-methods "$CALIBRATION_METHODS" \
  --class-weight "$CLASS_WEIGHT" \
  --score-scale "$SCORE_SCALE" \
  --clip-eps "$CLIP_EPS" \
  --device "$DEVICE" \
  --tabpfn-model-version "$TABPFN_MODEL_VERSION" \
  --tabpfn-kwargs "$TABPFN_KWARGS" \
  "${TABPFN_CACHE_ARGS[@]}" \
  --out-dir "$OUT_DIR"

echo "Finished TabPFN final train-val 5-CV at $(date)"
