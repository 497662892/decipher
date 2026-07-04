#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=deciphehr
#SBATCH --gres=gpu:1
#SBATCH --job-name=tabpfn_final_pred
#SBATCH --output=logs/tabpfn_final_holdout/%x-%J.log

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

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TABPFN_MODEL_CACHE_DIR" logs/tabpfn_final_holdout

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
TRAIN_PHENOTYPE="${TRAIN_PHENOTYPE:-data/processed/final/train_val_phenotype.csv}"
TEST_COVARIATES="${TEST_COVARIATES:-data/processed/final/holdout_test_covariates.csv}"
LABELS="${LABELS:-cad,bc,T2D}"
COVARIATES="${COVARIATES:-sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
ENSEMBLE_INPUT_DIR="${ENSEMBLE_INPUT_DIR:-outputs/tabpfn_final_ensemble_inputs}"
OUT_DIR="${OUT_DIR:-outputs/tabpfn_final_holdout}"
PREDICTION_ROOT="${PREDICTION_ROOT:-outputs/final_holdout_predictions/tabpfn}"
DEVICE="${DEVICE:-cuda}"
STRATEGIES="${STRATEGIES:-prs_only,covariate_adjusted}"
N_FOLDS="${N_FOLDS:-5}"
TRAIN_PREDICTION_MODE="${TRAIN_PREDICTION_MODE:-oof}"
SCORE_SCALE="${SCORE_SCALE:-train_sd}"
CLIP_EPS="${CLIP_EPS:-1e-5}"
CLASS_WEIGHT="${CLASS_WEIGHT:-none}"
WRITE_SUBMISSION="${WRITE_SUBMISSION:-1}"
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

echo "Started TabPFN final holdout prediction at $(date)"
echo "Using final ensemble inputs under $ENSEMBLE_INPUT_DIR"
echo "Writing TabPFN PRS outputs under $OUT_DIR"
"$PYTHON_BIN" scripts/build_tabpfn_ensemble_prs.py \
  --config "$CONFIG" \
  --labels "$LABELS" \
  --train-phenotype "$TRAIN_PHENOTYPE" \
  --validation-covariates "$TEST_COVARIATES" \
  --eligibility-config "$ELIGIBILITY_CONFIG" \
  --train-prs-long-scores "$ENSEMBLE_INPUT_DIR/{disease}.train_long.csv" \
  --validation-prs-long-scores "$ENSEMBLE_INPUT_DIR/{disease}.test_long.csv" \
  --strategies "$STRATEGIES" \
  --n-folds "$N_FOLDS" \
  --train-prediction-mode "$TRAIN_PREDICTION_MODE" \
  --score-scale "$SCORE_SCALE" \
  --clip-eps "$CLIP_EPS" \
  --device "$DEVICE" \
  --tabpfn-model-version "$TABPFN_MODEL_VERSION" \
  --tabpfn-kwargs "$TABPFN_KWARGS" \
  "${TABPFN_CACHE_ARGS[@]}" \
  --no-validation-commands \
  --out-dir "$OUT_DIR"

IFS=, read -r -a STRATEGY_LIST <<< "$STRATEGIES"
for STRATEGY in "${STRATEGY_LIST[@]}"; do
  STRATEGY_DIR="$OUT_DIR/$STRATEGY"
  PRED_DIR="$PREDICTION_ROOT/$STRATEGY"
  mkdir -p "$PRED_DIR"
  SUBMISSION_ARGS=()
  if [[ "$WRITE_SUBMISSION" == "1" ]]; then
    SUBMISSION_ARGS=(--submission-out "$PRED_DIR/submission.csv")
  fi
  "$PYTHON_BIN" scripts/predict_final_holdout_model.py \
    --model-name "tabpfn_ensemble_$STRATEGY" \
    --train-phenotype "$TRAIN_PHENOTYPE" \
    --test-covariates "$TEST_COVARIATES" \
    --patient-col "$PATIENT_COL" \
    --labels "$LABELS" \
    --covariates "$COVARIATES" \
    --train-prs-scores "$STRATEGY_DIR/train_prs_scores.csv" \
    --test-prs-scores "$STRATEGY_DIR/validation_prs_scores.csv" \
    --class-weight "$CLASS_WEIGHT" \
    --eligibility-config "$ELIGIBILITY_CONFIG" \
    --out-dir "$PRED_DIR" \
    --predictions-out "$PRED_DIR/predictions.csv" \
    "${SUBMISSION_ARGS[@]}"
done

echo "Finished TabPFN final holdout prediction at $(date)"
