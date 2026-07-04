#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=deciphehr
#SBATCH --gres=gpu:1
#SBATCH --job-name=tabpfn_oof_holdout
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
SPLITS="${SPLITS:-data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv}"
HOLDOUT_COVARIATES="${HOLDOUT_COVARIATES:-data/raw/holdout_phase/phenotype_files/Holdout_Cov_file.txt}"
PROCESSED_HOLDOUT_COVARIATES="${PROCESSED_HOLDOUT_COVARIATES:-data/processed/final/holdout_test_covariates.csv}"
LABELS="${LABELS:-cad,bc,T2D}"
COVARIATES="${COVARIATES:-sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
ENSEMBLE_INPUT_DIR="${ENSEMBLE_INPUT_DIR:-outputs/tabpfn3_final_trainval_selected_drop3_drop2_drop3_inputs}"
OUT_DIR="${OUT_DIR:-submissions/final_phase/tabpfn3_audited_pipeline}"
DEVICE="${DEVICE:-cuda}"
CLIP_EPS="${CLIP_EPS:-0.001}"
TABPFN_MODEL_VERSION="${TABPFN_MODEL_VERSION:-v3}"
TABPFN_MODEL_PATH="${TABPFN_MODEL_PATH:-}"
TABPFN_N_ESTIMATORS="${TABPFN_N_ESTIMATORS:-16}"
TABPFN_RANDOM_STATE="${TABPFN_RANDOM_STATE:-}"
TABPFN_KWARGS="${TABPFN_KWARGS:-}"
TEAM_NAME="${TEAM_NAME:-TeamName}"

if [[ -z "$TABPFN_KWARGS" ]]; then
  TABPFN_KWARGS_PARTS=()
  if [[ -n "$TABPFN_N_ESTIMATORS" ]]; then
    TABPFN_KWARGS_PARTS+=("\"n_estimators\":$TABPFN_N_ESTIMATORS")
  fi
  if [[ -n "$TABPFN_RANDOM_STATE" ]]; then
    TABPFN_KWARGS_PARTS+=("\"random_state\":$TABPFN_RANDOM_STATE")
  fi
  if [[ -n "$TABPFN_MODEL_PATH" ]]; then
    TABPFN_KWARGS_PARTS+=("\"model_path\":\"$TABPFN_MODEL_PATH\"")
  fi
  if ((${#TABPFN_KWARGS_PARTS[@]})); then
    printf -v TABPFN_KWARGS '{%s}' "$(IFS=,; echo "${TABPFN_KWARGS_PARTS[*]}")"
  else
    TABPFN_KWARGS="{}"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

echo "Started audited final TabPFN OOF+holdout pipeline at $(date)"
"$PYTHON_BIN" scripts/run_final_tabpfn_oof_holdout_pipeline.py \
  --config "$CONFIG" \
  --labels "$LABELS" \
  --covariates "$COVARIATES" \
  --patient-col "$PATIENT_COL" \
  --train-phenotype "$TRAIN_PHENOTYPE" \
  --splits "$SPLITS" \
  --holdout-covariates "$HOLDOUT_COVARIATES" \
  --processed-holdout-covariates "$PROCESSED_HOLDOUT_COVARIATES" \
  --eligibility-config "$ELIGIBILITY_CONFIG" \
  --train-prs-long-scores "$ENSEMBLE_INPUT_DIR/{disease}.train_long.csv" \
  --holdout-prs-long-scores "$ENSEMBLE_INPUT_DIR/{disease}.test_long.csv" \
  --clip-eps "$CLIP_EPS" \
  --device "$DEVICE" \
  --tabpfn-model-version "$TABPFN_MODEL_VERSION" \
  --tabpfn-kwargs "$TABPFN_KWARGS" \
  --tabpfn-model-cache-dir "$TABPFN_MODEL_CACHE_DIR" \
  --team-name "$TEAM_NAME" \
  --out-dir "$OUT_DIR"
echo "Finished audited final TabPFN OOF+holdout pipeline at $(date)"
