#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=gpu4_dev,gl40s_dev
#SBATCH --job-name=final_holdout_predict
#SBATCH --output=logs/final_holdout_predict/%x-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
METHOD="${METHOD:-}"
DISEASE="${DISEASE:-}"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

if [[ -z "$METHOD" ]]; then
  echo "METHOD is required: pt, prscs, sdpr, ldpred2, or lassosum2_ukbb." >&2
  exit 1
fi
if [[ -z "$DISEASE" ]]; then
  echo "DISEASE is required: cad, bc, or T2D." >&2
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
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
if [[ -x "$PYTHON_BIN" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  PYTHON_CMD=(uv run --extra dev python)
fi

TRAIN_PHENOTYPE="${TRAIN_PHENOTYPE:-data/processed/final/train_val_phenotype.csv}"
TEST_COVARIATES="${TEST_COVARIATES:-data/processed/final/holdout_test_covariates.csv}"
LABELS="${LABELS:-$DISEASE}"
COVARIATES="${COVARIATES:-sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd}"
OUT_DIR="${OUT_DIR:-outputs/final_holdout_predictions/$METHOD/$DISEASE}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-$OUT_DIR/predictions.csv}"
SUBMISSION_OUT="${SUBMISSION_OUT:-}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
CLASS_WEIGHT="${CLASS_WEIGHT:-none}"
SELECTED_PRS_PARAMS="${SELECTED_PRS_PARAMS:-}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR" logs/final_holdout_predict

case "$METHOD" in
  pt)
    SCORE_ROOT="${SCORE_ROOT:-outputs/pt_final_trainval_test}"
    TASK_MANIFEST="${TASK_MANIFEST:-$SCORE_ROOT/$DISEASE/commands/$DISEASE.pt_full_train.tasks.csv}"
    TEST_PRS_LONG_SCORES="${TEST_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.pt_full_train_holdout_test_scores_long.csv}"
    TRAIN_PRS_LONG_SCORES="${TRAIN_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.pt_full_train_trainval_scores_long.csv}"
    PRS_PARAM_COLS="${PRS_PARAM_COLS:-clump_r2,p_threshold}"
    test -s "$TASK_MANIFEST" || { echo "Missing P+T task manifest: $TASK_MANIFEST" >&2; exit 1; }
    "${PYTHON_CMD[@]}" scripts/collect_pt_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --out "$TEST_PRS_LONG_SCORES"
    "${PYTHON_CMD[@]}" scripts/collect_pt_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --sscore-column fit_sscore --out "$TRAIN_PRS_LONG_SCORES"
    ;;
  prscs)
    SCORE_ROOT="${SCORE_ROOT:-outputs/prscs_final_trainval_test_ukbb_eur}"
    TASK_MANIFEST="${TASK_MANIFEST:-$SCORE_ROOT/$DISEASE/commands/$DISEASE.prscs_full_train.tasks.csv}"
    SCORE_SCRIPT_LIST="${SCORE_SCRIPT_LIST:-$SCORE_ROOT/$DISEASE/commands/$DISEASE.prscs_full_train.score_scripts.txt}"
    TEST_PRS_LONG_SCORES="${TEST_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.prscs_full_train_holdout_test_scores_long.csv}"
    TRAIN_PRS_LONG_SCORES="${TRAIN_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.prscs_full_train_trainval_scores_long.csv}"
    PRS_PARAM_COLS="${PRS_PARAM_COLS:-phi}"
    test -s "$TASK_MANIFEST" || { echo "Missing PRS-CS task manifest: $TASK_MANIFEST" >&2; exit 1; }
    test -s "$SCORE_SCRIPT_LIST" || { echo "Missing PRS-CS score script list: $SCORE_SCRIPT_LIST" >&2; exit 1; }
    while IFS= read -r SCORE_SCRIPT; do
      [[ -n "$SCORE_SCRIPT" ]] || continue
      bash "$SCORE_SCRIPT"
    done < "$SCORE_SCRIPT_LIST"
    "${PYTHON_CMD[@]}" scripts/collect_prscs_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --out "$TEST_PRS_LONG_SCORES"
    "${PYTHON_CMD[@]}" scripts/collect_prscs_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --sscore-column fit_sscore --out "$TRAIN_PRS_LONG_SCORES"
    ;;
  sdpr)
    SCORE_ROOT="${SCORE_ROOT:-outputs/sdpr_final_trainval_test}"
    TASK_MANIFEST="${TASK_MANIFEST:-$SCORE_ROOT/$DISEASE/commands/$DISEASE.sdpr_full_train.tasks.csv}"
    SCORE_SCRIPT_LIST="${SCORE_SCRIPT_LIST:-$SCORE_ROOT/$DISEASE/commands/$DISEASE.sdpr_full_train.score_scripts.txt}"
    TEST_PRS_LONG_SCORES="${TEST_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.sdpr_full_train_holdout_test_scores_long.csv}"
    TRAIN_PRS_LONG_SCORES="${TRAIN_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.sdpr_full_train_trainval_scores_long.csv}"
    PRS_PARAM_COLS="${PRS_PARAM_COLS:-model}"
    test -s "$TASK_MANIFEST" || { echo "Missing SDPR task manifest: $TASK_MANIFEST" >&2; exit 1; }
    test -s "$SCORE_SCRIPT_LIST" || { echo "Missing SDPR score script list: $SCORE_SCRIPT_LIST" >&2; exit 1; }
    while IFS= read -r SCORE_SCRIPT; do
      [[ -n "$SCORE_SCRIPT" ]] || continue
      bash "$SCORE_SCRIPT"
    done < "$SCORE_SCRIPT_LIST"
    "${PYTHON_CMD[@]}" scripts/collect_sdpr_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --out "$TEST_PRS_LONG_SCORES"
    "${PYTHON_CMD[@]}" scripts/collect_sdpr_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --sscore-column fit_sscore --out "$TRAIN_PRS_LONG_SCORES"
    ;;
  ldpred2)
    SCORE_ROOT="${SCORE_ROOT:-outputs/ldpred2_final_trainval_test}"
    TASK_MANIFEST="${TASK_MANIFEST:-$SCORE_ROOT/$DISEASE/commands/$DISEASE.ldpred2_full_train.tasks.csv}"
    TEST_PRS_LONG_SCORES="${TEST_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.ldpred2_full_train_holdout_test_scores_long.csv}"
    TRAIN_PRS_LONG_SCORES="${TRAIN_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.ldpred2_full_train_trainval_scores_long.csv}"
    PRS_PARAM_COLS="${PRS_PARAM_COLS:-h2,p,sparse}"
    test -s "$TASK_MANIFEST" || { echo "Missing LDpred2 task manifest: $TASK_MANIFEST" >&2; exit 1; }
    "${PYTHON_CMD[@]}" scripts/collect_ldpred2_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --out "$TEST_PRS_LONG_SCORES"
    "${PYTHON_CMD[@]}" scripts/collect_ldpred2_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --score-file-column fit_score_file --out "$TRAIN_PRS_LONG_SCORES"
    ;;
  lassosum2_ukbb)
    SCORE_ROOT="${SCORE_ROOT:-outputs/lassosum2_ukbb_final_trainval_test}"
    TASK_MANIFEST="${TASK_MANIFEST:-$SCORE_ROOT/$DISEASE/commands/$DISEASE.lassosum2_full_train.tasks.csv}"
    TEST_PRS_LONG_SCORES="${TEST_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.lassosum2_full_train_holdout_test_scores_long.csv}"
    TRAIN_PRS_LONG_SCORES="${TRAIN_PRS_LONG_SCORES:-$SCORE_ROOT/$DISEASE/scores/$DISEASE.lassosum2_full_train_trainval_scores_long.csv}"
    PRS_PARAM_COLS="${PRS_PARAM_COLS:-delta,lambda}"
    test -s "$TASK_MANIFEST" || { echo "Missing lassosum2 task manifest: $TASK_MANIFEST" >&2; exit 1; }
    "${PYTHON_CMD[@]}" scripts/collect_lassosum2_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --out "$TEST_PRS_LONG_SCORES"
    "${PYTHON_CMD[@]}" scripts/collect_lassosum2_scores.py --task-manifest "$TASK_MANIFEST" --patient-col "$PATIENT_COL" --score-file-column fit_score_file --out "$TRAIN_PRS_LONG_SCORES"
    ;;
  *)
    echo "Unsupported METHOD=$METHOD" >&2
    exit 1
    ;;
esac

SUBMISSION_ARGS=()
if [[ -n "$SUBMISSION_OUT" ]]; then
  SUBMISSION_ARGS=(--submission-out "$SUBMISSION_OUT")
fi
SELECTED_PRS_ARGS=()
if [[ -n "$SELECTED_PRS_PARAMS" ]]; then
  SELECTED_PRS_ARGS=(--selected-prs-params "$SELECTED_PRS_PARAMS")
fi

"${PYTHON_CMD[@]}" scripts/predict_final_holdout_model.py \
  --model-name "$METHOD" \
  --train-phenotype "$TRAIN_PHENOTYPE" \
  --test-covariates "$TEST_COVARIATES" \
  --patient-col "$PATIENT_COL" \
  --labels "$LABELS" \
  --covariates "$COVARIATES" \
  --train-prs-long-scores "$TRAIN_PRS_LONG_SCORES" \
  --test-prs-long-scores "$TEST_PRS_LONG_SCORES" \
  --prs-param-cols "$PRS_PARAM_COLS" \
  "${SELECTED_PRS_ARGS[@]}" \
  --class-weight "$CLASS_WEIGHT" \
  --out-dir "$OUT_DIR" \
  --predictions-out "$PREDICTIONS_OUT" \
  "${SUBMISSION_ARGS[@]}"

echo "Finished final holdout prediction METHOD=$METHOD DISEASE=$DISEASE at $(date)"
