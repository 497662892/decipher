#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-1%1
#SBATCH --job-name=lassosum2_ukbb_full
#SBATCH --output=logs/lassosum2_ukbb_full_train/%x-%A_%a.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DISEASE="${DISEASE:-}"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

if [[ -z "$DISEASE" ]]; then
  echo "DISEASE is required; use a disease wrapper or export DISEASE=cad/bc/T2D." >&2
  exit 1
fi

cd "$REPO_DIR"

R_MODULE="${R_MODULE:-r/4.5.3}"
if command -v module >/dev/null 2>&1; then
  module load plink/2.0-20260228 || true
  module load "$R_MODULE" || true
fi
if ! command -v plink2 >/dev/null 2>&1 && [[ -n "${PLINK2_DIR:-}" && -x "$PLINK2_DIR/plink2" ]]; then
  export PATH="$PLINK2_DIR:$PATH"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"
export R_LIBS_USER="${R_LIBS_USER:-$REPO_DIR/data/resources/R/4.5.3-library}"
mkdir -p "$R_LIBS_USER"

MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
SCORE_GENOTYPE_PREFIX="${SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"
FIT_SCORE_GENOTYPE_PREFIX="${FIT_SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"
HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/sumstats/harmonized}"
TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/labeled_train.keep.psam}"
SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/unlabelled_validation.keep.psam}"
FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
OUT_DIR="${OUT_DIR:-outputs/lassosum2_ukbb_full_train}"
LD_CACHE_DIR="${LD_CACHE_DIR:-outputs/lassosum2_ukbb_ld_cache}"
PLINK2="${PLINK2:-plink2}"
R_SCRIPT="${R_SCRIPT:-Rscript}"
LASSOSUM2_SCRIPT="${LASSOSUM2_SCRIPT:-scripts/run_lassosum2_task.R}"
DELTA_GRID="${DELTA_GRID:-0.001,0.01,0.1,1}"
NLAMBDA="${NLAMBDA:-30}"
LAMBDA_MIN_RATIO="${LAMBDA_MIN_RATIO:-0.01}"
DFMAX="${DFMAX:-200000}"
MAXITER="${MAXITER:-1000}"
TOL="${TOL:-1e-5}"
LD_REFERENCE_MODE="${LD_REFERENCE_MODE:-ukb_hm3_plus}"
LD_REF_MAP="${LD_REF_MAP:-data/resources/ldpred2/map_hm3_plus.rds}"
LD_REF_DIR="${LD_REF_DIR:-data/resources/ldpred2/corr_hm3_plus}"
if [[ -z "${LD_REF_PATTERN:-}" ]]; then
  LD_REF_PATTERN='LD_with_blocks_chr{chr}.rds'
fi
VARIANT_EXTRACT="${VARIANT_EXTRACT:-data/processed/ldpred2/hm3plus_target.snplist}"
CHROMOSOMES="${CHROMOSOMES:-1-22}"
MATCH_BY="${MATCH_BY:-position}"
DEFAULT_N="${DEFAULT_N:-}"
SAVE_BETA_GRID="${SAVE_BETA_GRID:-0}"
REUSE_LASSOSUM2_LD="${REUSE_LASSOSUM2_LD:-1}"
REGENERATE_LASSOSUM2_TASKS="${REGENERATE_LASSOSUM2_TASKS:-0}"
TASKS_FILE="${TASKS_FILE:-$OUT_DIR/$DISEASE/commands/$DISEASE.lassosum2_full_train.array_tasks.txt}"
LOCK_FILE="$OUT_DIR/$DISEASE/commands/.generate.lock"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" logs/lassosum2_ukbb_full_train

if ! command -v "$R_SCRIPT" >/dev/null 2>&1; then
  echo "Cannot find Rscript command: $R_SCRIPT" >&2
  exit 1
fi

DEFAULT_N_ARGS=()
if [[ -n "$DEFAULT_N" ]]; then
  DEFAULT_N_ARGS=(--default-n "$DEFAULT_N")
fi
SAVE_BETA_ARGS=()
if [[ "$SAVE_BETA_GRID" == "1" ]]; then
  SAVE_BETA_ARGS=(--save-beta-grid)
fi
VARIANT_EXTRACT_ARGS=()
if [[ -n "$VARIANT_EXTRACT" ]]; then
  VARIANT_EXTRACT_ARGS=(--variant-extract "$VARIANT_EXTRACT")
fi
LD_CACHE_ARGS=()
if [[ -n "$LD_CACHE_DIR" ]]; then
  LD_CACHE_ARGS=(--ld-cache-dir "$LD_CACHE_DIR")
fi

(
  flock 9
  if [[ ! -s "$TASKS_FILE" || "$REGENERATE_LASSOSUM2_TASKS" == "1" ]]; then
    uv run --extra dev python -u scripts/write_lassosum2_full_train_commands.py \
      --disease "$DISEASE" \
      --manifest "$MANIFEST" \
      --patient-col "$PATIENT_COL" \
      --genotype-prefix "$GENOTYPE_PREFIX" \
      --score-genotype-prefix "$SCORE_GENOTYPE_PREFIX" \
      --fit-score-genotype-prefix "$FIT_SCORE_GENOTYPE_PREFIX" \
      --harmonized-dir "$HARMONIZED_DIR" \
      --train-keep "$TRAIN_KEEP" \
      --score-keep "$SCORE_KEEP" \
      --fit-score-keep "$FIT_SCORE_KEEP" \
      --out-dir "$OUT_DIR" \
      --plink2 "$PLINK2" \
      --rscript "$R_SCRIPT" \
      --lassosum2-script "$LASSOSUM2_SCRIPT" \
      --delta-grid "$DELTA_GRID" \
      --nlambda "$NLAMBDA" \
      --lambda-min-ratio "$LAMBDA_MIN_RATIO" \
      --dfmax "$DFMAX" \
      --maxiter "$MAXITER" \
      --tol "$TOL" \
      --ld-reference-mode "$LD_REFERENCE_MODE" \
      --ld-ref-map "$LD_REF_MAP" \
      --ld-ref-dir "$LD_REF_DIR" \
      --ld-ref-pattern "$LD_REF_PATTERN" \
      "${VARIANT_EXTRACT_ARGS[@]}" \
      "${LD_CACHE_ARGS[@]}" \
      --chromosomes "$CHROMOSOMES" \
      --threads "$OMP_NUM_THREADS" \
      --match-by "$MATCH_BY" \
      "${DEFAULT_N_ARGS[@]}" \
      "${SAVE_BETA_ARGS[@]}"
  fi
) 9>"$LOCK_FILE"

TASK_SCRIPT="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")"
if [[ -z "$TASK_SCRIPT" ]]; then
  echo "No task script for array index $SLURM_ARRAY_TASK_ID in $TASKS_FILE" >&2
  exit 1
fi

bash "$TASK_SCRIPT"
