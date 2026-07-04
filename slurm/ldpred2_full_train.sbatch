#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=72:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-1%1
#SBATCH --job-name=ldpred2_full_train
#SBATCH --output=logs/ldpred2_full_train/%x-%A_%a.log

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
HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/sumstats/harmonized}"
TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/labeled_train.keep.psam}"
SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/unlabelled_validation.keep.psam}"
FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
OUT_DIR="${OUT_DIR:-outputs/ldpred2_full_train}"
LD_CACHE_DIR="${LD_CACHE_DIR:-outputs/ldpred2_ld_cache}"
PLINK2="${PLINK2:-plink2}"
R_SCRIPT="${R_SCRIPT:-Rscript}"
LDPRED2_SCRIPT="${LDPRED2_SCRIPT:-scripts/run_ldpred2_task.R}"
H2_GRID="${H2_GRID:-0.05,0.1,0.2,0.4,0.8,1}"
P_GRID="${P_GRID:-0.0001,0.001,0.01,0.1,1}"
SPARSE_GRID="${SPARSE_GRID:-0,1}"
LDPRED2_MODE="${LDPRED2_MODE:-grid}"
LD_REFERENCE_MODE="${LD_REFERENCE_MODE:-target}"
LD_REF_MAP="${LD_REF_MAP:-}"
LD_REF_DIR="${LD_REF_DIR:-}"
if [[ -z "${LD_REF_PATTERN:-}" ]]; then
  LD_REF_PATTERN='LD_with_blocks_chr{chr}.rds'
fi
AUTO_H2_INIT="${AUTO_H2_INIT:-0.1}"
AUTO_P_INIT="${AUTO_P_INIT:-0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.2}"
AUTO_BURN_IN="${AUTO_BURN_IN:-500}"
AUTO_NUM_ITER="${AUTO_NUM_ITER:-200}"
AUTO_SPARSE="${AUTO_SPARSE:-1}"
AUTO_ALLOW_JUMP_SIGN="${AUTO_ALLOW_JUMP_SIGN:-0}"
AUTO_SHRINK_CORR="${AUTO_SHRINK_CORR:-0.95}"
AUTO_USE_MLE="${AUTO_USE_MLE:-1}"
AUTO_SEED="${AUTO_SEED:-42}"
AUTO_VERBOSE="${AUTO_VERBOSE:-0}"
LD_WINDOW_CM="${LD_WINDOW_CM:-3}"
LD_WINDOW_MB="${LD_WINDOW_MB:-3}"
VARIANT_EXTRACT="${VARIANT_EXTRACT:-data/processed/ldpred2/hm3plus_target.snplist}"
GENETIC_MAP_DIR="${GENETIC_MAP_DIR:-data/resources/genetic_maps}"
GENETIC_MAP_TYPE="${GENETIC_MAP_TYPE:-hg38_price}"
ALLOW_MB_FALLBACK="${ALLOW_MB_FALLBACK:-0}"
CHROMOSOMES="${CHROMOSOMES:-1-22}"
MATCH_BY="${MATCH_BY:-id}"
DEFAULT_N="${DEFAULT_N:-}"
SAVE_BETA_GRID="${SAVE_BETA_GRID:-0}"
REGENERATE_LDPRED2_TASKS="${REGENERATE_LDPRED2_TASKS:-0}"
TASKS_FILE="${TASKS_FILE:-$OUT_DIR/$DISEASE/commands/$DISEASE.ldpred2_full_train.array_tasks.txt}"
LOCK_FILE="$OUT_DIR/$DISEASE/commands/.generate.lock"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" "logs/ldpred2_full_train"

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
LD_REF_ARGS=()
if [[ "$LD_REFERENCE_MODE" != "target" ]]; then
  LD_REF_ARGS=(--ld-ref-map "$LD_REF_MAP" --ld-ref-dir "$LD_REF_DIR" --ld-ref-pattern "$LD_REF_PATTERN")
fi
GENETIC_MAP_ARGS=()
if [[ -n "$GENETIC_MAP_DIR" ]]; then
  GENETIC_MAP_ARGS=(--genetic-map-dir "$GENETIC_MAP_DIR")
fi
ALLOW_MB_FALLBACK_ARGS=()
if [[ "$ALLOW_MB_FALLBACK" == "1" ]]; then
  ALLOW_MB_FALLBACK_ARGS=(--allow-mb-fallback)
fi

(
  flock 9
  if [[ ! -s "$TASKS_FILE" || "$REGENERATE_LDPRED2_TASKS" == "1" ]]; then
    uv run --extra dev python -u scripts/write_ldpred2_full_train_commands.py \
      --disease "$DISEASE" \
      --manifest "$MANIFEST" \
      --patient-col "$PATIENT_COL" \
      --genotype-prefix "$GENOTYPE_PREFIX" \
      --score-genotype-prefix "$SCORE_GENOTYPE_PREFIX" \
      --harmonized-dir "$HARMONIZED_DIR" \
      --train-keep "$TRAIN_KEEP" \
      --score-keep "$SCORE_KEEP" \
      --fit-score-keep "$FIT_SCORE_KEEP" \
      --out-dir "$OUT_DIR" \
      --plink2 "$PLINK2" \
      --rscript "$R_SCRIPT" \
      --ldpred2-script "$LDPRED2_SCRIPT" \
      --h2-grid "$H2_GRID" \
      --p-grid "$P_GRID" \
      --sparse-grid "$SPARSE_GRID" \
      --ldpred2-mode "$LDPRED2_MODE" \
      --ld-reference-mode "$LD_REFERENCE_MODE" \
      "${LD_REF_ARGS[@]}" \
      --auto-h2-init "$AUTO_H2_INIT" \
      --auto-p-init "$AUTO_P_INIT" \
      --auto-burn-in "$AUTO_BURN_IN" \
      --auto-num-iter "$AUTO_NUM_ITER" \
      --auto-sparse "$AUTO_SPARSE" \
      --auto-allow-jump-sign "$AUTO_ALLOW_JUMP_SIGN" \
      --auto-shrink-corr "$AUTO_SHRINK_CORR" \
      --auto-use-mle "$AUTO_USE_MLE" \
      --auto-seed "$AUTO_SEED" \
      --auto-verbose "$AUTO_VERBOSE" \
      --ld-window-cm "$LD_WINDOW_CM" \
      --ld-window-mb "$LD_WINDOW_MB" \
      "${VARIANT_EXTRACT_ARGS[@]}" \
      "${LD_CACHE_ARGS[@]}" \
      "${GENETIC_MAP_ARGS[@]}" \
      --genetic-map-type "$GENETIC_MAP_TYPE" \
      "${ALLOW_MB_FALLBACK_ARGS[@]}" \
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
