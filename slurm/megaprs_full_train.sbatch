#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --partition=deciphehr
#SBATCH --job-name=megaprs_full
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/megaprs_full_train/%x-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
DISEASE="${DISEASE:-}"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  source /gpfs/home/jm10850/.bashrc
fi

set -u

if [[ -z "$DISEASE" ]]; then
  echo "DISEASE is required; use a disease wrapper or export DISEASE=cad/bc/T2D." >&2
  exit 1
fi

cd "$REPO_DIR"

PLINK_MODULE="${PLINK_MODULE:-plink/2.0-20260228}"
if command -v module >/dev/null 2>&1; then
  module load "$PLINK_MODULE" || true
fi
if ! command -v plink2 >/dev/null 2>&1 && [[ -x /gpfs/share/apps/plink/2.0-20260228/plink2 ]]; then
  export PATH="/gpfs/share/apps/plink/2.0-20260228:$PATH"
fi

LDAK_MODULE="${LDAK_MODULE:-}"
if command -v module >/dev/null 2>&1 && [[ -n "$LDAK_MODULE" ]]; then
  module load "$LDAK_MODULE" || true
fi

MEGAPRS_THREADS="${MEGAPRS_THREADS:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$MEGAPRS_THREADS}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"
export REUSE_MEGAPRS_FREQ="${REUSE_MEGAPRS_FREQ:-1}"
export REUSE_MEGAPRS_REFERENCE="${REUSE_MEGAPRS_REFERENCE:-1}"
export REUSE_MEGAPRS_CORS="${REUSE_MEGAPRS_CORS:-1}"
export REUSE_MEGAPRS_EFFECTS="${REUSE_MEGAPRS_EFFECTS:-1}"

MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/sumstats/harmonized}"
TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/labeled_train.keep.psam}"
SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/unlabelled_validation.keep.psam}"
FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
OUT_DIR="${OUT_DIR:-outputs/megaprs_full_train}"
REFERENCE_DIR="${REFERENCE_DIR:-$OUT_DIR/$DISEASE/reference}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
PLINK2="${PLINK2:-plink2}"
CHROMOSOMES="${CHROMOSOMES:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}"
MEGAPRS_PROFILE="${MEGAPRS_PROFILE:-target}"
MEGAPRS_RESOURCE_DIR="${MEGAPRS_RESOURCE_DIR:-data/resources/megaprs}"
LDAK_BIN="${LDAK_BIN:-$MEGAPRS_RESOURCE_DIR/ldak}"
BERISA_BREAKPOINTS="${BERISA_BREAKPOINTS:-$MEGAPRS_RESOURCE_DIR/berisa.txt}"
BASELINELD_PREFIX="${BASELINELD_PREFIX:-$MEGAPRS_RESOURCE_DIR/BaselineLD/BaselineLD}"
VARIANT_EXTRACT_DEFAULT="${VARIANT_EXTRACT_DEFAULT:-data/processed/ldpred2/hm3plus_target.snplist}"
MEGAPRS_MODEL="${MEGAPRS_MODEL:-bayesr}"
if [[ "$MEGAPRS_PROFILE" == "target" || "$MEGAPRS_PROFILE" == "official" ]]; then
  VARIANT_EXTRACT="${VARIANT_EXTRACT-}"
  BREAK_POINTS="${BREAK_POINTS:-$BERISA_BREAKPOINTS}"
  ANNOTATION_NUMBER="${ANNOTATION_NUMBER:-}"
  ANNOTATION_PREFIX="${ANNOTATION_PREFIX:-}"
  ALLOW_AMBIGUOUS="${ALLOW_AMBIGUOUS:-0}"
  DROP_AMBIGUOUS_SUMSTATS="${DROP_AMBIGUOUS_SUMSTATS:-1}"
elif [[ "$MEGAPRS_PROFILE" == "target_annotated" || "$MEGAPRS_PROFILE" == "annotated" ]]; then
  VARIANT_EXTRACT="${VARIANT_EXTRACT-}"
  BREAK_POINTS="${BREAK_POINTS:-$BERISA_BREAKPOINTS}"
  ANNOTATION_NUMBER="${ANNOTATION_NUMBER:-86}"
  ANNOTATION_PREFIX="${ANNOTATION_PREFIX:-$OUT_DIR/$DISEASE/annotations/BaselineLD_target/BaselineLD}"
  ALLOW_AMBIGUOUS="${ALLOW_AMBIGUOUS:-0}"
  DROP_AMBIGUOUS_SUMSTATS="${DROP_AMBIGUOUS_SUMSTATS:-1}"
elif [[ "$MEGAPRS_PROFILE" == "official_annotated" || "$MEGAPRS_PROFILE" == "rsid_annotated" ]]; then
  VARIANT_EXTRACT="${VARIANT_EXTRACT-}"
  BREAK_POINTS="${BREAK_POINTS:-$BERISA_BREAKPOINTS}"
  ANNOTATION_NUMBER="${ANNOTATION_NUMBER:-86}"
  ANNOTATION_PREFIX="${ANNOTATION_PREFIX:-$BASELINELD_PREFIX}"
  ALLOW_AMBIGUOUS="${ALLOW_AMBIGUOUS:-0}"
  DROP_AMBIGUOUS_SUMSTATS="${DROP_AMBIGUOUS_SUMSTATS:-1}"
else
  VARIANT_EXTRACT="${VARIANT_EXTRACT-$VARIANT_EXTRACT_DEFAULT}"
  BREAK_POINTS="${BREAK_POINTS:-}"
  ANNOTATION_NUMBER="${ANNOTATION_NUMBER:-}"
  ANNOTATION_PREFIX="${ANNOTATION_PREFIX:-}"
  ALLOW_AMBIGUOUS="${ALLOW_AMBIGUOUS:-1}"
  DROP_AMBIGUOUS_SUMSTATS="${DROP_AMBIGUOUS_SUMSTATS:-0}"
fi
MEGAPRS_CHECK_MCMC="${MEGAPRS_CHECK_MCMC:-0}"
MEGAPRS_MCMC_SOLVE="${MEGAPRS_MCMC_SOLVE:-}"
WINDOW_CM="${WINDOW_CM:-1}"
MIN_MAF="${MIN_MAF:-0.005}"
MAX_COR="${MAX_COR:-0.98}"
N_GWAS="${N_GWAS:-}"
CHUNKSIZE="${CHUNKSIZE:-250000}"
WORKFLOW_SCRIPT="${WORKFLOW_SCRIPT:-$OUT_DIR/$DISEASE/commands/$DISEASE.megaprs_full_train.commands.sh}"
MEGAPRS_EXECUTION_MODE="${MEGAPRS_EXECUTION_MODE:-submit_array}"
MEGAPRS_PARTITION="${MEGAPRS_PARTITION:-deciphehr}"
MEGAPRS_PRE_CORS_CPUS="${MEGAPRS_PRE_CORS_CPUS:-16}"
MEGAPRS_PRE_CORS_MEM="${MEGAPRS_PRE_CORS_MEM:-128GB}"
MEGAPRS_PRE_CORS_TIME="${MEGAPRS_PRE_CORS_TIME:-24:00:00}"
MEGAPRS_CORS_CPUS="${MEGAPRS_CORS_CPUS:-16}"
MEGAPRS_CORS_MEM="${MEGAPRS_CORS_MEM:-128GB}"
MEGAPRS_CORS_TIME="${MEGAPRS_CORS_TIME:-24:00:00}"
MEGAPRS_CORS_ARRAY_CONCURRENCY="${MEGAPRS_CORS_ARRAY_CONCURRENCY:-22}"
MEGAPRS_JOIN_CPUS="${MEGAPRS_JOIN_CPUS:-4}"
MEGAPRS_JOIN_MEM="${MEGAPRS_JOIN_MEM:-128GB}"
MEGAPRS_JOIN_TIME="${MEGAPRS_JOIN_TIME:-12:00:00}"
MEGAPRS_RUN_CPUS="${MEGAPRS_RUN_CPUS:-16}"
MEGAPRS_RUN_MEM="${MEGAPRS_RUN_MEM:-256GB}"
MEGAPRS_RUN_TIME="${MEGAPRS_RUN_TIME:-72:00:00}"
MEGAPRS_SCORE_CPUS="${MEGAPRS_SCORE_CPUS:-16}"
MEGAPRS_SCORE_MEM="${MEGAPRS_SCORE_MEM:-64GB}"
MEGAPRS_SCORE_TIME="${MEGAPRS_SCORE_TIME:-12:00:00}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" logs/megaprs_full_train

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable PYTHON_BIN=$PYTHON_BIN. Set PYTHON_BIN or create the repo .venv." >&2
  exit 1
fi
if [[ "$LDAK_BIN" == */* || "$LDAK_BIN" == .* ]]; then
  if [[ ! -x "$LDAK_BIN" ]]; then
    echo "Missing executable LDAK_BIN=$LDAK_BIN. Set LDAK_BIN or load an LDAK module." >&2
    exit 1
  fi
elif ! command -v "$LDAK_BIN" >/dev/null 2>&1; then
  echo "Cannot find LDAK executable '$LDAK_BIN'. Set LDAK_BIN or load an LDAK module." >&2
  exit 1
fi
if [[ "$MEGAPRS_PROFILE" == "target" || "$MEGAPRS_PROFILE" == "official" || "$MEGAPRS_PROFILE" == "target_annotated" || "$MEGAPRS_PROFILE" == "annotated" || "$MEGAPRS_PROFILE" == "official_annotated" || "$MEGAPRS_PROFILE" == "rsid_annotated" ]]; then
  test -s "$BREAK_POINTS" || {
    echo "Missing official MegaPRS Berisa breakpoints: $BREAK_POINTS. Run slurm/download_megaprs_resources.sbatch or set BREAK_POINTS." >&2
    exit 1
  }
fi
if [[ -n "$ANNOTATION_PREFIX" ]]; then
  test -n "$ANNOTATION_NUMBER" || {
    echo "ANNOTATION_NUMBER is required when ANNOTATION_PREFIX is set." >&2
    exit 1
  }
  test -s "${ANNOTATION_PREFIX}1" || {
    echo "Missing official MegaPRS BaselineLD annotation prefix: ${ANNOTATION_PREFIX}1. Run slurm/download_megaprs_resources.sbatch or set ANNOTATION_PREFIX/ANNOTATION_NUMBER." >&2
    exit 1
  }
fi

N_GWAS_ARGS=()
if [[ -n "$N_GWAS" ]]; then
  N_GWAS_ARGS=(--n-gwas "$N_GWAS")
fi
MAX_COR_ARGS=()
if [[ -n "$MAX_COR" ]]; then
  MAX_COR_ARGS=(--max-cor "$MAX_COR")
fi
VARIANT_EXTRACT_ARGS=()
if [[ -n "$VARIANT_EXTRACT" ]]; then
  VARIANT_EXTRACT_ARGS=(--variant-extract "$VARIANT_EXTRACT")
fi
BREAK_POINTS_ARGS=()
if [[ -n "$BREAK_POINTS" ]]; then
  BREAK_POINTS_ARGS=(--break-points "$BREAK_POINTS")
fi
ANNOTATION_ARGS=()
if [[ -n "$ANNOTATION_NUMBER" ]]; then
  ANNOTATION_ARGS+=(--annotation-number "$ANNOTATION_NUMBER")
fi
if [[ -n "$ANNOTATION_PREFIX" ]]; then
  ANNOTATION_ARGS+=(--annotation-prefix "$ANNOTATION_PREFIX")
fi
CHECK_MCMC_ARGS=()
if [[ -n "$MEGAPRS_CHECK_MCMC" ]]; then
  CHECK_MCMC_ARGS=(--check-mcmc "$MEGAPRS_CHECK_MCMC")
fi
MCMC_SOLVE_ARGS=()
if [[ -n "$MEGAPRS_MCMC_SOLVE" ]]; then
  MCMC_SOLVE_ARGS=(--mcmc-solve "$MEGAPRS_MCMC_SOLVE")
fi
LDAK_MODULE_ARGS=()
if [[ -n "$LDAK_MODULE" ]]; then
  LDAK_MODULE_ARGS=(--ldak-module "$LDAK_MODULE")
fi
PLINK_MODULE_ARGS=()
if [[ -n "$PLINK_MODULE" ]]; then
  PLINK_MODULE_ARGS=(--plink-module "$PLINK_MODULE")
fi
DROP_AMBIGUOUS_ARGS=()
if [[ "$DROP_AMBIGUOUS_SUMSTATS" == "1" ]]; then
  DROP_AMBIGUOUS_ARGS=(--drop-ambiguous-sumstats)
fi

echo "Started MegaPRS full-train workflow for $DISEASE at $(date)"

"$PYTHON_BIN" -u scripts/write_megaprs_full_train_commands.py \
  --disease "$DISEASE" \
  --manifest "$MANIFEST" \
  --genotype-prefix "$GENOTYPE_PREFIX" \
  --harmonized-dir "$HARMONIZED_DIR" \
  --train-keep "$TRAIN_KEEP" \
  --score-keep "$SCORE_KEEP" \
  --fit-score-keep "$FIT_SCORE_KEEP" \
  "${VARIANT_EXTRACT_ARGS[@]}" \
  --out-dir "$OUT_DIR" \
  --reference-dir "$REFERENCE_DIR" \
  --ldak "$LDAK_BIN" \
  --python "$PYTHON_BIN" \
  --plink2 "$PLINK2" \
  "${LDAK_MODULE_ARGS[@]}" \
  "${PLINK_MODULE_ARGS[@]}" \
  "${N_GWAS_ARGS[@]}" \
  --chromosomes "$CHROMOSOMES" \
  --model "$MEGAPRS_MODEL" \
  "${BREAK_POINTS_ARGS[@]}" \
  --window-cm "$WINDOW_CM" \
  --min-maf "$MIN_MAF" \
  "${MAX_COR_ARGS[@]}" \
  --allow-ambiguous "$ALLOW_AMBIGUOUS" \
  "${ANNOTATION_ARGS[@]}" \
  "${CHECK_MCMC_ARGS[@]}" \
  "${MCMC_SOLVE_ARGS[@]}" \
  --chunksize "$CHUNKSIZE" \
  --threads "$MEGAPRS_THREADS" \
  "${DROP_AMBIGUOUS_ARGS[@]}"

COMMAND_DIR="$OUT_DIR/$DISEASE/commands"
TASK_DIR="$OUT_DIR/$DISEASE/tasks"
PRE_CORS_SCRIPT="$COMMAND_DIR/$DISEASE.megaprs.pre_cors.sh"
CORS_ARRAY_SCRIPT="$COMMAND_DIR/$DISEASE.megaprs.calc_cors.array.sh"
CORS_TASK_LIST="$COMMAND_DIR/$DISEASE.megaprs.calc_cors.array_tasks.txt"
JOIN_CORS_SCRIPT="$COMMAND_DIR/$DISEASE.megaprs.join_cors.sh"
RUN_SCRIPT="$TASK_DIR/$DISEASE.megaprs.sh"
SCORE_SCRIPT="$COMMAND_DIR/$DISEASE.megaprs.score.sh"

if [[ "$MEGAPRS_EXECUTION_MODE" == "local" ]]; then
  bash "$WORKFLOW_SCRIPT"
  echo "Finished MegaPRS full-train workflow for $DISEASE at $(date)"
  exit 0
fi

if [[ "$MEGAPRS_EXECUTION_MODE" != "submit_array" ]]; then
  echo "Unknown MEGAPRS_EXECUTION_MODE=$MEGAPRS_EXECUTION_MODE; use submit_array or local." >&2
  exit 1
fi

command -v sbatch >/dev/null 2>&1 || {
  echo "Cannot find sbatch; set MEGAPRS_EXECUTION_MODE=local to run sequentially." >&2
  exit 1
}
test -s "$PRE_CORS_SCRIPT" || { echo "Missing pre-cors script: $PRE_CORS_SCRIPT" >&2; exit 1; }
test -s "$CORS_ARRAY_SCRIPT" || { echo "Missing calc-cors array script: $CORS_ARRAY_SCRIPT" >&2; exit 1; }
test -s "$CORS_TASK_LIST" || { echo "Missing calc-cors task list: $CORS_TASK_LIST" >&2; exit 1; }
test -s "$JOIN_CORS_SCRIPT" || { echo "Missing join-cors script: $JOIN_CORS_SCRIPT" >&2; exit 1; }
test -s "$RUN_SCRIPT" || { echo "Missing MegaPRS run script: $RUN_SCRIPT" >&2; exit 1; }
test -s "$SCORE_SCRIPT" || { echo "Missing MegaPRS score script: $SCORE_SCRIPT" >&2; exit 1; }

CORS_TASK_COUNT="$(wc -l < "$CORS_TASK_LIST" | tr -d ' ')"
if [[ "$CORS_TASK_COUNT" -lt 1 ]]; then
  echo "No calc-cors tasks listed in $CORS_TASK_LIST" >&2
  exit 1
fi

LOG_DIR="logs/megaprs_full_train/$DISEASE"
mkdir -p "$LOG_DIR"

SBATCH_BASE=(--parsable --chdir "$REPO_DIR" --partition "$MEGAPRS_PARTITION" --export=ALL)

PRE_JOB_ID="$(
  sbatch "${SBATCH_BASE[@]}" \
    --job-name "megaprs_${DISEASE}_pre" \
    --output "$LOG_DIR/%x-%j.log" \
    --cpus-per-task "$MEGAPRS_PRE_CORS_CPUS" \
    --mem "$MEGAPRS_PRE_CORS_MEM" \
    --time "$MEGAPRS_PRE_CORS_TIME" \
    "$PRE_CORS_SCRIPT"
)"
CORS_JOB_ID="$(
  sbatch "${SBATCH_BASE[@]}" \
    --dependency "afterok:$PRE_JOB_ID" \
    --array "1-${CORS_TASK_COUNT}%${MEGAPRS_CORS_ARRAY_CONCURRENCY}" \
    --job-name "megaprs_${DISEASE}_cors" \
    --output "$LOG_DIR/%x-%A_%a.log" \
    --cpus-per-task "$MEGAPRS_CORS_CPUS" \
    --mem "$MEGAPRS_CORS_MEM" \
    --time "$MEGAPRS_CORS_TIME" \
    "$CORS_ARRAY_SCRIPT"
)"
JOIN_JOB_ID="$(
  sbatch "${SBATCH_BASE[@]}" \
    --dependency "afterok:$CORS_JOB_ID" \
    --job-name "megaprs_${DISEASE}_join" \
    --output "$LOG_DIR/%x-%j.log" \
    --cpus-per-task "$MEGAPRS_JOIN_CPUS" \
    --mem "$MEGAPRS_JOIN_MEM" \
    --time "$MEGAPRS_JOIN_TIME" \
    "$JOIN_CORS_SCRIPT"
)"
RUN_JOB_ID="$(
  sbatch "${SBATCH_BASE[@]}" \
    --dependency "afterok:$JOIN_JOB_ID" \
    --job-name "megaprs_${DISEASE}_run" \
    --output "$LOG_DIR/%x-%j.log" \
    --cpus-per-task "$MEGAPRS_RUN_CPUS" \
    --mem "$MEGAPRS_RUN_MEM" \
    --time "$MEGAPRS_RUN_TIME" \
    "$RUN_SCRIPT"
)"
SCORE_JOB_ID="$(
  sbatch "${SBATCH_BASE[@]}" \
    --dependency "afterok:$RUN_JOB_ID" \
    --job-name "megaprs_${DISEASE}_score" \
    --output "$LOG_DIR/%x-%j.log" \
    --cpus-per-task "$MEGAPRS_SCORE_CPUS" \
    --mem "$MEGAPRS_SCORE_MEM" \
    --time "$MEGAPRS_SCORE_TIME" \
    "$SCORE_SCRIPT"
)"

echo "Submitted MegaPRS full-train workflow for $DISEASE:"
echo "  pre-cors:   $PRE_JOB_ID"
echo "  calc-cors:  $CORS_JOB_ID array=1-${CORS_TASK_COUNT}%${MEGAPRS_CORS_ARRAY_CONCURRENCY}"
echo "  join-cors:  $JOIN_JOB_ID"
echo "  mega-prs:   $RUN_JOB_ID"
echo "  score:      $SCORE_JOB_ID"

echo "Finished MegaPRS full-train submission for $DISEASE at $(date)"
