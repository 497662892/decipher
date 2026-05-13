#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --job-name=sdpr_prepare
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/sdpr_prepare/%x-%J.log

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
SDPR_MODULE="${SDPR_MODULE:-}"
if command -v module >/dev/null 2>&1; then
  module load "$PLINK_MODULE" || true
  if [[ -n "$SDPR_MODULE" ]]; then
    module load "$SDPR_MODULE" || true
  fi
fi
if ! command -v plink2 >/dev/null 2>&1 && [[ -x /gpfs/share/apps/plink/2.0-20260228/plink2 ]]; then
  export PATH="/gpfs/share/apps/plink/2.0-20260228:$PATH"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
SUMSTATS_ROOT="${SUMSTATS_ROOT:-/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/supplementary_files/summary_statistics}"
GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/sumstats/harmonized}"
TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/labeled_train.keep.psam}"
SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/unlabelled_validation.keep.psam}"
FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
OUT_DIR="${OUT_DIR:-outputs/sdpr_full_train}"
VARIANT_EXTRACT="${VARIANT_EXTRACT:-data/processed/ldpred2/hm3plus_target.snplist}"
SDPR_BIN="${SDPR_BIN:-data/resources/sdpr/bin/SDPR}"
REF_MODE="${REF_MODE:-target}"
PREBUILT_REF_DIR="${PREBUILT_REF_DIR:-data/resources/sdpr/ref_1kg_eur_mhc_included}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
PLINK2="${PLINK2:-plink2}"
CHROMOSOMES="${CHROMOSOMES:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}"
CHUNKSIZE="${CHUNKSIZE:-250000}"
N_GWAS="${N_GWAS:-}"
DEFAULT_N="${DEFAULT_N:-}"
M_COMPONENTS="${M_COMPONENTS:-1000}"
N_ITER="${N_ITER:-1000}"
N_BURN="${N_BURN:-200}"
THIN="${THIN:-5}"
R2="${R2:-0.1}"
A="${A:-0.1}"
C="${C:-1}"
A0K="${A0K:-0.5}"
B0K="${B0K:-0.5}"
OPT_LLK="${OPT_LLK:-1}"
PREPARE_SCRIPT="$OUT_DIR/$DISEASE/commands/$DISEASE.sdpr_prepare_reference.sh"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" logs/sdpr_prepare

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable PYTHON_BIN=$PYTHON_BIN. Set PYTHON_BIN or create the repo .venv." >&2
  exit 1
fi

if [[ "$SDPR_BIN" == */* || "$SDPR_BIN" == .* ]]; then
  test -x "$SDPR_BIN" || {
    echo "Missing executable SDPR_BIN=$SDPR_BIN. Install official SDPR or export SDPR_BIN." >&2
    exit 1
  }
else
  command -v "$SDPR_BIN" >/dev/null 2>&1 || {
    echo "Cannot find SDPR_BIN=$SDPR_BIN on PATH. Install official SDPR or export SDPR_BIN." >&2
    exit 1
  }
fi

N_GWAS_ARGS=()
if [[ -n "$N_GWAS" ]]; then
  N_GWAS_ARGS=(--n-gwas "$N_GWAS")
fi
DEFAULT_N_ARGS=()
if [[ -n "$DEFAULT_N" ]]; then
  DEFAULT_N_ARGS=(--default-n "$DEFAULT_N")
fi
VARIANT_EXTRACT_ARGS=()
if [[ -n "$VARIANT_EXTRACT" ]]; then
  VARIANT_EXTRACT_ARGS=(--variant-extract "$VARIANT_EXTRACT")
fi
REF_MODE_ARGS=(--ref-mode "$REF_MODE")
if [[ "$REF_MODE" == "prebuilt" ]]; then
  REF_MODE_ARGS+=(--prebuilt-ref-dir "$PREBUILT_REF_DIR")
fi

echo "Started SDPR reference preparation for $DISEASE at $(date)"

"$PYTHON_BIN" -u scripts/write_sdpr_full_train_commands.py \
  --disease "$DISEASE" \
  --manifest "$MANIFEST" \
  --sumstats-root "$SUMSTATS_ROOT" \
  --genotype-prefix "$GENOTYPE_PREFIX" \
  --harmonized-dir "$HARMONIZED_DIR" \
  --train-keep "$TRAIN_KEEP" \
  --score-keep "$SCORE_KEEP" \
  --fit-score-keep "$FIT_SCORE_KEEP" \
  --out-dir "$OUT_DIR" \
  "${VARIANT_EXTRACT_ARGS[@]}" \
  --sdpr-bin "$SDPR_BIN" \
  "${REF_MODE_ARGS[@]}" \
  --python "$PYTHON_BIN" \
  --plink2 "$PLINK2" \
  "${N_GWAS_ARGS[@]}" \
  "${DEFAULT_N_ARGS[@]}" \
  --chromosomes "$CHROMOSOMES" \
  --chunksize "$CHUNKSIZE" \
  --m-components "$M_COMPONENTS" \
  --n-iter "$N_ITER" \
  --n-burn "$N_BURN" \
  --thin "$THIN" \
  --r2 "$R2" \
  --a "$A" \
  --c "$C" \
  --a0k "$A0K" \
  --b0k "$B0K" \
  --opt-llk "$OPT_LLK" \
  --threads "$OMP_NUM_THREADS"

bash "$PREPARE_SCRIPT"

echo "Finished SDPR reference preparation for $DISEASE at $(date)"
