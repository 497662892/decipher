#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=72:00:00
#SBATCH --partition=cpu_long
#SBATCH --job-name=sbayesrc_full
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/sbayesrc_full_train/%x-%J.log

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

GCTB_MODULE="${GCTB_MODULE:-}"
if command -v module >/dev/null 2>&1 && [[ -n "$GCTB_MODULE" ]]; then
  module load "$GCTB_MODULE" || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"
export REUSE_SBAYESRC_FREQ="${REUSE_SBAYESRC_FREQ:-1}"
export REUSE_SBAYESRC_IMPUTED="${REUSE_SBAYESRC_IMPUTED:-1}"
export REUSE_SBAYESRC_EFFECTS="${REUSE_SBAYESRC_EFFECTS:-1}"

MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
SUMSTATS_ROOT="${SUMSTATS_ROOT:-/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/supplementary_files/summary_statistics}"
GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/sumstats/harmonized}"
TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/labeled_train.keep.psam}"
SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/unlabelled_validation.keep.psam}"
FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
VARIANT_EXTRACT_DEFAULT="${VARIANT_EXTRACT_DEFAULT:-data/processed/ldpred2/hm3plus_target.snplist}"
VARIANT_EXTRACT="${VARIANT_EXTRACT-$VARIANT_EXTRACT_DEFAULT}"
OUT_DIR="${OUT_DIR:-outputs/sbayesrc_full_train}"
LDM_EIGEN_DIR="${LDM_EIGEN_DIR:-}"
ANNOT_FILE="${ANNOT_FILE:-}"
GCTB_BIN="${GCTB_BIN:-gctb}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
PLINK2="${PLINK2:-plink2}"
CHROMOSOMES="${CHROMOSOMES:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}"
CHAIN_LENGTH="${CHAIN_LENGTH:-3000}"
BURN_IN="${BURN_IN:-1000}"
THIN="${THIN:-10}"
SEED="${SEED:-42}"
LDM_EIGEN_CUTOFF="${LDM_EIGEN_CUTOFF:-}"
CHUNKSIZE="${CHUNKSIZE:-250000}"
N_GWAS="${N_GWAS:-}"
NO_MCMC_BIN="${NO_MCMC_BIN:-1}"
WORKFLOW_SCRIPT="${WORKFLOW_SCRIPT:-$OUT_DIR/$DISEASE/commands/$DISEASE.sbayesrc_full_train.commands.sh}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" logs/sbayesrc_full_train

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable PYTHON_BIN=$PYTHON_BIN. Set PYTHON_BIN or create the repo .venv." >&2
  exit 1
fi
if [[ -z "$LDM_EIGEN_DIR" ]]; then
  echo "LDM_EIGEN_DIR is required for sBayesRC." >&2
  exit 1
fi
if [[ -z "$ANNOT_FILE" ]]; then
  echo "ANNOT_FILE is required for sBayesRC." >&2
  exit 1
fi

N_GWAS_ARGS=()
if [[ -n "$N_GWAS" ]]; then
  N_GWAS_ARGS=(--n-gwas "$N_GWAS")
fi
VARIANT_EXTRACT_ARGS=()
if [[ -n "$VARIANT_EXTRACT" ]]; then
  VARIANT_EXTRACT_ARGS=(--variant-extract "$VARIANT_EXTRACT")
fi
GCTB_MODULE_ARGS=()
if [[ -n "$GCTB_MODULE" ]]; then
  GCTB_MODULE_ARGS=(--gctb-module "$GCTB_MODULE")
fi
PLINK_MODULE_ARGS=()
if [[ -n "$PLINK_MODULE" ]]; then
  PLINK_MODULE_ARGS=(--plink-module "$PLINK_MODULE")
fi
NO_MCMC_BIN_ARGS=()
if [[ "$NO_MCMC_BIN" == "1" ]]; then
  NO_MCMC_BIN_ARGS=(--no-mcmc-bin)
fi
LDM_EIGEN_CUTOFF_ARGS=()
if [[ -n "$LDM_EIGEN_CUTOFF" ]]; then
  LDM_EIGEN_CUTOFF_ARGS=(--ldm-eigen-cutoff "$LDM_EIGEN_CUTOFF")
fi

echo "Started sBayesRC full-train workflow for $DISEASE at $(date)"

"$PYTHON_BIN" -u scripts/write_sbayesrc_full_train_commands.py \
  --disease "$DISEASE" \
  --manifest "$MANIFEST" \
  --sumstats-root "$SUMSTATS_ROOT" \
  --genotype-prefix "$GENOTYPE_PREFIX" \
  --harmonized-dir "$HARMONIZED_DIR" \
  --train-keep "$TRAIN_KEEP" \
  --score-keep "$SCORE_KEEP" \
  --fit-score-keep "$FIT_SCORE_KEEP" \
  "${VARIANT_EXTRACT_ARGS[@]}" \
  --out-dir "$OUT_DIR" \
  --ldm-eigen-dir "$LDM_EIGEN_DIR" \
  --annot-file "$ANNOT_FILE" \
  --gctb "$GCTB_BIN" \
  --python "$PYTHON_BIN" \
  --plink2 "$PLINK2" \
  "${GCTB_MODULE_ARGS[@]}" \
  "${PLINK_MODULE_ARGS[@]}" \
  "${N_GWAS_ARGS[@]}" \
  --chromosomes "$CHROMOSOMES" \
  --chain-length "$CHAIN_LENGTH" \
  --burn-in "$BURN_IN" \
  --thin "$THIN" \
  --seed "$SEED" \
  "${LDM_EIGEN_CUTOFF_ARGS[@]}" \
  --chunksize "$CHUNKSIZE" \
  --threads "$OMP_NUM_THREADS" \
  "${NO_MCMC_BIN_ARGS[@]}"

bash "$WORKFLOW_SCRIPT"

echo "Finished sBayesRC full-train workflow for $DISEASE at $(date)"
