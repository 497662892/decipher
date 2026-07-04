#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=prscs_prepare
#SBATCH --output=logs/prscs_prepare/%x-%J.log

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

PLINK_MODULE="${PLINK_MODULE:-plink/2.0-20260228}"
if command -v module >/dev/null 2>&1; then
  module load "$PLINK_MODULE" || true
fi
if ! command -v plink2 >/dev/null 2>&1 && [[ -n "${PLINK2_DIR:-}" && -x "$PLINK2_DIR/plink2" ]]; then
  export PATH="$PLINK2_DIR:$PATH"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
SUMSTATS_ROOT="${SUMSTATS_ROOT:-data/raw/training_phase/supplementary_files/summary_statistics}"
GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
SCORE_GENOTYPE_PREFIX="${SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"
FIT_SCORE_GENOTYPE_PREFIX="${FIT_SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"
HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/sumstats/harmonized}"
TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/labeled_train.keep.psam}"
SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/unlabelled_validation.keep.psam}"
FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
OUT_DIR="${OUT_DIR:-outputs/prscs_full_train_ukbb_eur}"
REF_DIR="${REF_DIR:-data/resources/prscs/ldblk_ukbb_eur}"
PRSCS_SCRIPT="${PRSCS_SCRIPT:-data/resources/prscs/PRScs/PRScs.py}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
PLINK2="${PLINK2:-plink2}"
PHI_GRID="${PHI_GRID:-auto,1e-4,1e-2}"
CHROMOSOMES="${CHROMOSOMES:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}"
N_ITER="${N_ITER:-1000}"
N_BURNIN="${N_BURNIN:-500}"
THIN="${THIN:-5}"
SUMSTATS_MODE="${SUMSTATS_MODE:-auto}"
CHUNKSIZE="${CHUNKSIZE:-250000}"
N_GWAS="${N_GWAS:-}"
PREPARE_SCRIPT="$OUT_DIR/$DISEASE/commands/$DISEASE.prscs_prepare_inputs.sh"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" "logs/prscs_prepare"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable PYTHON_BIN=$PYTHON_BIN. Set PYTHON_BIN or create the repo .venv." >&2
  exit 1
fi

N_GWAS_ARGS=()
if [[ -n "$N_GWAS" ]]; then
  N_GWAS_ARGS=(--n-gwas "$N_GWAS")
fi
MODULE_LOAD_ARGS=()
if [[ -n "$PLINK_MODULE" ]]; then
  MODULE_LOAD_ARGS=(--module-load "$PLINK_MODULE")
fi

echo "Started PRS-CS input preparation for $DISEASE at $(date)"

"$PYTHON_BIN" -u scripts/write_prscs_full_train_commands.py \
  --disease "$DISEASE" \
  --manifest "$MANIFEST" \
  --sumstats-root "$SUMSTATS_ROOT" \
  --genotype-prefix "$GENOTYPE_PREFIX" \
  --score-genotype-prefix "$SCORE_GENOTYPE_PREFIX" \
  --fit-score-genotype-prefix "$FIT_SCORE_GENOTYPE_PREFIX" \
  --harmonized-dir "$HARMONIZED_DIR" \
  --train-keep "$TRAIN_KEEP" \
  --score-keep "$SCORE_KEEP" \
  --fit-score-keep "$FIT_SCORE_KEEP" \
  --out-dir "$OUT_DIR" \
  --ref-dir "$REF_DIR" \
  --prscs-script "$PRSCS_SCRIPT" \
  --python "$PYTHON_BIN" \
  --plink2 "$PLINK2" \
  "${MODULE_LOAD_ARGS[@]}" \
  "${N_GWAS_ARGS[@]}" \
  --phi-grid "$PHI_GRID" \
  --chromosomes "$CHROMOSOMES" \
  --n-iter "$N_ITER" \
  --n-burnin "$N_BURNIN" \
  --thin "$THIN" \
  --sumstats-mode "$SUMSTATS_MODE" \
  --chunksize "$CHUNKSIZE" \
  --threads "$OMP_NUM_THREADS"

bash "$PREPARE_SCRIPT"

echo "Finished PRS-CS input preparation for $DISEASE at $(date)"
