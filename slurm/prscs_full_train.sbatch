#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-66%22
#SBATCH --job-name=prscs_full_train
#SBATCH --output=logs/prscs_full_train/%x-%A_%a.log

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
export REUSE_PRSCS_EFFECTS="${REUSE_PRSCS_EFFECTS:-1}"

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
TASKS_FILE="${TASKS_FILE:-$OUT_DIR/$DISEASE/commands/$DISEASE.prscs_full_train.array_tasks.txt}"
INPUT_PREFIX="$OUT_DIR/$DISEASE/inputs/$DISEASE.prscs"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" "logs/prscs_full_train/$DISEASE"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable PYTHON_BIN=$PYTHON_BIN. Set PYTHON_BIN or create the repo .venv." >&2
  exit 1
fi
"$PYTHON_BIN" -c "import h5py" || {
  echo "Missing Python dependency h5py in PYTHON_BIN=$PYTHON_BIN; run uv sync or install h5py before PRS-CS." >&2
  exit 1
}

test -s "$TASKS_FILE" || {
  echo "Missing PRS-CS task list: $TASKS_FILE" >&2
  echo "Run slurm/prscs_prepare_inputs_${DISEASE}.sbatch before this array job." >&2
  exit 1
}
test -s "$INPUT_PREFIX.sst" || { echo "Missing PRS-CS sumstats: $INPUT_PREFIX.sst" >&2; exit 1; }
test -s "$INPUT_PREFIX.bim" || { echo "Missing PRS-CS BIM: $INPUT_PREFIX.bim" >&2; exit 1; }
test -s "$INPUT_PREFIX.target_id_map.tsv" || {
  echo "Missing PRS-CS target ID map: $INPUT_PREFIX.target_id_map.tsv" >&2
  exit 1
}

TASK_SCRIPT="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")"
if [[ -z "$TASK_SCRIPT" ]]; then
  echo "No task script for array index $SLURM_ARRAY_TASK_ID in $TASKS_FILE" >&2
  exit 1
fi

bash "$TASK_SCRIPT"
