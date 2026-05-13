#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --partition=deciphehr
#SBATCH --array=1-4%4
#SBATCH --job-name=pt_full_train
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/pt_full_train/%x-%A_%a.log

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

if command -v module >/dev/null 2>&1; then
  module load plink/2.0-20260228 || true
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
GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/sumstats/harmonized}"
TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/labeled_train.keep.psam}"
SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/unlabelled_validation.keep.psam}"
FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
OUT_DIR="${OUT_DIR:-outputs/pt_full_train}"
PLINK2="${PLINK2:-plink2}"
P_THRESHOLDS="${P_THRESHOLDS:-1,0.5,0.05,0.0005,0.000005,0.00000005}"
R2_THRESHOLDS="${R2_THRESHOLDS:-0.2,0.4,0.6,0.8}"
WINDOW_KB="${WINDOW_KB:-250}"
TASKS_FILE="${TASKS_FILE:-$OUT_DIR/$DISEASE/commands/$DISEASE.pt_full_train.array_tasks.txt}"
LOCK_FILE="$OUT_DIR/$DISEASE/commands/.generate.lock"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" "logs/pt_full_train"

(
  flock 9
  if [[ ! -s "$TASKS_FILE" ]]; then
    uv run --extra dev python -u scripts/write_pt_full_train_commands.py \
      --disease "$DISEASE" \
      --manifest "$MANIFEST" \
      --genotype-prefix "$GENOTYPE_PREFIX" \
      --harmonized-dir "$HARMONIZED_DIR" \
      --train-keep "$TRAIN_KEEP" \
      --score-keep "$SCORE_KEEP" \
      --fit-score-keep "$FIT_SCORE_KEEP" \
      --out-dir "$OUT_DIR" \
      --plink2 "$PLINK2" \
      --p-thresholds "$P_THRESHOLDS" \
      --r2-thresholds "$R2_THRESHOLDS" \
      --window-kb "$WINDOW_KB" \
      --threads "$OMP_NUM_THREADS"
  fi
) 9>"$LOCK_FILE"

TASK_SCRIPT="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")"
if [[ -z "$TASK_SCRIPT" ]]; then
  echo "No task script for array index $SLURM_ARRAY_TASK_ID in $TASKS_FILE" >&2
  exit 1
fi

bash "$TASK_SCRIPT"
