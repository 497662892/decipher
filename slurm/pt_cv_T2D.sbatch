#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --partition=gl40s_short
#SBATCH --array=1-20%4
#SBATCH --job-name=pt_cv_T2D
#SBATCH --output=logs/pt_cv/T2D/pt_cv_T2D-%A_%a.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DISEASE="T2D"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module load plink/2.0-20260228 || true
fi
if ! command -v plink2 >/dev/null 2>&1 && [[ -n "${PLINK2_DIR:-}" && -x "$PLINK2_DIR/plink2" ]]; then
  export PATH="$PLINK2_DIR:$PATH"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"
export PLINK_MEMORY_MB="${PLINK_MEMORY_MB:-90000}"

MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
SPLITS="${SPLITS:-data/processed/splits/disease_specific_split_seed42.csv}"
GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/sumstats/harmonized}"
OUT_DIR="${OUT_DIR:-outputs/pt_cv}"
PLINK2="${PLINK2:-plink2}"
P_THRESHOLDS="${P_THRESHOLDS:-1,0.5,0.05,0.0005,0.000005,0.00000005}"
R2_THRESHOLDS="${R2_THRESHOLDS:-0.2,0.4,0.6,0.8}"
WINDOW_KB="${WINDOW_KB:-250}"
TASKS_FILE="${TASKS_FILE:-$OUT_DIR/$DISEASE/commands/$DISEASE.pt_cv.array_tasks.txt}"
LOCK_FILE="$OUT_DIR/$DISEASE/commands/.generate.lock"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" "logs/pt_cv/$DISEASE"

(
  flock 9
  regenerate_tasks=0
  if [[ ! -s "$TASKS_FILE" ]]; then
    regenerate_tasks=1
  else
    FIRST_TASK_SCRIPT="$(sed -n '1p' "$TASKS_FILE")"
    if [[ -z "$FIRST_TASK_SCRIPT" || ! -s "$FIRST_TASK_SCRIPT" ]]; then
      regenerate_tasks=1
    elif ! grep -q -- "--score-col-nums" "$FIRST_TASK_SCRIPT"; then
      echo "Regenerating stale P+T task scripts missing --score-col-nums"
      regenerate_tasks=1
    elif ! grep -q -- "PLINK_MEMORY_ARGS" "$FIRST_TASK_SCRIPT"; then
      echo "Regenerating stale P+T task scripts missing PLINK memory controls"
      regenerate_tasks=1
    fi
  fi

  if [[ "$regenerate_tasks" == "1" ]]; then
    uv run --extra dev python -u scripts/write_pt_cv_commands.py \
      --disease "$DISEASE" \
      --manifest "$MANIFEST" \
      --splits "$SPLITS" \
      --genotype-prefix "$GENOTYPE_PREFIX" \
      --harmonized-dir "$HARMONIZED_DIR" \
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
