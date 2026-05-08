#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=00-12:00:00
#SBATCH --partition=deciphehr
#SBATCH --array=1-20%4
#SBATCH --job-name=pt_cv_bc
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/pt_cv/bc/pt_cv_bc-%A_%a.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
DISEASE="bc"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  source /gpfs/home/jm10850/.bashrc
fi

set -u

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
