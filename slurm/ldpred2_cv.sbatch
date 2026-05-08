#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=03-00:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-5%1
#SBATCH --job-name=ldpred2_cv
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/ldpred2_cv/ldpred2_cv-%A_%a.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
DISEASE="${DISEASE:-}"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  source /gpfs/home/jm10850/.bashrc
fi

set -u

if [[ -z "$DISEASE" ]]; then
  echo "DISEASE is required; use slurm/ldpred2_cv_cad.sbatch, slurm/ldpred2_cv_bc.sbatch, or slurm/ldpred2_cv_T2D.sbatch." >&2
  exit 1
fi

cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module load plink/2.0-20260228 || true
  module load R || true
fi
if ! command -v plink2 >/dev/null 2>&1 && [[ -x /gpfs/share/apps/plink/2.0-20260228/plink2 ]]; then
  export PATH="/gpfs/share/apps/plink/2.0-20260228:$PATH"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
SPLITS="${SPLITS:-data/processed/splits/disease_specific_split_seed42.csv}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/sumstats/harmonized}"
OUT_DIR="${OUT_DIR:-outputs/ldpred2_cv}"
PLINK2="${PLINK2:-plink2}"
R_SCRIPT="${R_SCRIPT:-Rscript}"
LDPRED2_SCRIPT="${LDPRED2_SCRIPT:-scripts/run_ldpred2_task.R}"
H2_GRID="${H2_GRID:-0.05,0.1,0.2,0.4,0.8,1}"
P_GRID="${P_GRID:-0.0001,0.001,0.01,0.1,1}"
SPARSE_GRID="${SPARSE_GRID:-0,1}"
LD_WINDOW_CM="${LD_WINDOW_CM:-3}"
LD_WINDOW_MB="${LD_WINDOW_MB:-3}"
CHROMOSOMES="${CHROMOSOMES:-1-22}"
MATCH_BY="${MATCH_BY:-id}"
DEFAULT_N="${DEFAULT_N:-}"
SAVE_BETA_GRID="${SAVE_BETA_GRID:-0}"
TASKS_FILE="${TASKS_FILE:-$OUT_DIR/$DISEASE/commands/$DISEASE.ldpred2_cv.array_tasks.txt}"
LOCK_FILE="$OUT_DIR/$DISEASE/commands/.generate.lock"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR/$DISEASE/commands" "logs/ldpred2_cv/$DISEASE"

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

(
  flock 9
  if [[ ! -s "$TASKS_FILE" ]]; then
    uv run --extra dev python -u scripts/write_ldpred2_cv_commands.py \
      --disease "$DISEASE" \
      --manifest "$MANIFEST" \
      --splits "$SPLITS" \
      --patient-col "$PATIENT_COL" \
      --genotype-prefix "$GENOTYPE_PREFIX" \
      --harmonized-dir "$HARMONIZED_DIR" \
      --out-dir "$OUT_DIR" \
      --plink2 "$PLINK2" \
      --rscript "$R_SCRIPT" \
      --ldpred2-script "$LDPRED2_SCRIPT" \
      --h2-grid "$H2_GRID" \
      --p-grid "$P_GRID" \
      --sparse-grid "$SPARSE_GRID" \
      --ld-window-cm "$LD_WINDOW_CM" \
      --ld-window-mb "$LD_WINDOW_MB" \
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
