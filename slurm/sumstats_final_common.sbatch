#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=08:00:00
#SBATCH --partition=gl40s_short
#SBATCH --array=1-3%3
#SBATCH --job-name=sumstats_final_common
#SBATCH --output=logs/sumstats_final_common/%x-%A_%a.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

DISEASES=(cad bc T2D)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-1}"
DISEASE="${DISEASE:-${DISEASES[$((ARRAY_INDEX - 1))]}}"

cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
SUMSTATS_ROOT="${SUMSTATS_ROOT:-data/raw/training_phase/supplementary_files/summary_statistics}"
TARGET_PVAR="${TARGET_PVAR:-outputs/final_genotype/qc_final/train_val.common.pvar}"
OUT_DIR="${OUT_DIR:-data/processed/final/sumstats}"
STANDARDIZED_SOURCE_DIR="${STANDARDIZED_SOURCE_DIR:-data/processed/sumstats/standardized}"
CHUNKSIZE="${CHUNKSIZE:-250000}"
TARGET_PVAR_CHUNKSIZE="${TARGET_PVAR_CHUNKSIZE:-1000000}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR" "logs/sumstats_final_common"

echo "Started final-common sumstats prep for $DISEASE at $(date)"
echo "Repo: $REPO_DIR"
echo "Manifest: $MANIFEST"
echo "Summary statistics root: $SUMSTATS_ROOT"
echo "Target pvar: $TARGET_PVAR"
echo "Output dir: $OUT_DIR"
echo "Standardized source dir: $STANDARDIZED_SOURCE_DIR"

uv run --extra dev python scripts/prepare_sumstats_common.py \
  --disease "$DISEASE" \
  --manifest "$MANIFEST" \
  --sumstats-root "$SUMSTATS_ROOT" \
  --target-pvar "$TARGET_PVAR" \
  --out-dir "$OUT_DIR" \
  --standardized-source-dir "$STANDARDIZED_SOURCE_DIR" \
  --chunksize "$CHUNKSIZE" \
  --target-pvar-chunksize "$TARGET_PVAR_CHUNKSIZE"

echo "Finished final-common sumstats prep for $DISEASE at $(date)"
