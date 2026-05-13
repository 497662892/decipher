#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=08:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=sumstats_common_cad
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/sumstats_common/cad/sumstats_common_cad-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
DISEASE="cad"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  source /gpfs/home/jm10850/.bashrc
fi

set -u

cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
SUMSTATS_ROOT="${SUMSTATS_ROOT:-/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/supplementary_files/summary_statistics}"
TARGET_PVAR="${TARGET_PVAR:-outputs/genotype_qc/qc_final/all_chromosomes.qc.pvar}"
OUT_DIR="${OUT_DIR:-data/processed/sumstats}"
CHUNKSIZE="${CHUNKSIZE:-250000}"
TARGET_PVAR_CHUNKSIZE="${TARGET_PVAR_CHUNKSIZE:-1000000}"
REUSE_STANDARDIZED="${REUSE_STANDARDIZED:-0}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR" "logs/sumstats_common/$DISEASE"

echo "Started common sumstats prep for $DISEASE at $(date)"
echo "Repo: $REPO_DIR"
echo "Manifest: $MANIFEST"
echo "Summary statistics root: $SUMSTATS_ROOT"
echo "Target pvar: $TARGET_PVAR"
echo "Output dir: $OUT_DIR"

args=(
  uv run --extra dev python scripts/prepare_sumstats_common.py
  --disease "$DISEASE"
  --manifest "$MANIFEST"
  --sumstats-root "$SUMSTATS_ROOT"
  --target-pvar "$TARGET_PVAR"
  --out-dir "$OUT_DIR"
  --chunksize "$CHUNKSIZE"
  --target-pvar-chunksize "$TARGET_PVAR_CHUNKSIZE"
)
if [[ "$REUSE_STANDARDIZED" == "1" ]]; then
  args+=(--reuse-standardized)
fi
"${args[@]}"

echo "Finished common sumstats prep for $DISEASE at $(date)"
