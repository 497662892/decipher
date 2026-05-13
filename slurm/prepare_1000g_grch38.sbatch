#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_medium
#SBATCH --job-name=prepare_1000g_grch38
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/1000g_grch38/prepare-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
cd "$REPO_DIR"

mkdir -p logs/1000g_grch38 data/resources/1000g/grch38 outputs/uv-cache
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache}"

if command -v module >/dev/null 2>&1; then
  module load plink/2.0-20260228 || module load plink/2.0 || true
fi

if ! command -v plink2 >/dev/null 2>&1 && [[ -x /gpfs/share/apps/plink/2.0-20260228/plink2 ]]; then
  export PATH="/gpfs/share/apps/plink/2.0-20260228:$PATH"
fi

PLINK2="${PLINK2:-plink2}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
OUT_DIR="${OUT_DIR:-data/resources/1000g/grch38}"

uv run --extra dev python -u scripts/prepare_1000g_grch38.py \
  --out-dir "$OUT_DIR" \
  --threads "$THREADS" \
  --plink2 "$PLINK2" \
  "$@"
