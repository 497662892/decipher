#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=08:00:00
#SBATCH --partition=gl40s_short
#SBATCH --job-name=final_genotype_finalize
#SBATCH --output=logs/final_genotype/final_genotype_finalize-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs/final_genotype outputs/uv-cache outputs/final_genotype/reports
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if command -v module >/dev/null 2>&1; then
  module load plink/2.0-20260228 || module load plink/2.0 || true
fi

if ! command -v plink2 >/dev/null 2>&1 && [[ -n "${PLINK2_DIR:-}" && -x "$PLINK2_DIR/plink2" ]]; then
  export PATH="$PLINK2_DIR:$PATH"
fi

PYTHON="${PYTHON:-.venv/bin/python}"
PLINK2="${PLINK2:-plink2}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
TRAIN_PREFIX="${TRAIN_PREFIX:-outputs/final_genotype/qc_final/train_val.common}"
TEST_PREFIX="${TEST_PREFIX:-outputs/final_genotype/qc_final/holdout_test.common}"
REPORT_DIR="${REPORT_DIR:-outputs/final_genotype/reports}"

test -x "$PYTHON" || { echo "Missing Python executable: $PYTHON" >&2; exit 1; }
for suffix in pgen pvar psam; do
  test -s "${TRAIN_PREFIX}.${suffix}" || { echo "Missing train-val final genotype: ${TRAIN_PREFIX}.${suffix}" >&2; exit 1; }
  test -s "${TEST_PREFIX}.${suffix}" || { echo "Missing holdout-test final genotype: ${TEST_PREFIX}.${suffix}" >&2; exit 1; }
done

echo "Started final genotype finalize at $(date)"
echo "Train-val prefix: $TRAIN_PREFIX"
echo "Holdout-test prefix: $TEST_PREFIX"
echo "PLINK2: $PLINK2"

"$PYTHON" scripts/audit_pvar_compatibility.py \
  --left-pvar "${TRAIN_PREFIX}.pvar" \
  --right-pvar "${TEST_PREFIX}.pvar" \
  --left-name train_val \
  --right-name holdout_test \
  --out "$REPORT_DIR/trainval_vs_holdout_final_pvar_mismatches.csv" \
  --summary-out "$REPORT_DIR/trainval_vs_holdout_final_pvar.summary.json" \
  --fail-on-mismatch

"$PLINK2" \
  --pfile "$TRAIN_PREFIX" \
  --freq \
  --missing \
  --pgen-info \
  --threads "$THREADS" \
  --out "$REPORT_DIR/train_val.common.summary"

"$PLINK2" \
  --pfile "$TEST_PREFIX" \
  --freq \
  --missing \
  --pgen-info \
  --threads "$THREADS" \
  --out "$REPORT_DIR/holdout_test.common.summary"

echo "Final train-val genotype prefix:"
echo "$TRAIN_PREFIX"
echo "Final holdout-test genotype prefix:"
echo "$TEST_PREFIX"
echo "Finished final genotype finalize at $(date)"
