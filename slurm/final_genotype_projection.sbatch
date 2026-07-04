#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220GB
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_medium
#SBATCH --job-name=final_genotype_projection
#SBATCH --output=logs/final_genotype/final_genotype-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs/final_genotype outputs/final_genotype outputs/uv-cache
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

if [[ -z "${TRAIN_PREFIX_PATTERN+x}" ]]; then
  TRAIN_PREFIX_PATTERN='outputs/genotype_qc/qc_final/chr{chr}.qc'
fi
if [[ -z "${HOLDOUT_PREFIX_PATTERN+x}" ]]; then
  HOLDOUT_PREFIX_PATTERN='data/raw/holdout_phase/genotype_files/plink_files/Holdout_chr{chr}'
fi
TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/train_val.keep.psam}"
TEST_KEEP="${TEST_KEEP:-data/processed/sample_sets/holdout_test.keep.psam}"
OUT_DIR="${OUT_DIR:-outputs/final_genotype}"
COMMANDS_OUT="${COMMANDS_OUT:-$OUT_DIR/final_genotype_projection.commands.sh}"
TRAIN_FINAL_PREFIX_NAME="${TRAIN_FINAL_PREFIX_NAME:-train_val.common}"
TEST_FINAL_PREFIX_NAME="${TEST_FINAL_PREFIX_NAME:-holdout_test.common}"
CHROMOSOMES="${CHROMOSOMES:-1-22}"

test -x "$PYTHON" || { echo "Missing Python executable: $PYTHON" >&2; exit 1; }
test -s "$TRAIN_KEEP" || { echo "Missing train-val keep file: $TRAIN_KEEP" >&2; exit 1; }
test -s "$TEST_KEEP" || { echo "Missing holdout-test keep file: $TEST_KEEP" >&2; exit 1; }

echo "Started final genotype projection at $(date)"
echo "Train prefix pattern: $TRAIN_PREFIX_PATTERN"
echo "Holdout prefix pattern: $HOLDOUT_PREFIX_PATTERN"
echo "Train keep: $TRAIN_KEEP"
echo "Test keep: $TEST_KEEP"
echo "Output dir: $OUT_DIR"
echo "Final train-val prefix: $OUT_DIR/qc_final/$TRAIN_FINAL_PREFIX_NAME"
echo "Final holdout-test prefix: $OUT_DIR/qc_final/$TEST_FINAL_PREFIX_NAME"
echo "PLINK2: $PLINK2"
echo "Threads: $THREADS"

"$PYTHON" scripts/write_final_genotype_projection_commands.py \
  --train-prefix-pattern "$TRAIN_PREFIX_PATTERN" \
  --holdout-prefix-pattern "$HOLDOUT_PREFIX_PATTERN" \
  --train-keep "$TRAIN_KEEP" \
  --test-keep "$TEST_KEEP" \
  --out-dir "$OUT_DIR" \
  --train-final-prefix-name "$TRAIN_FINAL_PREFIX_NAME" \
  --test-final-prefix-name "$TEST_FINAL_PREFIX_NAME" \
  --chromosomes "$CHROMOSOMES" \
  --plink2 "$PLINK2" \
  --python "$PYTHON" \
  --threads "$THREADS" \
  --out "$COMMANDS_OUT"

bash "$COMMANDS_OUT"

echo "Finished final genotype projection at $(date)"
