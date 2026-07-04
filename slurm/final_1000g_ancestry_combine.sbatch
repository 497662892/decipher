#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu4_dev,gl40s_dev
#SBATCH --job-name=final_1000g_ancestry_combine
#SBATCH --output=logs/final_ancestry_1000g/combine-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs/final_ancestry_1000g outputs/uv-cache
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-.venv/bin/python}"
TRAIN_ANCESTRY_DIR="${TRAIN_ANCESTRY_DIR:-outputs/final_ancestry_1000g_trainval/ancestry}"
TEST_ANCESTRY_DIR="${TEST_ANCESTRY_DIR:-outputs/final_ancestry_1000g_holdout/ancestry}"
PREFIX="${PREFIX:-target_1000G}"
N_PCS="${N_PCS:-10}"
HARD_MARGIN="${HARD_MARGIN:-0.5}"
SOFT_TAU="${SOFT_TAU:-1.0}"
TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/train_val.keep.psam}"
TEST_KEEP="${TEST_KEEP:-data/processed/sample_sets/holdout_test.keep.psam}"
OUT_DIR="${OUT_DIR:-outputs/final_ancestry_1000g/combined}"

test -x "$PYTHON" || { echo "Missing Python executable: $PYTHON" >&2; exit 1; }
test -s "$TRAIN_KEEP" || { echo "Missing train-val keep file: $TRAIN_KEEP" >&2; exit 1; }
test -s "$TEST_KEEP" || { echo "Missing holdout-test keep file: $TEST_KEEP" >&2; exit 1; }

"$PYTHON" scripts/combine_1000g_ancestry_outputs.py \
  --train-ancestry-dir "$TRAIN_ANCESTRY_DIR" \
  --test-ancestry-dir "$TEST_ANCESTRY_DIR" \
  --prefix "$PREFIX" \
  --n-pcs "$N_PCS" \
  --hard-margin "$HARD_MARGIN" \
  --soft-tau "$SOFT_TAU" \
  --train-keep "$TRAIN_KEEP" \
  --test-keep "$TEST_KEEP" \
  --out-dir "$OUT_DIR"
