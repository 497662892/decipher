#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=24:00:00
#SBATCH --partition=gl40s_short
#SBATCH --job-name=final_trainval_test_pfile
#SBATCH --output=logs/final_trainval_test_pfile/%x-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u
cd "$REPO_DIR"

PLINK_MODULE="${PLINK_MODULE:-plink/2.0-20260228}"
if command -v module >/dev/null 2>&1; then
  module load "$PLINK_MODULE" || true
fi
if ! command -v plink2 >/dev/null 2>&1 && [[ -n "${PLINK2_DIR:-}" && -x "$PLINK2_DIR/plink2" ]]; then
  export PATH="$PLINK2_DIR:$PATH"
fi

TRAIN_PREFIX="${TRAIN_PREFIX:-outputs/final_genotype/qc_final/train_val.common}"
TEST_PREFIX="${TEST_PREFIX:-outputs/final_genotype/qc_final/holdout_test.common}"
OUT_PREFIX="${OUT_PREFIX:-outputs/final_genotype/qc_final/trainval_test.common}"
MERGE_LIST="${MERGE_LIST:-outputs/final_genotype/qc_final/trainval_test_common.pmerge_list.txt}"
PLINK_MEMORY_MB="${PLINK_MEMORY_MB:-170000}"

mkdir -p "$(dirname "$MERGE_LIST")" logs/final_trainval_test_pfile

test -s "$TRAIN_PREFIX.pgen" || { echo "Missing train pgen: $TRAIN_PREFIX.pgen" >&2; exit 1; }
test -s "$TEST_PREFIX.pgen" || { echo "Missing test pgen: $TEST_PREFIX.pgen" >&2; exit 1; }

printf '%s\n%s\n' "$TRAIN_PREFIX" "$TEST_PREFIX" > "$MERGE_LIST"

plink2 \
  --pmerge-list "$MERGE_LIST" \
  --make-pgen \
  --sort-vars \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --memory "$PLINK_MEMORY_MB" \
  --out "$OUT_PREFIX"

echo "Wrote merged train-val plus holdout-test pfile prefix: $OUT_PREFIX"
