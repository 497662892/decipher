#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=220GB
#SBATCH --time=08:00:00
#SBATCH --partition=cpu_medium
#SBATCH --job-name=external_pgs_reference
#SBATCH --output=logs/external_pgs/reference-%J.log

set -eo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then
  module load plink/2.0-20260228
fi
set -u

# Full, unpruned variant matching needs substantially more memory than PLINK --freq.
# Declare the verified target build explicitly when submitting this job.
uv run --no-sync python -u scripts/external_pgs.py prepare-reference \
  --target-pvar "${TARGET_PVAR:-outputs/final_genotype/qc_final/train_val.common.pvar}" \
  --reference-pfile "${REFERENCE_PFILE:-data/resources/1000g/grch38/1000g_GRCh38}" \
  --target-build "${TARGET_BUILD:?Set TARGET_BUILD to the verified genotype build}" \
  --reference-build GRCh38 \
  --reference-provenance '1000 Genomes GRCh38 20190312 biallelic release; Phase 3 panel restriction; actual sample count recorded from reference PSAM; all five superpopulations; external-only frequencies' \
  --threads "${SLURM_CPUS_PER_TASK:-4}" --memory-mb 32000 \
  --out-dir "${OUT_DIR:-outputs/external_pgs/reference_1000g_trainval}" \
  "$@"
