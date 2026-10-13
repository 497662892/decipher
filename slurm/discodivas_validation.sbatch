#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH --job-name=discodivas-validation
#SBATCH --output=logs/discodivas/validation-%A_%a.log

set -euo pipefail
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
module load r/4.4.2
export R_LIBS_USER="$PWD/outputs/external_pgs/discodivas/R-library"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1 TABPFN_MODEL_CACHE_DIR="$PWD/outputs/tabpfn-model-cache"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
diseases=(T2D cad bc)
uv run --no-sync python -u scripts/evaluate_discodivas_validation.py \
  --disease "${diseases[${SLURM_ARRAY_TASK_ID:?Array task is required}]}" \
  --scoring-dir outputs/external_pgs/run_20261010/scoring_v2 \
  --projection-dir "${PROJECTION_DIR:-outputs/external_pgs/discodivas/projection_1000g_refit_v2_20261010}" \
  --resource-dir outputs/external_pgs/discodivas_resource_audit_20261010 \
  --ancestry-labels outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv \
  --eligibility-config configs/disease_eligibility.yaml \
  --model-path "$TABPFN_MODEL_CACHE_DIR/tabpfn-v3-classifier-v3_default.ckpt" \
  --out-dir "${RUN_DIR:-outputs/external_pgs/discodivas/validation_20261010}/${diseases[$SLURM_ARRAY_TASK_ID]}" \
  "$@"
