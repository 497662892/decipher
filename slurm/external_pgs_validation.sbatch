#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH --job-name=external_pgs_validation
#SBATCH --output=logs/external_pgs/validation-%A_%a.log

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export TABPFN_NO_BROWSER=1
export TABPFN_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TABPFN_MODEL_CACHE_DIR="$REPO_DIR/outputs/tabpfn-model-cache"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

diseases=(T2D cad bc)
disease="${diseases[${SLURM_ARRAY_TASK_ID:?Array task is required}]}"
score_dir="${SCORE_DIR:-outputs/external_pgs/run_20261010/scoring_v2}"
run_dir="${RUN_DIR:-outputs/external_pgs/pc_primary/official_validation}"
uv run --no-sync python -u scripts/evaluate_external_pgs_validation.py \
  --weight-lock "$score_dir/weights.scored.lock.json" \
  --scores "$score_dir/scores.long.csv" --scoring-run "$score_dir/scoring_run.json" \
  --train-phenotype data/processed/phenotype.csv \
  --validation-phenotype data/processed/official_validation_phenotype.csv \
  --ancestry-labels outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv \
  --eligibility-config configs/disease_eligibility.yaml \
  --model-path "$TABPFN_MODEL_CACHE_DIR/tabpfn-v3-classifier-v3_default.ckpt" \
  --device cuda --diseases "$disease" --methods "${METHODS:-tabpfn_prs_pc}" --n-bootstraps 2000 \
  --out-dir "$run_dir/$disease" \
  "$@"
