#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%2
#SBATCH --job-name=ho2022_bc_eval
#SBATCH --output=logs/external_pgs/ho2022-bc-eval-%A_%a.log

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TABPFN_MODEL_CACHE_DIR="$PWD/outputs/tabpfn-model-cache"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
run_dir="${RUN_DIR:-outputs/external_pgs/bc_ho2022_preparation_20261012}"
test -f "$run_dir/ASSEMBLED.json"
panels=(baseline expanded baseline expanded)
scopes=(development development oof oof)
panel="${panels[$SLURM_ARRAY_TASK_ID]}"
scope="${scopes[$SLURM_ARRAY_TASK_ID]}"
bank="$run_dir/banks/$panel"
script=scripts/evaluate_external_pgs_validation.py
extra=()
if [[ "$scope" == oof ]]; then
  script=scripts/evaluate_external_pgs_cv.py
  extra=(--splits "$run_dir/shared_folds.csv")
fi
uv run --no-sync python -u "$script" \
  --weight-lock "$bank/weights.scored.lock.json" \
  --scores "$bank/scores.long.csv" --scoring-run "$bank/scoring_run.json" \
  --ancestry-labels outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv \
  --eligibility-config configs/disease_eligibility.yaml \
  --diseases bc --methods multi_prs_pc_lr,tabpfn_prs_pc \
  --model-path "$TABPFN_MODEL_CACHE_DIR/tabpfn-v3-classifier-v3_default.ckpt" \
  --device cuda --seed 42 --n-bootstraps 2000 \
  --out-dir "$run_dir/evaluation/$scope/$panel" "${extra[@]}"
