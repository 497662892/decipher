#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --job-name=prs_distillation
#SBATCH --output=logs/prs_distillation/%x-%j.log

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TABPFN_MODEL_CACHE_DIR="$PWD/outputs/tabpfn-model-cache"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
run_dir="${RUN_DIR:?Prepare a run with the audit command and set RUN_DIR}"
mode="${MODE:-core}"
common=(--config "${CONFIG:-configs/prs_distillation_v1.yaml}" --eligibility-config configs/disease_eligibility.yaml --run-dir "$run_dir")
if [[ "$mode" == smoke ]]; then
  uv run --offline --no-sync python -u scripts/run_prs_distillation.py smoke "${common[@]}" --trait T2D
else
  test -f "$run_dir/smoke/T2D/VERIFIED.json"
  # Exactly one teacher/GPU task at a time, including across traits.
  for trait in T2D cad bc; do
    uv run --offline --no-sync python -u scripts/run_prs_distillation.py run-core "${common[@]}" --trait "$trait"
    uv run --offline --no-sync python -u scripts/run_prs_distillation.py ablate "${common[@]}" --trait "$trait"
    uv run --offline --no-sync python -u scripts/run_prs_distillation.py report "${common[@]}" --trait "$trait"
  done
  if [[ -n "${QC_REFERENCE_RUN:-}" ]]; then
    uv run --offline --no-sync python -u scripts/run_prs_distillation_teacher_control.py \
      --config "${CONFIG:-configs/prs_distillation_v1.yaml}" --run-dir "$run_dir" \
      --reference-run "$QC_REFERENCE_RUN"
  fi
fi
