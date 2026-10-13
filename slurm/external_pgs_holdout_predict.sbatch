#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --partition=gl40s_short,a100_short
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%2
#SBATCH --job-name=external_holdout_wrapper
#SBATCH --output=logs/external_pgs/holdout-wrapper-%A_%a.log

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TABPFN_MODEL_CACHE_DIR="$PWD/outputs/tabpfn-model-cache"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --no-sync python - <<'PY'
import os
import torch
print({k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")}, flush=True)
probe = torch.ones(1, device="cuda")
probe.add_(1)
torch.cuda.synchronize()
print("CUDA allocation and kernel probe passed:", torch.cuda.get_device_name(0), flush=True)
PY
diseases=(T2D cad bc)
uv run --no-sync python -u scripts/predict_external_pgs_holdout.py \
  --disease "${diseases[$SLURM_ARRAY_TASK_ID]}" --eligibility-config configs/disease_eligibility.yaml
