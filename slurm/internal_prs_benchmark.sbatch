#!/usr/bin/env bash
#SBATCH --job-name=nyu_prs_internal
#SBATCH --partition=a100_short,gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-17%2
#SBATCH --output=outputs/internal_prs/nyu_20261011/logs/%A_%a.out
#SBATCH --error=outputs/internal_prs/nyu_20261011/logs/%A_%a.err
set -euo pipefail
umask 077
cd /gpfs/data/shenlab/Jiajian/HW/decipehr
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TABPFN_MODEL_CACHE_DIR="$PWD/outputs/tabpfn-model-cache"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --no-sync python - <<'PY'
import os
import torch
print({key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")}, flush=True)
x = torch.ones((32, 32), device="cuda")
assert float((x @ x).sum().item()) == 32768.0
torch.cuda.synchronize()
print("CUDA preflight passed:", torch.cuda.get_device_name(0), flush=True)
PY
diseases=(T2D cad bc)
partitions=(0 1 2 3 4 development)
task=${SLURM_ARRAY_TASK_ID}
uv run --no-sync python scripts/evaluate_internal_prs_benchmark.py \
  --disease "${diseases[$((task / 6))]}" --partition "${partitions[$((task % 6))]}"
