#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --job-name=prs_new_fm
#SBATCH --output=logs/prs_foundation_models/%x-%A_%a.log

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache" PYTHONPATH="$PWD/src"
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export HF_HOME="$PWD/outputs/kumo-model-cache" MPLCONFIGDIR="$PWD/outputs/mpl-cache"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export XDG_CACHE_HOME="$PWD/outputs/runtime/cache"
export TRITON_CACHE_DIR="$PWD/outputs/runtime/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs/runtime/torchinductor-cache"
backends=(tabpfn35 kumo)
backend="${backends[${SLURM_ARRAY_TASK_ID:?}]}"
uv run --offline --no-project --python "outputs/runtime/$backend/bin/python" python -u \
  scripts/benchmark_prs_foundation_models.py --backend "$backend" \
  --eligibility-config configs/disease_eligibility.yaml
