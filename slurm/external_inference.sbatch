#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gl40s_short
#SBATCH --job-name=prs_external
#SBATCH --output=logs/external-%j.log
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
module load plink/2.0-20260228
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export TABPFN_NO_BROWSER=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
# Supply CONFIG=/absolute/cohort.yaml; edit resources/partition for your cluster.
# For CPU preparation: sbatch --gres=none --partition=cpu_short with EXTERNAL_COMMAND=prepare.
"${UV_BIN:-/gpfs/data/shenlab/Jiajian/software/miniconda3/bin/uv}" run --offline --no-sync \
  python -u scripts/external_inference.py "${EXTERNAL_COMMAND:-run}" --config "${CONFIG:?}"

