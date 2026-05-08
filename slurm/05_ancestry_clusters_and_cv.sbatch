#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=00-04:00:00
#SBATCH --partition=gpu4_dev
#SBATCH --job-name=decipehr_jupyter_05
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/notebooks/05_ancestry_clusters_and_cv-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
NOTEBOOK_DIR="${NOTEBOOK_DIR:-$REPO_DIR}"
PORT="${PORT:-8890}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

cd "$REPO_DIR"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  # BigPurple's /etc/bashrc is not nounset-safe; mirror jupyter.sbatch by
  # sourcing the user shell environment before enabling `set -u`.
  source /gpfs/home/jm10850/.bashrc
fi

set -u

mkdir -p "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not available on PATH. Load your shell environment before submitting." >&2
  exit 1
fi

export UV_CACHE_DIR
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

XDG_RUNTIME_DIR=""
node="$(hostname -s)"
user="$(whoami)"
login_node="${SLURM_SUBMIT_HOST:-}"

case "${SLURM_SUBMIT_HOST:-}" in
bigpurple-ln1)
  login_node=bigpurple1.nyumc.org
  ;;
bigpurple-ln2)
  login_node=bigpurple2.nyumc.org
  ;;
bigpurple-ln3)
  login_node=bigpurple3.nyumc.org
  ;;
bigpurple-ln4)
  login_node=bigpurple4.nyumc.org
  ;;
esac

echo "Started decipEHR Jupyter notebook server at $(date)"
echo "Host: $node"
echo "Repo: $REPO_DIR"
echo "Notebook root: $NOTEBOOK_DIR"
echo "Target notebook: jupyter-notebook/05_ancestry_clusters_and_cv.ipynb"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "UV cache: $UV_CACHE_DIR"
echo ""
echo "MacOS or Linux terminal command to create your SSH tunnel:"
echo "ssh -N -L ${PORT}:${node}:${PORT} ${user}@${login_node}"
echo ""
echo "Then open this URL on your local machine:"
echo "http://localhost:${PORT}"
echo ""
echo "Use the token printed by Jupyter below, then open:"
echo "jupyter-notebook/05_ancestry_clusters_and_cv.ipynb"
echo ""

if [[ "$RUN_UV_SYNC" == "1" ]]; then
  uv sync --extra dev
fi

uv run --extra dev \
  --with ipykernel \
  --with notebook \
  jupyter-notebook \
  --no-browser \
  --port="$PORT" \
  --ip="$node" \
  --notebook-dir="$NOTEBOOK_DIR"
