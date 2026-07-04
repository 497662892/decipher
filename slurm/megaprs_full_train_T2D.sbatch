#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu4_dev,gl40s_dev
#SBATCH --job-name=megaprs_T2D
#SBATCH --output=logs/megaprs_full_train/T2D-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=T2D
source "$REPO_DIR/slurm/megaprs_full_train.sbatch"
