#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --partition=gpu4_dev,gl40s_dev
#SBATCH --job-name=pt_cv_eval_bc
#SBATCH --output=logs/pt_cv/bc/pt_cv_eval_bc-%J.log

export DISEASE="bc"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$REPO_DIR/slurm/pt_cv_eval.sbatch"
