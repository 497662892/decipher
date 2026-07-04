#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=pt_eval_bc
#SBATCH --output=logs/official_validation_pt/bc-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_DIR
export DISEASE=bc
source "$REPO_DIR/slurm/pt_full_train_eval.sbatch"
