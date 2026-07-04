#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=72:00:00
#SBATCH --partition=cpu_long
#SBATCH --job-name=sbayesrc_T2D
#SBATCH --output=logs/sbayesrc_full_train/T2D-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=T2D
source "$REPO_DIR/slurm/sbayesrc_full_train.sbatch"
