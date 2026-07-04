#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=1-22%22
#SBATCH --job-name=sdpr_pre_T2D
#SBATCH --output=logs/sdpr_full_train/T2D-prebuilt-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=T2D
source "$REPO_DIR/slurm/sdpr_prebuilt_full_train.sbatch"
