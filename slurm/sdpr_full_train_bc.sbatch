#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-22%22
#SBATCH --job-name=sdpr_full_bc
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/sdpr_full_train/bc-%A_%a.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=bc
source "$REPO_DIR/slurm/sdpr_full_train.sbatch"
