#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=72:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-1%1
#SBATCH --job-name=ldpred2_full_bc
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/ldpred2_full_train/bc-%A_%a.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=bc
source "$REPO_DIR/slurm/ldpred2_full_train.sbatch"
