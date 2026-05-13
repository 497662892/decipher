#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --partition=deciphehr
#SBATCH --array=1-4%4
#SBATCH --job-name=pt_full_bc
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/pt_full_train/bc-%A_%a.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=bc
source "$REPO_DIR/slurm/pt_full_train.sbatch"
