#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-1%1
#SBATCH --job-name=lassosum2_ukbb_T2D
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/lassosum2_ukbb_full_train/T2D-%A_%a.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=T2D
source "$REPO_DIR/slurm/lassosum2_ukbb_full_train.sbatch"
