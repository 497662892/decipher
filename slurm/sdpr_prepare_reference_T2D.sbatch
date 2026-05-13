#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --job-name=sdpr_prep_T2D
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/sdpr_prepare/T2D-%J.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=T2D
source "$REPO_DIR/slurm/sdpr_prepare_reference.sbatch"
