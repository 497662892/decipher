#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --partition=deciphehr
#SBATCH --job-name=megaprs_bc
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/megaprs_full_train/bc-%J.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=bc
source "$REPO_DIR/slurm/megaprs_full_train.sbatch"
