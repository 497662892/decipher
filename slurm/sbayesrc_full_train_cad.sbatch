#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=72:00:00
#SBATCH --partition=cpu_long
#SBATCH --job-name=sbayesrc_cad
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/sbayesrc_full_train/cad-%J.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=cad
source "$REPO_DIR/slurm/sbayesrc_full_train.sbatch"
