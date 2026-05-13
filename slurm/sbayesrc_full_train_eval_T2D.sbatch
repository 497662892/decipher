#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=sbayesrc_eval_T2D
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/official_validation_sbayesrc/T2D-%J.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=T2D
source "$REPO_DIR/slurm/sbayesrc_full_train_eval.sbatch"
