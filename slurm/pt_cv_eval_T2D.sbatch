#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00-04:00:00
#SBATCH --partition=deciphehr
#SBATCH --job-name=pt_cv_eval_T2D
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/pt_cv/T2D/pt_cv_eval_T2D-%J.log

export DISEASE="T2D"
REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
source "$REPO_DIR/slurm/pt_cv_eval.sbatch"
