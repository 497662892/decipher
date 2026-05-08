#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00-04:00:00
#SBATCH --partition=deciphehr
#SBATCH --job-name=pt_cv_eval_bc
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/pt_cv/bc/pt_cv_eval_bc-%J.log

export DISEASE="bc"
REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
source "$REPO_DIR/slurm/pt_cv_eval.sbatch"
