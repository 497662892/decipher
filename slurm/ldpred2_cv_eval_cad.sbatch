#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH --time=00-08:00:00
#SBATCH --partition=deciphehr
#SBATCH --job-name=ldpred2_eval_cad
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/ldpred2_cv/cad/ldpred2_eval_cad-%J.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
DISEASE="cad"
source "$REPO_DIR/slurm/ldpred2_cv_eval.sbatch"
