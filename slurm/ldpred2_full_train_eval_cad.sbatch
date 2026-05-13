#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=ldpred2_eval_cad
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/official_validation_ldpred2/cad-%J.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=cad
source "$REPO_DIR/slurm/ldpred2_full_train_eval.sbatch"
