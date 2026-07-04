#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=ldpred2_ukbb_eval_T2D
#SBATCH --output=logs/official_validation_ldpred2_ukbb/T2D-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=T2D
source "$REPO_DIR/slurm/ldpred2_ukbb_full_train_eval.sbatch"
