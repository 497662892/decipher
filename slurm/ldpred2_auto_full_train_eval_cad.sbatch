#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=ldpred2_auto_eval_cad
#SBATCH --output=logs/official_validation_ldpred2_auto/cad-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=cad
source "$REPO_DIR/slurm/ldpred2_auto_full_train_eval.sbatch"
