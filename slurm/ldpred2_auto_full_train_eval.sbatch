#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=ldpred2_auto_eval
#SBATCH --output=logs/official_validation_ldpred2_auto/ldpred2_auto_eval-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LDPRED2_DIR="${LDPRED2_DIR:-outputs/ldpred2_auto_full_train}"
export VALIDATION_OUT_ROOT="${VALIDATION_OUT_ROOT:-outputs/official_validation_ldpred2_auto}"
export MODEL_NAME="${MODEL_NAME:-ldpred2_auto}"
export PRS_PARAM_COLS="${PRS_PARAM_COLS:-model,h2,p,sparse}"
source "$REPO_DIR/slurm/ldpred2_full_train_eval.sbatch"
