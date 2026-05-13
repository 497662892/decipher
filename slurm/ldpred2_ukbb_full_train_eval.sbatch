#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=ldpred2_ukbb_eval
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/official_validation_ldpred2_ukbb/ldpred2_ukbb_eval-%J.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export LDPRED2_DIR="${LDPRED2_DIR:-outputs/ldpred2_ukbb_full_train}"
export VALIDATION_OUT_ROOT="${VALIDATION_OUT_ROOT:-outputs/official_validation_ldpred2_ukbb}"
export MODEL_NAME="${MODEL_NAME:-ldpred2_ukbb}"

source "$REPO_DIR/slurm/ldpred2_full_train_eval.sbatch"
