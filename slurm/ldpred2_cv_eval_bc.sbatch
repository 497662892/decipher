#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH --time=08:00:00
#SBATCH --partition=gl40s_short
#SBATCH --job-name=ldpred2_eval_bc
#SBATCH --output=logs/ldpred2_cv/bc/ldpred2_eval_bc-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DISEASE="bc"
source "$REPO_DIR/slurm/ldpred2_cv_eval.sbatch"
