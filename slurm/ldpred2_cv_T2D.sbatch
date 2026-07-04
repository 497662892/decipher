#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=72:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-5%1
#SBATCH --job-name=ldpred2_cv_T2D
#SBATCH --output=logs/ldpred2_cv/T2D/ldpred2_cv_T2D-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DISEASE="T2D"
source "$REPO_DIR/slurm/ldpred2_cv.sbatch"
