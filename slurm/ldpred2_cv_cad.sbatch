#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=03-00:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-5%1
#SBATCH --job-name=ldpred2_cv_cad
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/ldpred2_cv/cad/ldpred2_cv_cad-%A_%a.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
DISEASE="cad"
source "$REPO_DIR/slurm/ldpred2_cv.sbatch"
