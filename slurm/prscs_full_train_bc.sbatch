#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-88%22
#SBATCH --job-name=prscs_full_bc
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/prscs_full_train/bc-%A_%a.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=bc
export PHI_GRID="${PHI_GRID:-auto,1e-6,1e-4,1e-2}"
source "$REPO_DIR/slurm/prscs_full_train.sbatch"
