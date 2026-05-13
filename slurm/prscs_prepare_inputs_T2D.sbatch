#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=prscs_prep_T2D
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/prscs_prepare/T2D-%J.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=T2D
export PHI_GRID="${PHI_GRID:-auto,1e-4,1e-2}"
source "$REPO_DIR/slurm/prscs_prepare_inputs.sbatch"
