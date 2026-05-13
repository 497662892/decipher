#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=sdpr_pre_prep_bc
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/sdpr_prepare/bc-prebuilt-%J.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=bc
source "$REPO_DIR/slurm/sdpr_prebuilt_prepare_reference.sbatch"
