#!/bin/bash
#SBATCH --job-name=pt_full_cad
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/pt_full_train/cad-%A_%a.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export DISEASE=cad
source "$REPO_DIR/slurm/pt_full_train.sbatch"
