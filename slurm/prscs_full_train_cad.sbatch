#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-66%22
#SBATCH --job-name=prscs_full_cad
#SBATCH --output=logs/prscs_full_train/cad-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=cad
export PHI_GRID="${PHI_GRID:-auto,1e-4,1e-2}"
source "$REPO_DIR/slurm/prscs_full_train.sbatch"
