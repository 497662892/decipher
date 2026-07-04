#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=gl40s_short
#SBATCH --array=1-66%22
#SBATCH --job-name=prscs_final_trainval
#SBATCH --output=logs/prscs_final_trainval/%x-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export DISEASE="${DISEASE:-}"
export OUT_DIR="${OUT_DIR:-outputs/prscs_final_trainval_test_ukbb_eur}"
export PHI_GRID="${PHI_GRID:-auto,1e-4,1e-2}"

source "$REPO_DIR/slurm/prscs_full_train.sbatch"
