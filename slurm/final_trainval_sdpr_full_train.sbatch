#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --partition=gl40s_short
#SBATCH --array=1-22%22
#SBATCH --job-name=sdpr_final_trainval
#SBATCH --output=logs/sdpr_final_trainval/%x-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export DISEASE="${DISEASE:-}"
export OUT_DIR="${OUT_DIR:-outputs/sdpr_final_trainval_test}"
export REF_DIR="${REF_DIR:-data/resources/sdpr/ref_1kg_eur_mhc_included}"

source "$REPO_DIR/slurm/sdpr_full_train.sbatch"
