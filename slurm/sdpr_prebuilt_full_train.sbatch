#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --array=1-22%22
#SBATCH --job-name=sdpr_prebuilt_full
#SBATCH --output=logs/sdpr_full_train/%x-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PREBUILT_REF_DIR="${PREBUILT_REF_DIR:-data/resources/sdpr/ref_1kg_eur_mhc_included}"
export REF_DIR="${REF_DIR:-$PREBUILT_REF_DIR}"
source "$REPO_DIR/slurm/sdpr_full_train.sbatch"
