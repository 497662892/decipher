#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=sdpr_prebuilt_prep
#SBATCH --output=logs/sdpr_prepare/%x-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REF_MODE=prebuilt
export PREBUILT_REF_DIR="${PREBUILT_REF_DIR:-data/resources/sdpr/ref_1kg_eur_mhc_included}"
source "$REPO_DIR/slurm/sdpr_prepare_reference.sbatch"
