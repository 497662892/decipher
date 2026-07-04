#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=sdpr_pre_prep_T2D
#SBATCH --output=logs/sdpr_prepare/T2D-prebuilt-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=T2D
source "$REPO_DIR/slurm/sdpr_prebuilt_prepare_reference.sbatch"
