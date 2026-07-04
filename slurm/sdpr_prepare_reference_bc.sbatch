#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --job-name=sdpr_prep_bc
#SBATCH --output=logs/sdpr_prepare/bc-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=bc
source "$REPO_DIR/slurm/sdpr_prepare_reference.sbatch"
