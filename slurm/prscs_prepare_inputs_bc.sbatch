#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=prscs_prep_bc
#SBATCH --output=logs/prscs_prepare/bc-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=bc
export PHI_GRID="${PHI_GRID:-auto,1e-6,1e-4,1e-2}"
source "$REPO_DIR/slurm/prscs_prepare_inputs.sbatch"
