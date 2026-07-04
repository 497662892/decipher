#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=megaprs_eval_cad
#SBATCH --output=logs/official_validation_megaprs/cad-%J.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DISEASE=cad
source "$REPO_DIR/slurm/megaprs_full_train_eval.sbatch"
