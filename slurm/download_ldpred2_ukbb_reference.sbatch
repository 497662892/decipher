#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=download_ldpred2_ukbb
#SBATCH --output=logs/ldpred2_ukbb_reference/download-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs/ldpred2_ukbb_reference data/resources/ldpred2

python3 -u scripts/download_ldpred2_ukbb_reference.py "$@"
