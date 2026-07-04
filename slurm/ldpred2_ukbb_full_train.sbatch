#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-1%1
#SBATCH --job-name=ldpred2_ukbb_full
#SBATCH --output=logs/ldpred2_ukbb_full_train/%x-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OUT_DIR="${OUT_DIR:-outputs/ldpred2_ukbb_full_train}"
export LD_CACHE_DIR="${LD_CACHE_DIR:-outputs/ldpred2_ukbb_ld_cache}"
export LD_REFERENCE_MODE="${LD_REFERENCE_MODE:-ukb_hm3_plus}"
export LD_REF_MAP="${LD_REF_MAP:-data/resources/ldpred2/map_hm3_plus.rds}"
export LD_REF_DIR="${LD_REF_DIR:-data/resources/ldpred2/corr_hm3_plus}"
if [[ -z "${LD_REF_PATTERN:-}" ]]; then
  export LD_REF_PATTERN='LD_with_blocks_chr{chr}.rds'
fi
export VARIANT_EXTRACT="${VARIANT_EXTRACT:-data/processed/ldpred2/hm3plus_target.snplist}"
export MATCH_BY="${MATCH_BY:-position}"
export REUSE_LDPRED2_LD="${REUSE_LDPRED2_LD:-1}"

source "$REPO_DIR/slurm/ldpred2_full_train.sbatch"
