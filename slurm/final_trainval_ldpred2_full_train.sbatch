#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=72:00:00
#SBATCH --partition=gl40s_short
#SBATCH --array=1-1%1
#SBATCH --job-name=ldpred2_final_trainval
#SBATCH --output=logs/ldpred2_final_trainval/%x-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export DISEASE="${DISEASE:-}"
export MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
export GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/final_genotype/qc_final/train_val.common}"
export SCORE_GENOTYPE_PREFIX="${SCORE_GENOTYPE_PREFIX:-outputs/final_genotype/qc_final/holdout_test.common}"
export HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/final/sumstats/harmonized}"
export TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/train_val.keep.psam}"
export SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/holdout_test.keep.psam}"
export FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
export OUT_DIR="${OUT_DIR:-outputs/ldpred2_final_trainval_test}"
export LD_CACHE_DIR="${LD_CACHE_DIR:-outputs/ldpred2_final_trainval_ld_cache}"
export VARIANT_EXTRACT="${VARIANT_EXTRACT:-data/processed/ldpred2/hm3plus_target.snplist}"
export H2_GRID="${H2_GRID:-0.05,0.1,0.2,0.4,0.8,1}"
export P_GRID="${P_GRID:-0.0001,0.001,0.01,0.1,1}"
export SPARSE_GRID="${SPARSE_GRID:-0,1}"
export LD_REFERENCE_MODE="${LD_REFERENCE_MODE:-ukb_hm3_plus}"
export LD_REF_MAP="${LD_REF_MAP:-data/resources/ldpred2/map_hm3_plus.rds}"
export LD_REF_DIR="${LD_REF_DIR:-data/resources/ldpred2/corr_hm3_plus}"
if [[ -z "${LD_REF_PATTERN:-}" ]]; then
  export LD_REF_PATTERN='LD_with_blocks_chr{chr}.rds'
fi
export MATCH_BY="${MATCH_BY:-position}"
export REGENERATE_LDPRED2_TASKS="${REGENERATE_LDPRED2_TASKS:-1}"

source "$REPO_DIR/slurm/ldpred2_full_train.sbatch"
