#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --partition=gl40s_short
#SBATCH --array=1-1%1
#SBATCH --job-name=lassosum2_final_trainval
#SBATCH --output=logs/lassosum2_final_trainval/%x-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export DISEASE="${DISEASE:-}"
export MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
export GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/final_genotype/qc_final/train_val.common}"
export SCORE_GENOTYPE_PREFIX="${SCORE_GENOTYPE_PREFIX:-outputs/final_genotype/qc_final/holdout_test.common}"
export FIT_SCORE_GENOTYPE_PREFIX="${FIT_SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"
export HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/final/sumstats/harmonized}"
export TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/train_val.keep.psam}"
export SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/holdout_test.keep.psam}"
export FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
export OUT_DIR="${OUT_DIR:-outputs/lassosum2_ukbb_final_trainval_test}"
export LD_CACHE_DIR="${LD_CACHE_DIR:-outputs/lassosum2_ukbb_final_trainval_ld_cache}"
export VARIANT_EXTRACT="${VARIANT_EXTRACT:-data/processed/ldpred2/hm3plus_target.snplist}"
export DELTA_GRID="${DELTA_GRID:-0.001,0.01,0.1,1}"
export NLAMBDA="${NLAMBDA:-30}"
export REGENERATE_LASSOSUM2_TASKS="${REGENERATE_LASSOSUM2_TASKS:-1}"

source "$REPO_DIR/slurm/lassosum2_ukbb_full_train.sbatch"
