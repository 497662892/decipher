#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --partition=gl40s_short
#SBATCH --job-name=prscs_final_prepare
#SBATCH --output=logs/prscs_final_trainval/%x-%J.log

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
export OUT_DIR="${OUT_DIR:-outputs/prscs_final_trainval_test_ukbb_eur}"
export REF_DIR="${REF_DIR:-data/resources/prscs/ldblk_ukbb_eur}"
export PHI_GRID="${PHI_GRID:-auto,1e-4,1e-2}"

source "$REPO_DIR/slurm/prscs_prepare_inputs.sbatch"
