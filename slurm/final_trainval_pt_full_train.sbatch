#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --partition=gl40s_short
#SBATCH --array=1-4%4
#SBATCH --job-name=pt_final_trainval
#SBATCH --output=logs/pt_final_trainval/%x-%A_%a.log

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Set DISEASE=cad/bc/T2D and edit the grid values below to the CV-selected grid.
export DISEASE="${DISEASE:-}"
export MANIFEST="${MANIFEST:-configs/official_sumstats.yaml}"
export GENOTYPE_PREFIX="${GENOTYPE_PREFIX:-outputs/final_genotype/qc_final/train_val.common}"
export SCORE_GENOTYPE_PREFIX="${SCORE_GENOTYPE_PREFIX:-outputs/final_genotype/qc_final/holdout_test.common}"
export FIT_SCORE_GENOTYPE_PREFIX="${FIT_SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"
export HARMONIZED_DIR="${HARMONIZED_DIR:-data/processed/final/sumstats/harmonized}"
export TRAIN_KEEP="${TRAIN_KEEP:-data/processed/sample_sets/train_val.keep.psam}"
export SCORE_KEEP="${SCORE_KEEP:-data/processed/sample_sets/holdout_test.keep.psam}"
export FIT_SCORE_KEEP="${FIT_SCORE_KEEP:-$TRAIN_KEEP}"
export OUT_DIR="${OUT_DIR:-outputs/pt_final_trainval_test}"
export P_THRESHOLDS="${P_THRESHOLDS:-1,0.5,0.05,0.0005,0.000005,0.00000005}"
export R2_THRESHOLDS="${R2_THRESHOLDS:-0.2,0.4,0.6,0.8}"
export PLINK_MEMORY_MB="${PLINK_MEMORY_MB:-90000}"
export REGENERATE_PT_TASKS="${REGENERATE_PT_TASKS:-1}"

source "$REPO_DIR/slurm/pt_full_train.sbatch"
