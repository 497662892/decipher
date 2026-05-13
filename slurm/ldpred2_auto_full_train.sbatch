#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=72:00:00
#SBATCH --partition=cpu_medium
#SBATCH --array=1-1%1
#SBATCH --job-name=ldpred2_auto_full
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/ldpred2_auto_full_train/%x-%A_%a.log

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
export LDPRED2_MODE="${LDPRED2_MODE:-auto}"
export OUT_DIR="${OUT_DIR:-outputs/ldpred2_auto_full_train}"
export LD_CACHE_DIR="${LD_CACHE_DIR:-outputs/ldpred2_auto_ukbb_ld_cache}"
export LD_REFERENCE_MODE="${LD_REFERENCE_MODE:-ukb_hm3_plus}"
export LD_REF_MAP="${LD_REF_MAP:-data/resources/ldpred2/map_hm3_plus.rds}"
export LD_REF_DIR="${LD_REF_DIR:-data/resources/ldpred2/corr_hm3_plus}"
if [[ -z "${LD_REF_PATTERN:-}" ]]; then
  export LD_REF_PATTERN='LD_with_blocks_chr{chr}.rds'
fi
export MATCH_BY="${MATCH_BY:-position}"
export REGENERATE_LDPRED2_TASKS="${REGENERATE_LDPRED2_TASKS:-1}"
case "${DISEASE:-}" in
  T2D)
    DEFAULT_AUTO_H2_INIT=0.05
    DEFAULT_AUTO_P_INIT=0.1,0.3,1
    ;;
  cad)
    DEFAULT_AUTO_H2_INIT=0.2
    DEFAULT_AUTO_P_INIT=0.03,0.1,0.3
    ;;
  bc)
    DEFAULT_AUTO_H2_INIT=0.2
    DEFAULT_AUTO_P_INIT=0.003,0.01,0.03
    ;;
  *)
    DEFAULT_AUTO_H2_INIT=0.1
    DEFAULT_AUTO_P_INIT=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.2
    ;;
esac
export AUTO_H2_INIT="${AUTO_H2_INIT:-$DEFAULT_AUTO_H2_INIT}"
export AUTO_P_INIT="${AUTO_P_INIT:-$DEFAULT_AUTO_P_INIT}"
export AUTO_BURN_IN="${AUTO_BURN_IN:-500}"
export AUTO_NUM_ITER="${AUTO_NUM_ITER:-200}"
export AUTO_SPARSE="${AUTO_SPARSE:-1}"
export AUTO_ALLOW_JUMP_SIGN="${AUTO_ALLOW_JUMP_SIGN:-0}"
export AUTO_SHRINK_CORR="${AUTO_SHRINK_CORR:-0.95}"
export AUTO_USE_MLE="${AUTO_USE_MLE:-0}"
source "$REPO_DIR/slurm/ldpred2_full_train.sbatch"
