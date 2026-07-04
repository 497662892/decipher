#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH --time=08:00:00
#SBATCH --partition=gl40s_short
#SBATCH --job-name=ldpred2_cv_eval
#SBATCH --output=logs/ldpred2_cv/eval/ldpred2_cv_eval-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DISEASE="${DISEASE:-}"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

if [[ -z "$DISEASE" ]]; then
  echo "DISEASE is required; use slurm/ldpred2_cv_eval_cad.sbatch, slurm/ldpred2_cv_eval_bc.sbatch, or slurm/ldpred2_cv_eval_T2D.sbatch." >&2
  exit 1
fi

cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

LDPRED2_DIR="${LDPRED2_DIR:-outputs/ldpred2_cv}"
PHENOTYPE="${PHENOTYPE:-data/processed/phenotype.csv}"
SPLITS="${SPLITS:-data/processed/splits/disease_specific_split_seed42.csv}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
ELIGIBILITY_CONFIG="${ELIGIBILITY_CONFIG:-configs/disease_eligibility.yaml}"
COVARIATES="${COVARIATES:-sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd}"

TASK_MANIFEST="${TASK_MANIFEST:-$LDPRED2_DIR/$DISEASE/commands/$DISEASE.ldpred2_cv.tasks.csv}"
SCORES_OUT="${SCORES_OUT:-$LDPRED2_DIR/$DISEASE/scores/$DISEASE.ldpred2_cv_scores_long.csv}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$LDPRED2_DIR/$DISEASE/evaluation}"

mkdir -p "$UV_CACHE_DIR" "$(dirname "$SCORES_OUT")" "$EVAL_OUT_DIR" logs/ldpred2_cv/eval "logs/ldpred2_cv/$DISEASE"

echo "Started LDpred2 CV evaluation for $DISEASE at $(date)"
echo "Repo: $REPO_DIR"
echo "Task manifest: $TASK_MANIFEST"
echo "Scores out: $SCORES_OUT"
echo "Evaluation out dir: $EVAL_OUT_DIR"

test -s "$TASK_MANIFEST" || { echo "Missing task manifest: $TASK_MANIFEST" >&2; exit 1; }
test -s "$PHENOTYPE" || { echo "Missing phenotype file: $PHENOTYPE" >&2; exit 1; }
test -s "$SPLITS" || { echo "Missing splits file: $SPLITS" >&2; exit 1; }
test -s "$ELIGIBILITY_CONFIG" || { echo "Missing eligibility config: $ELIGIBILITY_CONFIG" >&2; exit 1; }

uv run --extra dev python -u scripts/collect_ldpred2_cv_scores.py \
  --task-manifest "$TASK_MANIFEST" \
  --patient-col "$PATIENT_COL" \
  --out "$SCORES_OUT"

uv run --extra dev python -u scripts/evaluate_ldpred2_cv.py \
  --disease "$DISEASE" \
  --phenotype "$PHENOTYPE" \
  --splits "$SPLITS" \
  --scores "$SCORES_OUT" \
  --patient-col "$PATIENT_COL" \
  --covariates "$COVARIATES" \
  --eligibility-config "$ELIGIBILITY_CONFIG" \
  --out-dir "$EVAL_OUT_DIR"

echo "Finished LDpred2 CV evaluation for $DISEASE at $(date)"
echo "Summary metrics: $EVAL_OUT_DIR/summary_metrics.csv"
echo "Best by method: $EVAL_OUT_DIR/best_by_method.csv"
echo "Best primary full model: $EVAL_OUT_DIR/best_primary_model.csv"
