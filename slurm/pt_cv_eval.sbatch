#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00-04:00:00
#SBATCH --partition=deciphehr
#SBATCH --job-name=pt_cv_eval
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/pt_cv/eval/pt_cv_eval-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"
DISEASE="${DISEASE:-}"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  source /gpfs/home/jm10850/.bashrc
fi

set -u

if [[ -z "$DISEASE" ]]; then
  echo "DISEASE is required; use slurm/pt_cv_eval_cad.sbatch, slurm/pt_cv_eval_bc.sbatch, or slurm/pt_cv_eval_T2D.sbatch." >&2
  exit 1
fi

cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

PT_DIR="${PT_DIR:-outputs/pt_cv}"
PHENOTYPE="${PHENOTYPE:-data/processed/phenotype.csv}"
SPLITS="${SPLITS:-data/processed/splits/disease_specific_split_seed42.csv}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
ELIGIBILITY_CONFIG="${ELIGIBILITY_CONFIG:-configs/disease_eligibility.yaml}"
COVARIATES="${COVARIATES:-sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd}"

TASK_MANIFEST="${TASK_MANIFEST:-$PT_DIR/$DISEASE/commands/$DISEASE.pt_cv.tasks.csv}"
SCORES_OUT="${SCORES_OUT:-$PT_DIR/$DISEASE/scores/$DISEASE.pt_cv_scores_long.csv}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$PT_DIR/$DISEASE/evaluation}"

mkdir -p "$UV_CACHE_DIR" "$(dirname "$SCORES_OUT")" "$EVAL_OUT_DIR" logs/pt_cv/eval "logs/pt_cv/$DISEASE"

echo "Started P+T CV evaluation for $DISEASE at $(date)"
echo "Repo: $REPO_DIR"
echo "Task manifest: $TASK_MANIFEST"
echo "Scores out: $SCORES_OUT"
echo "Evaluation out dir: $EVAL_OUT_DIR"

test -s "$TASK_MANIFEST" || { echo "Missing task manifest: $TASK_MANIFEST" >&2; exit 1; }
test -s "$PHENOTYPE" || { echo "Missing phenotype file: $PHENOTYPE" >&2; exit 1; }
test -s "$SPLITS" || { echo "Missing splits file: $SPLITS" >&2; exit 1; }
test -s "$ELIGIBILITY_CONFIG" || { echo "Missing eligibility config: $ELIGIBILITY_CONFIG" >&2; exit 1; }

uv run --extra dev python -u scripts/collect_pt_cv_scores.py \
  --task-manifest "$TASK_MANIFEST" \
  --patient-col "$PATIENT_COL" \
  --out "$SCORES_OUT"

uv run --extra dev python -u scripts/evaluate_pt_cv.py \
  --disease "$DISEASE" \
  --phenotype "$PHENOTYPE" \
  --splits "$SPLITS" \
  --scores "$SCORES_OUT" \
  --patient-col "$PATIENT_COL" \
  --covariates "$COVARIATES" \
  --eligibility-config "$ELIGIBILITY_CONFIG" \
  --out-dir "$EVAL_OUT_DIR"

echo "Finished P+T CV evaluation for $DISEASE at $(date)"
echo "Summary metrics: $EVAL_OUT_DIR/summary_metrics.csv"
echo "Best by method: $EVAL_OUT_DIR/best_by_method.csv"
