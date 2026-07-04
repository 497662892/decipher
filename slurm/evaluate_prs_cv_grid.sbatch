#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --partition=gl40s_short
#SBATCH --job-name=prs_cv_eval
#SBATCH --output=logs/prs_cv_eval/%x-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DISEASE="${DISEASE:-}"
METHOD_NAME="${METHOD_NAME:-}"
SCORE_MANIFEST="${SCORE_MANIFEST:-}"
SCORE_KIND="${SCORE_KIND:-}"
SCORE_PATH_COLUMN="${SCORE_PATH_COLUMN:-score_file}"
PARAM_COLS="${PARAM_COLS:-}"
PHENOTYPE="${PHENOTYPE:-data/processed/final/train_val_phenotype.csv}"
SPLITS="${SPLITS:-data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
ELIGIBILITY_CONFIG="${ELIGIBILITY_CONFIG:-configs/disease_eligibility.yaml}"
COVARIATES="${COVARIATES:-sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd}"
OUT_DIR="${OUT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

if [[ -z "$DISEASE" || -z "$METHOD_NAME" || -z "$SCORE_MANIFEST" || -z "$SCORE_KIND" || -z "$OUT_DIR" ]]; then
  echo "DISEASE, METHOD_NAME, SCORE_MANIFEST, SCORE_KIND, and OUT_DIR are required." >&2
  exit 1
fi

cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR" logs/prs_cv_eval

if [[ -x "$PYTHON_BIN" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  PYTHON_CMD=(uv run --extra dev python)
fi

"${PYTHON_CMD[@]}" scripts/evaluate_prs_cv_grid.py \
  --disease "$DISEASE" \
  --method-name "$METHOD_NAME" \
  --score-manifest "$SCORE_MANIFEST" \
  --score-kind "$SCORE_KIND" \
  --score-path-column "$SCORE_PATH_COLUMN" \
  --param-cols "$PARAM_COLS" \
  --phenotype "$PHENOTYPE" \
  --splits "$SPLITS" \
  --patient-col "$PATIENT_COL" \
  --covariates "$COVARIATES" \
  --eligibility-config "$ELIGIBILITY_CONFIG" \
  --out-dir "$OUT_DIR"
