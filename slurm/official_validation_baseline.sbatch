#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=official_validation
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/official_validation/official_validation-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  source /gpfs/home/jm10850/.bashrc
fi

set -u

cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not available on PATH after loading the shell environment." >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"

LABELLED_PHENOTYPE="${LABELLED_PHENOTYPE:-/gpfs/data/deciphEHRlab/deciphEHRdream/holdout_phase/phenotype_files/Training_Phase_Cov_file_with_Phenotype.txt}"
TRAIN_PHENOTYPE="${TRAIN_PHENOTYPE:-/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/phenotype_files/Training_Cov_file.txt}"
VALIDATION_COVARIATES="${VALIDATION_COVARIATES:-/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/phenotype_files/Testing_Cov_file.txt}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
LABELS="${LABELS:-cad,bc,T2D}"
COVARIATES="${COVARIATES:-sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd}"
ELIGIBILITY_CONFIG="${ELIGIBILITY_CONFIG:-configs/disease_eligibility.yaml}"
MODEL_NAME="${MODEL_NAME:-baseline}"
N_BOOTSTRAPS="${N_BOOTSTRAPS:-1000}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-outputs/official_validation_baseline}"
VALIDATION_PHENOTYPE_OUT="${VALIDATION_PHENOTYPE_OUT:-data/processed/official_validation_phenotype.csv}"
TRAIN_PRS_SCORES="${TRAIN_PRS_SCORES:-}"
PRS_SCORES="${PRS_SCORES:-}"
TRAIN_PRS_LONG_SCORES="${TRAIN_PRS_LONG_SCORES:-}"
PRS_LONG_SCORES="${PRS_LONG_SCORES:-}"
PRS_PARAM_COLS="${PRS_PARAM_COLS:-clump_r2,p_threshold}"

mkdir -p "$OUT_DIR" "$(dirname "$VALIDATION_PHENOTYPE_OUT")" "$UV_CACHE_DIR" logs/official_validation

cmd=(
  uv run --extra dev python scripts/validate_official_model.py
  --model-name "$MODEL_NAME"
  --train-phenotype "$TRAIN_PHENOTYPE"
  --labelled-phenotype "$LABELLED_PHENOTYPE"
  --validation-covariates "$VALIDATION_COVARIATES"
  --patient-col "$PATIENT_COL"
  --labels "$LABELS"
  --covariates "$COVARIATES"
  --eligibility-config "$ELIGIBILITY_CONFIG"
  --n-bootstraps "$N_BOOTSTRAPS"
  --seed "$SEED"
  --validation-phenotype-out "$VALIDATION_PHENOTYPE_OUT"
  --out-dir "$OUT_DIR"
)

if [[ -n "$TRAIN_PRS_SCORES" ]]; then
  cmd+=(--train-prs-scores "$TRAIN_PRS_SCORES")
fi
if [[ -n "$PRS_SCORES" ]]; then
  cmd+=(--validation-prs-scores "$PRS_SCORES")
fi
if [[ -n "$TRAIN_PRS_LONG_SCORES" ]]; then
  cmd+=(--train-prs-long-scores "$TRAIN_PRS_LONG_SCORES")
fi
if [[ -n "$PRS_LONG_SCORES" ]]; then
  cmd+=(--validation-prs-long-scores "$PRS_LONG_SCORES" --prs-param-cols "$PRS_PARAM_COLS")
fi

echo "Started official validation bootstrap scoring at $(date)"
echo "Model: $MODEL_NAME"
echo "Train phenotype: $TRAIN_PHENOTYPE"
echo "Labelled phenotype: $LABELLED_PHENOTYPE"
echo "Validation covariates: $VALIDATION_COVARIATES"
echo "Bootstrap resamples: $N_BOOTSTRAPS"
echo "Output: $OUT_DIR"
"${cmd[@]}"
echo "Finished official validation bootstrap scoring at $(date)"
