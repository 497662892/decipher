#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=decipehr_preprocess
#SBATCH --output=/gpfs/data/shenlab/Jiajian/HW/decipehr/logs/preprocess/preprocess-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/data/shenlab/Jiajian/HW/decipehr}"

if [[ -f /gpfs/home/jm10850/.bashrc ]]; then
  # BigPurple's /etc/bashrc is not nounset-safe; source the user environment
  # before enabling `set -u`.
  source /gpfs/home/jm10850/.bashrc
fi

set -u

cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module load git/2.49.0 || true
  module load git-lfs/3.3.0 || true
  module load plink/2.0-20260228 || true
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not available on PATH after loading the shell environment." >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

PHENOTYPE="${PHENOTYPE:-/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/phenotype_files/Training_Cov_file.txt}"
VALIDATION_COVARIATES="${VALIDATION_COVARIATES:-/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/phenotype_files/Testing_Cov_file.txt}"
PATIENT_COL="${PATIENT_COL:-sampleid}"
LABELS="${LABELS:-cad,bc,T2D}"
COVARIATES="${COVARIATES:-sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd}"
PC_COLS="${PC_COLS:-PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd}"
AGE_COL="${AGE_COL:-age_sd}"
SEX_COL="${SEX_COL:-sex}"
SEX_BALANCE_LABELS="${SEX_BALANCE_LABELS:-cad,T2D}"
ELIGIBILITY_CONFIG="${ELIGIBILITY_CONFIG:-configs/disease_eligibility.yaml}"

OUT_DIR="${OUT_DIR:-data/processed}"
REPORT_DIR="${REPORT_DIR:-$OUT_DIR/reports}"
SPLIT_DIR="${SPLIT_DIR:-$OUT_DIR/splits}"
ANCESTRY_DIR="${ANCESTRY_DIR:-outputs/ancestry}"
CV_DIR="${CV_DIR:-outputs/cv}"
BASELINE_OUT_DIR="${BASELINE_OUT_DIR:-outputs/cv_disease_specific_baseline}"
GENOTYPE_QC_DIR="${GENOTYPE_QC_DIR:-outputs/genotype_qc}"
UV_CACHE_DIR="${UV_CACHE_DIR:-outputs/uv-cache}"
SAMPLE_SET_DIR="${SAMPLE_SET_DIR:-$OUT_DIR/sample_sets}"
export UV_CACHE_DIR

N_FOLDS="${N_FOLDS:-5}"
SEED="${SEED:-42}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
RUN_ANCESTRY="${RUN_ANCESTRY:-1}"
RUN_SPLITS="${RUN_SPLITS:-1}"
RUN_BASELINE_CV="${RUN_BASELINE_CV:-0}"
RUN_GENOTYPE_QC_TEMPLATE="${RUN_GENOTYPE_QC_TEMPLATE:-1}"
RUN_GENOTYPE_QC="${RUN_GENOTYPE_QC:-0}"
RUN_SAMPLE_SETS="${RUN_SAMPLE_SETS:-1}"

PREPARED_PHENOTYPE="${PREPARED_PHENOTYPE:-$OUT_DIR/phenotype.csv}"
MISSINGNESS_REPORT="${MISSINGNESS_REPORT:-$REPORT_DIR/phenotype_missingness.csv}"
LABEL_REPORT="${LABEL_REPORT:-$REPORT_DIR/label_balance.csv}"
FEATURE_REPORT="${FEATURE_REPORT:-$REPORT_DIR/feature_summary.csv}"
FEATURE_COUNT_REPORT="${FEATURE_COUNT_REPORT:-$REPORT_DIR/feature_counts.csv}"
ANCESTRY_CLUSTERS="${ANCESTRY_CLUSTERS:-$OUT_DIR/ancestry_clusters.csv}"
ANCESTRY_REPORT="${ANCESTRY_REPORT:-$ANCESTRY_DIR/ancestry_cluster_report.csv}"
ANCESTRY_QUALITY="${ANCESTRY_QUALITY:-$ANCESTRY_DIR/ancestry_cluster_quality.csv}"
ANCESTRY_PLOT="${ANCESTRY_PLOT:-$ANCESTRY_DIR/pc1_pc2_clusters.svg}"
SPLIT_OUT="${SPLIT_OUT:-$SPLIT_DIR/disease_specific_split_seed${SEED}.csv}"
SPLIT_DIAGNOSTICS="${SPLIT_DIAGNOSTICS:-$CV_DIR/disease_specific_split_summary.csv}"
QC_CONFIG="${QC_CONFIG:-configs/qc_dream_training.yaml}"
GENOTYPE_QC_COMMANDS="${GENOTYPE_QC_COMMANDS:-$GENOTYPE_QC_DIR/genotype_qc.commands.sh}"
GENOTYPE_SAMPLE_PSAM="${GENOTYPE_SAMPLE_PSAM:-$GENOTYPE_QC_DIR/qc_final/all_chromosomes.qc.psam}"

mkdir -p \
  "$OUT_DIR" \
  "$REPORT_DIR" \
  "$SPLIT_DIR" \
  "$ANCESTRY_DIR" \
  "$CV_DIR" \
  "$BASELINE_OUT_DIR" \
  "$GENOTYPE_QC_DIR" \
  "$SAMPLE_SET_DIR" \
  "$UV_CACHE_DIR" \
  logs/preprocess

echo "Started decipEHR preprocessing at $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "Input phenotype: $PHENOTYPE"
echo "Validation covariates: $VALIDATION_COVARIATES"
echo "Prepared phenotype: $PREPARED_PHENOTYPE"
echo "Labels: $LABELS"
echo "Covariates: $COVARIATES"
echo "Eligibility config: $ELIGIBILITY_CONFIG"
echo "Sample set dir: $SAMPLE_SET_DIR"
echo "Genotype sample PSAM: $GENOTYPE_SAMPLE_PSAM"
echo "UV cache: $UV_CACHE_DIR"
echo "Partition snapshot:"
sinfo -o "%P %G %D %t" || true
echo ""

test -f "$PHENOTYPE" || { echo "ERROR: Missing phenotype file: $PHENOTYPE" >&2; exit 1; }
test -f "$VALIDATION_COVARIATES" || { echo "ERROR: Missing validation covariates: $VALIDATION_COVARIATES" >&2; exit 1; }
test -f "$ELIGIBILITY_CONFIG" || { echo "ERROR: Missing eligibility config: $ELIGIBILITY_CONFIG" >&2; exit 1; }

if [[ "$RUN_UV_SYNC" == "1" ]]; then
  uv sync --extra dev
fi

echo "[1/6] Preparing phenotype and covariate table"
uv run --extra dev python scripts/prepare_data.py \
  --phenotype "$PHENOTYPE" \
  --patient-col "$PATIENT_COL" \
  --labels "$LABELS" \
  --covariates "$COVARIATES" \
  --out "$PREPARED_PHENOTYPE" \
  --missingness-report "$MISSINGNESS_REPORT" \
  --label-report "$LABEL_REPORT" \
  --feature-report "$FEATURE_REPORT" \
  --feature-count-report "$FEATURE_COUNT_REPORT"
test -s "$PREPARED_PHENOTYPE"

if [[ "$RUN_ANCESTRY" == "1" ]]; then
  echo "[2/6] Creating ancestry clusters from PCs"
  uv run --extra dev python scripts/create_ancestry_clusters.py \
    --phenotype "$PREPARED_PHENOTYPE" \
    --patient-col "$PATIENT_COL" \
    --pc-cols "$PC_COLS" \
    --labels "$LABELS" \
    --age-col "$AGE_COL" \
    --sex-col "$SEX_COL" \
    --clusters-out "$ANCESTRY_CLUSTERS" \
    --report-out "$ANCESTRY_REPORT" \
    --quality-out "$ANCESTRY_QUALITY" \
    --plot-out "$ANCESTRY_PLOT"
  test -s "$ANCESTRY_CLUSTERS"
else
  echo "[2/6] Skipping ancestry clustering"
fi

if [[ "$RUN_SPLITS" == "1" ]]; then
  echo "[3/6] Creating disease-specific patient-level CV folds"
  split_args=(
    uv run --extra dev python scripts/make_splits.py
    --phenotype "$PREPARED_PHENOTYPE"
    --patient-col "$PATIENT_COL"
    --labels "$LABELS"
    --n-folds "$N_FOLDS"
    --seed "$SEED"
    --disease-specific
    --age-col "$AGE_COL"
    --sex-col "$SEX_COL"
    --sex-balance-labels "$SEX_BALANCE_LABELS"
    --eligibility-config "$ELIGIBILITY_CONFIG"
    --diagnostics-out "$SPLIT_DIAGNOSTICS"
    --out "$SPLIT_OUT"
  )
  if [[ "$RUN_ANCESTRY" == "1" ]]; then
    split_args+=(--ancestry-clusters "$ANCESTRY_CLUSTERS")
  fi
  "${split_args[@]}"
  test -s "$SPLIT_OUT"
else
  echo "[3/6] Skipping CV split creation"
fi

if [[ "$RUN_BASELINE_CV" == "1" ]]; then
  echo "[4/6] Running covariate-only baseline CV"
  test -s "$SPLIT_OUT" || { echo "ERROR: Baseline CV requires split file: $SPLIT_OUT" >&2; exit 1; }
  uv run --extra dev python scripts/run_cv.py \
    --phenotype "$PREPARED_PHENOTYPE" \
    --splits "$SPLIT_OUT" \
    --patient-col "$PATIENT_COL" \
    --labels "$LABELS" \
    --covariates "$COVARIATES" \
    --eligibility-config "$ELIGIBILITY_CONFIG" \
    --out-dir "$BASELINE_OUT_DIR"
else
  echo "[4/6] Skipping baseline CV; set RUN_BASELINE_CV=1 to run it"
fi

if [[ "$RUN_GENOTYPE_QC_TEMPLATE" == "1" ]]; then
  echo "[5/6] Writing genotype QC command template"
  uv run --extra dev python scripts/write_genotype_qc_commands.py \
    --config "$QC_CONFIG" \
    --out "$GENOTYPE_QC_COMMANDS"
  test -s "$GENOTYPE_QC_COMMANDS"
  if [[ "$RUN_GENOTYPE_QC" == "1" ]]; then
    echo "Running genotype QC command template"
    uv run --extra dev bash "$GENOTYPE_QC_COMMANDS"
  else
    echo "Genotype QC commands written to: $GENOTYPE_QC_COMMANDS"
    echo "Set RUN_GENOTYPE_QC=1 to run the PLINK2 QC commands in this job."
  fi
else
  echo "[5/6] Skipping genotype QC command template"
fi

if [[ "$RUN_SAMPLE_SETS" == "1" ]]; then
  if [[ -s "$GENOTYPE_SAMPLE_PSAM" ]]; then
    echo "[6/6] Creating labeled training and unlabelled validation sample sets"
    uv run --extra dev python scripts/make_sample_sets.py \
      --genotype-psam "$GENOTYPE_SAMPLE_PSAM" \
      --phenotype "$PREPARED_PHENOTYPE" \
      --validation-covariates "$VALIDATION_COVARIATES" \
      --patient-col "$PATIENT_COL" \
      --out-dir "$SAMPLE_SET_DIR"
  elif [[ "$RUN_GENOTYPE_QC" == "1" ]]; then
    echo "ERROR: Sample set creation requires genotype PSAM: $GENOTYPE_SAMPLE_PSAM" >&2
    exit 1
  else
    echo "[6/6] Skipping sample sets; genotype PSAM not available: $GENOTYPE_SAMPLE_PSAM"
  fi
else
  echo "[6/6] Skipping sample set creation"
fi

echo ""
echo "Preprocessing outputs:"
echo "  $PREPARED_PHENOTYPE"
echo "  $MISSINGNESS_REPORT"
echo "  $LABEL_REPORT"
echo "  $FEATURE_REPORT"
echo "  $FEATURE_COUNT_REPORT"
if [[ "$RUN_ANCESTRY" == "1" ]]; then
  echo "  $ANCESTRY_CLUSTERS"
  echo "  $ANCESTRY_REPORT"
  echo "  $ANCESTRY_QUALITY"
  echo "  $ANCESTRY_PLOT"
fi
if [[ "$RUN_SPLITS" == "1" ]]; then
  echo "  $SPLIT_OUT"
  echo "  $SPLIT_DIAGNOSTICS"
fi
if [[ "$RUN_GENOTYPE_QC_TEMPLATE" == "1" ]]; then
  echo "  $GENOTYPE_QC_COMMANDS"
fi
if [[ "$RUN_SAMPLE_SETS" == "1" && -s "$GENOTYPE_SAMPLE_PSAM" ]]; then
  echo "  $SAMPLE_SET_DIR/sample_sets.csv"
  echo "  $SAMPLE_SET_DIR/sample_set_counts.csv"
  echo "  $SAMPLE_SET_DIR/labeled_train.keep.psam"
  echo "  $SAMPLE_SET_DIR/unlabelled_validation.keep.psam"
fi
echo "Finished decipEHR preprocessing at $(date)"
