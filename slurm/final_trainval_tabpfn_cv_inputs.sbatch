#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu4_dev,gl40s_dev
#SBATCH --job-name=tabpfn_cv_inputs
#SBATCH --output=logs/tabpfn_final_trainval/%x-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache}"

LABELS="${LABELS:-cad,bc,T2D}"
FAMILIES="${FAMILIES:-pt,ldpred2,lassosum2,prscs,sdpr}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"
SELECTION_METHOD="${SELECTION_METHOD:-{family}_prs_covariate}"
SPLITS="${SPLITS:-data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv}"
OUT_DIR="${OUT_DIR:-outputs/tabpfn_cv_ensemble_inputs}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
PT_CV_ROOT="${PT_CV_ROOT:-}"
LDPRED2_CV_ROOT="${LDPRED2_CV_ROOT:-}"
LASSOSUM2_CV_ROOT="${LASSOSUM2_CV_ROOT:-}"
PRSCS_CV_ROOT="${PRSCS_CV_ROOT:-}"
SDPR_CV_ROOT="${SDPR_CV_ROOT:-}"

mkdir -p "$UV_CACHE_DIR" "$OUT_DIR" logs/tabpfn_final_trainval

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
if [[ -x "$PYTHON_BIN" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  PYTHON_CMD=(uv run --extra dev python)
fi

ALLOW_ARGS=()
if [[ "$ALLOW_MISSING" == "1" ]]; then
  ALLOW_ARGS=(--allow-missing)
fi

ROOT_ARGS=()
if [[ -n "$PT_CV_ROOT" ]]; then
  ROOT_ARGS+=(--pt-cv-root "$PT_CV_ROOT")
fi
if [[ -n "$LDPRED2_CV_ROOT" ]]; then
  ROOT_ARGS+=(--ldpred2-cv-root "$LDPRED2_CV_ROOT")
fi
if [[ -n "$LASSOSUM2_CV_ROOT" ]]; then
  ROOT_ARGS+=(--lassosum2-cv-root "$LASSOSUM2_CV_ROOT")
fi
if [[ -n "$PRSCS_CV_ROOT" ]]; then
  ROOT_ARGS+=(--prscs-cv-root "$PRSCS_CV_ROOT")
fi
if [[ -n "$SDPR_CV_ROOT" ]]; then
  ROOT_ARGS+=(--sdpr-cv-root "$SDPR_CV_ROOT")
fi

echo "Started TabPFN CV ensemble input build at $(date)"
"${PYTHON_CMD[@]}" scripts/build_prs_cv_ensemble_inputs.py \
  --labels "$LABELS" \
  --families "$FAMILIES" \
  --selection-metric "$SELECTION_METRIC" \
  --selection-method "$SELECTION_METHOD" \
  --splits "$SPLITS" \
  --out-dir "$OUT_DIR" \
  "${ROOT_ARGS[@]}" \
  "${ALLOW_ARGS[@]}"

echo "Finished TabPFN CV ensemble input build at $(date)"
