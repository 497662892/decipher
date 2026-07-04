#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180GB
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_medium
#SBATCH --job-name=patient_ancestry_1000g
#SBATCH --output=logs/ancestry_1000g/patient_ancestry-%J.log

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p logs/ancestry_1000g outputs/ancestry_1000g outputs/uv-cache
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_DIR/outputs/uv-cache}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if command -v module >/dev/null 2>&1; then
  module load plink/2.0-20260228 || module load plink/2.0 || true
fi

if ! command -v plink2 >/dev/null 2>&1 && [[ -n "${PLINK2_DIR:-}" && -x "$PLINK2_DIR/plink2" ]]; then
  export PATH="$PLINK2_DIR:$PATH"
fi

PLINK2="${PLINK2:-plink2}"
PYTHON="${PYTHON:-.venv/bin/python}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

TARGET_PREFIX="${TARGET_PREFIX:-outputs/genotype_qc/qc_final/all_chromosomes.qc}"
REF_PREFIX="${REF_PREFIX:-data/resources/1000g/grch38/1000g_GRCh38}"
PANEL="${PANEL:-data/resources/1000g/grch38/1000g_panel.tsv}"
OUT_DIR="${OUT_DIR:-outputs/ancestry_1000g}"
N_PCS_FIT="${N_PCS_FIT:-20}"
N_PCS_DISTANCE="${N_PCS_DISTANCE:-10}"
HARD_MARGIN="${HARD_MARGIN:-0.5}"
SOFT_TAU="${SOFT_TAU:-1.0}"

HARM_DIR="$OUT_DIR/harmonization"
PCA_DIR="$OUT_DIR/pca"
ANCESTRY_DIR="$OUT_DIR/ancestry"
mkdir -p "$HARM_DIR" "$PCA_DIR" "$ANCESTRY_DIR"

for suffix in pgen pvar psam; do
  test -s "${TARGET_PREFIX}.${suffix}" || { echo "Missing target ${TARGET_PREFIX}.${suffix}" >&2; exit 1; }
  test -s "${REF_PREFIX}.${suffix}" || { echo "Missing 1000G ${REF_PREFIX}.${suffix}" >&2; exit 1; }
done
test -x "$PYTHON" || { echo "Missing Python executable: $PYTHON" >&2; exit 1; }
test -s "$PANEL" || { echo "Missing 1000G panel $PANEL" >&2; exit 1; }

echo "Started patient ancestry projection at $(date)"
echo "Target prefix: $TARGET_PREFIX"
echo "1000G prefix: $REF_PREFIX"
echo "1000G panel: $PANEL"
echo "Output dir: $OUT_DIR"
echo "Fit PCs: $N_PCS_FIT; distance PCs: $N_PCS_DISTANCE"
echo "PLINK2: $PLINK2"
echo "Python: $PYTHON"

"$PYTHON" -u scripts/make_ancestry_overlap.py \
  --target-variants "${TARGET_PREFIX}.pvar" \
  --reference-variants "${REF_PREFIX}.pvar" \
  --out-dir "$HARM_DIR" \
  --prefix genome

"$PLINK2" \
  --pfile "$TARGET_PREFIX" \
  --extract "$HARM_DIR/genome.target_extract.snplist" \
  --make-pgen \
  --sort-vars \
  --threads "$THREADS" \
  --out "$PCA_DIR/target.overlap.raw_ids"

"$PLINK2" \
  --pfile "$PCA_DIR/target.overlap.raw_ids" \
  --update-name "$HARM_DIR/genome.target_rename_to_1000g.tsv" 2 1 1 \
  --make-pgen \
  --sort-vars \
  --threads "$THREADS" \
  --out "$PCA_DIR/target.overlap"

"$PLINK2" \
  --pfile "$REF_PREFIX" \
  --extract "$HARM_DIR/genome.1000g_extract.snplist" \
  --make-pgen \
  --sort-vars \
  --threads "$THREADS" \
  --out "$PCA_DIR/1000g.overlap"

HIGH_LD="$PCA_DIR/high_ld_regions_hg38.txt"
printf '6\t25000000\t34000000\tMHC\n' > "$HIGH_LD"

"$PLINK2" \
  --pfile "$PCA_DIR/1000g.overlap" \
  --maf 0.01 \
  --geno 0.02 \
  --exclude range "$HIGH_LD" \
  --indep-pairwise 200 50 0.1 \
  --threads "$THREADS" \
  --out "$PCA_DIR/1000g.ldprune"

"$PLINK2" \
  --pfile "$PCA_DIR/1000g.overlap" \
  --extract "$PCA_DIR/1000g.ldprune.prune.in" \
  --make-pgen \
  --sort-vars \
  --threads "$THREADS" \
  --out "$PCA_DIR/1000g.pca.snps"

"$PLINK2" \
  --pfile "$PCA_DIR/target.overlap" \
  --extract "$PCA_DIR/1000g.ldprune.prune.in" \
  --make-pgen \
  --sort-vars \
  --threads "$THREADS" \
  --out "$PCA_DIR/target.pca.snps"

"$PLINK2" \
  --pfile "$PCA_DIR/1000g.pca.snps" \
  --freq counts \
  --keep-founders \
  --pca "$N_PCS_FIT" allele-wts vcols=chrom,ref,alt \
  --threads "$THREADS" \
  --out "$PCA_DIR/1000g.ref_pca"

score_end_col=$((5 + N_PCS_FIT))

"$PLINK2" \
  --pfile "$PCA_DIR/1000g.pca.snps" \
  --read-freq "$PCA_DIR/1000g.ref_pca.acount" \
  --score "$PCA_DIR/1000g.ref_pca.eigenvec.allele" 2 5 header-read no-mean-imputation variance-standardize \
  --score-col-nums "6-${score_end_col}" \
  --threads "$THREADS" \
  --out "$PCA_DIR/1000g.projected"

"$PLINK2" \
  --pfile "$PCA_DIR/target.pca.snps" \
  --read-freq "$PCA_DIR/1000g.ref_pca.acount" \
  --score "$PCA_DIR/1000g.ref_pca.eigenvec.allele" 2 5 header-read no-mean-imputation variance-standardize \
  --score-col-nums "6-${score_end_col}" \
  --threads "$THREADS" \
  --out "$PCA_DIR/target.projected"

"$PYTHON" -u scripts/compute_ancestry_distances.py \
  --ref-sscore "$PCA_DIR/1000g.projected.sscore" \
  --target-sscore "$PCA_DIR/target.projected.sscore" \
  --panel "$PANEL" \
  --out-dir "$ANCESTRY_DIR" \
  --prefix target_1000G \
  --n-pcs "$N_PCS_DISTANCE" \
  --hard-margin "$HARD_MARGIN" \
  --soft-tau "$SOFT_TAU"

echo "Finished patient ancestry projection at $(date)"
echo "Ancestry outputs:"
ls -1 "$ANCESTRY_DIR"/target_1000G_*
