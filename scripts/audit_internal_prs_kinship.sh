#!/usr/bin/env bash
# Outcome-blind relatedness audit on the existing frozen ancestry marker panel.
set -euo pipefail
umask 077
cd /gpfs/data/shenlab/Jiajian/HW/decipehr
module load plink/2.0-20260228
out=outputs/internal_prs/nyu_20261011/kinship
mkdir -p "$out"
test ! -e "$out/king.kin0"
plink2 --pfile outputs/final_genotype/qc_final/train_val.common \
  --extract outputs/external_pgs/discodivas/projection_1000g_refit_v2_20261010/target.snplist \
  --autosome --maf 0.05 --geno 0.02 --indep-pairwise 200 50 0.2 \
  --threads 4 --memory 4000 --out "$out/prune"
plink2 --pfile outputs/final_genotype/qc_final/train_val.common \
  --extract "$out/prune.prune.in" --make-king-table --king-table-filter 0.0442 \
  --threads 4 --memory 4000 --out "$out/king"
