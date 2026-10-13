#!/usr/bin/env bash
# Preserve the first screen; regenerate hardcalls from dosages for the diagnostic.
set -euo pipefail
umask 077
cd /gpfs/data/shenlab/Jiajian/HW/decipehr
module load plink/2.0-20260228
out=outputs/internal_prs/nyu_20261011/kinship
test ! -e "$out/rehard.kin0"
plink2 --pfile outputs/final_genotype/qc_final/train_val.common \
  --extract outputs/external_pgs/discodivas/projection_1000g_refit_v2_20261010/target.snplist \
  --autosome --hard-call-threshold 0.1 --make-pgen erase-dosage \
  --threads 4 --memory 4000 --out "$out/rehard"
plink2 --pfile "$out/rehard" --sample-counts --missing --geno-counts \
  --threads 4 --memory 4000 --out "$out/rehard_qc"
plink2 --pfile "$out/rehard" --make-king-table --king-table-filter 0.0442 \
  --threads 4 --memory 4000 --out "$out/rehard"
