# Recommended Minimal Robust PRS QC Pipeline for Codex

## Summary

This is the recommended **current implementation target** for the PRS project.

The goal is **not** to build a full raw-genotype GWAS QC pipeline now.
The goal is to build a **minimal robust PRS-ready QC pipeline** that is:

1. Safe for imputed PGEN data.
2. Simple enough to implement quickly.
3. Hard to misuse.
4. Extensible later if we need paper-grade QC.
5. Focused on the most important PRS failure mode: allele direction mismatch.

Key idea:

```text
Do variant-level target genotype QC first.
Then harmonize GWAS summary statistics to the target genotype before PRS scoring.
```

Do not over-complicate sample-level QC, HWE, or strand-flip rescue in the first implementation.

---

# 1. Recommended current version

## Version name

```text
V1: minimal robust PRS-ready QC
```

## What V1 should do

```text
Target genotype QC:
  1. Filter variants by imputation quality.
  2. Filter variants by allele frequency.
  3. Keep only biallelic A/C/G/T SNPs.
  4. Use dosage-aware variant missingness filtering for imputed PGENs.
  5. Merge chromosomes.
  6. Generate QC reports.

PRS harmonization:
  1. Match GWAS summary statistics to target genotype.
  2. Require allele-pair consistency.
  3. Remove A/T and C/G ambiguous SNPs by default.
  4. Keep strand flipping disabled by default.
  5. Output a PLINK2 score file: SNP, A1, BETA.

PRS scoring:
  1. Use PLINK2 --score with the harmonized score file.
```

## What V1 should NOT do by default

```text
Do not apply sample removal by default.
Do not run per-chromosome --mind.
Do not perform HWE filtering by default.
Do not rescue ambiguous A/T or C/G SNPs.
Do not enable strand flipping by default.
Do not do relatedness filtering, sex check, or heterozygosity filtering now.
```

Rationale:

```text
Allele harmonization is required for PRS correctness.

HWE and sample-level QC are useful later, but they are not the bottleneck for
the first robust PRS baseline, especially if the target genotype data have
already gone through upstream imputation and basic QC.
```

---

# 2. Recommended default config

Update or create:

```text
configs/qc.yaml
```

Recommended config:

```yaml
variant_qc:
  min_info: 0.8
  min_maf: 0.01
  max_af: 0.99

  max_variant_missingness: 0.05
  genotype_missingness_mode: dosage

  keep_snps_only: true
  biallelic_only: true

  info_score:
    variant_col: ID
    info_col: info_IMPUTE
    freq_col: MAF
    freq_is_maf: true

  apply_hwe: false
  hwe_p: 1e-6
  controls_keep_file: null

sample_qc:
  apply_mind: false
  max_sample_missingness: 0.02
  keep_file: null

prs_alignment:
  remove_ambiguous_snps: true
  require_allele_match: true
  remove_duplicate_gwas_variants: true
  keep_only_acgt: true

  match_strategy: id_then_position_alleles

  allow_strand_flip: false
  strand_flip_requires_frequency_check: true
  max_frequency_diff_for_flip: 0.15

paths:
  plink2: plink2

  raw_pgen_dir: data/pgen
  raw_pgen_pattern: chr{chr}

  info_score_dir: data/info_scores
  info_score_pattern: chr{chr}.info_scores.txt

  qc_work_dir: outputs/genotype_qc

  gwas_sumstats: null
  harmonized_score_dir: outputs/prs_scores
```

Backward compatibility requirement:

```text
If old top-level config fields exist, keep supporting them:

min_info
min_maf
max_variant_missingness

Map them internally to:

variant_qc.min_info
variant_qc.min_maf
variant_qc.max_variant_missingness
```

---

# 3. Directory structure

Recommended outputs:

```text
outputs/
  genotype_qc/
    run_genotype_qc.sh

    lists/
      chr1.info_maf.snplist
      chr1.info_maf.summary.json

    qc_step1/
      chr1.step1.pgen
      chr1.step1.pvar
      chr1.step1.psam

    qc_final/
      chr1.qc.pgen
      chr1.qc.pvar
      chr1.qc.psam
      qc_pfiles.txt
      all_chromosomes.qc.pgen
      all_chromosomes.qc.pvar
      all_chromosomes.qc.psam

    reports/
      chr1.step1_missingness.smiss
      chr1.step1_missingness.vmiss
      all_chromosomes.qc_summary.*

  prs_scores/
    trait.harmonized.score.tsv
    trait.harmonized.summary.json
    run_score_trait.sh
    trait.prs.sscore
```

---

# 4. Script 1: update `scripts/filter_info_scores.py`

## Goal

Read official imputation info score file and output a variant snplist.

## Required CLI

```bash
python scripts/filter_info_scores.py \
  --info-file data/info_scores/chr1.info_scores.txt \
  --out outputs/genotype_qc/lists/chr1.info_maf.snplist \
  --min-info 0.8 \
  --min-maf 0.01 \
  --max-af 0.99 \
  --variant-id-col ID \
  --info-col info_IMPUTE \
  --freq-col MAF \
  --freq-is-maf true
```

## Backward-compatible aliases

Keep these aliases if they already exist:

```text
--info-score      alias for --info-file
--variant-col     alias for --variant-id-col
--maf-col         alias for --freq-col and implies --freq-is-maf true
```

## Required filtering logic

If the frequency column is true MAF:

```python
keep = (info >= min_info) and (maf >= min_maf)
```

If the frequency column is allele frequency rather than MAF:

```python
maf = min(af, 1 - af)
keep = (info >= min_info) and (maf >= min_maf)
```

If the project wants a simple AF bound:

```python
keep = (info >= min_info) and (af >= min_maf) and (af <= max_af)
```

Recommended:

```text
Prefer computing MAF = min(AF, 1 - AF) when freq_is_maf is false.
```

## Output

Main output:

```text
One variant ID per line.
```

Example:

```text
rs123
rs456
chr1:12345:A:G
```

Also write a JSON summary next to the snplist:

```json
{
  "input_file": "chr1.info_scores.txt",
  "n_total": 1234567,
  "n_pass_info": 1000000,
  "n_pass_maf": 900000,
  "n_pass_all": 850000,
  "min_info": 0.8,
  "min_maf": 0.01
}
```

---

# 5. Script 2: update `scripts/write_genotype_qc_commands.py`

## Goal

Generate a shell script containing PLINK2 genotype QC commands.

This script should generate commands only.
It should not execute PLINK2 directly.

## Required CLI

```bash
python scripts/write_genotype_qc_commands.py \
  --config configs/qc.yaml \
  --chromosomes 1-22 \
  --out outputs/genotype_qc/run_genotype_qc.sh
```

## Backward compatibility

Existing CLI usage should continue to work if present, e.g.:

```text
--pfile-pattern
--info-pattern
--out-dir
--min-info
--min-maf
--geno
```

New `--config` mode should be additive, not a breaking replacement.

## Chromosome parser

Support:

```text
1-22   -> 1, 2, ..., 22
1,2,3  -> 1, 2, 3
```

## Shell script header

Generated script should start with:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/genotype_qc/lists
mkdir -p outputs/genotype_qc/qc_step1
mkdir -p outputs/genotype_qc/qc_final
mkdir -p outputs/genotype_qc/reports
mkdir -p outputs/genotype_qc/logs
```

---

# 6. Genotype QC command templates

## Step 1. Generate INFO/MAF snplist

For each chromosome:

```bash
python scripts/filter_info_scores.py \
  --info-file ${INFO_SCORE_DIR}/chr${CHR}.info_scores.txt \
  --out ${QC_WORK_DIR}/lists/chr${CHR}.info_maf.snplist \
  --min-info ${MIN_INFO} \
  --min-maf ${MIN_MAF} \
  --max-af ${MAX_AF} \
  --variant-id-col ${INFO_VARIANT_COL} \
  --info-col ${INFO_COL} \
  --freq-col ${INFO_FREQ_COL} \
  --freq-is-maf ${INFO_FREQ_IS_MAF}
```

## Step 2. First-pass variant QC

For imputed PGENs, default command should include `dosage`:

```bash
${PLINK2} \
  --pfile ${RAW_PGEN_DIR}/chr${CHR} \
  --extract ${QC_WORK_DIR}/lists/chr${CHR}.info_maf.snplist \
  --snps-only just-acgt \
  --max-alleles 2 \
  --geno ${MAX_VARIANT_MISSINGNESS} dosage \
  --make-pgen \
  --out ${QC_WORK_DIR}/qc_step1/chr${CHR}.step1
```

If `genotype_missingness_mode: hardcall`, generate:

```bash
--geno ${MAX_VARIANT_MISSINGNESS}
```

If `keep_snps_only: false`, omit:

```text
--snps-only just-acgt
```

If `biallelic_only: false`, omit:

```text
--max-alleles 2
```

## Step 3. Missingness reports

Always generate missingness reports:

```bash
${PLINK2} \
  --pfile ${QC_WORK_DIR}/qc_step1/chr${CHR}.step1 \
  --missing \
  --out ${QC_WORK_DIR}/reports/chr${CHR}.step1_missingness
```

## Step 4. Final per-chromosome output

Default V1 does not apply HWE or sample removal.
So final output can simply copy/rewrite the step1 pgen:

```bash
${PLINK2} \
  --pfile ${QC_WORK_DIR}/qc_step1/chr${CHR}.step1 \
  --make-pgen \
  --out ${QC_WORK_DIR}/qc_final/chr${CHR}.qc
```

Important:

```text
Do not run per-chromosome --mind.
Do not remove samples by default.
```

If a shared sample keep file is explicitly configured later, apply the same keep file to every chromosome:

```bash
${PLINK2} \
  --pfile ${QC_WORK_DIR}/qc_step1/chr${CHR}.step1 \
  --keep ${SAMPLE_KEEP_FILE} \
  --make-pgen \
  --out ${QC_WORK_DIR}/qc_final/chr${CHR}.qc
```

But V1 default should not do this.

## Step 5. Create pmerge list

Generate:

```text
${QC_WORK_DIR}/qc_final/qc_pfiles.txt
```

Contents:

```text
outputs/genotype_qc/qc_final/chr1.qc
outputs/genotype_qc/qc_final/chr2.qc
...
outputs/genotype_qc/qc_final/chr22.qc
```

## Step 6. Merge chromosomes

```bash
${PLINK2} \
  --pmerge-list ${QC_WORK_DIR}/qc_final/qc_pfiles.txt \
  --sort-vars \
  --make-pgen \
  --out ${QC_WORK_DIR}/qc_final/all_chromosomes.qc
```

## Step 7. Post-QC reports

```bash
${PLINK2} \
  --pfile ${QC_WORK_DIR}/qc_final/all_chromosomes.qc \
  --missing \
  --freq \
  --hardy \
  --pgen-info \
  --out ${QC_WORK_DIR}/reports/all_chromosomes.qc_summary
```

Note:

```text
The --freq report is useful later if we want frequency-based allele checks.
For V1, frequency-based strand-flip rescue remains disabled by default.
```

---

# 7. Script 3: add `scripts/harmonize_gwas_for_prs.py`

## Goal

Generate a harmonized PLINK2 score file from GWAS summary statistics and the final target genotype `.pvar`.

This script should not modify genotype files.

Output:

```text
SNP  A1  BETA
```

where:

```text
SNP  = target genotype variant ID
A1   = GWAS effect allele after harmonization
BETA = GWAS effect size for A1
```

## Required CLI

```bash
python scripts/harmonize_gwas_for_prs.py \
  --gwas gwas/trait.cleaned.tsv \
  --pvar outputs/genotype_qc/qc_final/all_chromosomes.qc.pvar \
  --out outputs/prs_scores/trait.harmonized.score.tsv \
  --snp-col SNP \
  --ea-col EA \
  --oa-col OA \
  --beta-col BETA \
  --remove-ambiguous true \
  --match-strategy id_then_position_alleles \
  --allow-strand-flip false
```

Optional columns for position fallback:

```text
--chr-col CHR
--bp-col BP
```

Optional later columns for frequency checks:

```text
--gwas-af-col EAF
--target-af-file outputs/genotype_qc/reports/all_chromosomes.qc_summary.afreq
```

Frequency-based checks should not be required for V1.

---

## Required `.pvar` parsing

Support PLINK2 `.pvar` columns:

```text
#CHROM or CHROM
POS
ID
REF
ALT
```

Normalize:

```python
CHROM = str(CHROM)
POS = int(POS)
ID = str(ID)
REF = REF.upper()
ALT = ALT.upper()
```

---

## Matching strategy

Use:

```text
id_then_position_alleles
```

Detailed logic:

```text
1. First try to match GWAS SNP to target pvar ID.
2. For GWAS rows not matched by ID:
   If CHR and BP are provided, fallback to CHR/POS matching.
3. For position fallback:
   Require allele-pair consistency between GWAS EA/OA and target REF/ALT.
4. If multiple target variants match the same ID or same CHR/POS/allele pair:
   remove that GWAS row and count it as duplicate/multimatch.
```

Rationale:

```text
GWAS summary statistics and target pvar files may use different variant ID conventions.
ID-only matching is sometimes insufficient.
Position + allele fallback is safer.
```

---

## Allele harmonization rules

For each GWAS row and target pvar row:

```text
GWAS:
  SNP
  EA = effect allele
  OA = other allele
  BETA

Target:
  ID
  REF
  ALT
```

Normalize alleles:

```python
EA = EA.upper()
OA = OA.upper()
REF = REF.upper()
ALT = ALT.upper()
```

If `keep_only_acgt: true`, remove if any allele is not one of:

```python
{"A", "C", "G", "T"}
```

Remove ambiguous SNPs by default:

```text
A/T
T/A
C/G
G/C
```

Definition:

```python
def is_ambiguous(a1, a2):
    pair = {a1, a2}
    return pair == {"A", "T"} or pair == {"C", "G"}
```

Allele pair matching:

```python
gwas_pair = {EA, OA}
target_pair = {REF, ALT}

if gwas_pair == target_pair:
    keep
else:
    remove
```

For V1:

```text
Do not strand flip.
Do not rescue ambiguous SNPs.
Do not flip beta just because EA is REF instead of ALT.
```

Important rule:

```text
For PLINK2 --score, A1 can be REF or ALT.

As long as A1 is the GWAS effect allele and BETA is the effect size for that
allele, beta should remain unchanged.
```

Example:

```text
GWAS:
  SNP rs123
  EA A
  OA G
  BETA +0.20

Target:
  ID rs123
  REF A
  ALT G

Output:
  rs123 A +0.20

Do NOT change beta to -0.20 just because A is REF.
```

---

## Harmonized score output

Tab-delimited:

```text
SNP    A1    BETA
rs123  A     0.023
rs456  G    -0.017
rs789  T     0.041
```

## Harmonization summary JSON

Write:

```text
outputs/prs_scores/trait.harmonized.summary.json
```

Required fields:

```json
{
  "n_gwas_input": 1000000,
  "n_after_required_columns": 990000,
  "n_after_duplicate_removal": 980000,
  "n_matched_to_target_pvar": 700000,
  "n_matched_by_id": 650000,
  "n_matched_by_position_alleles": 50000,
  "n_removed_non_acgt": 1000,
  "n_removed_ambiguous": 80000,
  "n_removed_duplicate_or_multimatch": 100,
  "n_removed_allele_mismatch": 20000,
  "n_harmonized_output": 599000,
  "strand_flipping_enabled": false
}
```

Warnings:

```text
Warn if n_harmonized_output < 50,000.
Warn if more than 50% of matched variants are removed due to allele mismatch.
Warn if ID matching is very low and CHR/BP fallback is unavailable.
Warn if pvar IDs are duplicated.
Warn if GWAS SNP IDs are duplicated.
```

---

# 8. Script 4: add `scripts/write_prs_score_commands.py`

## Goal

Generate PLINK2 scoring commands after GWAS harmonization.

## CLI

```bash
python scripts/write_prs_score_commands.py \
  --config configs/qc.yaml \
  --score-file outputs/prs_scores/trait.harmonized.score.tsv \
  --trait-name CAD \
  --out outputs/prs_scores/run_score_CAD.sh
```

## Command template

```bash
${PLINK2} \
  --pfile ${QC_WORK_DIR}/qc_final/all_chromosomes.qc \
  --score ${SCORE_FILE} 1 2 3 header-read cols=+scoresums \
  --out ${HARMONIZED_SCORE_DIR}/${TRAIT_NAME}.prs
```

Expected output:

```text
outputs/prs_scores/CAD.prs.sscore
```

---

# 9. Validation requirements

Codex should implement checks for:

```text
1. Config exists.
2. Required config fields are present or have defaults.
3. Input pgen prefix exists:
   .pgen, .pvar, .psam

4. Info score file exists for each chromosome.
5. Generated snplist is not empty.
6. Generated shell scripts are executable or clearly runnable.
7. GWAS summary statistics required columns exist.
8. Target pvar required columns exist.
9. Harmonized score file has at least one variant.
10. Summary JSON files are written.
```

Recommended sanity warnings:

```text
Warn if variant count after INFO/MAF filtering is unexpectedly low.
Warn if harmonized score file contains <50,000 variants.
Warn if allele mismatch removes >50% of matched GWAS variants.
Warn if many duplicated IDs are detected in GWAS or pvar.
Warn if strand flipping is requested in V1 mode.
```

---

# 10. Acceptance criteria

Implementation is acceptable if:

```text
1. `write_genotype_qc_commands.py` can generate a runnable shell script.

2. The genotype QC script includes:
   - INFO/MAF snplist generation
   - --snps-only just-acgt
   - --max-alleles 2
   - --geno 0.05 dosage by default
   - per-chromosome missingness reports
   - chromosome merge with --pmerge-list
   - post-QC reports

3. The genotype QC script does NOT:
   - run per-chromosome --mind
   - perform HWE by default
   - remove samples by default

4. `harmonize_gwas_for_prs.py` outputs:
   - SNP, A1, BETA score file
   - summary JSON
   - removes ambiguous A/T and C/G SNPs by default
   - keeps strand flipping disabled by default
   - does not flip beta when EA is REF

5. `write_prs_score_commands.py` generates:
   - PLINK2 --score command using the harmonized score file
   - output path for trait.prs.sscore

6. Existing older CLI and config usage continue to work when possible.
```

---

# 11. Recommended Codex prompt

Copy this into Codex:

```text
Please implement the V1 minimal robust PRS-ready QC pipeline described in this spec.

Main goal:
Build a simple, safe, command-generation-based pipeline for target genotype QC and PRS allele harmonization.

Please implement:

1. Update configs/qc.yaml to support:
   - variant_qc
   - sample_qc
   - prs_alignment
   - paths

2. Preserve backward compatibility with old top-level keys:
   - min_info
   - min_maf
   - max_variant_missingness

3. Update scripts/filter_info_scores.py:
   - generate INFO/MAF snplists
   - support freq_is_maf
   - support aliases: --info-score, --variant-col, --maf-col
   - write JSON summaries

4. Update scripts/write_genotype_qc_commands.py:
   - generate commands only; do not execute PLINK2
   - generate INFO/MAF snplist commands
   - run PLINK2 with:
       --snps-only just-acgt
       --max-alleles 2
       --geno 0.05 dosage by default
   - generate missingness reports
   - merge chromosomes with --pmerge-list and --sort-vars
   - generate post-QC reports
   - do not run per-chromosome --mind
   - do not apply HWE by default

5. Add scripts/harmonize_gwas_for_prs.py:
   - parse GWAS summary statistics
   - parse target .pvar
   - match by ID first
   - fallback to CHR/BP + allele-pair matching when available
   - remove non-A/C/G/T variants
   - remove ambiguous A/T and C/G SNPs by default
   - require allele-pair consistency
   - keep strand flipping disabled by default
   - output PLINK2 score file with columns SNP, A1, BETA
   - write harmonization summary JSON

6. Add scripts/write_prs_score_commands.py:
   - generate PLINK2 --score command using:
       --score score.tsv 1 2 3 header-read cols=+scoresums

7. Add validation and helpful error messages:
   - missing input files
   - missing required columns
   - empty snplist
   - empty harmonized score file
   - duplicated GWAS or pvar IDs

8. Keep the implementation simple and maintainable.
   Do not implement full HWE/sample-level QC/strand-flip rescue now.
   Those are later optional extensions.
```

---

# 12. Later optional extensions

Do not implement these in V1 unless specifically requested:

```text
1. Controls-only HWE filtering.
2. Shared sample keep file generation.
3. Sample-level missingness filtering.
4. Sex check.
5. Heterozygosity outlier detection.
6. Relatedness filtering.
7. Ancestry outlier filtering.
8. Frequency-based strand-flip rescue.
9. Frequency-based ambiguous SNP rescue.
10. INFO/MAF threshold sensitivity analysis.
```

Recommended later order:

```text
V2:
  Add controls-only HWE if hardcalls are reliable.

V3:
  Add sample-level QC only if sample removal is allowed.

V4:
  Add frequency-based strand-flip rescue after confirming genome build and allele frequency availability.
```

---

# 13. Short explanation for README

```text
We use a minimal robust PRS-ready QC pipeline. We retain well-imputed variants
(INFO >= 0.8) with MAF >= 1%, restrict analyses to biallelic A/C/G/T SNPs, and
remove variants with dosage-aware missingness >5%. We do not remove samples or
apply HWE filtering by default in the first baseline. Before PRS scoring, GWAS
summary statistics are harmonized to the target genotype data by matching variant
IDs and genomic positions where available, requiring allele-pair consistency, and
removing ambiguous A/T and C/G variants by default. PRS is then computed using
PLINK2 --score with the harmonized effect allele and effect-size columns.
```

---

# 14. Key PLINK2 command meanings

```text
--snps-only just-acgt
  Keep simple A/C/G/T SNPs.

--max-alleles 2
  Keep biallelic variants.

--geno 0.05 dosage
  Remove variants with missingness >5%, counting available dosages as nonmissing.

--score score.tsv 1 2 3 header-read
  Use column 1 as variant ID, column 2 as score/effect allele, and column 3 as beta.
```

---

# 15. Official references

PLINK2 filtering documentation:
https://www.cog-genomics.org/plink/2.0/filter

PLINK2 scoring documentation:
https://www.cog-genomics.org/plink/2.0/score

PLINK2 data management documentation:
https://www.cog-genomics.org/plink/2.0/data

PLINK2 order of operations:
https://www.cog-genomics.org/plink/2.0/order
