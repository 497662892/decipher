# PRS Genotype QC and Allele Harmonization: Codex Implementation Spec

## Purpose

Implement a robust PRS-ready target genotype QC pipeline.

Current pipeline already generates PLINK2 commands for genotype variant QC. The next version should extend it to:

1. Keep high-quality imputed variants.
2. Keep only simple biallelic A/C/G/T SNPs.
3. Optionally perform HWE filtering, preferably in controls only.
4. Merge QC-ed chromosomes.
5. Generate QC summary reports.
6. Add a separate PRS allele harmonization step before PLINK2 `--score`.

The key distinction:

```text
Genotype QC checks whether a target genotype variant is high quality.

Allele harmonization checks whether the GWAS effect allele is aligned with the target genotype allele coding before PRS scoring.
```

---

## Current known files

```text
scripts/write_genotype_qc_commands.py
scripts/filter_info_scores.py
configs/qc.yaml
```

Current default config:

```yaml
min_info: 0.8
min_maf: 0.01
max_variant_missingness: 0.05
```

Current intended command pattern:

```bash
plink2 \
  --pfile ... \
  --extract snplist \
  --geno 0.05 \
  --make-pgen \
  --out chrN.qc
```

Need to extend this.

---

# 1. Recommended target behavior

## 1.1 Minimal robust baseline

This should be the default behavior.

```text
Variant QC:
  INFO >= 0.8
  MAF >= 0.01
  AF <= 0.99 if using allele frequency column
  keep only A/C/G/T SNPs
  keep only biallelic variants
  variant missingness <= 0.05
  use PLINK2 dosage-aware missingness for imputed PGENs when dosages are present

PRS harmonization:
  match GWAS variants to target genotype variants
  remove allele mismatches
  remove ambiguous A/T and C/G SNPs by default
  keep strand-flip matching disabled by default unless allele convention and frequencies support it
  output harmonized PLINK2 score file
```

## 1.2 Optional robust mode

If controls are available:

```text
HWE:
  perform HWE filtering in controls only
  use HWE p-value threshold 1e-6
  output HWE-passed snplist
  apply HWE-passed snplist to full target dataset
```

## 1.3 Do not remove samples by default

For challenge/benchmark settings, sample removal may be undesirable.

Default:

```yaml
sample_qc:
  apply_mind: false
  max_sample_missingness: 0.02
  keep_file: null
```

Generate missingness reports, but do not remove samples by default.

Important:

```text
Do not run per-chromosome --mind independently.

If sample removal is enabled, use one shared genome-wide sample keep file and apply
that same keep file to every chromosome before merge/scoring. This avoids different
chromosomes having different retained sample sets.
```

---

# 2. Proposed config format

Update `configs/qc.yaml` to support the following structure.

```yaml
variant_qc:
  min_info: 0.8
  min_maf: 0.01
  max_af: 0.99
  max_variant_missingness: 0.05
  genotype_missingness_mode: dosage  # dosage for imputed PGENs; hardcall omits the modifier

  keep_snps_only: true
  biallelic_only: true

  info_score:
    variant_col: col0
    info_col: info_IMPUTE
    freq_col: MAF
    freq_is_maf: true

  apply_hwe: false
  hwe_p: 1e-6
  hwe_controls_only: true
  controls_keep_file: null
  hwe_midp: true
  hwe_keep_fewhet: true

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
If old config keys exist at top level:
  min_info
  min_maf
  max_variant_missingness
  info_score.variant_col
  info_score.info_col
  info_score.maf_col

The scripts should still work by mapping them to:
  variant_qc.min_info
  variant_qc.min_maf
  variant_qc.max_variant_missingness
  variant_qc.info_score.variant_col
  variant_qc.info_score.info_col
  variant_qc.info_score.freq_col
```

CLI backward compatibility requirement:

```text
Existing command-line usage should continue to work:

scripts/filter_info_scores.py:
  --info-score should remain an alias for --info-file.
  --variant-col should remain an alias for --variant-id-col.
  --maf-col should remain an alias for --freq-col with --freq-is-maf true.

scripts/write_genotype_qc_commands.py:
  Existing --pfile-pattern, --info-pattern, --out-dir, --min-info, --min-maf,
  and --geno arguments should still be accepted.
  New --config mode should be additive, not a breaking replacement.
```

---

# 3. Pipeline overview

Recommended command-generation pipeline:

```text
Step 1. Generate INFO/MAF snplists per chromosome.
Step 2. Run first-pass PLINK2 variant QC per chromosome.
Step 3. Generate missingness reports.
Step 4. Optionally apply one shared sample keep file if sample removal is enabled.
Step 5. Optionally run controls-only HWE filtering.
Step 6. Generate final per-chromosome QC pgen files.
Step 7. Merge final per-chromosome pgen files.
Step 8. Generate post-QC summary reports.
Step 9. Harmonize GWAS summary statistics against final target pvar.
Step 10. Run PLINK2 PRS scoring.
```

---

# 4. Detailed implementation tasks for Codex

## Task A. Update `scripts/filter_info_scores.py`

### Goal

Read official imputation info score file and output a variant snplist.

### Required inputs

```text
--info-file
--out
--min-info
--min-maf
--max-af
--variant-id-col
--info-col
--freq-col
--freq-is-maf
```

Backward-compatible aliases:

```text
--info-score      alias for --info-file
--variant-col     alias for --variant-id-col
--maf-col         alias for --freq-col and implies --freq-is-maf true unless overridden
```

### Suggested defaults

```text
variant-id-col: ID
info-col: info_IMPUTE
freq-col: MAF
freq-is-maf: true
min-info: 0.8
min-maf: 0.01
max-af: 0.99
```

### Required logic

If `freq_is_maf == true`:

```python
keep = (info >= min_info) and (maf >= min_maf)
```

If `freq_is_maf == false`:

```python
maf = min(af, 1 - af)
keep = (info >= min_info) and (maf >= min_maf) and (af <= max_af) and (af >= 1 - max_af if needed)
```

Simpler AF logic:

```python
keep = (info >= min_info) and (af >= min_maf) and (af <= max_af)
```

But if possible, prefer:

```python
maf = min(af, 1 - af)
keep = (info >= min_info) and (maf >= min_maf)
```

### Output

One variant ID per line:

```text
rs123
rs456
chr1:12345:A:G
```

### Also write a small JSON summary

Example:

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

Suggested output path:

```text
lists/chr${CHR}.info_maf.summary.json
```

---

## Task B. Update `scripts/write_genotype_qc_commands.py`

### Goal

Generate a shell script containing all PLINK2 genotype QC commands.

### Required behavior

The command generator should support:

```text
1. Per-chromosome info/MAF snplist creation
2. First-pass QC
3. Missingness report
4. Optional shared sample keep-file application
5. Optional HWE filtering
6. Final QC output
7. Genome-wide merge
8. Post-QC reports
```

It should support two input styles:

```text
Preferred new style:
  --config configs/qc.yaml --chromosomes 1-22 --out outputs/genotype_qc/run_genotype_qc.sh

Legacy style:
  --pfile-pattern ... --info-pattern ... --out-dir ... --out ...

The legacy style must keep working for existing tests and existing local command snippets.
```

### Required CLI

Example:

```bash
python scripts/write_genotype_qc_commands.py \
  --config configs/qc.yaml \
  --chromosomes 1-22 \
  --out outputs/genotype_qc/run_genotype_qc.sh
```

### Suggested helper behavior

Parse chromosome argument:

```text
"1-22" -> [1, 2, ..., 22]
"1,2,3" -> [1, 2, 3]
```

Create output directories in command script:

```bash
mkdir -p outputs/genotype_qc/lists
mkdir -p outputs/genotype_qc/qc_step1
mkdir -p outputs/genotype_qc/hwe
mkdir -p outputs/genotype_qc/qc_final
mkdir -p outputs/genotype_qc/reports
mkdir -p outputs/genotype_qc/logs
```

---

# 5. Exact command templates

## 5.1 Step 1: Generate INFO/MAF snplist

Command template:

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

---

## 5.2 Step 2: First-pass PLINK2 variant QC

Command template:

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

`dosage` should be included when `variant_qc.genotype_missingness_mode: dosage`.
If `genotype_missingness_mode: hardcall`, omit the modifier:

```text
--geno ${MAX_VARIANT_MISSINGNESS}
```

If `genotype_missingness_mode: hh-missing`, use:

```text
--geno ${MAX_VARIANT_MISSINGNESS} hh-missing
```

If `keep_snps_only: false`, omit:

```text
--snps-only just-acgt
```

If `biallelic_only: false`, omit:

```text
--max-alleles 2
```

---

## 5.3 Step 3: Missingness report

Always generate reports:

```bash
${PLINK2} \
  --pfile ${QC_WORK_DIR}/qc_step1/chr${CHR}.step1 \
  --missing \
  --out ${QC_WORK_DIR}/reports/chr${CHR}.step1_missingness
```

Sample removal should not be performed independently per chromosome.

Recommended v1 behavior:

```text
If sample_qc.apply_mind is false:
  Do not remove samples.

If sample_qc.apply_mind is true and sample_qc.keep_file is provided:
  Apply the same --keep file to every chromosome when writing final pgen files.

If sample_qc.apply_mind is true and sample_qc.keep_file is null:
  Fail with a clear error explaining that per-chromosome --mind is not allowed.
  A future implementation may create a genome-wide keep file first, but v1 should
  avoid chromosome-specific sample sets.
```

If not applying sample QC, downstream input should remain:

```text
${QC_WORK_DIR}/qc_step1/chr${CHR}.step1
```

If applying sample QC, downstream input should become:

```text
${QC_WORK_DIR}/qc_step1/chr${CHR}.step1 with the shared --keep ${SAMPLE_KEEP_FILE}
```

---

## 5.4 Step 4: Optional controls-only HWE

Only run if:

```yaml
variant_qc.apply_hwe: true
variant_qc.controls_keep_file: path/to/controls.keep
```

Command template:

```bash
${PLINK2} \
  --pfile ${STEP1_OR_SAMPLEQC_PREFIX} \
  ${HWE_KEEP_ARGS} \
  --hwe ${HWE_P} midp keep-fewhet \
  --write-snplist \
  --out ${QC_WORK_DIR}/hwe/chr${CHR}.controls_hwe
```

`HWE_KEEP_ARGS` should expand to:

```text
--keep ${CONTROLS_KEEP_FILE}
```

If both `CONTROLS_KEEP_FILE` and `SAMPLE_KEEP_FILE` are present, it should
expand to:

```text
--keep ${CONTROLS_KEEP_FILE} ${SAMPLE_KEEP_FILE}
```

so PLINK2 uses their intersection.

If `hwe_midp: false`, omit `midp`.

If `hwe_keep_fewhet: false`, omit `keep-fewhet`.

Expected output:

```text
${QC_WORK_DIR}/hwe/chr${CHR}.controls_hwe.snplist
```

---

## 5.5 Step 5: Final per-chromosome QC

If HWE is enabled:

```bash
${PLINK2} \
  --pfile ${STEP1_OR_SAMPLEQC_PREFIX} \
  ${SAMPLE_KEEP_ARGS} \
  --extract ${QC_WORK_DIR}/hwe/chr${CHR}.controls_hwe.snplist \
  --make-pgen \
  --out ${QC_WORK_DIR}/qc_final/chr${CHR}.qc
```

If HWE is disabled:

```bash
${PLINK2} \
  --pfile ${STEP1_OR_SAMPLEQC_PREFIX} \
  ${SAMPLE_KEEP_ARGS} \
  --make-pgen \
  --out ${QC_WORK_DIR}/qc_final/chr${CHR}.qc
```

`SAMPLE_KEEP_ARGS` should be empty when sample removal is disabled. If a shared
sample keep file is configured, it should expand to:

```text
--keep ${SAMPLE_KEEP_FILE}
```

---

## 5.6 Step 6: Create pmerge list

Generate this file:

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

Command:

```bash
${PLINK2} \
  --pmerge-list ${QC_WORK_DIR}/qc_final/qc_pfiles.txt \
  --sort-vars \
  --make-pgen \
  --out ${QC_WORK_DIR}/qc_final/all_chromosomes.qc
```

`--sort-vars` is included to make merge output deterministic and to avoid
position-order problems if any per-chromosome files are not already in the same
sort order.

---

## 5.7 Step 7: Post-QC reports

Command:

```bash
${PLINK2} \
  --pfile ${QC_WORK_DIR}/qc_final/all_chromosomes.qc \
  --missing \
  --freq \
  --hardy \
  --pgen-info \
  --out ${QC_WORK_DIR}/reports/all_chromosomes.qc_summary
```

Use `--freq` rather than `--freq counts` by default so the resulting `.afreq`
can be used for optional allele-frequency checks during harmonization. Add a
separate counts report later if allele counts are needed.

---

# 6. New script: `scripts/harmonize_gwas_for_prs.py`

## Goal

Generate a harmonized PLINK2 score file from GWAS summary statistics and target genotype `.pvar`.

This script should not modify genotype files. It only outputs a score file:

```text
SNP  A1  BETA
```

This score file is then used by PLINK2 `--score`.

---

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

Optional columns:

```text
--chr-col CHR
--bp-col BP
--gwas-af-col EAF
--target-af-file outputs/genotype_qc/reports/all_chromosomes.qc_summary.afreq
--target-id-col ID
--target-af-col ALT_FREQS
--max-frequency-diff-for-flip 0.15
```

---

## Required pvar parsing

PLINK2 `.pvar` typically has columns like:

```text
#CHROM
POS
ID
REF
ALT
```

The script should support:

```text
#CHROM or CHROM
POS
ID
REF
ALT
```

Target variant key should be by `ID` first.

Required v1 matching strategy:

```text
1. Match by GWAS SNP ID to target pvar ID.
2. For unmatched rows, if GWAS CHR/BP columns are provided, fallback to CHR/POS
   matching and require the allele pair to match the target REF/ALT pair.
3. If multiple target variants match the same ID or same CHR/POS/allele pair,
   remove that GWAS row and count it as ambiguous/duplicate.
```

Do not rely on ID-only matching as the only implementation path; official GWAS
summary statistics and target `.pvar` IDs often use different conventions.

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

Keep only A/C/G/T alleles if configured:

```python
valid = {"A", "C", "G", "T"}
```

Remove if:

```text
EA not in A/C/G/T
OA not in A/C/G/T
REF not in A/C/G/T
ALT not in A/C/G/T
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

Allele pair must match:

```python
gwas_pair = {EA, OA}
target_pair = {REF, ALT}

if gwas_pair == target_pair:
    keep
elif allow_strand_flip and {comp(EA), comp(OA)} == target_pair:
    keep_with_a1 = comp(EA)
else:
    remove
```

If pairs match:

```text
Output:
  SNP = target ID
  A1 = EA
  BETA = BETA
```

Important:

```text
Do NOT flip beta just because EA equals REF instead of ALT.

For PLINK2 --score, A1 can be REF or ALT as long as the score file explicitly gives the allele to count.

Therefore:
  if EA is present in target alleles and OA is also present, keep BETA unchanged.
```

If strand flip is enabled and the complemented GWAS alleles match the target
alleles:

```text
Output:
  SNP = target ID
  A1 = complement(EA)
  BETA = BETA
```

Do not flip the sign of `BETA` for strand-complement matching. Only flip beta in
advanced mode if converting all scores to the non-effect allele or to ALT dosage.
That is not needed for default PLINK2 `--score`.

Default:

```text
allow_strand_flip: false
```

Only enable strand flipping after confirming genome build, allele convention,
and frequency consistency. When `allow_strand_flip: true`,
`strand_flip_requires_frequency_check: true` should require both GWAS EAF and
target allele frequency inputs and reject flips where frequency difference is
greater than `max_frequency_diff_for_flip`. The target allele frequency should
usually come from a PLINK2 `--freq`/`.afreq` report, not from the `.pvar` itself.

---

## Harmonization output

Output file:

```text
SNP    A1    BETA
rs123  A     0.023
rs456  G    -0.017
rs789  T     0.041
```

Tab-delimited.

---

## Harmonization summary JSON

Write:

```text
outputs/prs_scores/trait.harmonized.summary.json
```

Example fields:

```json
{
  "n_gwas_input": 1000000,
  "n_after_required_columns": 990000,
  "n_after_duplicate_removal": 980000,
  "n_matched_to_target_pvar": 700000,
  "n_matched_by_id": 650000,
  "n_matched_by_position_alleles": 50000,
  "n_strand_flipped": 0,
  "n_removed_non_acgt": 1000,
  "n_removed_ambiguous": 80000,
  "n_removed_duplicate_or_multimatch": 100,
  "n_removed_allele_mismatch": 20000,
  "n_harmonized_output": 599000
}
```

---

# 7. New script: `scripts/write_prs_score_commands.py`

## Goal

Generate PLINK2 `--score` commands after harmonization.

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

# 8. Suggested final file structure

```text
configs/
  qc.yaml

scripts/
  filter_info_scores.py
  write_genotype_qc_commands.py
  harmonize_gwas_for_prs.py
  write_prs_score_commands.py

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
    hwe/
      chr1.controls_hwe.snplist
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

# 9. Safety checks and validation

Codex should add checks for:

```text
1. Config fields exist and have reasonable values.
2. Input pgen prefix exists:
   .pgen, .pvar, .psam

3. Info score file exists for each chromosome.
4. Generated snplist is not empty.
5. PLINK2 command outputs expected files.
6. HWE controls file exists if HWE is enabled.
7. GWAS summary statistics required columns exist.
8. Target pvar required columns exist.
9. Harmonized score file has at least one variant.
10. PLINK2 --score output .sscore exists.
11. If genotype_missingness_mode is dosage, generated PLINK2 commands include
    `--geno ... dosage`.
12. If sample_qc.apply_mind is true, a shared sample_qc.keep_file is present;
    otherwise command generation fails instead of producing per-chromosome
    sample removal commands.
```

Recommended threshold warnings:

```text
Warn if n_harmonized_output < 50,000 for common disease PRS.
Warn if more than 50% of GWAS variants are removed due to allele mismatch.
Warn if ambiguous SNP removal removes unexpectedly many variants.
Warn if pvar IDs are duplicated.
Warn if GWAS SNP IDs are duplicated.
Warn if ID matching is very low and CHR/BP fallback is unavailable.
Warn if strand flipping is enabled without both GWAS and target frequency inputs.
```

---

# 10. Acceptance criteria

The implementation is complete if:

```text
1. `python scripts/write_genotype_qc_commands.py --config configs/qc.yaml --chromosomes 1-22 --out ...`
   creates a runnable shell script.

2. The generated shell script includes:
   - INFO/MAF snplist generation
   - PLINK2 first-pass QC
   - --snps-only just-acgt
   - --max-alleles 2
   - --geno 0.05 dosage for imputed PGEN defaults
   - missingness reports
   - optional HWE commands if enabled
   - final chromosome outputs
   - pmerge-list command with deterministic sorted output
   - post-QC reports
   - no per-chromosome --mind sample removal unless a shared keep file is applied

3. `python scripts/harmonize_gwas_for_prs.py ...`
   creates:
   - harmonized score file with columns SNP, A1, BETA
   - summary JSON with removal counts
   - ID-first matching plus CHR/BP/allele fallback when CHR/BP columns are provided
   - strand flipping disabled by default

4. `python scripts/write_prs_score_commands.py ...`
   creates a PLINK2 --score command using:
   - final all_chromosomes.qc pfile
   - harmonized score file
   - output trait.prs.sscore

5. Existing behavior is preserved when only old config fields are present.
6. Existing legacy CLI invocations for `filter_info_scores.py` and
   `write_genotype_qc_commands.py` still work.
```

---

# 11. Example Codex instruction

Copy this into Codex:

```text
Please implement the PRS-ready genotype QC pipeline described in this spec.

Main goals:
1. Update configs/qc.yaml to support structured variant_qc, sample_qc, prs_alignment, and paths sections.
2. Preserve backward compatibility with old top-level keys: min_info, min_maf, max_variant_missingness.
3. Update scripts/filter_info_scores.py to generate INFO/MAF snplists and JSON summaries.
   Preserve aliases: --info-score, --variant-col, and --maf-col.
4. Update scripts/write_genotype_qc_commands.py to generate a full PLINK2 QC shell script:
   - info/MAF filtering
   - --snps-only just-acgt
   - --max-alleles 2
   - --geno with dosage modifier by default for imputed PGENs
   - missingness reports
   - optional controls-only HWE
   - final per-chromosome pgen
   - pmerge-list genome-wide merge
   - post-QC reports
5. Add scripts/harmonize_gwas_for_prs.py to harmonize GWAS summary statistics against final target .pvar and output PLINK2 score file columns SNP, A1, BETA.
   Use ID-first matching plus CHR/BP/allele fallback when columns are available.
6. Add scripts/write_prs_score_commands.py to generate PLINK2 --score commands.
7. Add robust validation, helpful error messages, and summary JSON outputs.
8. Do not remove samples by default. Do not run per-chromosome --mind; if sample removal is enabled, require a shared keep file.
9. Remove ambiguous A/T and C/G SNPs by default during PRS harmonization.
10. Keep strand flipping disabled by default; only allow it when explicitly enabled and supported by frequency checks.
11. Add minimal tests or dry-run examples if the repository already has a test framework.

Please keep the implementation simple and maintainable. Prefer command generation over executing PLINK2 directly.
```

---

# 12. Practical defaults for this project

Recommended current defaults:

```yaml
variant_qc:
  min_info: 0.8
  min_maf: 0.01
  max_af: 0.99
  max_variant_missingness: 0.05
  genotype_missingness_mode: dosage
  keep_snps_only: true
  biallelic_only: true

  apply_hwe: false
  hwe_p: 1e-6
  hwe_controls_only: true
  controls_keep_file: null
  hwe_midp: true
  hwe_keep_fewhet: true

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
```

Recommended first run:

```text
Run genotype QC without HWE first.
Run allele harmonization.
Run PRS scoring.
Check how many variants remain after harmonization.

Then, if controls.keep is available, enable controls-only HWE and rerun.
```

---

# 13. Notes for interpretation

```text
INFO >= 0.8:
  Keeps well-imputed variants.

MAF >= 0.01:
  Removes very rare variants, which are often unstable for common disease PRS.

--snps-only just-acgt:
  Keeps only simple A/C/G/T SNPs.

--max-alleles 2:
  Removes multiallelic variants.

--geno 0.05:
  Removes variants with >5% missing genotype rate.
  For imputed PGENs with dosages, prefer `--geno 0.05 dosage` so variants are
  not incorrectly removed just because hardcalls are missing while dosages exist.

--hwe 1e-6:
  Removes variants strongly deviating from Hardy-Weinberg equilibrium.
  Prefer controls-only HWE.

Allele harmonization:
  Ensures GWAS beta corresponds to the allele counted by PLINK2 --score.

Ambiguous SNP removal:
  Removes A/T and C/G SNPs by default to avoid strand-flip errors.
```

---

# 14. Official references

PLINK2 filtering documentation:
https://www.cog-genomics.org/plink/2.0/filter

PLINK2 data management documentation:
https://www.cog-genomics.org/plink/2.0/data

PLINK2 scoring documentation:
https://www.cog-genomics.org/plink/2.0/score

PLINK2 order of operations:
https://www.cog-genomics.org/plink/2.0/order
