# Frozen external PGS × ancestry-aware TabPFN

This workflow integrates published scores for T2D, CAD, and female breast cancer
using TabPFN with age, sex, and PC1–10. SNP weights remain frozen:
internal labels fit only the downstream integrator, inner-fold score selection,
and calibration. SPLENDID is a methodological reference, not a runtime dependency.

The primary protocol uses the official train cohort as the full TabPFN context
and evaluates on the fixed official validation cohort. Validation patients never
enter context, tuning, preprocessing fits, or calibration fits. The user's
2026-10-09 protocol update replaces the original pooled train/val five-fold plan
for this external-PGS benchmark.

On 2026-10-10, after reviewing the initial validation results, the user selected
tabpfn_prs_pc as the subsequent primary model. Soft genetic similarities are
optional sensitivity features. This is a development decision made after viewing
validation outcomes, not a predeclared selection in the original experiment.
The completed eight-arm run under outputs/external_pgs/run_20261010 remains unchanged;
its already evaluated PC arm can be reused without rescoring or refitting.

For a linear control with the same age, sex, and PC1–10 features as
`tabpfn_prs_pc`, request `multi_prs_pc_lr`. Its regularization is selected within
training folds. The legacy `multi_prs_lr` also uses soft ancestry similarities;
the original eight-arm roster and defaults are retained.

Existing pipelines and the sampleid,disease,model,prs interface remain intact.
Reusable logic lives in decipehr.external_pgs and decipehr.external_pgs_benchmark.
The existing bootstrap interface gains optional family_size, which adds
Bonferroni percentile bounds while retaining the existing 95% bounds.

## Inventory and review

The resource manifest, configs/external_pgs.yaml, defines 34 candidates:

| Disease | IDs | Interpretation |
|---|---|---|
| T2D | PGS005353–PGS005377 | D-PRISM target-population/source-weight components |
| T2D | PGS002308 | Final PRS-CSx score |
| CAD | PGS003725, PGS004696, PGS012558 | GPS_Mult, PRS-CSx, EAS-targeted MIXPRS+ |
| BC | PGS000004, PGS005378, PGS005379 | PRS313 and two African-ancestry-targeted overall-BC models |
| BC | PGS004869, PGS018538 | European MegaPRS and East Asian PRS-CS models |

The original bank contained 32 candidates. Before any validation evaluation on
2026-10-10, PRS313 and PGS005379 failed the fixed 75% matching threshold (67.73%
and 74.98%). PGS004869 and PGS018538 were added to retain multiple BC scores and
broaden their ancestry sources; the threshold was unchanged. The frozen run
records this outcome-blind amendment in protocol_amendment.json. The TPMI model's
source ICD definition and inclusion of male controls are recorded in its metadata;
the target BC analysis still uses eligible females only.

D-PRISM's 25 columns retain distinct trainedfor/source-weight meanings and are
not independent discovery studies. Catalog snapshots retain original score names,
methods, ancestry descriptions, discovery/development samples, and publication
and usage information.

~~~bash
uv run python scripts/external_pgs.py inventory \
  --manifest configs/external_pgs.yaml --build GRCh38 \
  --out-dir outputs/external_pgs/inventory
~~~

Use the verified genotype build, not an inferred default. This command retrieves
metadata only. In its generated review.yaml, identify the target cohort and
research/competition use context. For each included score, document evidence for:

- trait_match: confirmed — target phenotype correspondence.
- allowed_use: allowed — permitted use in the declared context.
- sample_overlap: excluded — no overlap in discovery, tuning, or score selection.

Set include: true only with supporting citations or cohort checks. Unknown overlap
is not evidence of independence. Catalog availability does not establish
competition permission. The initial template makes no affirmative claims.
An explicit project-owner confirmation of cohort independence is acceptable
provenance: record its date, wording, and scope, and identify it as owner-confirmed
rather than an independent participant-linkage audit.

~~~bash
uv run python scripts/external_pgs.py fetch \
  --inventory outputs/external_pgs/inventory/inventory.json \
  --review outputs/external_pgs/inventory/review.yaml \
  --out-dir data/external_pgs/weights
~~~

weights.lock.json freezes exact URLs, headers, SHA256 checksums, reviewed metadata,
and review/inventory hashes. Changed accessions, builds, or checksums fail.

### Ho 2022 breast-cancer supplemental components

`configs/external_pgs_ho2022_bc.yaml` separately records the published ASN/EUR
PRS-CSx components from Supplementary Tables S3/S4, their source hashes, the
cohort review, and the UCSC hg19-to-hg38 chain hash. These use local identifiers
`LOCAL_HO2022_BC_ASN` and `LOCAL_HO2022_BC_EUR`, not Catalog accessions; do not
pass this manifest to the Catalog inventory command.

~~~bash
uv run --no-sync --with openpyxl==3.1.5 --with pyliftover==0.4.1 \
  python scripts/prepare_ho2022_bc.py \
  --out-dir outputs/external_pgs/YOUR_NEW_HO_RUN/weights
~~~

The importer uses the source workbooks and chain at the paths in the manifest
and CLI defaults. It preserves published betas, complements both alleles for
reverse-strand mappings, rejects ambiguous/noncanonical mappings, and retains
unmapped rows in the original-variant coverage denominator. Local weight locks
verify the original workbook and chain in addition to the converted weights.
Coordinate-conversion success is not genotype-matching coverage: the existing
official matcher and fixed 75% gate still apply before PLINK2 scoring.

The prepared 2026-10-12 comparison uses the existing three BC scores plus these
two components for all eligible women, with shared training folds and separate
development evaluation. Its frozen protocol and launch commands are under
`outputs/external_pgs/bc_ho2022_preparation_20261012/`. Preparation does not submit
Slurm jobs; the scoring and evaluation launchers are `slurm/ho2022_bc_*.sbatch`.

The subsequent authorized comparison and holdout workflow are documented in
[`docs/1012/Ho2022_BC_and_External_PRS_Holdout_20261012_CN.md`](1012/Ho2022_BC_and_External_PRS_Holdout_20261012_CN.md).
Holdout scoring reuses the exact training-matched coefficient matrices and
external allele frequencies, checks canonical PVAR axes, and verifies used
variant sets. `predict_external_pgs_holdout.py` rejects holdout outcome columns
and fits the benchmark-equivalent three-fold OOF wrapper on labelled
train+development only. `verify_external_pgs_holdout.py` independently replays
the wrapper and checks eligible patient coverage and fit-set hashes.

## Score with mature tools

Use an already QC'd merged PLINK2 fileset with an uncompressed .pvar. PGS Catalog's
official formatter and matcher handle variant matching; PLINK2 computes scores.
Tools run in pinned isolated environments through uv tool run:
pgscatalog.core 1.1.1 and pgscatalog.match 0.4.0. A streamed five-column PVAR copy
avoids the matcher's metadata/optional-column parser issue. It includes only
variants with external frequencies and matching allele pairs. PLINK uses the
original genotype files; no alleles or stored genotypes are changed.

The local target and 1000G files use different variant IDs. Prepare frequencies
with the existing conservative chromosome/position/allele-pair matching routine,
then run PLINK --freq on the external reference alone. Only the frequency-table
IDs are translated to target IDs; REF, ALT, and ALT_FREQS keep their reference
orientation. PLINK resolves REF/ALT swaps by allele labels.

~~~bash
uv run python scripts/external_pgs.py prepare-reference \
  --target-pvar outputs/final_genotype/qc_final/train_val.common.pvar \
  --reference-pfile data/resources/1000g/grch38/1000g_GRCh38 \
  --target-build GRCh38 --reference-build GRCh38 \
  --reference-provenance '1000 Genomes GRCh38 20190312; restricted to Phase 3 panel samples' \
  --threads 4 --memory-mb 32000 \
  --out-dir outputs/external_pgs/reference_1000g_trainval
~~~

Run the whole-genome preparation in a compute allocation. The supplied
slurm/external_pgs_reference.sbatch requests 220 GB for the unpruned variant-table
join and requires TARGET_BUILD to be set explicitly after checking the genotype
build. Create logs/external_pgs before submitting. The preparation records the
variant map, matching counts, exact PLINK command, external-reference input
hashes, reference_run.json, and reference.target_ids.afreq. PCA-pruned frequency
files do not cover a genome-wide PGS bank.
The reference sample count comes from the actual PSAM, not the requested panel
size: the current merged fileset contains 2,503 of the panel's 2,504 individuals.

~~~bash
module load plink/2.0-20260228
uv run python scripts/external_pgs.py score \
  --weight-lock data/external_pgs/weights/weights.lock.json \
  --pfile data/processed/YOUR_QC_FILESET \
  --target-build GRCh38 \
  --reference-freq data/reference/YOUR_EXTERNAL_REFERENCE.afreq \
  --reference-provenance 'External cohort, release, and frequency provenance' \
  --threads 4 --min-overlap 0.75 \
  --out-dir outputs/external_pgs/scoring
~~~

The frequency file must come from an external reference, use target variant IDs,
and cover every scored allele. Missing or allele-incompatible reference variants
are excluded before matching and counted in target_reference_qc.json; the fixed
minimum PGS coverage still applies. Invalid frequencies fail; no fallback to
validation-cohort frequencies is allowed. This version accepts additive scores
and biallelic .afreq files; unsupported models fail explicitly.
PLINK is restricted to the matched variants and receives --error-on-freq-calc,
so it cannot silently estimate an unavailable frequency from target patients.

Conservative matcher defaults exclude palindromic and unresolved multiple matches.
No keep-first-match or duplicate-ID override is enabled. Default minimum overlap
is 0.75, checked against both the matcher and the published variant count. Declare
any changed threshold before evaluation and hold it fixed across methods.

Score every weight origin in every patient. The collector aligns patient IDs and
sums allele partitions using PLINK *_SUM, never *_AVG. Actual scored variant lists
must equal matched lists. Outputs include scores.long.csv, scoring_qc.csv,
the official per-score matching summary and variant log, scored-variant lists, commands.json, PLINK version,
and scoring_run.json with input/output hashes. Use fresh output directories.

The scoring bank is matched jointly and scored using PLINK's multiple coefficient
columns. This avoids rereading the genotype matrix for every published score.
The 0.75 coverage rule is applied separately to each score using both the official
matcher and the published variant count. Low-coverage candidates are recorded in
scoring_qc.csv and excluded before any outcome is read. weights.scored.lock.json
contains the exact retained bank, links back to the original frozen lock, and
records exclusions; use this generated lock for downstream evaluation.
The adapter accepts canonical model IDs and the exact build-qualified IDs emitted
by the official formatter. It preserves the tool's allele/coefficient columns,
records matcher_accession in QC, and rejects unexpected IDs or genome builds.

## Audit and run the official train-to-validation comparison

Use the supplied official split, with no new random holdout:

| Trait | Train context | Validation query | Train cases | Validation cases |
|---|---:|---:|---:|---:|
| T2D | 5,811 | 2,852 | 1,701 | 847 |
| CAD | 5,811 | 2,852 | 2,313 | 1,072 |
| Breast cancer, eligible females | 3,308 | 1,628 | 602 | 320 |

Counts were verified against the current processed files on 2026-10-09. The
phenotype inputs are read-only. Train/validation patient overlap fails before
disease eligibility filtering, and every eligible patient must have complete
scores and ancestry inputs. No row is silently dropped.

Supply the existing 1000G-only projection's hard labels for subgroup reporting.
The default PC model requires no soft-weight file. Supply --ancestry-weights only
for an explicitly requested soft-similarity method or soft_z normalization.
Soft columns are w_AFR,w_AMR,w_EAS,w_EUR,w_SAS: these are genetic similarities,
not ancestry proportions. Hard labels support subgroup reporting; AMBIGUOUS is
retained. IDs may be sampleid or IID and are read as strings. Covariates use
age_sd, sex, and PC1_sd through PC10_sd.

~~~bash
uv run python scripts/evaluate_external_pgs_validation.py \
  --weight-lock outputs/external_pgs/scoring/weights.scored.lock.json \
  --scores outputs/external_pgs/scoring/scores.long.csv \
  --scoring-run outputs/external_pgs/scoring/scoring_run.json \
  --train-phenotype data/processed/phenotype.csv \
  --validation-phenotype data/processed/official_validation_phenotype.csv \
  --ancestry-labels PATH_TO_TRAIN_VAL_HARD_LABELS.tsv \
  --eligibility-config configs/disease_eligibility.yaml \
  --diseases T2D --preflight-only \
  --out-dir outputs/external_pgs/pc_primary/t2d_preflight
~~~

The two phenotype paths shown are the defaults. No outer-fold file is used by
this primary entry point. The existing train_val genotype fileset may still be
scored once for efficiency: SNP weights and imputation frequencies are external
and fixed, and the two patient groups are separated before any downstream fit.

After input audit, run inside a GPU allocation, remove --preflight-only, use a new
output directory, and add --device cuda and --model-path for a local v3 checkpoint
when available. Install the local extra with uv sync --extra dev --extra tabpfn;
run the benchmark with uv run --extra tabpfn python to preserve that extra. The
Python entry points do not submit Slurm jobs. For production runs,
slurm/external_pgs_score.sbatch and slurm/external_pgs_validation.sbatch support
an afterok dependency chain. The
validation array runs the three diseases with the same locked protocol and
isolated output directories; a synthetic full-context GPU preflight gates it.
The validation and report launchers default to outputs/external_pgs/pc_primary,
separate from the completed eight-arm run. New computations should only be
submitted when needed; the current PC results already exist.

The default --methods tabpfn_prs_pc runs only the primary PC model. To reproduce
the original eight-arm comparison, explicitly use --methods all together with
--ancestry-weights and a new output directory. The original methods retain their
feature definitions for reproducibility:

| Method | Inputs |
|---|---|
| covariates_lr, covariates_tabpfn | Age, sex, PCs, soft similarities |
| best_single_lr | Inner-CV-selected score plus all covariates |
| multi_prs_lr | All scores plus all covariates |
| multi_prs_interactions_lr | Above plus PRS × PC and PRS × soft-similarity interactions |
| tabpfn_prs_demographics | All scores, age, sex |
| tabpfn_prs_pc | All scores, age, sex, PCs; current primary model |
| tabpfn_prs_pc_soft | Above plus soft similarities; optional sensitivity model |

Ridge logistic C is selected from 0.1, 1, 10 by pooled inner three-fold AUROC.
TabPFN is v3 with 16 estimators. Imputation, encoding, scaling, and interaction
features use training-fitted transforms.

The full official train set supplies the final context for every method. The
age/sex/PC-plus-logit wrapper fits to genuine inner OOF predictions from train
only. Score and C selection are repeated inside each inner training subset;
no in-context predictions train the wrapper. Validation labels are removed from
the query passed to prediction and calibration code and attached only to the
reported predictions. Wrapper logits clip at 0.001; primary validation AUROC uses
the raw probability ranking.

Use --score-set dprism_t2d --diseases T2D to isolate same-set components from the
pooled multi-method bank. Use --normalization soft_z for the existing training-only
ancestry normalization. These sensitivity runs are exploratory and require
separate output directories. --methods permits explicit development subsets or
all for the archived eight-arm design; the current default is tabpfn_prs_pc.

The older scripts/evaluate_external_pgs_cv.py entry point remains available for
optional development CV. Its phenotype default is now official train, and a
training-only shared five-fold file must be supplied explicitly with --splits.
It rejects fold files containing patients outside that training input. Its OOF
outputs stay in a separate directory and are not pooled with official validation
results. Inner three-fold selection/calibration within official train does not
change the fixed train/val evaluation boundary.

## Results and verification

Primary outputs include per-disease validation_predictions.csv, fit/context hashes, selected scores
and hyperparameters, preprocessing parameters, pooled/subgroup metrics, paired
AUROC differences, run provenance, and REPORT.md. run.json receives status:
complete only after every requested disease finishes.
All primary outputs identify evaluation_protocol=official_train_val. The shared
prediction schema retains fold=0 as a single-holdout marker; these predictions
are not described as OOF. Summaries reject mixtures of validation and CV results.

In the original eight-arm experiment, the two pooled primary contrasts, soft
versus PC-only TabPFN and soft versus linear interactions, formed a family of six
across three diseases. They remain available as sensitivity comparisons when
those arms are explicitly requested. Report paired
patient-bootstrap 95% intervals and Bonferroni percentile intervals (99.1667%
individual coverage). A one-trait pilot retains family size six. Bootstrap
uncertainty conditions on frozen validation predictions and excludes refitting
uncertainty. Small groups (<20 cases or controls) are flagged exploratory; subgroup
comparisons are secondary. Report OR90 and calibration for raw and wrapped
probabilities separately, together with AUC and Nagelkerke R². The official val
cohort has been used in prior project comparisons; this protocol does not make it
a newly untouched external test set.

After all three array tasks complete, generate the verified combined report:

~~~bash
uv run --no-sync python scripts/summarize_external_pgs_validation.py \
  --runs-root outputs/external_pgs/pc_primary/official_validation \
  --scoring-dir outputs/external_pgs/run_20261010/scoring_v2 \
  --eligibility-config configs/disease_eligibility.yaml \
  --out-dir outputs/external_pgs/pc_primary/report
~~~

slurm/external_pgs_report.sbatch runs the same verification and can depend on the
validation array. The verifier checks every eligible patient's predictions,
train/validation and inner-fold context hashes, frozen inputs, score-bank
membership, the requested method roster, the shared checkpoint, and recomputed
metrics. It accepts either the default PC-only run or the original eight-arm run;
all three diseases must use the same roster. REPORT.md shows candidate exclusions
and raw metrics; comparison tables are included only when their arms were fitted.
VERIFIED.json records source-run and output hashes only after these checks pass.

~~~bash
uv run --extra dev pytest
uv run --extra dev ruff check .
module load plink/2.0-20260228
DECIPEHR_EXTERNAL_PGS_INTEGRATION=1 uv run --extra dev pytest \
  tests/test_external_pgs.py::test_real_matcher_plink_alleles_missingness_and_reproducibility
~~~

The optional integration test uses known synthetic genotypes, a palindromic SNP,
a strand complement, a reference-allele weight, and a missing genotype. It checks
external-frequency imputation and repeatability with the real matcher and PLINK2.
Default tests require neither network nor GPU. Real data, snapshots, weights, and
results stay in ignored data/ or outputs/ directories.
Holdout tests verify that every fitting call uses train patients only, all final
predictions use the full eligible train context, changing validation labels
leaves predictions/selection/calibration unchanged, BC eligibility is enforced
on both sides, and the CLI writes only validation predictions for the primary run.

References: [PGS Catalog matching](https://pygscatalog.readthedocs.io/en/latest/how-to/guides/match.html),
[PLINK2 scoring](https://www.cog-genomics.org/plink/2.0/score),
[SPLENDID](https://github.com/chen-tony/SPLENDID).

## DiscoDivas comparison on official validation

The implemented comparison uses official train as context and official val for
prediction, with PC-only TabPFN and no extra soft-similarity features. External
SNP weights remain frozen. The official validation cohort has already informed
project development, so this is a further development comparison.

### Frozen inputs and matched methods

| Trait | Matched inputs | Reference proxies |
|---|---|---|
| T2D | Five D-PRISM metascores from all 25 retained components, PGS005353–PGS005377, with published combination coefficients | AFR, AMR, EAS, EUR, SAS |
| BC | PGS005378, PGS004869, PGS018538; eligible females only | AFR, EUR, EAS |
| CAD | PGS003725 (EUR-tuned GPS_Mult), PGS012558 (EAS-targeted MIXPRS+) | EUR, EAS |

All three matched methods receive the same train-fitted PC-residualized and
standardized scores and age/sex/PC1–10 covariates. DiscoDivas uses the official
individual interpolation coefficients to combine these scores before logistic
regression. The linear baseline fits all component scores directly. TabPFN v3
uses all component scores with 16 estimators and the existing local checkpoint.
Logistic C is selected from 0.1, 1, and 10 by three-fold CV within train. The
secondary wrapper uses nested OOF predictions from train for calibration.
Every score transformation is refitted inside each training subset.

The original full-bank PC-only TabPFN results remain a separate whole-pipeline
comparison: they use 26 T2D scores, three CAD scores, and three BC scores, with
different score preprocessing. PGS002308 is excluded from the matched T2D bank
because that comparison uses five complete D-PRISM metascores. PGS004696 is
excluded from the matched CAD bank because its mixed-ancestry tuning cohort has
no identified PC median; it cannot be assigned an arbitrary ancestry center.
BC is a heterogeneous-model extension and retains female eligibility on both
sides of the split via `configs/disease_eligibility.yaml`.

D-PRISM's five-by-five coefficients are external and fixed. Rows identify the
metascore's tuning population, and columns identify the GWAS-weight source.
The exact mapping and coefficients are in the resource audit's
`dprism_component_mapping.csv` and `dprism_published_coefficients.csv`.
Component scaling is fitted on train and reused on val, an explicit adaptation
of the authors' cohort-wise scaling. No internal GWAS or SNP-weight fitting is
performed. Train labels fit only downstream prediction and calibration models.

### Official interpolation and common reference coordinates

Pinned sources are DiscoDivas commit
`ee8e5d996e6fc6ffaf04255e1cf0fa8ecce68524` and D-PRISM commit
`75917168b64e0858cad2db7e61234050a81dbec1`, stored with SHA256 checksums under
`outputs/external_pgs/discodivas_resource_audit_20261010/`.

Only 53,571 of 59,529 upstream PCA markers match both GRCh38 cohorts. This failed
the original 90% coverage gate. Before disease evaluation, PLINK2 PCA was
therefore fitted on 2,503 1000G samples alone using these common markers.
All 8,663 target patients and the reference samples were projected using the
same external loadings and fixed 1000G allele frequencies; population medians
were recomputed in that space. Target patients never entered PCA fitting.
The reference projection agrees with its fitted eigenvectors up to arbitrary
axis sign. The final sign-invariant QC was recovered from already completed
scores; `projection.json` records that recovery and its hashes.

1000G medians are proxies for model development/tuning populations, not actual
fine-tuning-cohort medians. This applies official DiscoDivas interpolation to
our frozen models; it is not an exact reproduction of the paper's cohorts.

The upstream R file is unchanged. `scripts/run_official_discodivas.R` sets
`datatable.fread.datatable=FALSE` because the upstream `--regress.PCA FALSE`
branch assumes data.frame positional indexing. A real-script integration test
caught constant output under default data.table semantics. The launcher fixes
that I/O incompatibility without changing the interpolation algorithm.
Sorted nonnumeric surrogate IDs preserve ordering through upstream merges.

Upstream's default `--regress.PCA TRUE` fits residualization in the supplied
query cohort. Our adapter instead fits OLS residualization and scaling using
only each training context, then calls official interpolation with FALSE.
The quality parameters are fixed at one. Tests cover patient ordering,
query-batch independence, validation-label invariance, and nested fit boundaries.
Actual final scores are checked against official R output; the report verifier
independently reconstructs train-fitted transformations from frozen raw scores.

### Run and verify

The completed projection is
`outputs/external_pgs/discodivas/projection_1000g_refit_v2_20261010/`.
Use a new `RUN_DIR` for each evaluation; existing result directories are rejected.
The R environment is `module load r/4.4.2`, with local rio/writexl in
`outputs/external_pgs/discodivas/R-library`. Site Makevars overrides package
linker settings; installing these packages required per-command
`R_MAKEVARS_USER=/dev/null R_MAKEVARS_SITE=/dev/null`, without editing site files.

~~~bash
mkdir -p logs/discodivas
PROJECTION_DIR=outputs/external_pgs/discodivas/projection_1000g_refit_v2_20261010 \
RUN_DIR=outputs/external_pgs/discodivas/validation_20261010 \
sbatch --parsable slurm/discodivas_validation.sbatch

# Run after all three trait jobs complete.
uv run --no-sync python scripts/summarize_discodivas_validation.py \
  --runs-root outputs/external_pgs/discodivas/validation_20261010 \
  --previous-runs-root outputs/external_pgs/run_20261010/official_validation \
  --out-dir outputs/external_pgs/discodivas/report_20261010
~~~

To rebuild reference PCs in a new directory, submit
`slurm/discodivas_projection.sbatch --refit-reference-pca` with an explicit new
`PROJECTION_DIR`. The original projection mode still enforces its coverage gate.

Primary contrasts are raw AUROC differences: matched TabPFN versus DiscoDivas
and matched TabPFN versus linear integration, across three traits (family size
six). Report 2,000 paired patient-bootstrap 95% intervals and family-adjusted
percentile intervals. These condition on stored predictions and exclude
refitting uncertainty. OR90, Nagelkerke R², Brier/calibration, wrapper results,
ancestry subgroups, and full-bank contrasts are secondary. Groups with fewer
than 20 cases or controls are exploratory. AMBIGUOUS does not prove admixture.

The report verifier requires complete runs for all three traits, unchanged
inputs/code, exact patient sets and labels, female BC eligibility, full train
context, all nested training subsets, official R score agreement, independently
reconstructed score preprocessing, and recomputed metrics before writing
`VERIFIED.json` and `REPORT.md`.

References: [DiscoDivas code](https://github.com/YunfengRuan/DiscoDivas),
[DiscoDivas paper](https://doi.org/10.1016/j.ajhg.2026.05.006),
[D-PRISM](https://github.com/ahuertach/D-PRISM-T2D-PRS),
[MIXPRS+](https://doi.org/10.1038/s41588-026-02637-4).

### Completed comparison (2026-10-10)

Slurm array `28319084` completed all three tasks with exit code 0. The independently
verified report is `outputs/external_pgs/discodivas/report_20261010/REPORT.md`,
with numeric tables and provenance in that directory. The execution protocol was
snapshotted before submission in
`outputs/external_pgs/discodivas/execution_20261010/protocol_before_evaluation.md`.

| Trait | DiscoDivas + LR AUROC | Matched TabPFN AUROC | Difference (95% paired CI) |
|---|---:|---:|---|
| T2D | 0.747072 | 0.765747 | +0.018676 [+0.011437, +0.026260] |
| CAD | 0.799862 | 0.811875 | +0.012014 [+0.007208, +0.016723] |
| BC | 0.677535 | 0.716607 | +0.039072 [+0.020070, +0.058271] |

All three family-adjusted intervals also exclude zero. The matched linear
baseline has AUROC 0.747498, 0.800209, and 0.685464, respectively. The results
support TabPFN integration in this configuration, without establishing ancestry
awareness as the cause. Subgroup intervals are often wide; overall improvements
do not imply improvement in every ancestry group. The report includes raw and
wrapped OR90, Nagelkerke R², calibration, and the original full-bank TabPFN
comparison, together with the 1000G proxy and heterogeneous-input limitations.
