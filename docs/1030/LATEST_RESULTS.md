# Latest PRS integration and distillation results — 2026-10-30

This release publishes the accumulated workflow code and an aggregate summary of
verified experiments through October 27–28. It includes saved-student inference,
without publishing patient records, fitted bundles or prediction files. See the
[inference guide](../prs_inference.md) for inputs and executable entry points.

## Model coverage

| Layer | Published implementation |
|---|---|
| Internal PRS | P+T, LDpred2, PRS-CS, lassosum2, SDPR, sBayesRC and MegaPRS command/scoring workflows |
| External PRS | Frozen 26 T2D / 7 CAD / 5 female BC panels, including D-PRISM components and Ho2022 ASN/EUR BC components |
| TabPFN integration | v3 PRS/covariate integration, shared-fold evaluation, nested OOF wrapper, holdout prediction and standardized-logit submission export |
| Students | v1 prototypes, sign-aware v3, monotone biological-prior M0/A6/G/G_A6 models, GAM/pair boosting, supervised/KD controls and constrained ablations |
| Additional comparisons | TabPFN 3.5 and Kumo evaluation/probe adapters; these are separate comparators, not replacements for the frozen v3 teacher |

External SNP weights remain fixed. Student fitting learns combinations of the
computed PRSs and covariates; it does not estimate new SNP effects. Resource
coverage and allowed-use checks remain in the [external PGS protocol](../external_pgs_benchmark.md).

## Frozen internal versus external panels

These paired five-fold results use combined train+development, the same patients,
folds, TabPFN v3 context settings and OOF-trained Logistic wrapper. The primary
output here is the wrapper probability, unlike the direct outputs below.

| Trait | Patients | Internal-panel wrapper AUROC | External-panel wrapper AUROC |
|---|---:|---:|---:|
| T2D | 8663 | 0.752499 | 0.758216 |
| cad | 8663 | 0.798443 | 0.804783 |
| bc | 4936 | 0.654643 | 0.709261 |

All 30 disease/panel/fold tasks were verified; independent wrapper replay had
maximum absolute error zero. The score panels had historical development-based
selection, which is not repeated within these folds. This is retrospective
comparison conditional on frozen panels, not end-to-end nested PRS development or
an independent test. See the [comparison contract](../1013/Frozen_Internal_External_PRS_Comparison.md).

## Direct original-validation results

Train and validation remain separate. Full training contains 5,811 patients for
T2D/CAD and 3,308 eligible women for BC. Original validation contains 2,852 patients
(847 T2D cases; 1,072 CAD cases) and 1,628 women (320 BC cases). All methods within
each comparison use identical eligible rows and information. Student primary seed
is 17; the teacher uses full original train as context and validation as query.

Each cell is **AUROC / Nagelkerke pseudo-R²**. These are direct probabilities,
without an additional Logistic wrapper or validation-set recalibration.

| Model | T2D | CAD | Female BC |
|---|---:|---:|---:|
| LR_UNPENALIZED | 0.7463 / 0.2126 | 0.8008 / 0.3382 | 0.6859 / 0.1093 |
| M0_SUP | 0.7450 / 0.2120 | 0.8023 / 0.3448 | 0.6977 / 0.1214 |
| M0_KD | 0.7548 / 0.2257 | 0.8023 / 0.3465 | 0.7075 / 0.1329 |
| BIO_POLICY | 0.7548 / 0.2257 | 0.8009 / 0.3454 | 0.7023 / 0.1257 |
| TabPFN_v3 | 0.7643 / 0.2475 | 0.8143 / 0.3769 | 0.7188 / 0.1465 |
| TabPFN_v3.5 | 0.7664 / 0.2511 | 0.8148 / 0.3788 | 0.7218 / 0.1655 |
| Kumo_Tabular | 0.7691 / 0.2554 | 0.8128 / 0.3722 | 0.7163 / 0.1558 |

`M0_KD` is the common interpretable student used in the methods report. Its gate
uses ancestry PCs to allocate positive PRS coefficients; its baseline is monotone
in standardized age. It has no extra age-by-PRS multiplier. `M0_SUP` uses the same
structure with true-label supervision and is needed to assess the KD contrast.
Ordinary unpenalized logistic regression uses the same inputs.

`BIO_POLICY` is a different train-selected strategy. Its full-training choices
are T2D **M0_KD**, CAD **HGB_PAIR_KD**, and female BC **G_A6_KD**. Those choices were
locked using internal training loss, not the original-validation table. Do not
assign the common M0 metrics to the CAD/BC policy models. The policy choices are
not changed to whichever model looks best on validation.

M0's point estimates exceed ordinary logistic regression in these three traits,
but it retains a gap to TabPFN 3 on full validation. The table does not establish
uniform statistical superiority, external transportability or clinical utility.
Pseudo-R² is not heritability or a percentage of variance explained.

## Nested training OOF

The following AUROCs use only original training patients: 5,811 T2D/CAD and 3,308
female BC. Every outer fold repeats the required downstream selection and fits
within that fold's training context. Teacher OOF probabilities generate student
soft targets; they are not the full-train-context validation predictions above.
The BIO_POLICY can select a different arm in each outer fold.

| Model | T2D | CAD | Female BC |
|---|---:|---:|---:|
| M0_KD | 0.7381 | 0.7856 | 0.6921 |
| BIO_POLICY | 0.7389 | 0.7849 | 0.6917 |
| TabPFN_v3 | 0.7367 | 0.7884 | 0.6967 |

The shared patient/known-kinship-group folds prevent known training-family splits;
unresolved kinship is explicitly retained as unresolved. Historically selected
external score panels and repeated research decisions still limit an end-to-end
independence claim. The stronger teacher family has no uniform winner across
all diseases and readouts.

## Validation-only kinship sensitivity

This is a post-hoc diagnostic. Remove validation patients with a direct
train–validation KING edge >= 0.177 while keeping training, fitted models and
selection fixed. T2D/CAD lose 105 validation patients; BC loses 47. Unresolved
kinship cases remain. The groups/edges are inferred from genotypes, not verified
pedigrees; changes in performance do not identify a causal mechanism.

Each metric cell is **AUROC / Nagelkerke pseudo-R²**.

| Trait | Retained patients / cases | Ordinary LR | M0_KD | TabPFN 3 |
|---|---:|---:|---:|---:|
| T2D | 2747 / 809 | 0.7439 / 0.2081 | 0.7521 / 0.2214 | 0.7510 / 0.2156 |
| cad | 2747 / 1031 | 0.8036 / 0.3440 | 0.8047 / 0.3511 | 0.8062 / 0.3569 |
| bc | 1581 / 315 | 0.6892 / 0.1134 | 0.7092 / 0.1339 | 0.7085 / 0.1265 |

The separate shared-component diagnostic removes 159 T2D/CAD and 75 BC validation
patients, retaining 2,693 / 1,553 respectively. It is not interchangeable with the
direct-edge cutoff. Reports retain paired uncertainty estimates conditional on
saved predictions; those intervals do not account for model redevelopment or
all unresolved family dependence.

## Inference and verification

The new `scripts/predict_prs_student.py` and
`decipehr.prs_student_inference.predict_student` load the saved M0 or selected
student without training labels, training context or a TabPFN checkpoint. They
reuse frozen preprocessing and enforce disease eligibility. The
[inference guide](../prs_inference.md) documents model paths and commands.

For this publication, both M0 and BIO_POLICY were replayed on all original
validation inputs: six trait/mode combinations, each with 2,852, 2,852 or 1,628
patients. All maximum absolute probability differences from the saved outputs
were exactly zero. No training jobs, new model selection or holdout predictions
were performed for this release.

The sealed exploration records 8,280 candidate fits and 1,584 final models across
primary and secondary analyses, with 105 completed scheduler tasks. These are
historical experiment counts, not additional work launched during publication.
Phase B and independent external confirmation remain incomplete.

The publication checks passed: **285 tests passed, 4 skipped**, Ruff, lockfile
consistency and Bash syntax for all 14 added Slurm launchers. Two skipped tests
are opt-in external R/PLINK integrations; two require optional `openpyxl`, absent
in this environment. See [the verification record](VERIFICATION.json) for details.

## Provenance

The values above were read from the following local aggregate sources after
checking their containing verification receipts. Paths under `outputs/` are local
references and are not included in GitHub.

- `outputs/prs_distillation_bio/summary_20261027/main_metrics.csv`
  - SHA-256: `c4b518dba3f16524d656aba39a1abdf46d54fa4147f5bf7ff5db3b9bb973ccac`
- `outputs/prs_distillation_bio/methods_20261027/metrics.csv`
  - SHA-256: `667c2124f2baa94c379c52afafbef6f0a739005a6a4f5f6f25c4d45b71da235d`
- `outputs/external_pgs/frozen_internal_external_trainval_20261013/report/overall_metrics.csv`
  - SHA-256: `aca8e683a1d46312a1953749b44b7eeb3a26e79429d4b9fdc36b035b8c4693f5`
- `outputs/prs_distillation_bio/exploration_20261026_3569648f99a8/report/val/POLICY_LOCK.json`
  - SHA-256: `0b0ae8f6dfb32b4066b6f028bfe2053ad529a5e674aa960731aeedbbb9bc49ec`

The publication audit rechecked 79 hash entries for the overview, 79 for the
methods report, and 224 for the experiment completion receipt, with no mismatch.
These counts include references shared by receipts. The overview covers 114 main
metric rows and 576 rows across seven archived families; the methods report covers
24 metric rows. Aggregate results in this page are a curated summary, not a copy
of private patient-level predictions.

Code entry points for the full local bilingual reports are
`scripts/build_prs_results_overview.py` and `scripts/build_prs_distillation_methods.py`.
Rendering requires the pinned build-time KaTeX installation described in
`scripts/prs_report_math.py`; trained experiment artifacts must already be present.
The existing standalone HTML reports remain under local `outputs/`.

