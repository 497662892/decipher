# Phase A implementation

The executable adapter implements the fixed formula in the supplied v1 plan.
All new selection, teacher soft targets, preprocessing, early stopping and refits
use official **train only**. Official **val** labels are loaded by `report`, after
predictions and selection locks exist. The existing external panels were selected
historically using development evidence; the new experiment is therefore a
post-hoc mechanism study, not a new independent validation.

## Full original training cohort sensitivity

Following the user's later request, `configs/prs_distillation_fulltrain.yaml`
retains every original training patient: 5,811 for T2D/CAD and 3,308 eligible
women for BC. Original validation is prediction-only (2,852 / 1,628), and the
previous QC validation subset is additionally evaluated from the same saved
predictions. Train and validation are never merged.

This configuration restores historical Teacher seed 42 and directly reuses
`TabPFNFeatureTransformer` with the original PRS, age, sex, PC column order.
Checkpoint, 26/7/5 panels, package version and 16 estimators remain frozen.
The student structure and candidate grid remain the same as Phase A.

Known QC-valid families remain in one training fold. The 154 source-QC-flagged
patients have unresolved kinship; patient placeholders are explicitly labeled
`UNRESOLVED_QC` and are not evidence of unrelatedness. Known train/validation
family overlap is allowed only in the final D-to-E prediction scope. Patient
overlap is always rejected. This is a full-cohort sensitivity analysis, without
a claim of complete kinship isolation; intervals do not cover unresolved family
dependence. Previous QC results and caches are preserved in their original run.

Use the full-training config for audit and pass
`CONFIG=configs/prs_distillation_fulltrain.yaml` to the existing Slurm launcher.
All other run and smoke commands below apply with that config and a new run
directory. A historical Teacher replay comparison and the original-QC validation
subset comparison are performed only after predictions are frozen.
Setting `QC_REFERENCE_RUN` to the previous QC run adds three Teacher-only fits
using the restored historical settings on that same QC training cohort. This
control separates the training-cohort effect from seed/column-order changes;
it does not select any configuration on validation outcomes.

The user approved two explicit choices on 2026-10-26:

- Reuse the audited outcome-blind heterozygosity QC and kinship components from
  `outputs/internal_prs/nyu_20261011/prepared`. Remove training relatives of
  validation patients without moving patients between official splits. Retained
  train/val counts are 5,489/2,810 for T2D and CAD and 3,119/1,602 for female BC.
- Model and plot the supplied standardized `age_sd`. Original age in years and
  its conversion parameters are unavailable. Figures explicitly use `age_sd`.

The fixed external panels contain 26 T2D, 7 CAD and 5 BC scores. The receipt under
`holdout_wrapper_20261012/SELECTION.json` supplies panel metadata only; no holdout
patient data or outcomes are opened. Scores retain their published directions and
the existing context-only mean/SD scaling, without new ancestry residualization.
The recorded weight type for CAD score PGS000296 is **NR**, not beta. Its positive
risk direction is documented by [Wang et al.](https://pubmed.ncbi.nlm.nih.gov/32762905/)
and its source metadata by the [PGS Catalog](https://www.pgscatalog.org/score/PGS000296/).
No score direction is inferred from validation outcomes.

The supplied categorical `sex=Unknown` value occurs once in training. In its OOF
query fold it is absent from context; it uses the existing pipeline's all-zero
one-hot representation and receives an explicit `sex_out_of_support` flag. It is
not mapped to Female or Male. Truly missing values and other unsupported categories
fail, and the female-only BC cohort still rejects other sex values.

The local teacher is explicitly TabPFN V3 in package 8.0.1 with the existing
`tabpfn-v3-classifier-v3_default.ckpt`, 16 estimators and probability balancing off.
The complete actual constructor parameters and checkpoint checksum are saved.
Teacher and student receive the same information. Every teacher context fits its
own preprocessing. No inference API or automatic download is used.

Use the existing `uv` environment without upgrading it:

```bash
uv run --offline --no-sync --extra dev pytest -q
uv run --offline --no-sync --extra dev ruff check .
uv run --offline --no-sync python scripts/run_prs_distillation.py audit \
  --config configs/prs_distillation_v1.yaml \
  --eligibility-config configs/disease_eligibility.yaml
```

The audit prints the unique run directory. It writes `discovery_report.md`,
`resolved_config.yaml`, source/environment hashes, feature manifests and a shared
patient/group split manifest before fitting. Existing results are not overwritten.
Use that printed directory as `RUN_DIR` below:

```bash
mkdir -p logs/prs_distillation
sbatch --export=ALL,MODE=smoke,RUN_DIR=outputs/prs_distillation_v1/PRINTED_RUN \
  slurm/prs_distillation.sbatch
# Only after the smoke job exits successfully:
sbatch --export=ALL,MODE=core,RUN_DIR=outputs/prs_distillation_v1/PRINTED_RUN \
  slurm/prs_distillation.sbatch
```

The smoke namespace uses a deterministic whole-group subsample of T2D, at least
256 train and 64 validation patients, and 40 student epochs. It exercises the
complete interface, but its performance is not evidence for the full experiment.
The core job processes T2D, CAD and BC sequentially on one GPU. For each trait it
runs the 45 prescribed student configurations, locks selections, fits three seeds
for META/direct/pure/mixed models, and then the two locked deletion ablations.
Primary predictions always use seed 17; seeds 29/43 measure optimization stability.

The CLI also exposes individual `run-core`, `ablate` and `report` commands, each
with `--run-dir`, `--trait`, `--config` and `--eligibility-config`. `--dry-run`
validates configuration and prints the request without fitting. Completed teacher
and tuning caches are reused only with matching provenance. A partial teacher
cache requires explicit quarantine before retry; mismatched caches never overwrite
old results. `report` only verifies and summarizes saved predictions and bundles.

Each trait's `core/<trait>/` directory contains the selection lock, context/query
scope records, pooled teacher OOF predictions/metrics, 18 fitted student bundles,
patient-level exact decompositions, direct metrics, paired group-bootstrap
intervals, subgroup diagnostics, PRS correlations, initialization stability,
four figures and `VERIFIED.json`. All patient-level artifacts are under ignored
`outputs/`, with private file permissions. No commit or publication is implied.

Load a trusted local bundle without TabPFN or labels:

```python
from decipehr.prs_distillation.student import load_student_bundle

bundle = load_student_bundle("RUN/core/T2D/models/S_KD_MIX/seed17")
columns = bundle.preprocessing.source_columns
components = bundle.predict(patient_table[["sampleid", *columns]])
```

Required columns, their order, finite values and supported covariate categories
are checked. Each bundle contains score provenance, feature order, preprocessing,
spline parameters, centered gate parameters, model state and file checksums.

The main comparison is mixed minus directly supervised, with paired log loss,
AUROC and Brier differences. Intervals resample kinship groups and condition on
fixed predictions. Small subgroups are marked exploratory. Weight differences
across seeds are not confidence intervals or unique causal attributions.

Phase B is deliberately **not implemented or run**. `evaluate-official` fails
explicitly. Direct student metrics must not be compared to historical Stage-2
teacher metrics. A matched metadata-only TabPFN is also outside this core run, so
the shared simple META comparison cannot isolate the teacher's genetic contribution.
