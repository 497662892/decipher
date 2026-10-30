# PRS, TabPFN and student inference

This guide covers the code published on 2026-10-30. The repository contains
workflow code and aggregate result summaries. Supply trusted local model bundles,
PRS resources and patient inputs separately. Serialized joblib bundles must come
from a trusted source.

## Environment

```bash
uv sync --extra dev
# For the TabPFN teacher workflows as well:
uv sync --extra dev --extra tabpfn
```

The `dev` extra includes PyTorch for the student test suite and inference.
Student prediction runs on CPU, without a TabPFN checkpoint, training context,
outcome labels, or access to an inference service. The recorded student runs used
PyTorch 2.9.1 and scikit-learn 1.8.0; keep the repository lockfile and use the
recorded environment when loading fitted scikit-learn/joblib objects.

## 1. Compute PRS inputs

The models consume already computed PRS columns. They do not accept raw genotypes
directly. Use the existing mature-tool scoring workflow first:

| Input family | Entry points |
|---|---|
| P+T, LDpred2, PRS-CS, lassosum2, SDPR, sBayesRC, MegaPRS | `scripts/write_*_full_train_commands.py`, method-specific Slurm launchers and score collectors |
| Frozen external PGS | `scripts/external_pgs.py`, `configs/external_pgs.yaml` |
| Ho2022 BC components | `scripts/prepare_ho2022_bc.py`, `scripts/assemble_ho2022_bc_expansion.py` |
| Frozen external scores for holdout | `scripts/score_external_pgs_holdout.py` |

See [external PGS scoring](external_pgs_benchmark.md) and the
[Ho2022/holdout workflow](1012/Ho2022_BC_and_External_PRS_Holdout_20261012_CN.md)
for build/allele matching, fixed published weights and scoring-frequency rules.
Preserve the exact frozen score definitions and risk directions. Passing a
different PRS under the same column name is not a valid input conversion.

## 2. TabPFN integration

The external-panel command fits the downstream wrapper using labelled
train+development context and predicts the unlabelled holdout:

```bash
uv run --extra tabpfn python scripts/predict_external_pgs_holdout.py \
  --run-dir outputs/external_pgs/holdout_wrapper_20261012 \
  --disease T2D \
  --model-path outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt \
  --device cuda \
  --eligibility-config configs/disease_eligibility.yaml
```

The run directory must already contain the verified selection/scoring inputs;
the script also resolves the existing local phenotype and ancestry files. Run
`--preflight-only` to check those inputs. This is the established competition
pipeline, with its documented local layout, rather than a saved standalone
TabPFN model loader. Use a new prepared run directory when repeating a run.

For the earlier internal-PRS pipeline, use
`scripts/run_final_tabpfn_oof_holdout_pipeline.py`. The newer
`scripts/export_external_pgs_logit_submission.py` exports frozen external-panel
probabilities as standardized logits using training OOF mean/SD. Read
[the export contract](1013/External_PRS_Standardized_Logit_Submission.md): wrapper
probabilities and standardized direct logits are different outputs.

## 3. Predict with the latest saved students

The common interpretable model is `M0_KD` for all three traits. To use the
train-selected disease-specific alternative, request `BIO_POLICY`:

| Trait | `M0_KD` | `BIO_POLICY` at full-training refit |
|---|---|---|
| `T2D` | M0_KD | M0_KD |
| `cad` | M0_KD | HGB_PAIR_KD |
| `bc` | M0_KD, eligible females only | G_A6_KD, eligible females only |

The historical experiment root is
`outputs/prs_distillation_bio/exploration_20261026_3569648f99a8`.
It is ignored by Git and is not created by cloning the repository.

```bash
uv run --no-sync python scripts/predict_prs_student.py \
  --run-dir outputs/prs_distillation_bio/exploration_20261026_3569648f99a8 \
  --trait T2D --arm M0_KD --seed 17 \
  --input data/processed/new_patients/T2D_features.parquet \
  --output outputs/student_inference/new_cohort_T2D.csv \
  --eligibility-config configs/disease_eligibility.yaml
```

Use `--trait cad --arm BIO_POLICY` for the selected CAD pair booster, or
`--trait bc --arm BIO_POLICY` for the selected BC student. `M0_SUP` is the
same-structure true-label control. The CLI also accepts explicit `G_A6_KD` and
`HGB_PAIR_KD` bundles where they exist. Primary predictions use seed 17; saved
parametric seeds 29/43 are stability analyses, and HGB supports seed 17 only.

Input CSV, TSV or Parquet must contain unique, nonmissing `sampleid`, all frozen
PRS columns for the trait, `age_sd`, `PC1_sd` through `PC10_sd`, and `sex`.
Supply age and PCs in the same upstream definition as the training inputs;
`age_sd` is not age in years. The adapter arranges features in the saved order,
ignores extra columns (including outcomes), and reuses saved training transforms.
It does not refit scaling on the query. Missing/nonfinite features and unsupported
sex categories fail. The supplied `Unknown` category follows the saved encoder
and is flagged when outside its fitted categories.

BC uses `--eligibility-config` to omit ineligible rows. Output keeps eligible
patients in input order and contains `sampleid, trait, arm, seed, probability,
logit, age_out_of_support, sex_out_of_support`. These are direct student outputs;
no Logistic wrapper or standardized submission-logit transform is added.
Age beyond fitted support uses the saved constant age extrapolation and is
flagged. The sidecar `<output>.json` records counts, input/output hashes and the
model-manifest hash. Existing outputs are never overwritten.

The Python entry point is:

```python
from decipehr.prs_student_inference import predict_student

predictions = predict_student(
    run_dir,
    patient_features,
    trait="bc",
    arm="M0_KD",
    eligibility_config="configs/disease_eligibility.yaml",
)
```

For a portable local copy, preserve these relative paths beneath `run_dir`:

- Parametric model: `core/<trait>/models/<arm>/seed17/` with `state.pt`,
  `preprocessing.joblib`, `metadata.json` and `HASHES.json`.
- HGB model: `low_order/<trait>/models/HGB_PAIR_KD/` with `model.joblib` and
  `FIT.json`.
- Policy selection: `report/val/POLICY_LOCK.json` when using `BIO_POLICY`.

Bundle hashes are checked before prediction. Fitted bundles can include training
preprocessing statistics and remain private artifacts. For the earlier v1 bundle
API, see [Phase A implementation](1026/IMPLEMENTATION.md); sign-aware v3 exposes
its own load/predict functions in `src/decipehr/prs_distillation_v3.py`.

## Verification and interpretation

The current adapter reproduced the saved original-validation probabilities
exactly for both M0 and BIO_POLICY: T2D 2,852, CAD 2,852 and female BC 1,628
patients per mode. All six maximum absolute errors were zero; no model was
retrained. Synthetic tests cover outcome isolation, input order, missing or
nonfinite features, identifiers, female eligibility, checksum failures and
output preservation.

The [results summary](1030/LATEST_RESULTS.md) distinguishes original validation,
training OOF, and post-hoc kinship diagnostics. The models predict recorded
disease status; these outputs are not a calibrated future-incidence estimate.
