# Frozen internal versus external PRS on train + development

This analysis compares the previously selected internal PRS panel with the current
external panel using identical patients, outer folds, covariates, TabPFN settings,
and final Logistic wrappers. The primary endpoint is pooled outer-OOF wrapper AUC;
native TabPFN AUC is supplementary. Breast cancer follows the repository's female
eligibility rule.

The user requested that missing historical candidate files be left aside. The
analysis therefore conditions on both existing frozen score panels. Historical
development-based PRS parameter/panel selection and score construction are not
repeated inside CV. These results are retrospective development evidence, not an
end-to-end nested evaluation or a new independent test.

The outer split is the same patient-level train + development five-fold split used
by the May audited submission. Each outer training set fits all downstream
preprocessing and TabPFN context. Three-fold cross-predictions inside that training
set supply the logit feature for the Logistic wrapper. The final outer query is
passed to the prediction API without any outcome columns. Both arms use the same
v3 default checkpoint, 16 estimators, continuous PC1–PC10, age, sex, and fold seeds.
Public SNP weights remain frozen. No hold-out files enter this analysis.

Prepare a new, ignored run directory only after tests pass:

```bash
uv run --no-sync python scripts/run_frozen_prs_comparison.py \
  --prepare --eligibility-config configs/disease_eligibility.yaml \
  --run-dir outputs/external_pgs/frozen_internal_external_trainval_20261013
```

Preparation verifies that the phenotype file equals the disjoint union of original
train and development, checks identical eligible rows between arms, verifies the
external scoring receipts, and freezes input/code/checkpoint hashes. It records
historical candidate paths that are unavailable, without substituting a different
PRS method. Already prepared or completed directories are never overwritten.

`slurm/frozen_prs_comparison.sbatch` maps array tasks 0–9 to T2D, 10–19 to CAD,
and 20–29 to BC. Within each block the first five tasks are internal folds 0–4;
the next five are external folds 0–4. Every task saves raw and wrapper predictions,
the wrapper's training OOF predictions, preprocessing parameters, fit patient-set
hashes, software versions, and a completion receipt. Use `RUN_DIR` for a different
prepared directory. The default array permits two concurrent single-GPU tasks.

After all tasks complete:

```bash
uv run --no-sync python scripts/report_frozen_prs_comparison.py \
  --run-dir outputs/external_pgs/frozen_internal_external_trainval_20261013
```

Reporting independently reconstructs each Logistic wrapper from saved inner OOF
predictions and outer-query native probabilities. It verifies exact patient and
label coverage, the shared outer and inner split hashes, source checksums, and
software versions before computing results. Native float32 probabilities are
promoted to exact float64 values before serialization, preserving ranking ties and
wrapper replay inputs.

The report contains overall/fold/ancestry metrics, paired AUC differences, historical
native metrics, private OOF predictions, and a verification receipt. Paired intervals
use 2,000 outcome-stratified patient bootstrap draws (seed 42); simultaneous intervals
correct for six contrasts (three diseases × two output scales). They condition on
the stored predictions and omit retraining, historical selection, and family
dependence. Ancestry estimates are descriptive.

The historical native results remain a separate reference. The internal arm is
refitted with the current common downstream implementation, so its metrics need
not reproduce the old submission's numbers exactly. This work does not alter the
already completed train + development fit for final hold-out predictions.
