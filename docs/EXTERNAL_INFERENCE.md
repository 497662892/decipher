# External-cohort inference handbook

This is the entry point for the **November 2026 frozen standard-reference-PC models**. The October guides describe historical competition workflows and are retained for reproducibility.

The pipeline takes GRCh38 genotypes and a patient covariate table, computes the exact effective 26 T2D / 7 CAD / 5 female-BC raw PRS panels and fixed 1000G PCs, then predicts with TabPFN v3 and the distilled student. The default is the **train + validation refit** and common **M0_KD** student. A bundle exported with both scopes also supports the train-only models. No external labels enter preparation or inference.

[中文手册](EXTERNAL_INFERENCE_CN.md) · [All of Us adapter](EXTERNAL_ALLOFUS.md) · [Configuration](../configs/external_cohort.example.yaml)

## 1. Obtain a private bundle once

The repository contains code, not fitted models or genetic resources. A maintainer with the sealed local assets exports them:

~~~bash
uv run --frozen --extra tabpfn python scripts/external_inference.py export \
  --sources configs/external_model_sources.yaml \
  --scope both --out-dir models/external_reference_pc_bundle
~~~

Use a new output directory. The export verifies the reviewed experiment, source weights, feature panels, context hashes and model implementations. It copies relative-path resources and a runnable source/lockfile snapshot into the bundle. It does not train models.

The bundle includes effective matched SNP matrices, fixed frequencies, SNP coordinates, PCA loadings, saved students, TabPFN checkpoint, and **private training features and disease labels used as the TabPFN context**. A TabPFN checkpoint alone is insufficient. Keep the entire bundle and all cohort outputs inside an appropriate private workspace; it is not a public release. The exporter only writes locally.

For another authorized environment, place the bundle there through its approved transfer mechanism. Enter its runtime directory:

~~~bash
cd /path/to/bundle/runtime
uv sync --frozen --extra tabpfn
uv run --frozen --extra tabpfn python scripts/external_inference.py verify-bundle \
  --bundle /path/to/bundle
~~~

Python/package versions are checked at inference. The lockfile pins the tested environment; Linux is the supported runtime. PLINK2 is a separate executable. The local replay uses PLINK v2.0.0-a.6.33LM (28 February 2026). Set its path explicitly or put it on PATH. The runtime needs no model download or inference service. Dependency installation may require network access before moving into an offline environment.

Student inference runs on CPU. TabPFN should normally use an allocated GPU. Both still require the same raw PRS and PC preparation. The shipped bundle contains both methods even when only the student is requested.

## 2. Prepare two inputs

**Genotypes:** one whole-cohort fileset, or disjoint chromosome shards with the same patient roster.

| Config format | Files / settings |
|---|---|
| pfile | Prefix of .pgen, .pvar (or .pvar.zst), .psam |
| bfile | Prefix of .bed, .bim, .fam |
| vcf | .vcf or .vcf.gz; set dosage_field: DS when DS is intended |
| bgen | .bgen, explicit .sample file, and ref_mode: ref-first or ref-last |

Input must be GRCh38 with correct forward-strand alleles. This workflow starts from called/imputed genotypes, not FASTQ/CRAM. Perform platform/release sample QC, genotype QC and imputation-quality filtering upstream. Do not strand-flip ambiguous SNPs by guessing, merge duplicate variants arbitrarily, or silently liftover. PLINK failures and coverage reports identify problems requiring upstream correction.

Different SNP IDs are acceptable. PLINK aligns coordinates and allele pairs against the frozen BIM map after clearing old IDs; misleading same-name variants do not match merely by ID. REF/ALT swaps are allowed. Nonmatching alleles remain unmatched. See [PLINK ID recovery](https://www.cog-genomics.org/plink/2.0/data#recover_var_ids).

Patient IIDs must be unique across FIDs and exactly match the covariate sampleid; leading zeros are preserved. Every nonempty shard must contain every requested patient. This implementation supports chromosome-concatenation shards. Interleaving/overlapping genomic shards need consolidation upstream because the tested PLINK2 cannot perform a general non-concatenating PGEN merge.

**Covariates:** CSV/TSV with unique sampleid and explicitly mapped sex. Extra official PC columns are ignored. Outcome columns T2D/cad/bc/y/y_true are rejected. Example uses synthetic IDs:

~~~text
sampleid,sex
example001,Female
example002,Male
example003,Unknown
~~~

Female BC eligibility comes from the bundled disease_eligibility configuration. Male and unknown-sex participants receive no BC prediction. T2D and CAD retain all otherwise eligible participants. Ensure the declared sex field has the same intended biological meaning as the original eligibility criterion; do not map another demographic variable implicitly.

## 3. Edit one configuration and run

~~~bash
cp configs/external_cohort.example.yaml cohort.yaml
# Edit bundle, run_dir, genotypes, covariates and runtime.plink2.
uv run --frozen --extra tabpfn python scripts/external_inference.py run --config cohort.yaml
~~~

Paths are resolved relative to **the YAML file**, not the current shell directory. Absolute paths are convenient when moving the runtime. All generated data should stay in ignored/private directories.

Default age mode is deferred. The command completes genetic preparation and writes:
**preparation_complete_prediction_deferred_age**. This is a successful preparation, not completed prediction.

Useful separate commands:

~~~bash
uv run --frozen --extra tabpfn python scripts/external_inference.py preflight --config cohort.yaml
uv run --frozen --extra tabpfn python scripts/external_inference.py prepare --config cohort.yaml
uv run --frozen --extra tabpfn python scripts/external_inference.py predict --config cohort.yaml
~~~

Preflight checks files, schema and patient metadata. Actual SNP matching, coverage and dosage QC happen in prepare. On this HPC, slurm/external_inference.sbatch accepts the same configuration. Preparation itself is CPU work.

## 4. Age is deliberately unresolved

The original age_sd was supplied already transformed. The original years-to-age_sd mapping is not available. Neither years nor a z-score fitted to the new cohort is compatible.

Preparation does not use age. Once an independently verified mapping or genuinely compatible source values exist, add the age_sd column and declare:

~~~yaml
age:
  mode: training_scale
  contract_id: decipehr_provided_age_sd_v1
  provenance: "Actual source or independently verified conversion, with version"
~~~

Then rerun run/predict. Genetic preparation is reused when genotype inputs and ordered patient IDs are unchanged. The declaration documents responsibility; the software cannot prove equivalence of a claimed age transformation. We have not invented or estimated that mapping. A real external cohort with age in years therefore cannot yet complete valid model inference.

## 5. Choose models without retraining

~~~yaml
inference:
  scope: trainval_refit      # train_only if exported
  models: [tabpfn, student]  # [student] gives the CPU-only path
  student: M0_KD            # selected or both are optional
  traits: [T2D, cad, bc]
  device: cuda             # cpu for student-only
  batch_size: 1024
  name: primary
~~~

The selected student is M0_KD for T2D, HGB_PAIR_KD for CAD, and G_A6_KD for BC. both includes common M0 and selected, without duplicating T2D M0.

To compare train-only and merged models, change scope and use a new name, e.g. train_only. The same genetic preparation is shared. Saved training transforms and feature orders are reused. TabPFN fits its internal inference context from the frozen source features/labels; this is not fitting on the external query. External samples, labels and covariate distributions never update the model or scaling.

Prediction is batched with a saved receipt per batch. Reduce batch_size if GPU memory is insufficient, and use a new inference name. Model construction and context setup still have a nonzero memory cost. Student models remain CPU models.

**Measured batch sensitivity:** on the 3,922-patient holdout (2,187 female BC), both training scopes reproduce the frozen student predictions and whole-trait TabPFN predictions to below 5e-16. With the default 1,024-row batches, TabPFN differs from whole-trait inference by at most 0.002629 in probability; minimum Pearson and Spearman correlations both exceed 0.9999998. The frozen package uses auto precision. This experiment establishes batch-size sensitivity, not its full numerical mechanism. Batching is not an exact replay of whole-trait predictions. For a strict replay of this holdout, set batch_size: 5000 (at least the eligible cohort size); for large cohorts, prespecify and retain a bounded batch size and patient order. Do not silently change precision or batch size between analyses.

## 6. Coverage and exact score definitions

The 38 inputs are the **effective matched SNP subsets actually used to train these models**, with the original coefficients. Recomputing the full published PGS, using a similarly named score, changing allele conventions, residualizing on ancestry, or scaling PRS in the external cohort changes the input definition.

PLINK calculates raw dosage-weighted **SUM**, not AVG, with fixed reference frequencies. Each allele-part matrix and the native scored-variant list are verified. Parts are summed by patient and score. The PC projection uses fixed 1000G loadings/frequencies and the frozen variance-standardized SUM convention, with no new PCA and no OADP. See [PLINK scoring and projection](https://www.cog-genomics.org/plink/2.0/score).

Strict missing_variants: error is the default: all frozen markers must be present. An explicit reference_mean policy allows missing sites subject to the configured gates. It creates an all-missing native PLINK stream for absent sites and scores that separately. PRS uses the frozen reference allele frequency; target-cohort AF is never estimated.

For PCs, the exact historical PLINK command is retained, including its default missing-call behavior. In the tested version, missing entries under variance-standardize contribute a frequency-proportional term; this differs from the no-mean-imputation option in the current PLINK PCA example. The adapter does not silently switch conventions or promise zero contribution for missing PC sites. Prefer complete PC coverage. Individual missing calls follow the same frozen command as the training projection.

Reports include nonzero-term coverage and absolute-coefficient-weight coverage per PRS, plus PC coverage and missing SNP IDs. The threshold uses nonzero-term coverage. Defaults (99% PRS/PC coverage and 5% sample dosage missingness) are engineering gates, **not validated scientific equivalence thresholds**. Strict mode still rejects any absent SNP even above 99%. Sample QC includes globally absent SNPs; patients above the threshold trigger an error and are not silently dropped. Review coverage and distributions before interpreting predictions.

## 7. Outputs and recovery

| Path under run_dir | Meaning |
|---|---|
| RUN_STATUS.json | Latest command and explicit completion/deferred status |
| preparation/coverage/prs_coverage.csv | Per-PRS coverage and coefficient-weight coverage |
| preparation/coverage/missing.snplist | Absent or mismatching frozen markers |
| preparation/qc/ | Dosage missingness and configured-gate result |
| preparation/scores.long.csv | Raw PRS, sampleid/disease/model/prs |
| preparation/reference_pcs.tsv | Fixed PC1_sd through PC10_sd; column names are model contract names |
| preparation/genetic_features.parquet | Joined genetic features; no age or labels |
| preparation/PREPARED.json | Counts, checksums, input identity and method provenance |
| inference/NAME/predictions.csv | sampleid, trait, model, training_scope, probability, native/teacher logit, support flags |
| inference/NAME/cohort_counts.csv | Input, eligible and ineligible counts per disease |
| inference/NAME/feature_distribution.csv | External feature distributions compared with frozen context |
| inference/NAME/timing.json | Per-model prediction and reused-batch timing |
| inference/NAME/COMPLETE.json | Completion and output checksums |

Each preparation stage records the exact command, log, output identity and elapsed time. Completed stages are reused after verification. Interrupted stage directories are retained with an incomplete suffix before retry. The run directory is locked against concurrent use. Changes to genetic inputs, patient roster, tool binary or preparation code require a new run_dir. Different inference settings require a new inference.name. A completed output that fails verification is not silently overwritten.

Small files and final scores/predictions have SHA256. Genotype files above 128 MiB use size/mtime identity to avoid repeatedly rereading large streams; this is explicitly **not a full content-hash guarantee**. Input PGENs remain read-only and may be symlinked by the preparation view, so retain the original files until the run is complete.

## 8. Evaluate separately, when labels exist

~~~bash
uv run --frozen --extra tabpfn python scripts/external_inference.py evaluate \
  --prediction-dir /path/to/run/inference/primary \
  --outcomes /path/to/outcomes.csv \
  --phenotype-note "Versioned case/control definitions and observation window" \
  --out-dir /path/to/run/evaluation/primary
~~~

Outcomes have unique sampleid and 0/1/missing columns named T2D, cad, bc for the requested traits. Missing labels remain unknown. The command reports labeled/unlabeled denominators, AUROC, AUPRC, Brier score and log loss, with no calibration or refitting. These are descriptive point estimates, not confidence intervals.

Before claiming external validation, establish phenotype comparability, source-training/discovery-cohort overlap, follow-up, ascertainment and relevant ancestry strata. PGS metadata/resources.json records source information; the old decipEHR overlap review does not establish independence from All of Us or another cohort. Raw probability calibration may change with ascertainment/prevalence. Prediction agreement and a successful local replay do not establish external disease-prediction performance.

## 9. Verification and capacity

The local acceptance report is generated under outputs/external_inference_20261105/; it records synthetic PLINK tests, source-to-portable holdout replay, runtime versions, Slurm receipts and precise tolerances. Consult its final VERIFIED.json for completed evidence. Merely exporting a bundle is not an end-to-end test.

The union contains millions of SNPs despite the final table having only 38 raw PRS plus covariates. Budget disk, memory and CPU for genotype preparation. The example uses 4 threads and 32 GB PLINK memory; larger cohorts need workload-specific resources. Stage timing reflects preparation and inference separately. All of Us VDS extraction is a separate distributed workload; no current workspace dataset/release path is hardcoded.

