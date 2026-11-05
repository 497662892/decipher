# All of Us: workspace VDS to frozen models

This adapter runs **inside an authorized data workspace**. It accepts current release paths supplied by the researcher; it does not discover credentials, fetch participant data to this HPC, or assume that an old AoU release path remains valid. Local synthetic/replay verification is separate from a real AoU run.

Consult the current [access process](https://researchallofus.org/data-tools/data-access/) and [genomic data documentation](https://support.researchallofus.org/hc/en-us/sections/6088314608020-Working-with-Genomic-Data) for your chosen release.

## Route A: existing genotype files

If suitable GRCh38 PLINK/VCF/BGEN files already exist inside the authorized workspace, use them directly with [the cohort configuration](../configs/external_cohort.example.yaml). This is also the route for other external cohorts and does not require Hail.

## Route B: sparse Hail VDS

A VDS stores variant entries and reference blocks separately. A sparse missing entry is not automatically a missing genotype or a reference call. The adapter preserves reference blocks and uses Hail densification after filtering patients and loci. Do not export variant_data alone or fill all sparse missing entries with 0/0. See the [AoU VDS explanation](https://support.researchallofus.org/hc/en-us/articles/14927774297620-The-new-VariantDataset-VDS-format-for-All-of-Us-short-read-WGS-data) and [Hail to_dense_mt](https://hail.is/docs/0.2/vds/hail.vds.to_dense_mt.html).

Use the release-supported Hail/Spark environment for extraction, separately from the locked model runtime. Provide:

- Authorized current-release GRCh38 VDS path.
- TSV with one string sampleid column containing the final upstream-QC roster.
- Bundle genetics/variants.bim at a path accessible to Hail workers.
- Workspace output prefix and temporary directory.
- Source note with release, upstream sample/variant QC and cohort definition.

For a standalone local environment, the synthetic integration uses Python 3.11, Hail 0.2.139, Spark 3.5.3 and **Java 11**. Use a Java version supported by your Hail release; the local Java 17 trial completed an export but stalled during a cached rerun's shutdown. Do not silently replace the Workbench's supported environment. An isolated local setup is:

~~~bash
uv venv --python 3.11 outputs/hail-env
uv pip install --python outputs/hail-env/bin/python hail==0.2.139 pyspark==3.5.3
export JAVA_HOME=/path/to/jdk11
outputs/hail-env/bin/python scripts/export_aou_prs_inputs.py --help
~~~

The BIM is a SNP coordinate/allele map with no participant data. The full model bundle also contains private source-training context: consider the destination's data-use rules before transferring it. This script performs no transfer.

The following paths are placeholders, not current release URLs:

~~~bash
python export_aou_prs_inputs.py \
  --vds gs://AUTHORIZED_RELEASE/path/to/cohort.vds \
  --samples gs://WORKSPACE_BUCKET/inputs/samples.tsv \
  --variant-map gs://WORKSPACE_BUCKET/resources/variants.bim \
  --source-note "Release identifier; upstream QC version; prespecified cohort selection" \
  --output-prefix gs://WORKSPACE_BUCKET/derived/prs_inputs/cohort \
  --tmp-dir gs://WORKSPACE_BUCKET/tmp/prs_inputs \
  --log /private/workbench/hail-prs-export.log
~~~

The script processes autosomes individually and writes cohort.chrN.bed/.bim/.fam plus cohort.chrN.EXPORTED.json. Use --chromosomes 1 for an initial extraction. Rerunning all chromosomes reuses completed exports after checking their input contract and file sizes. Incomplete exports require a new prefix for that chromosome; no existing files are automatically deleted.

Hail splits multiallelic sites, filters representations whose loci change during splitting, retains biallelic SNPs, and converts LGT/LA to GT when needed. PLINK performs final coordinate/allele matching. The adapter exports **WGS hardcalls**. For imputed arrays use dosage-preserving PGEN, BGEN or VCF DS. It does not create entirely absent source loci; later coverage checks handle these.

Reference APIs: [VDS split_multi](https://hail.is/docs/0.2/vds/hail.vds.split_multi.html), [LGT to GT](https://hail.is/docs/0.2/vds/hail.vds.lgt_to_gt.html), [PLINK export](https://hail.is/docs/0.2/methods/impex.html#hail.methods.export_plink).

## Connect extraction to scoring

Within the authorized workspace, make exported files available on its compute disk or mounted filesystem. PLINK needs filesystem paths, not gs:// URLs. If permitted, use the workspace storage client to localize from its own bucket; this pipeline does not move data out of that workspace.

Create one bfile entry per nonempty chromosome:

~~~yaml
genotypes:
  - {format: bfile, path: /private/workbench/inputs/cohort.chr1}
  - {format: bfile, path: /private/workbench/inputs/cohort.chr2}
# Continue through the nonempty autosomes.
~~~

Supply covariates with matching IDs and an explicit sex mapping. Retain age.mode: deferred until the original age_sd scale can be established.

~~~bash
uv run --frozen --extra tabpfn python scripts/external_inference.py run --config cohort.yaml
~~~

This prepares all frozen raw PRS and reference PCs. Review coverage, missingness and feature distributions. Start with strict missing_variants: error; do not relax it merely to obtain output.

## Scientific inputs

The adapter expects a prespecified roster and release-appropriate upstream QC. It does not choose an AoU population, infer consent/access, construct T2D/CAD/BC outcomes, select controls or replace age_sd. Use the current release dictionary and study protocol for these definitions.

Evaluate labels only after prediction with the separate evaluate command. Audit discovery-data overlap, eligibility, ascertainment, follow-up, ancestry and calibration. The previous decipEHR overlap review is not proof of independence from AoU.

## Verification boundary

The local synthetic Hail test checks recovery of a hom-reference genotype from a reference block, preservation of a truly missing call, requested-patient selection and exported PLINK genotypes. Its Hail version and result are recorded with the local acceptance artifacts.

No real AoU workspace, participant VDS, cloud IAM policy, distributed cluster or current release schema has been exercised from this HPC. Start the real deployment with a small authorized cohort/chromosome and inspect its receipt before scaling. Valid external inference still awaits a compatible age mapping.

