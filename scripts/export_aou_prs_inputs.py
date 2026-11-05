"""Export requested autosomal SNPs from a GRCh38 Hail VDS inside its data workspace.

Uses reference blocks through Hail densification. It never downloads participant
data, constructs phenotypes, fits PRS, or turns missing calls into reference calls.
Run with the workspace's Hail/Spark environment, separately from model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time


def file_hash(hl, path):
    digest = hashlib.sha256()
    with hl.hadoop_open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_chromosome(hl, vds, loci, samples, chromosome, prefix):
    """Return aggregate counts; export biallelic SNP hardcalls without outcomes."""
    subset = hl.vds.filter_intervals(vds, [hl.parse_locus_interval(f"chr{chromosome}", reference_genome="GRCh38")])
    subset = hl.vds.filter_samples(subset, samples, remove_dead_alleles=False)
    if subset.variant_data.count_cols() != samples.count():
        raise ValueError("Some requested patient IDs are absent from the VDS")
    variant = subset.variant_data
    variant = variant.filter_rows(hl.is_defined(loci[variant.locus]))
    # Keep reference blocks. Filtering variant_data alone is not densification.
    subset = hl.vds.VariantDataset(subset.reference_data, variant)
    subset = hl.vds.split_multi(subset, filter_changed_loci=True)
    dense = hl.vds.to_dense_mt(subset)
    if "GT" not in dense.entry:
        if not {"LGT", "LA"}.issubset(set(dense.entry)):
            raise ValueError("VDS needs GT or LGT/LA calls after densification")
        dense = dense.annotate_entries(GT=hl.vds.lgt_to_gt(dense.LGT, dense.LA))
    dense = dense.filter_rows((hl.len(dense.alleles) == 2) & hl.is_snp(dense.alleles[0], dense.alleles[1]))
    dense = dense.select_entries("GT")
    n_variants, n_patients = dense.count()
    if n_variants:
        hl.export_plink(dense, prefix, call=dense.GT, fam_id="0", ind_id=dense.s,
                        pat_id="0", mat_id="0", is_female=hl.missing(hl.tbool), pheno=hl.missing(hl.tfloat64),
                        varid=hl.delimit([hl.str(dense.locus), dense.alleles[0], dense.alleles[1]], ":"))
    return {"n_exported_variants": n_variants, "n_patients": n_patients, "format": "bfile",
            "path": prefix if n_variants else None, "empty_chromosome": not n_variants}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vds", required=True, help="Current authorized release VDS path; not inferred")
    parser.add_argument("--samples", required=True, help="TSV with one unique string sampleid column")
    parser.add_argument("--variant-map", required=True, help="Bundle genetics/variants.bim, accessible to Hail workers")
    parser.add_argument("--output-prefix", required=True, help="New workspace path, e.g. gs://WORKSPACE/export/cohort")
    parser.add_argument("--source-note", required=True, help="Release, upstream sample/variant QC, and cohort definition")
    parser.add_argument("--chromosomes", nargs="+", type=int, default=list(range(1, 23)))
    parser.add_argument("--tmp-dir", required=True, help="Workspace temporary storage accessible to Hail workers")
    parser.add_argument("--log", default="hail-external-prs.log")
    parser.add_argument("--master", help="For a local synthetic smoke test, e.g. local[2]")
    args = parser.parse_args()
    if not args.source_note.strip() or not args.chromosomes or set(args.chromosomes) - set(range(1, 23)) or len(set(args.chromosomes)) != len(args.chromosomes):
        parser.error("Document source QC and select distinct autosomes 1-22")
    import hail as hl

    init = {"tmp_dir": args.tmp_dir, "log": args.log, "default_reference": "GRCh38"}
    if args.master:
        init["master"] = args.master
    hl.init(**init)
    samples = hl.import_table(args.samples, types={"sampleid": hl.tstr}).key_by("sampleid").select()
    if samples.count() == 0 or samples.count() != samples.distinct().count():
        raise ValueError("Sample table must be nonempty and unique")
    if samples.aggregate(hl.agg.any(hl.is_missing(samples.sampleid) | samples.sampleid.matches(r"\s") | (samples.sampleid == ""))):
        raise ValueError("Missing or whitespace-containing patient IDs")
    samples = samples.key_by(s=samples.sampleid).select()
    bim = hl.import_table(args.variant_map, no_header=True, types={"f0": hl.tstr, "f3": hl.tint32})
    loci = bim.key_by(locus=hl.locus("chr" + bim.f0.replace("^chr", ""), bim.f3, reference_genome="GRCh38")).select().distinct()
    vds = hl.vds.read_vds(args.vds)
    if vds.variant_data.locus.dtype.reference_genome.name != "GRCh38" or list(vds.variant_data.col_key) != ["s"] or vds.variant_data.s.dtype != hl.tstr:
        raise ValueError("Expected GRCh38 VDS with a unique string column key s")
    contract = {"vds": args.vds, "sample_sha256": file_hash(hl, args.samples),
                "variant_map_sha256": file_hash(hl, args.variant_map), "source_note": args.source_note,
                "hail_version": hl.__version__, "adapter_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    for chromosome in args.chromosomes:
        prefix = f"{args.output_prefix}.chr{chromosome}"
        receipt = prefix + ".EXPORTED.json"
        if hl.hadoop_exists(receipt):
            with hl.hadoop_open(receipt) as stream:
                saved = json.load(stream)
            if saved["contract"] != contract:
                raise ValueError("Completed export inputs changed; use a new output prefix")
            for path, size in saved["output_bytes"].items():
                if not hl.hadoop_exists(path) or hl.hadoop_stat(path)["size_bytes"] != size:
                    raise ValueError("Completed export files changed")
            print(f"Reuse chromosome {chromosome}", flush=True)
            continue
        if any(hl.hadoop_exists(prefix + suffix) for suffix in [".bed", ".bim", ".fam"]):
            raise FileExistsError("Unsealed partial chromosome export exists; use a new prefix for that chromosome")
        started = time.monotonic()
        result = export_chromosome(hl, vds, loci, samples, chromosome, prefix)
        outputs = {} if result["empty_chromosome"] else {prefix + suffix: hl.hadoop_stat(prefix + suffix)["size_bytes"] for suffix in [".bed", ".bim", ".fam"]}
        with hl.hadoop_open(receipt, "w") as stream:
            json.dump({"status": "chromosome_export_complete", "contract": contract, **result,
                "seconds": time.monotonic()-started, "output_bytes": outputs, "calls": "GT hardcalls",
                "reference_blocks_used": True, "missing_calls_replaced_with_zero": False,
                "multiallelic_changed_loci": "filtered by Hail split_multi", "outcomes_exported": False}, stream, indent=2)
        print(f"Chromosome {chromosome}: {result['n_exported_variants']:,} variants; {result['n_patients']:,} patients", flush=True)
    hl.stop()


if __name__ == "__main__":
    main()

