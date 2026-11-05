"""Real PLINK integration on synthetic genotypes, with independent score expectations."""

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from decipehr.external_bundle import AGE_CONTRACT, contained, load_bundle
from decipehr.external_cohort import load_cohort_config, prepare_cohort, read_covariates
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_prediction import compatible_age


@pytest.fixture
def plink():
    executable = shutil.which("plink2")
    if executable is None:
        candidate = Path("/gpfs/share/apps/plink/2.0-20260228/plink2")
        executable = str(candidate) if candidate.exists() else None
    if executable is None:
        pytest.skip("PLINK2 required for mature-tool genotype integration tests")
    return executable


def make_case(tmp_path, plink, *, absent=False, shards=False, format="vcf"):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    variants = [("1", "sourceA", 100, "A", "G"), ("1", "sourceB", 200, "C", "T"),
                ("2", "sourceC", 300, "A", "C"), ("2", "sourceD", 400, "G", "T")]
    (bundle / "map.bim").write_text("".join(f"{c}\t{k}\t0\t{p}\t{a}\t{r}\n" for c, k, p, r, a in variants))
    (bundle / "all.snplist").write_text("sourceA\nsourceB\nsourceC\nsourceD\n")
    (bundle / "pc.snplist").write_text("sourceA\nsourceB\nsourceC\n")
    (bundle / "part1.snplist").write_text("sourceA\n")
    aliases = {f"PGS00000{i}": f"PGS00000{i}" for i in [1, 2, 3]}
    weights = np.array([[.2, .1, .3], [-.1, .05, .1], [.4, .5, -.1], [.05, .3, .2]])
    header = "ID\teffect_allele\tPGS000001\tPGS000002\tPGS000003\n"
    with gzip.open(bundle / "part0.tsv.gz", "wt") as stream:
        stream.write(header)
        for key, allele, row in zip([x[1] for x in variants], ["G", "C", "A", "T"], weights, strict=True):
            stream.write(f"{key}\t{allele}\t" + "\t".join(map(str, row)) + "\n")
    with gzip.open(bundle / "part1.tsv.gz", "wt") as stream:
        stream.write(header + "sourceA\tA\t0.07\t0\t0\n")
    (bundle / "afreq").write_text("#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n" + "".join(
        f"{c}\t{k}\t{r}\t{a}\t{freq}\t100\n" for (c, k, _, r, a), freq in zip(variants, [.25, .2, .4, .25], strict=True)))
    loadings = np.array([.1, .2, -.3])[:, None] * np.arange(1, 11)[None, :]
    (bundle / "pc.tsv").write_text("ID\tA1\t" + "\t".join(f"PC{i}" for i in range(1, 11)) + "\n" + "".join(
        f"{key}\t{allele}\t" + "\t".join(map(str, row)) + "\n"
        for key, allele, row in zip([x[1] for x in variants[:3]], ["G", "T", "C"], loadings, strict=True)))
    manifest = {"schema_version": 1, "status": "complete", "genome_build": "GRCh38", "pc_mode": "reference_pc", "oadp": False,
        "prs": {"T2D": ["PGS000001"], "cad": ["PGS000002"], "bc": ["PGS000003"]}, "score_aliases": aliases,
        "reference_frequencies": "afreq", "required_variants": "all.snplist", "variant_map": "map.bim",
        "pc": {"weights": "pc.tsv", "frequencies": "afreq", "variants": "pc.snplist"},
        "score_parts": [{"weights": "part0.tsv.gz", "variants": "all.snplist", "score_columns": [3, 4, 5]},
                        {"weights": "part1.tsv.gz", "variants": "part1.snplist", "score_columns": [3, 4, 5]}],
        "age_contract": {"id": AGE_CONTRACT}, "file_sha256": {p.name: sha256_file(p) for p in bundle.iterdir()}}
    write_json(bundle / "BUNDLE.json", manifest)
    ids = [f"000{i}" for i in range(1, 5)]
    records = ["chr1\t100\trs_external_A\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t0/1\n",
               "chr1\t200\trs_external_B\tT\tC\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t0/0\n",
               "chr2\t300\trs_external_C\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\t1/1\t0/1\n"]
    if not absent:
        records.append("chr2\t400\trs_external_D\tG\tT\t.\tPASS\t.\tGT\t0/0\t./.\t1/1\t0/1\n")
    else:
        records.append("chr2\t400\tsourceD\tG\tC\t.\tPASS\t.\tGT\t1/1\t1/1\t1/1\t1/1\n")
    sources = []
    for number, subset in enumerate([records[:2], records[2:]] if shards else [records]):
        path = tmp_path / f"input{number}.vcf"
        path.write_text('##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n##contig=<ID=chr2,length=1000>\n'
                        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + "\t".join(ids) + "\n" + "".join(subset))
        if format in {"pfile", "bfile"}:
            prefix = tmp_path / f"input{number}"
            subprocess.run([plink, "--vcf", str(path), "--make-pgen" if format == "pfile" else "--make-bed",
                            "--threads", "1", "--memory", "1024", "--out", str(prefix)], check=True, capture_output=True)
            sources.append({"format": format, "path": str(prefix)})
        else:
            sources.append({"format": format, "path": str(path)})
    cov = tmp_path / "covariates.csv"
    pd.DataFrame({"sampleid": ids[::-1], "sex": ["Female", "Unknown", "Male", "Female"]}).to_csv(cov, index=False)
    config_path = tmp_path / "cohort.yaml"
    config_path.write_text(yaml.safe_dump({"schema_version": 1, "cohort": "synthetic", "genome_build": "GRCh38",
        "bundle": str(bundle), "run_dir": str(tmp_path / "run"), "genotypes": sources, "covariates": str(cov),
        "runtime": {"plink2": plink, "threads": 1, "memory_mb": 1024}, "qc": {"max_sample_missingness": .5},
        "coverage": {"missing_variants": "reference_mean" if absent else "error", "min_prs_fraction": .5, "min_pc_fraction": 1}}))
    config = load_cohort_config(config_path)
    dosage = np.array([[0, 0, 1, 0], [1, 1, 2, .5], [2, 2, 0, 2], [1, 0, 1, 1]], dtype=float)
    if absent:
        dosage[:, 3] = .5
    expected = dosage @ weights
    expected[:, 0] += .07 * (2-dosage[:, 0])
    alt = np.column_stack([dosage[:, 0], 2-dosage[:, 1], 2-dosage[:, 2]])
    freq = np.array([.25, .2, .4])
    expected_pcs = ((alt-2*freq)/np.sqrt(2*freq*(1-freq))) @ loadings
    return config_path, config, expected[::-1], expected_pcs[::-1]


@pytest.mark.parametrize("format,shards,absent", [("vcf", False, False), ("pfile", False, False),
                                                  ("bfile", True, False), ("pfile", False, True)])
def test_real_plink_matches_frozen_scores_and_pcs(tmp_path, plink, format, shards, absent):
    _, config, expected, expected_pcs = make_case(tmp_path, plink, format=format, shards=shards, absent=absent)
    result = prepare_cohort(config)
    output = Path(config["run_dir"]) / "preparation"
    frame = pd.read_parquet(output / "genetic_features.parquet")
    assert frame.sampleid.tolist() == ["0004", "0003", "0002", "0001"]
    np.testing.assert_allclose(frame[[f"PGS00000{i}" for i in [1, 2, 3]]], expected, atol=2e-6, rtol=0)
    np.testing.assert_allclose(frame[[f"PC{i}_sd" for i in range(1, 11)]], expected_pcs, atol=5e-5, rtol=0)
    assert result["outcomes_used"] is False and result["age_used"] is False
    assert result["coverage"]["n_missing"] == int(absent)
    before = (output / "score0/scores.sscore").stat().st_mtime_ns
    covariates = pd.read_csv(config["covariates"], dtype={"sampleid": str}).assign(age_sd=[-1, 0, 1, 2])
    covariates.to_csv(config["covariates"], index=False)
    config["age"] = {"mode": "training_scale", "contract_id": AGE_CONTRACT, "provenance": "synthetic compatible scale"}
    compatible_age(config, load_bundle(config["bundle"]))
    assert prepare_cohort(config) == result
    assert (output / "score0/scores.sscore").stat().st_mtime_ns == before


def test_missing_variants_fail_in_strict_mode_and_report_coverage(tmp_path, plink):
    _, config, _, _ = make_case(tmp_path, plink, absent=True)
    config["coverage"]["missing_variants"] = "error"
    with pytest.raises(ValueError, match="required SNPs are absent"):
        prepare_cohort(config)
    report = Path(config["run_dir"]) / "preparation/coverage"
    assert (report / "missing.snplist").read_text() == "sourceD\n"
    assert len(pd.read_csv(report / "prs_coverage.csv")) == 3
    assert not (Path(config["run_dir"]) / "preparation/PREPARED.json").exists()


def test_duplicate_shards_are_rejected(tmp_path, plink):
    _, config, _, _ = make_case(tmp_path, plink)
    config["genotypes"] *= 2
    with pytest.raises(ValueError, match="Duplicate matched variants"):
        prepare_cohort(config)


def test_age_outcomes_and_bundle_boundaries(tmp_path, plink):
    path, config, _, _ = make_case(tmp_path, plink)
    bundle = load_bundle(config["bundle"])
    with pytest.raises(ValueError, match="awaits compatible age"):
        compatible_age(config, bundle)
    config["age"] = {"mode": "training_scale", "contract_id": "raw_years", "provenance": "age at enrollment"}
    with pytest.raises(ValueError, match="contract_id"):
        compatible_age(config, bundle)
    with pytest.raises(ValueError, match="Nonportable"):
        contained(config["bundle"], "../outside")
    raw = pd.read_csv(config["covariates"], dtype={"sampleid": str}).assign(T2D=[0, 1, 0, 1])
    raw.to_csv(config["covariates"], index=False)
    with pytest.raises(ValueError, match="outcome columns"):
        read_covariates(config)
    changed = yaml.safe_load(path.read_text())
    changed["genome_build"] = "GRCh37"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="GRCh38"):
        load_cohort_config(path)
    (Path(config["bundle"]) / "pc.tsv").write_text("changed")
    with pytest.raises(ValueError, match="Bundle file changed"):
        load_bundle(config["bundle"])


def test_prepared_output_tampering_is_detected(tmp_path, plink):
    _, config, _, _ = make_case(tmp_path, plink)
    prepare_cohort(config)
    target = Path(config["run_dir"]) / "preparation/scores.long.csv"
    with target.open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="Prepared input changed"):
        prepare_cohort(config)
    receipt = json.loads((target.parent / "PREPARED.json").read_text())
    assert receipt["output_sha256"]["scores.long.csv"] != sha256_file(target)


@pytest.mark.parametrize("kind", ["vcf", "pfile", "bgen"])
def test_fractional_dosage_survives_conversion_and_metadata_view(tmp_path, plink, kind):
    _, config, expected, pcs = make_case(tmp_path, plink)
    source = tmp_path / "input0.vcf"
    lines = source.read_text().splitlines()
    lines.insert(3, '##FORMAT=<ID=DS,Number=1,Type=Float,Description="ALT dosage">')
    for i, line in enumerate(lines):
        if line.startswith("chr1\t100"):
            fields = line.split("\t")
            fields[8:] = ["GT:DS", "0/0:0.25", "0/1:0.75", "1/1:1.75", "0/1:1.25"]
            lines[i] = "\t".join(fields)
    source.write_text("\n".join(lines)+"\n")
    if kind == "vcf":
        config["genotypes"] = [{"format": kind, "path": str(source), "dosage_field": "DS"}]
    else:
        prefix = tmp_path / "dosage"
        args = ["--make-pgen"] if kind == "pfile" else ["--export", "bgen-1.3", "bits=16", "ref-first"]
        subprocess.run([plink, "--vcf", str(source), "dosage=DS", *args, "--threads", "1", "--memory", "1024",
                        "--out", str(prefix)], check=True, capture_output=True)
        config["genotypes"] = [{"format": kind, "path": str(prefix) + (".bgen" if kind == "bgen" else "")}]
        if kind == "bgen":
            config["genotypes"][0].update(sample=str(prefix)+".sample", ref_mode="ref-first")
    delta = np.array([.25, -.25, -.25, .25])[::-1]
    expected += delta[:, None] * np.array([.13, .1, .3])
    pcs += delta[:, None] / np.sqrt(.375) * .1 * np.arange(1, 11)
    prepare_cohort(config)
    frame = pd.read_parquet(Path(config["run_dir"]) / "preparation/genetic_features.parquet")
    np.testing.assert_allclose(frame[[f"PGS00000{i}" for i in [1, 2, 3]]], expected, atol=5e-5, rtol=0)
    np.testing.assert_allclose(frame[[f"PC{i}_sd" for i in range(1, 11)]], pcs, atol=4e-4, rtol=0)


def test_absent_pc_preserves_frozen_plink_missing_call_semantics(tmp_path, plink):
    _, config, expected, pcs = make_case(tmp_path, plink)
    source = tmp_path / "input0.vcf"
    source.write_text(source.read_text().replace("rs_external_C\tA\tC", "sourceC\tA\tT"))
    config["coverage"].update(missing_variants="reference_mean", min_pc_fraction=.5, min_prs_fraction=.5)
    delta_effect = np.array([.2, -.8, 1.2, .2])[::-1]
    expected += delta_effect[:, None] * np.array([.4, .5, -.1])
    pcs -= delta_effect[:, None] / np.sqrt(.48) * (-.3) * np.arange(1, 11)
    # Preserve the frozen PLINK variance-standardize command exactly: its default
    # missing-call term is frequency-proportional, not the no-mean-imputation
    # zero used in some PCA recipes. This distinction is part of this bundle.
    pcs += .8 / np.sqrt(.48) * (-.3) * np.arange(1, 11)
    result = prepare_cohort(config)
    assert result["coverage"]["pc"]["fraction"] == 2/3
    frame = pd.read_parquet(Path(config["run_dir"]) / "preparation/genetic_features.parquet")
    np.testing.assert_allclose(frame[[f"PGS00000{i}" for i in [1, 2, 3]]], expected, atol=2e-6, rtol=0)
    np.testing.assert_allclose(frame[[f"PC{i}_sd" for i in range(1, 11)]], pcs, atol=5e-5, rtol=0)


