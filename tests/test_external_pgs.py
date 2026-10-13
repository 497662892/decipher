import csv
import gzip
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decipehr.external_pgs import (
    collect_score_parts,
    load_candidates,
    load_weight_lock,
    plink_score_command,
    prepare_reference_frequencies,
    read_weight_header,
    reviewed_resources,
    score_frozen_weights,
    sha256_file,
    write_json,
    write_matching_pvar,
)


def synthetic_lock(tmp_path):
    weight_path = tmp_path / "PGS000004_hmPOS_GRCh38.txt.gz"
    text = (
        "#format_version=2.0\n#pgs_id=PGS000004\n#pgs_name=synthetic\n"
        "#trait_reported=synthetic\n#genome_build=GRCh38\n#variants_number=4\n"
        "#weight_type=beta\n#HmPOS_build=GRCh38\n"
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight"
        "\thm_source\thm_rsID\thm_chr\thm_pos\n"
        "rs1\t1\t100\tG\tA\t0.5\tAuthor-reported\trs1\t1\t100\n"
        "rs2\t1\t200\tC\tT\t-1.0\tAuthor-reported\trs2\t1\t200\n"
        "rs3\t1\t300\tA\tT\t99.0\tAuthor-reported\trs3\t1\t300\n"
        "rs4\t1\t400\tT\tC\t0.25\tAuthor-reported\trs4\t1\t400\n"
    )
    with gzip.open(weight_path, "wt") as stream:
        stream.write(text)
    review = {
        "include": True, "trait_match": "confirmed", "allowed_use": "allowed",
        "sample_overlap": "excluded", "evidence": ["Synthetic fixture only"],
    }
    resource = {
        "id": "PGS000004", "disease": "bc", "name": "synthetic",
        "weight_url": "https://example.org/synthetic", "review": review,
        "path": str(weight_path), "sha256": sha256_file(weight_path),
        "header": read_weight_header(weight_path, accession="PGS000004", build="GRCh38"),
    }
    lock = {
        "schema_version": 1, "build": "GRCh38", "resources": [resource],
        "target_cohort": "synthetic", "use_context": "research",
    }
    path = tmp_path / "weights.lock.json"
    write_json(path, lock)
    return path, lock


def test_candidate_manifest_preserves_three_traits_and_component_identity():
    candidates = load_candidates("configs/external_pgs.yaml")
    assert len(candidates) == 34
    assert {item["disease"] for item in candidates} == {"T2D", "cad", "bc"}
    components = [item for item in candidates if item["name"] == "dprism_t2d"]
    assert len(components) == 25
    assert all(item["role"] == "published_components" for item in components)
    assert len(load_candidates("configs/external_pgs.yaml", ["bc"])) == 5


@pytest.mark.parametrize("field,bad", [
    ("sample_overlap", "unknown"), ("sample_overlap", "present"),
    ("allowed_use", "unknown"), ("trait_match", "unknown"), ("evidence", []),
])
def test_unresolved_resource_cannot_enter_main_analysis(tmp_path, field, bad):
    _, lock = synthetic_lock(tmp_path)
    decision = dict(lock["resources"][0]["review"])
    decision[field] = bad
    review = {"target_cohort": "synthetic", "use_context": "research", "scores": {"PGS000004": decision}}
    with pytest.raises(ValueError, match="incomplete resource review"):
        reviewed_resources(lock, review)


def test_weight_lock_detects_build_mismatch_and_changed_weights(tmp_path):
    path, lock = synthetic_lock(tmp_path)
    assert load_weight_lock(path)["resources"][0]["id"] == "PGS000004"
    weight_path = Path(lock["resources"][0]["path"])
    with pytest.raises(ValueError, match="genome build mismatch"):
        read_weight_header(weight_path, accession="PGS000004", build="GRCh37")
    with gzip.open(weight_path, "at") as stream:
        stream.write("extra_variant\n")
    with pytest.raises(ValueError, match="Frozen weights have changed"):
        load_weight_lock(path)


def test_score_collection_aligns_patients_and_sums_allele_partitions(tmp_path):
    a, b = tmp_path / "a.sscore", tmp_path / "b.sscore"
    a.write_text("#IID PGS000004_AVG PGS000004_SUM\n001 99 1.5\n002 99 -2\n")
    b.write_text("#IID PGS000004_AVG PGS000004_SUM\n002 99 0.5\n001 99 1\n")
    scores = collect_score_parts([a, b], "PGS000004", "bc").set_index("sampleid")
    assert scores.loc["001", "prs"] == 2.5
    assert scores.loc["002", "prs"] == -1.5
    named = tmp_path / "harmonized.sscore"
    named.write_text("#IID PGS000004_hmPOS_GRCh38_SUM\n001 2.5\n002 -1.5\n")
    pd.testing.assert_frame_equal(
        collect_score_parts([named], "PGS000004", "bc", tool_accession="PGS000004_hmPOS_GRCh38"),
        scores.reset_index(),
    )
    b.write_text("#IID PGS000004_SUM\n003 1\n")
    with pytest.raises(ValueError, match="different patient sets"):
        collect_score_parts([a, b], "PGS000004", "bc")
    b.write_text("#FID IID PGS000004_SUM\nf1 001 1\nf2 001 2\n")
    with pytest.raises(ValueError, match="uniquely identify"):
        collect_score_parts([b], "PGS000004", "bc")


def test_imputation_requires_external_frequency_file():
    with pytest.raises(ValueError, match="External reference"):
        plink_score_command(pfile="p", scorefile="s", frequency_file="", out="o")
    command = plink_score_command(pfile="p", scorefile="s", frequency_file="reference.afreq", out="o")
    assert command[command.index("--read-freq") + 1] == "reference.afreq"
    assert "--error-on-freq-calc" in command
    assert "cols=+scoresums" in command
    assert "--glm" not in command


def test_score_rejects_build_mismatch_before_running_tools(tmp_path):
    path, _ = synthetic_lock(tmp_path)
    with pytest.raises(ValueError, match="genome builds must match"):
        score_frozen_weights(
            path, pfile="absent", target_build="GRCh37", reference_freq="absent",
            reference_provenance="synthetic external", out_dir=tmp_path / "result",
        )


def test_matching_pvar_preserves_variant_identity_with_metadata_and_optional_columns(tmp_path):
    source, destination = tmp_path / "source.pvar", tmp_path / "matching.pvar"
    source.write_text(
        "##contig=<ID=1,length=1000>\n#CHROM\tPOS\tID\tREF\tALT\tFILTER\n"
        "1\t100\trs1\tA\tG\tPASS\n1\t200\trs2\tC\tT\tPASS\n"
    )
    write_matching_pvar(source, destination)
    assert destination.read_text() == "#CHROM\tPOS\tID\tREF\tALT\n1\t100\trs1\tA\tG\n1\t200\trs2\tC\tT\n"


def test_matching_requires_both_reference_alleles_and_reports_missing_frequencies(tmp_path):
    source, destination = tmp_path / "source.pvar", tmp_path / "matching.pvar"
    source.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\n1\t100\tswapped\tA\tG\n"
        "1\t200\tmissing\tC\tT\n1\t300\twrong_pair\tA\tC\n"
    )
    counts = write_matching_pvar(source, destination, {"swapped": ("G", "A"), "wrong_pair": ("A", "G")})
    assert counts == {"n_input": 3, "n_written": 1, "n_missing_reference": 1, "n_allele_mismatch": 1}
    assert destination.read_text().splitlines()[1:] == ["1\t100\tswapped\tA\tG"]


def test_reference_preparation_rejects_build_mismatch_before_reading_genotypes(tmp_path):
    with pytest.raises(ValueError, match="genome builds must match"):
        prepare_reference_frequencies(
            target_pvar="absent", reference_pfile="absent", target_build="GRCh37",
            reference_build="GRCh38", reference_provenance="synthetic", out_dir=tmp_path / "ref",
        )


@pytest.mark.skipif(
    os.environ.get("DECIPEHR_EXTERNAL_PGS_INTEGRATION") != "1",
    reason="Opt-in real PLINK2/PGS Catalog integration; default tests require no network or binary.",
)
def test_real_matcher_plink_alleles_missingness_and_reproducibility(tmp_path, monkeypatch):
    plink = os.environ.get("PLINK2") or shutil.which("plink2")
    assert plink, "Set PLINK2 to an existing PLINK2 executable"
    lock_path, _ = synthetic_lock(tmp_path)
    vcf = tmp_path / "synthetic.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=1,length=1000>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t1/1\t./.\t0/1\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t0/0\t1/1\t0/1\n"
        "1\t300\trs3\tA\tT\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n"
        "1\t400\trs4\tG\tA\t.\tPASS\t.\tGT\t0/1\t0/0\t1/1\n"
    )
    prefix = str(tmp_path / "genotypes")
    subprocess.run([plink, "--vcf", str(vcf), "--make-pgen", "--out", prefix], check=True, capture_output=True)
    reference_vcf = tmp_path / "external.vcf"
    reference_vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=1,length=1000>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tE1\tE2\n"
        "1\t100\texternal1\tG\tA\t.\tPASS\t.\tGT\t0/1\t1/1\n"
        "1\t200\texternal2\tC\tT\t.\tPASS\t.\tGT\t0/0\t1/1\n"
        "1\t300\texternal3\tA\tT\t.\tPASS\t.\tGT\t0/1\t0/1\n"
        "1\t400\texternal4\tG\tA\t.\tPASS\t.\tGT\t0/1\t0/1\n"
    )
    reference_prefix = str(tmp_path / "external")
    subprocess.run(
        [plink, "--vcf", str(reference_vcf), "--make-pgen", "--out", reference_prefix],
        check=True, capture_output=True,
    )
    reference_result = prepare_reference_frequencies(
        target_pvar=prefix + ".pvar", reference_pfile=reference_prefix,
        target_build="GRCh38", reference_build="GRCh38",
        reference_provenance="Two independent synthetic external individuals E1/E2",
        out_dir=tmp_path / "reference", plink2=plink, threads=1,
    )
    frequencies = Path(reference_result["frequency_file"])
    external_freq = pd.read_csv(frequencies, sep="\t").set_index("ID")
    assert external_freq.loc["rs1", "REF"] == "G"  # REF/ALT swap retained by allele labels
    assert external_freq.loc["rs1", "ALT_FREQS"] == 0.75
    assert external_freq.loc["rs1", "OBS_CT"] == 4  # external E1/E2 only
    for run in ("one", "two"):
        score_frozen_weights(
            lock_path, pfile=prefix, target_build="GRCh38", reference_freq=str(frequencies),
            reference_provenance="Synthetic fixed external frequencies", out_dir=tmp_path / run,
            plink2=plink, min_overlap=0.75, threads=1,
        )
    result = pd.read_csv(tmp_path / "one/scores.long.csv").set_index("sampleid")
    np.testing.assert_allclose(result.loc[["S1", "S2", "S3"], "prs"], [-0.75, 0.25, 0.0])
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "one/scores.long.csv"), pd.read_csv(tmp_path / "two/scores.long.csv"),
    )
    qc = pd.read_csv(tmp_path / "one/scoring_qc.csv").iloc[0]
    assert qc.n_matched_variants == 3  # palindromic rs3 excluded before matching
    assert qc.coverage == 0.75
    assert json.loads((tmp_path / "one/scoring_run.json").read_text())["build"] == "GRCh38"
    assert json.loads((tmp_path / "one/target_reference_qc.json").read_text())["n_missing_reference"] == 1

    # Real batch scoring must equal individual scoring even when two scores use
    # opposite effect alleles. A third, unmatched score is excluded before labels.
    batch_lock = json.loads(lock_path.read_text())
    original = batch_lock["resources"][0]
    with gzip.open(original["path"], "rt") as stream:
        original_text = stream.read()
    second = original_text.replace("PGS000004", "PGS000005")
    second = second.replace("G\tA\t0.5", "A\tG\t-0.25")
    second = second.replace("C\tT\t-1.0", "T\tC\t0.2")
    second = second.replace("T\tC\t0.25", "T\tC\t0.0")
    unmatched = original_text.replace("PGS000004", "PGS000006")
    for old, new in ((100, 900), (200, 910), (300, 920), (400, 930)):
        unmatched = unmatched.replace(f"\t{old}", f"\t{new}")
    for accession, text in (("PGS000005", second), ("PGS000006", unmatched)):
        resource = deepcopy(original)
        path = tmp_path / f"{accession}_hmPOS_GRCh38.txt.gz"
        with gzip.open(path, "wt") as stream:
            stream.write(text)
        resource.update(id=accession, path=str(path), sha256=sha256_file(path))
        resource["header"] = read_weight_header(path, accession=accession, build="GRCh38")
        batch_lock["resources"].append(resource)
    local_id = "LOCAL_HO_SYNTHETIC"
    local_text = original_text.replace("#pgs_id=PGS000004", f"#score_id={local_id}")
    local_text = local_text.replace("#HmPOS_build=GRCh38", "#HmPOS_build=GRCh38\n#HmPOS_date=2026-10-12")
    local_path = tmp_path / f"{local_id}_hmPOS_GRCh38.txt.gz"
    with gzip.open(local_path, "wt") as stream:
        stream.write(local_text)
    source_path, chain_path = tmp_path / "published.xlsx", tmp_path / "lift.chain"
    source_path.write_bytes(b"synthetic published source")
    chain_path.write_bytes(b"synthetic chain provenance")
    batch_lock["resources"].append({
        **deepcopy(original), "id": local_id, "source_type": "publication_supplement",
        "path": str(local_path), "sha256": sha256_file(local_path),
        "header": read_weight_header(local_path, accession=local_id, build="GRCh38"),
        "source_path": str(source_path), "source_sha256": sha256_file(source_path),
        "chain_path": str(chain_path), "chain_sha256": sha256_file(chain_path),
    })
    batch_path = tmp_path / "batch.lock.json"
    write_json(batch_path, batch_lock)
    batch_run = score_frozen_weights(
        batch_path, pfile=prefix, target_build="GRCh38", reference_freq=str(frequencies),
        reference_provenance="Synthetic independent reference", out_dir=tmp_path / "batch",
        plink2=plink, threads=1, min_overlap=0.75,
    )
    batch_scores = pd.read_csv(tmp_path / "batch/scores.long.csv").pivot(index="sampleid", columns="model", values="prs")
    np.testing.assert_allclose(batch_scores["PGS000004"], result["prs"])
    np.testing.assert_allclose(batch_scores["PGS000005"], [0, 0.025, -0.05], atol=1e-7)
    np.testing.assert_allclose(batch_scores[local_id], result["prs"])
    assert batch_run["n_requested_scores"] == 4 and batch_run["n_scored"] == 3
    scored_lock = load_weight_lock(batch_run["scored_weight_lock"])
    assert {item["id"] for item in scored_lock["resources"]} == {"PGS000004", "PGS000005", local_id}
    assert scored_lock["excluded_by_variant_qc"][0]["model"] == "PGS000006"
    assert scored_lock["excluded_by_variant_qc"][0]["coverage"] == 0

    # Production formatter output may qualify accessions with the build. Emulate
    # only this naming variation, then use the real matcher and PLINK unchanged.
    original_run = subprocess.run

    def qualify_formatter_accessions(command, **kwargs):
        result = original_run(command, **kwargs)
        if "pgscatalog-format" in command:
            directory = Path(command[command.index("-o") + 1])
            for path in directory.glob("*.gz"):
                with gzip.open(path, "rt") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    fieldnames, rows = reader.fieldnames, list(reader)
                for row in rows:
                    row["accession"] = row["accession"].split("_hmPOS_")[0] + "_hmPOS_GRCh38"
                with gzip.open(path, "wt") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
                    writer.writeheader()
                    writer.writerows(rows)
        return result

    monkeypatch.setattr(subprocess, "run", qualify_formatter_accessions)
    score_frozen_weights(
        batch_path, pfile=prefix, target_build="GRCh38", reference_freq=str(frequencies),
        reference_provenance="Synthetic independent reference", out_dir=tmp_path / "qualified",
        plink2=plink, threads=1, min_overlap=0.75,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "batch/scores.long.csv"), pd.read_csv(tmp_path / "qualified/scores.long.csv"),
    )
