import csv
import gzip
from copy import deepcopy

import pytest

from decipehr.external_pgs import load_weight_lock, sha256_file, write_json
from decipehr.ho_bc_weights import convert_workbook, lift_snv


class SyntheticLift:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def convert_coordinate(self, chrom, pos):
        self.calls.append((chrom, pos))
        return self.hits.get((chrom, pos), [])


def test_liftover_uses_zero_based_coordinates_and_complements_both_alleles():
    converter = SyntheticLift({("chr1", 99): [("chr1", 199, "-", 123)]})
    assert lift_snv(1, 100, "A", "C", converter) == ("1", 200, "T", "G", "mapped_-")
    assert converter.calls == [("chr1", 99)]


@pytest.mark.parametrize("hits,reason", [
    ([], "unmapped"),
    ([("chr1", 10, "+", 1), ("chr1", 20, "+", 2)], "multiple_mappings"),
    ([("chr1_alt", 10, "+", 1)], "noncanonical_mapping"),
    ([("chr2", 10, "+", 1)], "noncanonical_mapping"),
])
def test_liftover_does_not_choose_an_ambiguous_or_noncanonical_mapping(hits, reason):
    converter = SyntheticLift({("chr1", 99): hits})
    assert lift_snv(1, 100, "G", "A", converter) == ("", "", "G", "A", reason)


def make_workbook(path, rows):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    book.active.title = "Description"
    book.active.append(["Build 37 position; PRSGW_ASN"])
    for chrom in range(1, 23):
        sheet = book.create_sheet(f"Chr{chrom}")
        sheet.append(["CHR", "SNV", "Position1", "Eff. Allele2", "Ref. allele3", "PRSGW_ASN4"])
        if chrom == 1:
            for row in rows:
                sheet.append(row)
    book.save(path)


def test_published_betas_and_unmapped_denominator_survive_workbook_conversion(tmp_path):
    source, output = tmp_path / "source.xlsx", tmp_path / "weights.txt.gz"
    make_workbook(source, [[1, "rs1", 100, "A", "C", -0.125],
                           [1, "rs2", 200, "G", "A", 0.25]])
    source_hash = sha256_file(source)
    qc = convert_workbook(source, output, component="ASN", score_id="LOCAL_TEST",
                          expected_count=2, converter=SyntheticLift({
                              ("chr1", 99): [("chr1", 299, "-", 10)],
                          }))
    with gzip.open(output, "rt") as stream:
        rows = list(csv.DictReader((line for line in stream if not line.startswith("#")), delimiter="\t"))
    assert len(rows) == qc["published_variants"] == 2
    assert qc["unmapped"] == 1
    assert rows[0]["effect_allele"] == "T" and rows[0]["other_allele"] == "G"
    assert rows[0]["hm_pos"] == "300" and float(rows[0]["effect_weight"]) == -0.125
    assert rows[1]["hm_pos"] == "" and float(rows[1]["effect_weight"]) == 0.25
    assert sha256_file(source) == source_hash


def test_changed_workbook_row_count_cannot_produce_a_frozen_weight_file(tmp_path):
    source, output = tmp_path / "source.xlsx", tmp_path / "weights.txt.gz"
    make_workbook(source, [[1, "rs1", 100, "A", "C", 0.1]])
    with pytest.raises(ValueError, match="variant count"):
        convert_workbook(source, output, component="ASN", score_id="LOCAL_TEST",
                         expected_count=2, converter=SyntheticLift({}))
    assert not output.exists()


def test_local_score_lock_requires_frozen_source_and_chain_and_uses_own_namespace(tmp_path):
    weight = tmp_path / "local.txt.gz"
    with gzip.open(weight, "wt") as stream:
        stream.write("#score_id=LOCAL_HO_TEST\n#HmPOS_build=GRCh38\n#variants_number=1\n")
    source, chain = tmp_path / "source.xlsx", tmp_path / "chain.gz"
    source.write_bytes(b"synthetic source")
    chain.write_bytes(b"synthetic chain")
    item = {
        "id": "LOCAL_HO_TEST", "source_type": "publication_supplement",
        "path": str(weight), "sha256": sha256_file(weight),
        "weight_url": "https://example.org/supplement",
        "source_path": str(source), "source_sha256": sha256_file(source),
        "chain_path": str(chain), "chain_sha256": sha256_file(chain),
        "review": {"include": True, "trait_match": "confirmed", "allowed_use": "allowed",
                   "sample_overlap": "excluded", "evidence": ["synthetic fixture"]},
    }
    lock = {"schema_version": 1, "build": "GRCh38", "target_cohort": "synthetic",
            "use_context": "research", "resources": [item]}
    path = tmp_path / "lock.json"
    write_json(path, lock)
    assert load_weight_lock(path)["resources"][0]["id"] == "LOCAL_HO_TEST"
    invalid = deepcopy(lock)
    invalid["resources"][0].pop("source_sha256")
    write_json(path, invalid)
    with pytest.raises(ValueError, match="source provenance"):
        load_weight_lock(path)
    write_json(path, lock)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source has changed"):
        load_weight_lock(path)
