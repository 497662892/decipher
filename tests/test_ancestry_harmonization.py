import json
import subprocess
import sys

import pandas as pd

from decipehr.ancestry_harmonization import match_target_to_reference


def test_match_target_to_reference_uses_position_alleles_and_removes_unsafe_variants():
    target = pd.DataFrame(
        {
            "variant_id": ["target1", "target2", "target3", "target4", "dup", "dup"],
            "chrom": [1, 1, 1, "X", 2, 2],
            "bp": [100, 200, 300, 400, 500, 501],
            "allele1": ["A", "C", "A", "A", "A", "A"],
            "allele2": ["G", "G", "T", "C", "C", "C"],
        }
    )
    reference = pd.DataFrame(
        {
            "variant_id": ["rs100", "rs200", "rs300", "rs500"],
            "chrom": [1, 1, 1, 2],
            "bp": [100, 200, 300, 500],
            "allele1": ["G", "C", "T", "A"],
            "allele2": ["A", "G", "A", "C"],
        }
    )

    matched, summary = match_target_to_reference(target, reference)

    assert matched[["target_id", "ref_id", "same_allele_order"]].to_dict("records") == [
        {"target_id": "target1", "ref_id": "rs100", "same_allele_order": False}
    ]
    assert summary["n_matched"] == 1
    assert summary["n_target_rename_to_reference_id"] == 1
    assert summary["target"]["n_removed_ambiguous"] == 2
    assert summary["target"]["n_removed_non_autosomal"] == 1
    assert summary["target"]["n_removed_duplicate_ids"] == 2


def test_make_ancestry_overlap_cli_writes_extract_lists_and_rename_map(tmp_path):
    target_pvar = tmp_path / "target.pvar"
    target_pvar.write_text(
        "#CHROM POS ID REF ALT\n"
        "1 100 target1 A G\n"
        "1 200 shared C T\n"
        "1 300 ambiguous A T\n",
        encoding="utf-8",
    )
    ref_bim = tmp_path / "1000g.bim"
    ref_bim.write_text(
        "1 rs100 0 100 G A\n"
        "1 shared 0 200 C T\n"
        "1 rs300 0 300 T A\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "ancestry_overlap"

    subprocess.run(
        [
            sys.executable,
            "scripts/make_ancestry_overlap.py",
            "--target-variants",
            str(target_pvar),
            "--reference-variants",
            str(ref_bim),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "genome",
        ],
        check=True,
    )

    assert (out_dir / "genome.target_extract.snplist").read_text(encoding="utf-8").splitlines() == [
        "target1",
        "shared",
    ]
    assert (out_dir / "genome.1000g_extract.snplist").read_text(encoding="utf-8").splitlines() == [
        "rs100",
        "shared",
    ]
    rename = pd.read_csv(out_dir / "genome.target_rename_to_1000g.tsv", sep="\t")
    assert rename.to_dict("records") == [
        {"old_id": "target1", "new_id": "rs100"},
        {"old_id": "shared", "new_id": "shared"},
    ]
    summary = json.loads((out_dir / "genome.summary.json").read_text(encoding="utf-8"))
    assert summary["n_matched"] == 2
    assert summary["n_same_variant_id"] == 1
