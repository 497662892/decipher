import json
import subprocess
import sys

import pandas as pd

from decipehr.io import read_table
from decipehr.sumstats_harmonization import harmonize_sumstats_common, read_target_pvar_from_frame


def test_harmonize_sumstats_common_outputs_method_agnostic_table():
    pvar = pd.DataFrame(
        {
            "#CHROM": [1, 1, 1],
            "POS": [100, 200, 300],
            "ID": ["rs1", "rs2", "target3"],
            "REF": ["A", "C", "T"],
            "ALT": ["G", "G", "C"],
        }
    )
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "other3", "missing"],
            "CHR": [1, 1, 1, 1],
            "BP": [100, 200, 300, 400],
            "A1": ["A", "C", "C", "A"],
            "A2": ["G", "G", "T", "C"],
            "P": [0.01, 0.02, 0.03, 0.5],
            "BETA": [0.2, 0.3, -0.1, 0.5],
            "N": [1000, 1000, 1000, 1000],
        }
    )

    harmonized, summary = harmonize_sumstats_common(
        sumstats,
        read_target_pvar_from_frame(pvar),
    )

    assert harmonized[["SNP", "CHR", "BP", "A1", "BETA", "target_ref", "target_alt", "matched_by"]].to_dict(
        "records"
    ) == [
        {
            "SNP": "rs1",
            "CHR": 1,
            "BP": 100,
            "A1": "A",
            "BETA": 0.2,
            "target_ref": "A",
            "target_alt": "G",
            "matched_by": "id",
        },
        {
            "SNP": "target3",
            "CHR": 1,
            "BP": 300,
            "A1": "C",
            "BETA": -0.1,
            "target_ref": "T",
            "target_alt": "C",
            "matched_by": "position_alleles",
        },
    ]
    assert "N" in harmonized.columns
    assert summary["n_removed_ambiguous"] == 1
    assert summary["n_matched_by_id"] == 1
    assert summary["n_matched_by_position_alleles"] == 1
    assert summary["n_unmatched_to_target_pvar"] == 1


def test_harmonize_sumstats_common_cli_writes_table_and_summary(tmp_path):
    pvar = tmp_path / "target.pvar"
    pvar.write_text(
        "#CHROM POS ID REF ALT\n"
        "1 100 rs1 A G\n"
        "1 200 target2 C T\n",
        encoding="utf-8",
    )
    sumstats = tmp_path / "trait.standardized.tsv"
    pd.DataFrame(
        {
            "SNP": ["rs1", "other2"],
            "CHR": [1, 1],
            "BP": [100, 200],
            "A1": ["G", "T"],
            "A2": ["A", "C"],
            "P": [0.01, 0.02],
            "BETA": [0.2, -0.4],
        }
    ).to_csv(sumstats, sep="\t", index=False)
    out = tmp_path / "trait.harmonized.tsv"
    summary_out = tmp_path / "trait.summary.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/harmonize_sumstats_common.py",
            "--sumstats",
            str(sumstats),
            "--target-pvar",
            str(pvar),
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ],
        check=True,
    )

    harmonized = read_table(out)
    summary = json.loads(summary_out.read_text(encoding="utf-8"))

    assert harmonized["SNP"].tolist() == ["rs1", "target2"]
    assert harmonized["matched_by"].tolist() == ["id", "position_alleles"]
    assert summary["n_harmonized_output"] == 2


def test_prepare_sumstats_common_can_reuse_standardized_source_dir(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "summary_statistics:\n"
        "  cad:\n"
        "    file: raw.tsv\n"
        "    accession: GCSTTEST\n"
        "    variant_id_strategy: decipehr\n",
        encoding="utf-8",
    )
    source_dir = tmp_path / "source_standardized"
    source_dir.mkdir()
    pd.DataFrame(
        {
            "SNP": ["rs1"],
            "CHR": [1],
            "BP": [100],
            "A1": ["A"],
            "A2": ["G"],
            "P": [0.01],
            "BETA": [0.2],
        }
    ).to_csv(source_dir / "cad.GCSTTEST.standardized.tsv.gz", sep="\t", index=False)
    pvar = tmp_path / "target.pvar"
    pvar.write_text("#CHROM POS ID REF ALT\n1 100 rs1 A G\n", encoding="utf-8")
    out_dir = tmp_path / "final" / "sumstats"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_sumstats_common.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--sumstats-root",
            str(tmp_path / "raw_root"),
            "--target-pvar",
            str(pvar),
            "--out-dir",
            str(out_dir),
            "--standardized-source-dir",
            str(source_dir),
        ],
        check=True,
    )

    harmonized = read_table(out_dir / "harmonized" / "cad.GCSTTEST.harmonized.tsv.gz")
    summary = json.loads(
        (out_dir / "reports" / "cad.GCSTTEST.harmonized.summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert harmonized["SNP"].tolist() == ["rs1"]
    assert summary["n_harmonized_output"] == 1
    assert not (out_dir / "standardized" / "cad.GCSTTEST.standardized.tsv.gz").exists()
