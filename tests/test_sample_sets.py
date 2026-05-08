import subprocess
import sys

import pandas as pd

from decipehr.io import read_table
from decipehr.sample_sets import (
    LABELED_TRAIN,
    UNLABELLED_VALIDATION,
    build_sample_set_tables,
)


def test_build_sample_set_tables_preserves_genotype_order():
    genotype = pd.DataFrame({"#IID": ["P2", "P1", "P3", "P4"], "SEX": ["NA"] * 4})
    phenotype = pd.DataFrame({"sampleid": ["P1", "P3", "P5"], "cad": [0, 1, 0]})

    tables = build_sample_set_tables(genotype, phenotype, patient_col="sampleid")

    assert tables.labeled_train_samples["sampleid"].tolist() == ["P1", "P3"]
    assert tables.unlabelled_validation_samples["sampleid"].tolist() == ["P2", "P4"]
    assert tables.phenotype_without_qc_samples["sampleid"].tolist() == ["P5"]
    assert tables.manifest["sample_set"].tolist() == [
        UNLABELLED_VALIDATION,
        LABELED_TRAIN,
        LABELED_TRAIN,
        UNLABELLED_VALIDATION,
    ]


def test_make_sample_sets_cli_writes_manifests_and_keep_files(tmp_path):
    psam = tmp_path / "all.qc.psam"
    phenotype = tmp_path / "phenotype.csv"
    out_dir = tmp_path / "sample_sets"
    pd.DataFrame({"#IID": ["P1", "P2", "P3"], "SEX": ["NA", "NA", "NA"]}).to_csv(
        psam, sep="\t", index=False
    )
    pd.DataFrame({"sampleid": ["P3", "P1"], "cad": [1, 0]}).to_csv(phenotype, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/make_sample_sets.py",
            "--genotype-psam",
            str(psam),
            "--phenotype",
            str(phenotype),
            "--patient-col",
            "sampleid",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    manifest = read_table(out_dir / "sample_sets.csv")
    counts = read_table(out_dir / "sample_set_counts.csv")
    train_keep = read_table(out_dir / "labeled_train.keep.psam")
    validation_keep = read_table(out_dir / "unlabelled_validation.keep.psam")

    assert manifest[["sampleid", "sample_set"]].to_dict("records") == [
        {"sampleid": "P1", "sample_set": LABELED_TRAIN},
        {"sampleid": "P2", "sample_set": UNLABELLED_VALIDATION},
        {"sampleid": "P3", "sample_set": LABELED_TRAIN},
    ]
    assert dict(zip(counts["sample_set"], counts["n"], strict=True)) == {
        "all_qc": 3,
        "phenotype": 2,
        LABELED_TRAIN: 2,
        UNLABELLED_VALIDATION: 1,
        "phenotype_without_qc": 0,
    }
    assert train_keep["#IID"].tolist() == ["P1", "P3"]
    assert validation_keep["#IID"].tolist() == ["P2"]
