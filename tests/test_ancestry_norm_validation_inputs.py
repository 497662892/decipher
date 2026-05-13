import subprocess
import sys

import pandas as pd


def test_build_ancestry_norm_validation_inputs_writes_long_tables(tmp_path):
    raw_train = tmp_path / "raw_train.tsv"
    raw_val = tmp_path / "raw_val.tsv"
    soft_train = tmp_path / "soft_train.tsv"
    soft_val = tmp_path / "soft_val.tsv"
    pd.DataFrame({"IID": ["S1", "S2"], "pt": [1.0, 2.0], "ldpred2": [3.0, 4.0]}).to_csv(
        raw_train, sep="\t", index=False
    )
    pd.DataFrame({"IID": ["S3"], "pt": [1.5], "ldpred2": [3.5]}).to_csv(
        raw_val, sep="\t", index=False
    )
    pd.DataFrame({"IID": ["S1", "S2"], "pt": [0.1, 0.2], "ldpred2": [0.3, 0.4]}).to_csv(
        soft_train, sep="\t", index=False
    )
    pd.DataFrame({"IID": ["S3"], "pt": [0.15], "ldpred2": [0.35]}).to_csv(
        soft_val, sep="\t", index=False
    )
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_ancestry_norm_validation_inputs.py",
            "--disease",
            "T2D",
            "--setting",
            "raw",
            str(raw_train),
            str(raw_val),
            "--setting",
            "soft_pc10_tau1.0_shrink100",
            str(soft_train),
            str(soft_val),
            "--patient-col",
            "sampleid",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    train = pd.read_csv(out_dir / "T2D.train_long.csv")
    validation = pd.read_csv(out_dir / "T2D.validation_long.csv")

    assert len(train) == 8
    assert len(validation) == 4
    assert set(train.columns) == {
        "sampleid",
        "disease",
        "prs",
        "model",
        "base_model",
        "normalization",
        "source_score_file",
        "source_score_column",
    }
    assert set(train["model"]) == {
        "pt__raw",
        "ldpred2__raw",
        "pt__soft_pc10_tau1.0_shrink100",
        "ldpred2__soft_pc10_tau1.0_shrink100",
    }
    assert set(validation["sampleid"]) == {"S3"}
