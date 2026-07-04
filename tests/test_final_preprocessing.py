import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_prepare_final_trainval_test_writes_keep_files_and_pc_ancestry(tmp_path):
    train_val = tmp_path / "train_val.tsv"
    holdout = tmp_path / "holdout.tsv"
    train_psam = tmp_path / "train.psam"
    holdout_psam = tmp_path / "holdout.psam"
    pc_cols = [f"PC{i}_sd" for i in range(1, 11)]

    train_rows = []
    for index, sample_id in enumerate(["T1", "T2", "T3", "T4"]):
        base = -2.0 if index < 2 else 2.0
        row = {
            "sampleid": sample_id,
            "cad": index % 2,
            "bc": (index + 1) % 2,
            "T2D": index % 2,
            "sex": "Female" if index % 2 else "Male",
            "age_sd": float(index),
        }
        row.update({pc_col: base + pc_index * 0.01 for pc_index, pc_col in enumerate(pc_cols)})
        train_rows.append(row)
    test_rows = []
    for index, sample_id in enumerate(["H1", "H2"]):
        base = -1.8 if index == 0 else 2.2
        row = {"sampleid": sample_id, "sex": "Female", "age_sd": float(index)}
        row.update({pc_col: base + pc_index * 0.01 for pc_index, pc_col in enumerate(pc_cols)})
        test_rows.append(row)

    _write_table(train_val, train_rows)
    _write_table(holdout, test_rows)
    pd.DataFrame({"#IID": ["T1", "T2", "T3", "T4", "EXTRA"]}).to_csv(
        train_psam, sep="\t", index=False
    )
    pd.DataFrame({"#IID": ["H1", "H2"]}).to_csv(holdout_psam, sep="\t", index=False)

    out_dir = tmp_path / "final"
    sample_dir = tmp_path / "sample_sets"
    ancestry_dir = tmp_path / "ancestry"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_final_trainval_test.py",
            "--train-val-phenotype",
            str(train_val),
            "--holdout-covariates",
            str(holdout),
            "--train-qc-psam",
            str(train_psam),
            "--holdout-psam",
            str(holdout_psam),
            "--out-dir",
            str(out_dir),
            "--sample-set-dir",
            str(sample_dir),
            "--ancestry-dir",
            str(ancestry_dir),
            "--n-clusters",
            "2",
            "--quality-k-min",
            "2",
            "--quality-k-max",
            "2",
            "--n-init",
            "5",
        ],
        check=True,
    )

    train_keep = pd.read_csv(sample_dir / "train_val.keep.psam", sep="\t")
    holdout_keep = pd.read_csv(sample_dir / "holdout_test.keep.psam", sep="\t")
    assert train_keep["#IID"].tolist() == ["T1", "T2", "T3", "T4"]
    assert holdout_keep["#IID"].tolist() == ["H1", "H2"]

    alignment = pd.read_csv(out_dir / "reports" / "sample_alignment_counts.csv")
    assert alignment.set_index("metric").loc["train_qc_not_train_val", "n"] == 1
    assignments = pd.read_csv(ancestry_dir / "pc_ancestry_assignments.csv")
    assert set(assignments["split"]) == {"train_val", "holdout_test"}
    assert assignments["ancestry_cluster"].nunique() == 2


def test_audit_pvar_compatibility_detects_exact_match(tmp_path):
    left = tmp_path / "left.pvar"
    right = tmp_path / "right.pvar"
    text = "#CHROM\tPOS\tID\tREF\tALT\n1\t10\trs1\tA\tG\n1\t20\trs2\tC\tT\n"
    left.write_text(text, encoding="utf-8")
    right.write_text("##source=test\n" + text, encoding="utf-8")
    out = tmp_path / "mismatch.csv"
    summary = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_pvar_compatibility.py",
            "--left-pvar",
            str(left),
            "--right-pvar",
            str(right),
            "--out",
            str(out),
            "--summary-out",
            str(summary),
            "--fail-on-mismatch",
        ],
        check=True,
    )

    assert pd.read_csv(out).empty
    assert '"exact_ordered_match": true' in summary.read_text(encoding="utf-8")


def test_write_final_genotype_projection_commands_generates_plink_workflow(tmp_path):
    out = tmp_path / "project.sh"
    subprocess.run(
        [
            sys.executable,
            "scripts/write_final_genotype_projection_commands.py",
            "--train-prefix-pattern",
            "train/chr{chr}",
            "--holdout-prefix-pattern",
            "holdout/Holdout_chr{chr}",
            "--train-keep",
            "train.keep.psam",
            "--test-keep",
            "test.keep.psam",
            "--out-dir",
            str(tmp_path / "final_genotype"),
            "--chromosomes",
            "1,2",
            "--out",
            str(out),
        ],
        check=True,
    )

    text = out.read_text(encoding="utf-8")
    assert "--pfile holdout/Holdout_chr1 --keep test.keep.psam --extract" in text
    assert "--pfile train/chr2 --keep train.keep.psam --extract" in text
    assert "scripts/audit_pvar_compatibility.py" in text
    assert "train_val.common" in text
    assert "holdout_test.common" in text


def test_split_1000g_ancestry_outputs_writes_train_and_holdout_tables(tmp_path):
    ancestry_dir = tmp_path / "ancestry"
    ancestry_dir.mkdir()
    train_keep = tmp_path / "train.keep.psam"
    test_keep = tmp_path / "test.keep.psam"
    pd.DataFrame({"#IID": ["T1", "T2"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["H1"]}).to_csv(test_keep, sep="\t", index=False)

    pd.DataFrame(
        {
            "IID": ["T1", "T2", "H1"],
            "d_mahal_AFR": [1.0, 2.0, 3.0],
            "nearest_1000G_superpop": ["AFR", "EUR", "EUR"],
        }
    ).to_csv(ancestry_dir / "target_1000G_distances_pc10.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "IID": ["T1", "T2", "H1"],
            "nearest_1000G_superpop": ["AFR", "EUR", "EUR"],
            "hard_label": ["AFR-like", "EUR-like", "EUR-like"],
        }
    ).to_csv(ancestry_dir / "target_1000G_hard_labels_pc10_margin0.5.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"IID": ["T1", "T2", "H1"], "w_AFR": [0.8, 0.1, 0.2], "w_EUR": [0.2, 0.9, 0.8]}
    ).to_csv(ancestry_dir / "target_1000G_soft_weights_pc10_tau1.tsv", sep="\t", index=False)

    out_dir = tmp_path / "split"
    subprocess.run(
        [
            sys.executable,
            "scripts/split_1000g_ancestry_outputs.py",
            "--ancestry-dir",
            str(ancestry_dir),
            "--train-keep",
            str(train_keep),
            "--test-keep",
            str(test_keep),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    train = pd.read_csv(out_dir / "target_1000G_hard_labels_pc10_margin0.5.train_val.tsv", sep="\t")
    holdout = pd.read_csv(
        out_dir / "target_1000G_hard_labels_pc10_margin0.5.holdout_test.tsv", sep="\t"
    )
    assert train["sampleid"].tolist() == ["T1", "T2"]
    assert holdout["sampleid"].tolist() == ["H1"]


def test_combine_1000g_ancestry_outputs_writes_combined_tables(tmp_path):
    train_dir = tmp_path / "train" / "ancestry"
    test_dir = tmp_path / "test" / "ancestry"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    train_keep = tmp_path / "train.keep.psam"
    test_keep = tmp_path / "test.keep.psam"
    pd.DataFrame({"#IID": ["T1", "T2"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["H1"]}).to_csv(test_keep, sep="\t", index=False)

    for root, ids in [(train_dir, ["T1", "T2"]), (test_dir, ["H1"])]:
        pd.DataFrame(
            {
                "IID": ids,
                "d_mahal_AFR": [1.0] * len(ids),
                "nearest_1000G_superpop": ["AFR"] * len(ids),
            }
        ).to_csv(root / "target_1000G_distances_pc10.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "IID": ids,
                "nearest_1000G_superpop": ["AFR"] * len(ids),
                "hard_label": ["AFR-like"] * len(ids),
            }
        ).to_csv(root / "target_1000G_hard_labels_pc10_margin0.5.tsv", sep="\t", index=False)
        pd.DataFrame({"IID": ids, "w_AFR": [1.0] * len(ids)}).to_csv(
            root / "target_1000G_soft_weights_pc10_tau1.tsv", sep="\t", index=False
        )

    out_dir = tmp_path / "combined"
    subprocess.run(
        [
            sys.executable,
            "scripts/combine_1000g_ancestry_outputs.py",
            "--train-ancestry-dir",
            str(train_dir),
            "--test-ancestry-dir",
            str(test_dir),
            "--train-keep",
            str(train_keep),
            "--test-keep",
            str(test_keep),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    combined = pd.read_csv(out_dir / "target_1000G_hard_labels_pc10_margin0.5.combined.tsv", sep="\t")
    assert combined["split"].tolist() == ["train_val", "train_val", "holdout_test"]
