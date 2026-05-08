import subprocess
import sys

import pandas as pd
import yaml


def test_write_genotype_qc_commands_generates_templates(tmp_path):
    out = tmp_path / "qc.sh"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_genotype_qc_commands.py",
            "--pfile-pattern",
            "plink/Testing_Chr{chr}",
            "--info-pattern",
            "info/chr{chr}.txt.gz",
            "--out-dir",
            str(tmp_path / "qc"),
            "--out",
            str(out),
            "--chromosomes",
            "1,2",
        ],
        check=True,
    )

    text = out.read_text(encoding="utf-8")
    assert "scripts/filter_info_scores.py --info-score info/chr1.txt.gz" in text
    assert "plink2 --pfile plink/Testing_Chr2" in text
    assert "--snps-only just-acgt" in text
    assert "--max-alleles 2" in text
    assert "--geno 0.05 dosage" in text
    assert "--sort-vars" in text
    assert "--pmerge-list" in text
    assert "--mind" not in text


def test_write_genotype_qc_commands_uses_structured_config(tmp_path):
    config = tmp_path / "qc.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "variant_qc": {
                    "min_info": 0.7,
                    "min_maf": 0.02,
                    "max_variant_missingness": 0.03,
                    "info_score": {
                        "variant_col": "ID",
                        "info_col": "R2",
                        "freq_col": "AF",
                        "freq_is_maf": False,
                    },
                },
                "paths": {
                    "raw_pgen_dir": "pgen",
                    "raw_pgen_pattern": "chr{chr}",
                    "info_score_dir": "info",
                    "info_score_pattern": "chr{chr}.txt",
                    "qc_work_dir": str(tmp_path / "qc"),
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run_qc.sh"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_genotype_qc_commands.py",
            "--config",
            str(config),
            "--chromosomes",
            "1",
            "--out",
            str(out),
        ],
        check=True,
    )

    text = out.read_text(encoding="utf-8")
    assert "--info-file info/chr1.txt" in text
    assert "--variant-id-col ID" in text
    assert "--info-col R2" in text
    assert "--freq-col AF" in text
    assert "--freq-is-maf false" in text
    assert "--geno 0.03 dosage" in text


def test_filter_info_scores_writes_summary_and_supports_aliases(tmp_path):
    info = tmp_path / "chr1.info.txt"
    info.write_text(
        "col0 info_IMPUTE MAF\nv1 0.9 0.02\nv2 0.7 0.02\nv3 0.9 0.001\n",
        encoding="utf-8",
    )
    out = tmp_path / "chr1.info_maf.snplist"

    subprocess.run(
        [
            sys.executable,
            "scripts/filter_info_scores.py",
            "--info-score",
            str(info),
            "--out",
            str(out),
            "--variant-col",
            "col0",
            "--maf-col",
            "MAF",
        ],
        check=True,
    )

    assert out.read_text(encoding="utf-8").splitlines() == ["v1"]
    summary = yaml.safe_load((tmp_path / "chr1.info_maf.summary.json").read_text(encoding="utf-8"))
    assert summary["n_total"] == 3
    assert summary["n_pass_all"] == 1


def test_harmonize_gwas_for_prs_outputs_score_and_summary(tmp_path):
    pvar = tmp_path / "target.pvar"
    pvar.write_text(
        "#CHROM POS ID REF ALT\n"
        "1 100 rs1 A G\n"
        "1 200 rs2 C G\n"
        "1 300 target3 T C\n",
        encoding="utf-8",
    )
    gwas = tmp_path / "gwas.tsv"
    pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "other3", "missing"],
            "CHR": [1, 1, 1, 1],
            "BP": [100, 200, 300, 400],
            "EA": ["A", "C", "C", "A"],
            "OA": ["G", "G", "T", "C"],
            "BETA": [0.2, 0.3, -0.1, 0.5],
        }
    ).to_csv(gwas, sep="\t", index=False)
    out = tmp_path / "trait.harmonized.score.tsv"

    subprocess.run(
        [
            sys.executable,
            "scripts/harmonize_gwas_for_prs.py",
            "--gwas",
            str(gwas),
            "--pvar",
            str(pvar),
            "--out",
            str(out),
            "--chr-col",
            "CHR",
            "--bp-col",
            "BP",
        ],
        check=True,
    )

    score = pd.read_csv(out, sep="\t")
    assert score.to_dict("records") == [
        {"SNP": "rs1", "A1": "A", "BETA": 0.2},
        {"SNP": "target3", "A1": "C", "BETA": -0.1},
    ]
    summary = yaml.safe_load((tmp_path / "trait.harmonized.summary.json").read_text(encoding="utf-8"))
    assert summary["n_removed_ambiguous"] == 1
    assert summary["n_matched_by_position_alleles"] == 1


def test_write_prs_score_commands_generates_plink_score_command(tmp_path):
    config = tmp_path / "qc.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "qc_work_dir": str(tmp_path / "qc"),
                    "harmonized_score_dir": str(tmp_path / "prs"),
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run_score.sh"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_prs_score_commands.py",
            "--config",
            str(config),
            "--score-file",
            str(tmp_path / "trait.score.tsv"),
            "--trait-name",
            "CAD",
            "--out",
            str(out),
        ],
        check=True,
    )

    text = out.read_text(encoding="utf-8")
    assert "--score" in text
    assert "1 2 3 header-read cols=+scoresums" in text
    assert "CAD.prs" in text


def test_write_official_prs_commands_uses_manifest(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "variant_id_strategy": "decipehr",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "prs.sh"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_official_prs_commands.py",
            "--manifest",
            str(manifest),
            "--sumstats-root",
            "sumstats",
            "--pfile-prefix",
            "qc/all",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--score-dir",
            str(tmp_path / "pt"),
            "--out",
            str(out),
        ],
        check=True,
    )

    text = out.read_text(encoding="utf-8")
    assert "--variant-id-strategy decipehr" in text
    assert "scripts/write_pt_commands.py --genotype-kind pfile" in text


def test_write_sumstats_common_commands_keeps_method_specific_steps_out(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "variant_id_strategy": "decipehr",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "common.sh"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_sumstats_common_commands.py",
            "--manifest",
            str(manifest),
            "--sumstats-root",
            "sumstats",
            "--target-pvar",
            "qc/all.pvar",
            "--out-dir",
            str(tmp_path / "processed" / "sumstats"),
            "--out",
            str(out),
        ],
        check=True,
    )

    text = out.read_text(encoding="utf-8")
    assert "scripts/prepare_sumstats_common.py" in text
    assert "--disease cad" in text
    assert "--clump" not in text
    assert "--score" not in text


def test_write_pt_cv_commands_generates_fold_aware_array_tasks(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    splits = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "sampleid": ["P1", "P2", "P3", "P4"],
            "disease": ["cad", "cad", "cad", "cad"],
            "fold": [0, 0, 1, 1],
        }
    ).to_csv(splits, index=False)
    out_dir = tmp_path / "pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_pt_cv_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--genotype-prefix",
            "qc/all",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--out-dir",
            str(out_dir),
            "--p-thresholds",
            "1,0.05",
            "--r2-thresholds",
            "0.2",
        ],
        check=True,
    )

    task_list = out_dir / "cad" / "commands" / "cad.pt_cv.array_tasks.txt"
    tasks = task_list.read_text(encoding="utf-8").splitlines()
    assert len(tasks) == 2
    task_text = (out_dir / "cad" / "tasks" / "cad.fold0_r20p2.sh").read_text(encoding="utf-8")
    assert "--keep" in task_text
    assert "--clump-p1 1 --clump-p2 1 --clump-r2 0.2" in task_text
    assert "scripts/build_pt_scorefile.py" in task_text
    assert "--score" in task_text
    assert "1 2 header-read cols=+scoresums --score-col-nums 3 4" in task_text
    train_keep = pd.read_csv(out_dir / "cad" / "keep" / "cad.fold0.train.keep.psam", sep="\t")
    assert train_keep["#IID"].tolist() == ["P3", "P4"]


def test_write_ldpred2_cv_commands_generates_fold_aware_array_tasks(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    splits = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "sampleid": ["P1", "P2", "P3", "P4"],
            "disease": ["cad", "cad", "cad", "cad"],
            "fold": [0, 0, 1, 1],
        }
    ).to_csv(splits, index=False)
    out_dir = tmp_path / "ldpred2"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_ldpred2_cv_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--genotype-prefix",
            "qc/all",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--out-dir",
            str(out_dir),
            "--h2-grid",
            "0.1",
            "--p-grid",
            "0.01",
            "--sparse-grid",
            "0,1",
            "--match-by",
            "position",
        ],
        check=True,
    )

    task_list = out_dir / "cad" / "commands" / "cad.ldpred2_cv.array_tasks.txt"
    tasks = task_list.read_text(encoding="utf-8").splitlines()
    assert len(tasks) == 2
    task_text = (out_dir / "cad" / "tasks" / "cad.fold0.ldpred2.sh").read_text(encoding="utf-8")
    assert "--make-bed" in task_text
    assert "scripts/run_ldpred2_task.R" in task_text
    assert "--h2-grid" in task_text
    assert "--p-grid" in task_text
    assert "--sparse-grid" in task_text
    assert "--join-by-pos" in task_text
    assert "true" in task_text
    assert "REUSE_LDPRED2_SCORES" in task_text
    train_keep = pd.read_csv(out_dir / "cad" / "keep" / "cad.fold0.train.keep.psam", sep="\t")
    assert train_keep["#IID"].tolist() == ["P3", "P4"]
    manifest_out = pd.read_csv(out_dir / "cad" / "commands" / "cad.ldpred2_cv.tasks.csv")
    assert manifest_out[["disease", "fold", "n_models"]].to_dict("records") == [
        {"disease": "cad", "fold": 0, "n_models": 2},
        {"disease": "cad", "fold": 1, "n_models": 2},
    ]


def test_collect_pt_cv_scores_maps_multiscore_sscore(tmp_path):
    sscore = tmp_path / "fold0.sscore"
    sscore.write_text(
        "#IID SCORE1_SUM SCORE2_SUM\nP1 0.1 0.2\nP2 -0.1 -0.2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad"],
            "fold": [0],
            "clump_r2": [0.2],
            "p_thresholds": ["1,0.05"],
            "sscore": [str(sscore)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_pt_cv_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )

    scores = pd.read_csv(out)
    assert scores[["sampleid", "fold", "clump_r2", "p_threshold", "prs"]].to_dict("records") == [
        {"sampleid": "P1", "fold": 0, "clump_r2": 0.2, "p_threshold": 1.0, "prs": 0.1},
        {"sampleid": "P2", "fold": 0, "clump_r2": 0.2, "p_threshold": 1.0, "prs": -0.1},
        {"sampleid": "P1", "fold": 0, "clump_r2": 0.2, "p_threshold": 0.05, "prs": 0.2},
        {"sampleid": "P2", "fold": 0, "clump_r2": 0.2, "p_threshold": 0.05, "prs": -0.2},
    ]


def test_collect_pt_cv_scores_maps_named_plink2_sscore_columns(tmp_path):
    sscore = tmp_path / "fold0.sscore"
    sscore.write_text(
        "#IID BETA_p1_AVG BETA_p0p05_AVG BETA_p1_SUM BETA_p0p05_SUM\n"
        "P1 0.01 0.02 0.1 0.2\n"
        "P2 -0.01 -0.02 -0.1 -0.2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad"],
            "fold": [0],
            "clump_r2": [0.2],
            "p_thresholds": ["1,0.05"],
            "sscore": [str(sscore)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_pt_cv_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )

    scores = pd.read_csv(out)
    assert scores[["sampleid", "fold", "clump_r2", "p_threshold", "prs"]].to_dict("records") == [
        {"sampleid": "P1", "fold": 0, "clump_r2": 0.2, "p_threshold": 1.0, "prs": 0.1},
        {"sampleid": "P2", "fold": 0, "clump_r2": 0.2, "p_threshold": 1.0, "prs": -0.1},
        {"sampleid": "P1", "fold": 0, "clump_r2": 0.2, "p_threshold": 0.05, "prs": 0.2},
        {"sampleid": "P2", "fold": 0, "clump_r2": 0.2, "p_threshold": 0.05, "prs": -0.2},
    ]


def test_collect_ldpred2_cv_scores_concatenates_fold_outputs(tmp_path):
    score0 = tmp_path / "fold0.csv"
    score1 = tmp_path / "fold1.csv"
    pd.DataFrame(
        {
            "sampleid": ["P1", "P2"],
            "disease": ["cad", "cad"],
            "fold": [0, 0],
            "h2": [0.1, 0.1],
            "p": [0.01, 0.01],
            "sparse": [0, 0],
            "model": ["h20p1_p0p01_sparse0", "h20p1_p0p01_sparse0"],
            "prs": [0.2, -0.2],
        }
    ).to_csv(score0, index=False)
    pd.DataFrame(
        {
            "sampleid": ["P1", "P2"],
            "disease": ["cad", "cad"],
            "fold": [1, 1],
            "h2": [0.1, 0.1],
            "p": [0.01, 0.01],
            "sparse": [1, 1],
            "model": ["h20p1_p0p01_sparse1", "h20p1_p0p01_sparse1"],
            "prs": [0.3, -0.3],
        }
    ).to_csv(score1, index=False)
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad", "cad"],
            "fold": [0, 1],
            "score_file": [str(score0), str(score1)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_ldpred2_cv_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )

    scores = pd.read_csv(out)
    assert scores[["sampleid", "fold", "h2", "p", "sparse", "prs"]].to_dict("records") == [
        {"sampleid": "P1", "fold": 0, "h2": 0.1, "p": 0.01, "sparse": 0, "prs": 0.2},
        {"sampleid": "P2", "fold": 0, "h2": 0.1, "p": 0.01, "sparse": 0, "prs": -0.2},
        {"sampleid": "P1", "fold": 1, "h2": 0.1, "p": 0.01, "sparse": 1, "prs": 0.3},
        {"sampleid": "P2", "fold": 1, "h2": 0.1, "p": 0.01, "sparse": 1, "prs": -0.3},
    ]


def test_evaluate_ldpred2_cv_writes_summary_metrics(tmp_path):
    phenotype = tmp_path / "phenotype.csv"
    splits = tmp_path / "splits.csv"
    scores = tmp_path / "scores.csv"
    eligibility = tmp_path / "eligibility.yaml"
    out_dir = tmp_path / "evaluation"
    rows = []
    split_rows = []
    score_rows = []
    for i in range(60):
        patient = f"P{i:03d}"
        label = i % 2
        rows.append({"sampleid": patient, "cad": label})
        split_rows.append({"sampleid": patient, "disease": "cad", "fold": i % 5})
    for fold in range(5):
        for i in range(60):
            patient = f"P{i:03d}"
            score_rows.append(
                {
                    "sampleid": patient,
                    "disease": "cad",
                    "fold": fold,
                    "h2": 0.1,
                    "p": 0.01,
                    "sparse": 0,
                    "prs": float((i % 2) + (i % 7) / 100),
                }
            )
    pd.DataFrame(rows).to_csv(phenotype, index=False)
    pd.DataFrame(split_rows).to_csv(splits, index=False)
    pd.DataFrame(score_rows).to_csv(scores, index=False)
    eligibility.write_text("disease_eligibility: {}\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_ldpred2_cv.py",
            "--disease",
            "cad",
            "--phenotype",
            str(phenotype),
            "--splits",
            str(splits),
            "--scores",
            str(scores),
            "--eligibility-config",
            str(eligibility),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    summary = pd.read_csv(out_dir / "summary_metrics.csv")
    assert summary[["disease", "method", "h2", "p", "sparse", "n"]].to_dict("records") == [
        {
            "disease": "cad",
            "method": "ldpred2_prs_only",
            "h2": 0.1,
            "p": 0.01,
            "sparse": 0,
            "n": 60,
        }
    ]


def test_run_cv_accepts_disease_specific_splits(tmp_path):
    phenotype = tmp_path / "phenotype.tsv"
    splits = tmp_path / "splits.tsv"
    out_dir = tmp_path / "cv"
    rows = []
    split_rows = []
    for i in range(50):
        patient = f"P{i:03d}"
        rows.append(
            {
                "patient_id": patient,
                "cad": i % 2,
                "T2D": (i // 2) % 2,
                "age": float(i),
            }
        )
        for disease in ["cad", "T2D"]:
            split_rows.append({"patient_id": patient, "disease": disease, "fold": i % 5})
    pd.DataFrame(rows).to_csv(phenotype, sep="\t", index=False)
    pd.DataFrame(split_rows).to_csv(splits, sep="\t", index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_cv.py",
            "--phenotype",
            str(phenotype),
            "--splits",
            str(splits),
            "--patient-col",
            "patient_id",
            "--labels",
            "cad,T2D",
            "--covariates",
            "age",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    oof = pd.read_csv(out_dir / "oof_predictions.csv")
    assert set(oof["disease"]) == {"cad", "T2D"}
    assert len(oof) == 100
