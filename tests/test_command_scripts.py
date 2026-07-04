import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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
            "--standardized-source-dir",
            str(tmp_path / "processed" / "existing_standardized"),
            "--target-pvar-chunksize",
            "500000",
            "--reuse-standardized",
            "--out",
            str(out),
        ],
        check=True,
    )

    text = out.read_text(encoding="utf-8")
    assert "scripts/prepare_sumstats_common.py" in text
    assert "--disease cad" in text
    assert "--standardized-source-dir" in text
    assert "--target-pvar-chunksize 500000" in text
    assert "--reuse-standardized" in text
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


def test_write_pt_full_train_commands_generates_full_training_tasks(tmp_path):
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
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1", "T2"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1", "V2"]}).to_csv(score_keep, sep="\t", index=False)
    out_dir = tmp_path / "pt_full"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_pt_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--genotype-prefix",
            "qc/all",
            "--score-genotype-prefix",
            "qc/test",
            "--fit-score-genotype-prefix",
            "qc/trainval",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--out-dir",
            str(out_dir),
            "--p-thresholds",
            "1,0.05",
            "--r2-thresholds",
            "0.2",
        ],
        check=True,
    )

    task_list = out_dir / "cad" / "commands" / "cad.pt_full_train.array_tasks.txt"
    assert len(task_list.read_text(encoding="utf-8").splitlines()) == 1
    task_text = (out_dir / "cad" / "tasks" / "cad.full_train.r20p2.sh").read_text(encoding="utf-8")
    assert f"--keep {train_keep}" in task_text
    assert f"--keep {score_keep}" in task_text
    assert "--pfile qc/all --keep" in task_text
    assert "--pfile qc/test --keep" in task_text
    assert "--pfile qc/trainval --keep" in task_text
    assert task_text.count("--score-col-nums 3 4") == 2
    manifest_out = pd.read_csv(out_dir / "cad" / "commands" / "cad.pt_full_train.tasks.csv")
    assert manifest_out[["disease", "model_scope", "clump_r2"]].to_dict("records") == [
        {"disease": "cad", "model_scope": "full_train", "clump_r2": 0.2}
    ]
    assert manifest_out["fit_score_keep"].tolist() == [str(train_keep)]
    assert manifest_out["fit_sscore"].str.endswith(".fit.sscore").all()


def test_prepare_prscs_inputs_maps_raw_rsids_to_target_ids(tmp_path):
    raw = tmp_path / "raw.tsv"
    pd.DataFrame(
        {
            "chromosome": [22, 22, 1],
            "base_pair_location": [100, 200, 300],
            "effect_allele": ["A", "G", "T"],
            "other_allele": ["C", "A", "C"],
            "odds_ratio": [1.1, 0.9, 1.2],
            "p_value": [0.1, 0.2, 0.3],
            "rsid": ["rs100", "rs200", "rs300"],
            "alt": ["A", "G", "T"],
        }
    ).to_csv(raw, sep="\t", index=False)
    harmonized = tmp_path / "harmonized.tsv"
    pd.DataFrame(
        {
            "SNP": ["c22_100_snv_A_C", "c22_200_snv_A_G", "c1_300_snv_C_T"],
            "CHR": [22, 22, 1],
            "BP": [100, 200, 300],
            "A1": ["A", "G", "T"],
            "A2": ["C", "A", "C"],
            "P": [0.1, 0.2, 0.3],
            "BETA": [0.2, -0.1, 0.4],
            "source_snp": ["c22_100_snv_C_A", "c22_200_snv_A_G", "c1_300_snv_C_T"],
            "target_ref": ["A", "A", "C"],
            "target_alt": ["C", "G", "T"],
            "SE": [0.05, 0.06, 0.07],
        }
    ).to_csv(harmonized, sep="\t", index=False)
    out_prefix = tmp_path / "cad.prscs"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_prscs_inputs.py",
            "--raw-sumstats",
            str(raw),
            "--harmonized",
            str(harmonized),
            "--out-prefix",
            str(out_prefix),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--chromosomes",
            "22",
        ],
        check=True,
    )

    sst = pd.read_csv(f"{out_prefix}.sst", sep="\t")
    assert sst.to_dict("records") == [
        {"SNP": "rs100", "A1": "A", "A2": "C", "BETA": 0.2, "SE": 0.05},
        {"SNP": "rs200", "A1": "G", "A2": "A", "BETA": -0.1, "SE": 0.06},
    ]
    bim = pd.read_csv(f"{out_prefix}.bim", sep="\t", header=None)
    assert bim.iloc[:, :6].values.tolist() == [
        [22, "rs100", 0, 100, "C", "A"],
        [22, "rs200", 0, 200, "G", "A"],
    ]
    id_map = pd.read_csv(f"{out_prefix}.target_id_map.tsv", sep="\t")
    assert id_map.to_dict("records") == [
        {"rsid": "rs100", "target_snp": "c22_100_snv_A_C"},
        {"rsid": "rs200", "target_snp": "c22_200_snv_A_G"},
    ]


def test_prepare_prscs_inputs_auto_falls_back_to_p_when_se_is_empty(tmp_path):
    raw = tmp_path / "raw.tsv"
    pd.DataFrame(
        {
            "chromosome": [22, 22],
            "base_pair_location": [100, 200],
            "effect_allele": ["A", "G"],
            "other_allele": ["C", "A"],
            "odds_ratio": [1.1, 0.9],
            "p_value": [0.1, 0.2],
            "rsid": ["rs100", "rs200"],
            "alt": ["A", "G"],
        }
    ).to_csv(raw, sep="\t", index=False)
    harmonized = tmp_path / "harmonized.tsv"
    pd.DataFrame(
        {
            "SNP": ["c22_100_snv_A_C", "c22_200_snv_A_G"],
            "CHR": [22, 22],
            "BP": [100, 200],
            "A1": ["A", "G"],
            "A2": ["C", "A"],
            "P": [0.1, 0.2],
            "BETA": [0.2, -0.1],
            "source_snp": ["c22_100_snv_C_A", "c22_200_snv_A_G"],
            "target_ref": ["A", "A"],
            "target_alt": ["C", "G"],
            "SE": ["", ""],
        }
    ).to_csv(harmonized, sep="\t", index=False)
    out_prefix = tmp_path / "bc.prscs"
    summary = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_prscs_inputs.py",
            "--raw-sumstats",
            str(raw),
            "--harmonized",
            str(harmonized),
            "--out-prefix",
            str(out_prefix),
            "--summary-out",
            str(summary),
            "--chromosomes",
            "22",
            "--mode",
            "auto",
        ],
        check=True,
    )

    sst = pd.read_csv(f"{out_prefix}.sst", sep="\t")
    assert sst.to_dict("records") == [
        {"SNP": "rs100", "A1": "A", "A2": "C", "BETA": 0.2, "P": 0.1},
        {"SNP": "rs200", "A1": "G", "A2": "A", "BETA": -0.1, "P": 0.2},
    ]
    assert json.loads(summary.read_text(encoding="utf-8"))["mode"] == "beta-p"


def test_build_prscs_scorefile_maps_effects_back_to_target_ids(tmp_path):
    effects = tmp_path / "cad_pst_eff_a1_b0.5_phi1e-02_chr22.txt"
    effects.write_text("22\trs100\t100\tA\tC\t2.0e-02\n22\trs200\t200\tG\tA\t-1.0e-02\n")
    target_map = tmp_path / "map.tsv"
    pd.DataFrame(
        {
            "rsid": ["rs100", "rs200"],
            "target_snp": ["c22_100_snv_C_A", "c22_200_snv_A_G"],
        }
    ).to_csv(target_map, sep="\t", index=False)
    out = tmp_path / "score.tsv"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_prscs_scorefile.py",
            "--effect-files",
            str(effects),
            "--target-id-map",
            str(target_map),
            "--out",
            str(out),
            "--summary-out",
            str(tmp_path / "summary.json"),
        ],
        check=True,
    )

    score = pd.read_csv(out, sep="\t")
    assert score.to_dict("records") == [
        {"SNP": "c22_100_snv_C_A", "A1": "A", "BETA": 0.02},
        {"SNP": "c22_200_snv_A_G", "A1": "G", "BETA": -0.01},
    ]


def test_collect_prscs_scores_reads_named_beta_sum(tmp_path):
    sscore = tmp_path / "cad.sscore"
    sscore.write_text(
        "#IID\tALLELE_CT\tNAMED_ALLELE_DOSAGE_SUM\tBETA_AVG\tBETA_SUM\n"
        "S1\t10\t4.0\t0.01\t0.1\n"
        "S2\t10\t5.0\t0.02\t0.2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad"],
            "model_scope": ["full_train"],
            "phi": ["1e-2"],
            "sscore": [str(sscore)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_prscs_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )

    scores = pd.read_csv(out)
    assert scores[["sampleid", "disease", "model_scope", "phi", "prs"]].to_dict("records") == [
        {"sampleid": "S1", "disease": "cad", "model_scope": "full_train", "phi": 0.01, "prs": 0.1},
        {"sampleid": "S2", "disease": "cad", "model_scope": "full_train", "phi": 0.01, "prs": 0.2},
    ]


def test_write_prscs_full_train_commands_generates_chr_tasks_and_scoring(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "prscs"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_prscs_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--sumstats-root",
            "sumstats",
            "--genotype-prefix",
            "qc/all",
            "--score-genotype-prefix",
            "qc/test",
            "--fit-score-genotype-prefix",
            "qc/trainval",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--out-dir",
            str(out_dir),
            "--ref-dir",
            "refs/ldblk_1kg_eur",
            "--prscs-script",
            "tools/PRScs.py",
            "--chromosomes",
            "21,22",
            "--phi-grid",
            "auto,1e-2",
            "--n-iter",
            "50",
            "--n-burnin",
            "25",
        ],
        check=True,
    )

    task_list = out_dir / "cad" / "commands" / "cad.prscs_full_train.array_tasks.txt"
    assert len(task_list.read_text(encoding="utf-8").splitlines()) == 4
    auto_task_text = (out_dir / "cad" / "tasks" / "cad.phiauto.chr22.sh").read_text(
        encoding="utf-8"
    )
    assert "--chrom=22" in auto_task_text
    assert "--phi=" not in auto_task_text
    assert "phiauto_chr22.txt" in auto_task_text
    task_text = (out_dir / "cad" / "tasks" / "cad.phi1e-02.chr22.sh").read_text(
        encoding="utf-8"
    )
    assert "--chrom=22" in task_text
    assert "--phi=1e-2" in task_text
    assert "--n_iter=50 --n_burnin=25" in task_text
    assert "ldblk_1kg_chr22.hdf5" in task_text
    score_text = (out_dir / "cad" / "commands" / "cad.phi1e-02.score.sh").read_text(
        encoding="utf-8"
    )
    assert "scripts/build_prscs_scorefile.py" in score_text
    assert "--score" in score_text
    assert "--pfile qc/test" in score_text
    assert "--pfile qc/trainval" in score_text
    manifest_out = pd.read_csv(out_dir / "cad" / "commands" / "cad.prscs_full_train.tasks.csv")
    assert manifest_out[["disease", "model_scope", "phi", "n_gwas"]].to_dict("records") == [
        {"disease": "cad", "model_scope": "full_train", "phi": "auto", "n_gwas": 12345},
        {"disease": "cad", "model_scope": "full_train", "phi": "1e-2", "n_gwas": 12345},
    ]


def test_prscs_bc_slurm_defaults_include_auto_phi():
    prepare_text = Path("slurm/prscs_prepare_inputs_bc.sbatch").read_text(encoding="utf-8")
    train_text = Path("slurm/prscs_full_train_bc.sbatch").read_text(encoding="utf-8")

    assert 'export PHI_GRID="${PHI_GRID:-auto,1e-6,1e-4,1e-2}"' in prepare_text
    assert 'export PHI_GRID="${PHI_GRID:-auto,1e-6,1e-4,1e-2}"' in train_text
    assert "#SBATCH --array=1-88%22" in train_text


def test_full_train_slurm_passes_separate_score_genotype_prefixes():
    pt_text = Path("slurm/pt_full_train.sbatch").read_text(encoding="utf-8")
    prscs_text = Path("slurm/prscs_prepare_inputs.sbatch").read_text(encoding="utf-8")
    sdpr_text = Path("slurm/sdpr_prepare_reference.sbatch").read_text(encoding="utf-8")

    for text in [pt_text, prscs_text, sdpr_text]:
        assert "SCORE_GENOTYPE_PREFIX" in text
        assert "FIT_SCORE_GENOTYPE_PREFIX" in text
        assert "--score-genotype-prefix" in text
        assert "--fit-score-genotype-prefix" in text


def test_prepare_sdpr_inputs_writes_default_n_and_deduplicates(tmp_path):
    harmonized = tmp_path / "harmonized.tsv"
    pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs1", "rs3"],
            "CHR": [22, 22, 22, 1],
            "A1": ["A", "C", "A", "G"],
            "A2": ["G", "T", "G", "A"],
            "BETA": [0.1, -0.2, 0.1, 0.3],
            "P": [0.01, 1e-320, 0.01, 0.03],
        }
    ).to_csv(harmonized, sep="\t", index=False)
    out = tmp_path / "sdpr.ss"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_sdpr_inputs.py",
            "--harmonized",
            str(harmonized),
            "--out",
            str(out),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--chromosomes",
            "22",
            "--default-n",
            "12345",
            "--chunksize",
            "2",
        ],
        check=True,
    )

    sdpr = pd.read_csv(out, sep="\t")
    assert sdpr.to_dict("records") == [
        {"SNP": "rs1", "A1": "A", "A2": "G", "BETA": 0.1, "P": 0.01, "N": 12345.0},
        {"SNP": "rs2", "A1": "C", "A2": "T", "BETA": -0.2, "P": 1e-300, "N": 12345.0},
    ]
    summary = yaml.safe_load((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_sdpr_rows_written"] == 2
    assert summary["n_p_values_clipped_to_sdpr_min"] == 1
    assert float(summary["sdpr_min_p"]) == 1e-300


def test_prepare_sdpr_inputs_can_write_rsid_inputs_for_prebuilt_reference(tmp_path):
    raw = tmp_path / "raw.tsv"
    pd.DataFrame(
        {
            "SNP": ["raw1", "raw2"],
            "CHR": [1, 1],
            "BP": [100, 200],
            "A1": ["G", "T"],
            "A2": ["A", "C"],
            "ALT": ["G", "T"],
            "RSID": ["rs100", "rs200"],
        }
    ).to_csv(raw, sep="\t", index=False)
    harmonized = tmp_path / "harmonized.tsv"
    pd.DataFrame(
        {
            "SNP": ["c1_100_snv_A_G", "c1_200_snv_C_T"],
            "source_snp": ["c1_100_snv_A_G", "c1_200_snv_C_T"],
            "CHR": [1, 1],
            "A1": ["G", "T"],
            "A2": ["A", "C"],
            "BETA": [0.1, -0.2],
            "P": [0.01, 0.02],
        }
    ).to_csv(harmonized, sep="\t", index=False)
    target_bim = tmp_path / "target.bim"
    target_bim.write_text(
        "1\tc1_100_snv_A_G\t0\t100\tG\tA\n1\tc1_200_snv_C_T\t0\t200\tT\tC\n",
        encoding="utf-8",
    )
    out = tmp_path / "sdpr.ss"
    target_map = tmp_path / "target_id_map.tsv"
    valid_bim = tmp_path / "valid.bim"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_sdpr_inputs.py",
            "--harmonized",
            str(harmonized),
            "--raw-sumstats",
            str(raw),
            "--out",
            str(out),
            "--target-id-map-out",
            str(target_map),
            "--target-bim",
            str(target_bim),
            "--valid-bim-out",
            str(valid_bim),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--default-n",
            "12345",
        ],
        check=True,
    )

    assert pd.read_csv(out, sep="\t")["SNP"].tolist() == ["rs100", "rs200"]
    assert pd.read_csv(target_map, sep="\t").to_dict("records") == [
        {"rsid": "rs100", "target_snp": "c1_100_snv_A_G"},
        {"rsid": "rs200", "target_snp": "c1_200_snv_C_T"},
    ]
    assert pd.read_csv(valid_bim, sep="\t", header=None).iloc[:, 1].tolist() == ["rs100", "rs200"]


def test_build_sdpr_scorefile_reads_effects(tmp_path):
    effects = tmp_path / "cad.chr22.sdpr.txt"
    effects.write_text("SNP\tA1\tbeta\nrs1\tA\t0.1\nrs2\tC\t-0.2\n", encoding="utf-8")
    out = tmp_path / "score.tsv"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_sdpr_scorefile.py",
            "--effect-files",
            str(effects),
            "--out",
            str(out),
            "--summary-out",
            str(tmp_path / "summary.json"),
        ],
        check=True,
    )

    score = pd.read_csv(out, sep="\t")
    assert score.to_dict("records") == [
        {"SNP": "rs1", "A1": "A", "BETA": 0.1},
        {"SNP": "rs2", "A1": "C", "BETA": -0.2},
    ]


def test_build_sdpr_scorefile_maps_prebuilt_rsid_effects_to_target_ids(tmp_path):
    effects = tmp_path / "cad.chr22.sdpr.txt"
    effects.write_text("SNP\tA1\tbeta\nrs1\tA\t0.1\nrs2\tC\t-0.2\n", encoding="utf-8")
    target_map = tmp_path / "target_id_map.tsv"
    pd.DataFrame(
        {
            "rsid": ["rs1", "rs2"],
            "target_snp": ["c22_100_snv_G_A", "c22_200_snv_T_C"],
        }
    ).to_csv(target_map, sep="\t", index=False)
    out = tmp_path / "score.tsv"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_sdpr_scorefile.py",
            "--effect-files",
            str(effects),
            "--target-id-map",
            str(target_map),
            "--out",
            str(out),
            "--summary-out",
            str(tmp_path / "summary.json"),
        ],
        check=True,
    )

    score = pd.read_csv(out, sep="\t")
    assert score.to_dict("records") == [
        {"SNP": "c22_100_snv_G_A", "A1": "A", "BETA": 0.1},
        {"SNP": "c22_200_snv_T_C", "A1": "C", "BETA": -0.2},
    ]


def test_collect_sdpr_scores_reads_named_beta_sum(tmp_path):
    sscore = tmp_path / "cad.sscore"
    sscore.write_text(
        "#IID\tALLELE_CT\tNAMED_ALLELE_DOSAGE_SUM\tBETA_AVG\tBETA_SUM\n"
        "S1\t10\t4.0\t0.01\t0.1\n"
        "S2\t10\t5.0\t0.02\t0.2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad"],
            "model_scope": ["full_train"],
            "model": ["sdpr"],
            "sscore": [str(sscore)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_sdpr_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )

    scores = pd.read_csv(out)
    assert scores[["sampleid", "disease", "model_scope", "model", "prs"]].to_dict("records") == [
        {"sampleid": "S1", "disease": "cad", "model_scope": "full_train", "model": "sdpr", "prs": 0.1},
        {"sampleid": "S2", "disease": "cad", "model_scope": "full_train", "model": "sdpr", "prs": 0.2},
    ]


def test_prepare_sbayesrc_inputs_maps_rsids_and_effect_frequencies(tmp_path):
    raw = tmp_path / "raw.tsv"
    pd.DataFrame(
        {
            "SNP": ["raw1", "raw2"],
            "CHR": [1, 1],
            "BP": [100, 200],
            "A1": ["G", "T"],
            "A2": ["A", "C"],
            "ALT": ["G", "T"],
            "RSID": ["rs100", "rs200"],
        }
    ).to_csv(raw, sep="\t", index=False)
    harmonized = tmp_path / "harmonized.tsv"
    pd.DataFrame(
        {
            "SNP": ["c1_100_snv_A_G", "c1_200_snv_T_C"],
            "CHR": [1, 1],
            "A1": ["G", "C"],
            "A2": ["A", "T"],
            "BETA": [0.1, -0.2],
            "SE": [None, None],
            "P": [0.001, 0.02],
            "source_snp": ["c1_100_snv_A_G", "c1_200_snv_C_T"],
            "target_snp": ["c1_100_snv_A_G", "c1_200_snv_T_C"],
        }
    ).to_csv(harmonized, sep="\t", index=False)
    freq = tmp_path / "train.afreq"
    freq.write_text(
        "#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
        "1\tc1_100_snv_A_G\tA\tG\t0.25\t10\n"
        "1\tc1_200_snv_T_C\tT\tC\t0.80\t10\n",
        encoding="utf-8",
    )
    out = tmp_path / "cad.ma"
    target_map = tmp_path / "target_id_map.tsv"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_sbayesrc_inputs.py",
            "--raw-sumstats",
            str(raw),
            "--harmonized",
            str(harmonized),
            "--freq",
            str(freq),
            "--out",
            str(out),
            "--target-id-map-out",
            str(target_map),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--default-n",
            "12345",
            "--chunksize",
            "1",
        ],
        check=True,
    )

    ma = pd.read_csv(out, sep="\t")
    ma["freq"] = ma["freq"].round(6)
    assert ma[["SNP", "A1", "A2", "freq", "b", "p", "N"]].to_dict("records") == [
        {"SNP": "rs100", "A1": "G", "A2": "A", "freq": 0.25, "b": 0.1, "p": 0.001, "N": 12345.0},
        {"SNP": "rs200", "A1": "C", "A2": "T", "freq": 0.8, "b": -0.2, "p": 0.02, "N": 12345.0},
    ]
    assert ma["se"].gt(0).all()
    assert pd.read_csv(target_map, sep="\t").to_dict("records") == [
        {"rsid": "rs100", "target_snp": "c1_100_snv_A_G"},
        {"rsid": "rs200", "target_snp": "c1_200_snv_T_C"},
    ]
    summary = yaml.safe_load((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_sbayesrc_rows_written"] == 2
    assert summary["rsid_join_column"] == "source_snp"
    assert summary["se_source"] == "derived_from_beta_p"


def test_build_sbayesrc_scorefile_maps_snpres_to_target_ids(tmp_path):
    snpres = tmp_path / "cad.sbayesrc.snpRes"
    snpres.write_text(
        "Id Name Chrom Position A1 A2 A1Frq A1Effect SE PIP\n"
        "1 rs1 1 100 A G 0.1 0.03 0.01 0.9\n"
        "2 rs2 1 200 C T 0.2 -0.04 0.02 0.8\n",
        encoding="utf-8",
    )
    target_map = tmp_path / "target_id_map.tsv"
    pd.DataFrame(
        {
            "rsid": ["rs1", "rs2"],
            "target_snp": ["c1_100_snv_G_A", "c1_200_snv_T_C"],
        }
    ).to_csv(target_map, sep="\t", index=False)
    out = tmp_path / "score.tsv"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_sbayesrc_scorefile.py",
            "--snpres",
            str(snpres),
            "--target-id-map",
            str(target_map),
            "--out",
            str(out),
            "--summary-out",
            str(tmp_path / "summary.json"),
        ],
        check=True,
    )

    score = pd.read_csv(out, sep="\t")
    assert score.to_dict("records") == [
        {"SNP": "c1_100_snv_G_A", "A1": "A", "BETA": 0.03},
        {"SNP": "c1_200_snv_T_C", "A1": "C", "BETA": -0.04},
    ]


def test_collect_sbayesrc_scores_reads_named_beta_sum(tmp_path):
    sscore = tmp_path / "cad.sscore"
    sscore.write_text(
        "#IID\tALLELE_CT\tNAMED_ALLELE_DOSAGE_SUM\tBETA_AVG\tBETA_SUM\n"
        "S1\t10\t4.0\t0.01\t0.1\n"
        "S2\t10\t5.0\t0.02\t0.2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad"],
            "model_scope": ["full_train"],
            "model": ["sbayesrc"],
            "sscore": [str(sscore)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_sbayesrc_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )

    scores = pd.read_csv(out)
    assert scores[["sampleid", "disease", "model_scope", "model", "prs"]].to_dict("records") == [
        {"sampleid": "S1", "disease": "cad", "model_scope": "full_train", "model": "sbayesrc", "prs": 0.1},
        {"sampleid": "S2", "disease": "cad", "model_scope": "full_train", "model": "sbayesrc", "prs": 0.2},
    ]


def test_write_sbayesrc_full_train_commands_generates_workflow(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1"]}).to_csv(score_keep, sep="\t", index=False)
    variant_extract = tmp_path / "hm3plus.snplist"
    variant_extract.write_text("rs1\n", encoding="utf-8")
    ldm = tmp_path / "ldm"
    ldm.mkdir()
    annot = tmp_path / "annot.txt"
    annot.write_text("SNP\tannot\nrs1\t1\n", encoding="utf-8")
    out_dir = tmp_path / "sbayesrc"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_sbayesrc_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--sumstats-root",
            str(tmp_path / "sumstats"),
            "--genotype-prefix",
            "qc/all",
            "--score-genotype-prefix",
            "qc/validation",
            "--fit-score-genotype-prefix",
            "qc/train_fit",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--variant-extract",
            str(variant_extract),
            "--out-dir",
            str(out_dir),
            "--ldm-eigen-dir",
            str(ldm),
            "--annot-file",
            str(annot),
            "--gctb",
            "tools/gctb",
            "--chromosomes",
            "21,22",
            "--chain-length",
            "50",
            "--burn-in",
            "10",
            "--thin",
            "2",
            "--no-mcmc-bin",
        ],
        check=True,
    )

    command_dir = out_dir / "cad" / "commands"
    frequency_text = (command_dir / "cad.sbayesrc.frequency.sh").read_text(encoding="utf-8")
    assert "--freq" in frequency_text
    assert str(variant_extract) in frequency_text
    prepare_text = (command_dir / "cad.sbayesrc.prepare_ma.sh").read_text(encoding="utf-8")
    assert "scripts/prepare_sbayesrc_inputs.py" in prepare_text
    impute_text = (command_dir / "cad.sbayesrc.impute_summary.sh").read_text(encoding="utf-8")
    assert "--impute-summary" in impute_text
    task_text = (out_dir / "cad" / "tasks" / "cad.sbayesrc.sh").read_text(encoding="utf-8")
    assert "--sbayes RC" in task_text
    assert "--annot" in task_text
    assert "--chain-length 50" in task_text
    assert "--no-mcmc-bin" in task_text
    score_text = (command_dir / "cad.sbayesrc.score.sh").read_text(encoding="utf-8")
    assert "scripts/build_sbayesrc_scorefile.py" in score_text
    assert "--score" in score_text
    assert "--pfile qc/validation --keep" in score_text
    assert "--pfile qc/train_fit --keep" in score_text
    manifest_out = pd.read_csv(command_dir / "cad.sbayesrc_full_train.tasks.csv")
    assert manifest_out[["disease", "model_scope", "model", "n_gwas"]].to_dict("records") == [
        {"disease": "cad", "model_scope": "full_train", "model": "sbayesrc", "n_gwas": 12345.0}
    ]
    assert manifest_out.loc[0, "score_genotype_prefix"] == "qc/validation"
    assert manifest_out.loc[0, "fit_score_genotype_prefix"] == "qc/train_fit"


def test_sbayesrc_slurm_allows_empty_variant_extract_override():
    text = Path("slurm/sbayesrc_full_train.sbatch").read_text(encoding="utf-8")

    assert 'VARIANT_EXTRACT_DEFAULT="${VARIANT_EXTRACT_DEFAULT:-data/processed/ldpred2/hm3plus_target.snplist}"' in text
    assert 'VARIANT_EXTRACT="${VARIANT_EXTRACT-$VARIANT_EXTRACT_DEFAULT}"' in text
    assert 'SCORE_GENOTYPE_PREFIX="${SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"' in text
    assert 'FIT_SCORE_GENOTYPE_PREFIX="${FIT_SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"' in text
    assert 'if [[ -n "$VARIANT_EXTRACT" ]]; then' in text
    assert "--score-genotype-prefix" in text
    assert "--fit-score-genotype-prefix" in text


def test_prepare_megaprs_inputs_writes_ldak_summary_format(tmp_path):
    harmonized = tmp_path / "harmonized.tsv"
    pd.DataFrame(
        {
            "SNP": ["c1_100_snv_A_G", "c1_200_snv_C_T"],
            "CHR": [1, 1],
            "A1": ["G", "C"],
            "A2": ["A", "T"],
            "BETA": [0.1, -0.2],
            "SE": [0.01, 0.02],
            "P": [0.001, 0.02],
        }
    ).to_csv(harmonized, sep="\t", index=False)
    freq = tmp_path / "train.afreq"
    freq.write_text(
        "#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
        "1\tc1_100_snv_A_G\tA\tG\t0.25\t10\n"
        "1\tc1_200_snv_C_T\tC\tT\t0.80\t10\n",
        encoding="utf-8",
    )
    out = tmp_path / "cad.megaprs.summaries"
    predictors = tmp_path / "cad.megaprs.predictors"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_megaprs_inputs.py",
            "--harmonized",
            str(harmonized),
            "--freq",
            str(freq),
            "--out",
            str(out),
            "--predictors-out",
            str(predictors),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--default-n",
            "12345",
            "--chunksize",
            "1",
        ],
        check=True,
    )

    summaries = pd.read_csv(out, sep="\t")
    summaries["A1Freq"] = summaries["A1Freq"].round(6)
    assert summaries.to_dict("records") == [
        {
            "Predictor": "c1_100_snv_A_G",
            "A1": "G",
            "A2": "A",
            "BETA": 0.1,
            "SE": 0.01,
            "n": 12345.0,
            "A1Freq": 0.25,
        },
        {
            "Predictor": "c1_200_snv_C_T",
            "A1": "C",
            "A2": "T",
            "BETA": -0.2,
            "SE": 0.02,
            "n": 12345.0,
            "A1Freq": 0.2,
        },
    ]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_megaprs_rows_written"] == 2
    assert predictors.read_text(encoding="utf-8").splitlines() == [
        "c1_100_snv_A_G",
        "c1_200_snv_C_T",
    ]


def test_remap_megaprs_annotations_maps_rsids_to_target_predictors(tmp_path):
    target_map = tmp_path / "target_id_map.tsv"
    pd.DataFrame(
        {
            "rsid": ["rs1", "rs2", "rs3", "rsdup", "rsdup"],
            "target_snp": [
                "c1_100_snv_A_G",
                "c1_200_snv_C_T",
                "c1_300_snv_A_C",
                "c1_400_snv_A_G",
                "c1_500_snv_A_G",
            ],
        }
    ).to_csv(target_map, sep="\t", index=False)
    predictors = tmp_path / "target.cors.bim"
    predictors.write_text(
        "1 c1_100_snv_A_G 0 100 A G\n"
        "1 c1_200_snv_C_T 0 200 C T\n"
        "1 c1_300_snv_A_C 0 300 A C\n"
        "1 c1_400_snv_A_G 0 400 A G\n"
        "1 c1_500_snv_A_G 0 500 A G\n",
        encoding="utf-8",
    )
    annotation_prefix = tmp_path / "BaselineLD"
    (tmp_path / "BaselineLD1").write_text("rs1 1\nrs2 1\nrsdup 1\n", encoding="utf-8")
    (tmp_path / "BaselineLD2").write_text("rs3 1\nrs_missing 1\n", encoding="utf-8")
    (tmp_path / "BaselineLD.names").write_text("ann1\nann2\n", encoding="utf-8")
    out_prefix = tmp_path / "mapped" / "BaselineLD"
    summary = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/remap_megaprs_annotations.py",
            "--annotation-prefix",
            str(annotation_prefix),
            "--annotation-number",
            "2",
            "--target-id-map",
            str(target_map),
            "--predictors",
            str(predictors),
            "--out-prefix",
            str(out_prefix),
            "--summary-out",
            str(summary),
        ],
        check=True,
    )

    assert (tmp_path / "mapped" / "BaselineLD1").read_text(encoding="utf-8").splitlines() == [
        "c1_100_snv_A_G 1",
        "c1_200_snv_C_T 1",
    ]
    assert (tmp_path / "mapped" / "BaselineLD2").read_text(encoding="utf-8").splitlines() == [
        "c1_300_snv_A_C 1",
    ]
    assert (tmp_path / "mapped" / "BaselineLD.names").read_text(encoding="utf-8") == "ann1\nann2\n"
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["n_duplicate_rsids_removed"] == 1
    assert report["n_total_mapped_rows"] == 3


def test_build_megaprs_scorefile_reads_ldak_effects(tmp_path):
    effects = tmp_path / "cad.megaprs.effects"
    effects.write_text(
        "Predictor A1 A2 Centre Effect\n"
        "c1_100_snv_A_G G A 0.25 0.03\n"
        "c1_200_snv_C_T C T 0.20 -0.04\n",
        encoding="utf-8",
    )
    out = tmp_path / "score.tsv"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_megaprs_scorefile.py",
            "--effects",
            str(effects),
            "--out",
            str(out),
            "--summary-out",
            str(tmp_path / "summary.json"),
        ],
        check=True,
    )

    score = pd.read_csv(out, sep="\t")
    assert score.to_dict("records") == [
        {"SNP": "c1_100_snv_A_G", "A1": "G", "BETA": 0.03},
        {"SNP": "c1_200_snv_C_T", "A1": "C", "BETA": -0.04},
    ]


def test_collect_megaprs_scores_reads_named_beta_sum(tmp_path):
    sscore = tmp_path / "cad.sscore"
    sscore.write_text(
        "#IID\tALLELE_CT\tNAMED_ALLELE_DOSAGE_SUM\tBETA_AVG\tBETA_SUM\n"
        "S1\t10\t4.0\t0.01\t0.1\n"
        "S2\t10\t5.0\t0.02\t0.2\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad"],
            "model_scope": ["full_train"],
            "model": ["megaprs_bayesr"],
            "sscore": [str(sscore)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_megaprs_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )

    scores = pd.read_csv(out)
    assert scores[["sampleid", "disease", "model_scope", "model", "prs"]].to_dict("records") == [
        {"sampleid": "S1", "disease": "cad", "model_scope": "full_train", "model": "megaprs_bayesr", "prs": 0.1},
        {"sampleid": "S2", "disease": "cad", "model_scope": "full_train", "model": "megaprs_bayesr", "prs": 0.2},
    ]


def test_write_megaprs_full_train_commands_generates_workflow(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1"]}).to_csv(score_keep, sep="\t", index=False)
    variant_extract = tmp_path / "hm3plus.snplist"
    variant_extract.write_text("rs1\n", encoding="utf-8")
    break_points = tmp_path / "berisa.txt"
    break_points.write_text("1 1 100\n", encoding="utf-8")
    out_dir = tmp_path / "megaprs"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_megaprs_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--genotype-prefix",
            "qc/all",
            "--score-genotype-prefix",
            "qc/validation",
            "--fit-score-genotype-prefix",
            "qc/train_fit",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--variant-extract",
            str(variant_extract),
            "--out-dir",
            str(out_dir),
            "--reference-dir",
            str(tmp_path / "reference"),
            "--ldak",
            "tools/ldak",
            "--break-points",
            str(break_points),
            "--max-cor",
            "0.98",
            "--model",
            "bayesr",
            "--annotation-number",
            "86",
            "--annotation-prefix",
            "refs/baselineLD.",
            "--check-mcmc",
            "0",
            "--mcmc-solve",
            "1",
        ],
        check=True,
    )

    command_dir = out_dir / "cad" / "commands"
    assert "--freq" in (command_dir / "cad.megaprs.frequency.sh").read_text(encoding="utf-8")
    reference_text = (command_dir / "cad.megaprs.make_reference.sh").read_text(encoding="utf-8")
    assert "--make-bed" in reference_text
    assert str(variant_extract) in reference_text
    prepare_text = (command_dir / "cad.megaprs.prepare_sumstats.sh").read_text(encoding="utf-8")
    assert "scripts/prepare_megaprs_inputs.py" in prepare_text
    assert "--predictors-out" in prepare_text
    cors_array_text = (command_dir / "cad.megaprs.calc_cors.array.sh").read_text(encoding="utf-8")
    assert "SLURM_ARRAY_TASK_ID" in cors_array_text
    cors_task_list = command_dir / "cad.megaprs.calc_cors.array_tasks.txt"
    cors_tasks = cors_task_list.read_text(encoding="utf-8").strip().splitlines()
    assert len(cors_tasks) == 22
    cors_text = Path(cors_tasks[0]).read_text(encoding="utf-8")
    assert "--calc-cors" in cors_text
    assert "--max-cor \\\n  0.98" in cors_text
    assert "--chr \\\n  1" in cors_text
    assert str(break_points) in cors_text
    join_text = (command_dir / "cad.megaprs.join_cors.sh").read_text(encoding="utf-8")
    assert "--join-cors" in join_text
    assert "--corslist" in join_text
    task_text = (out_dir / "cad" / "tasks" / "cad.megaprs.sh").read_text(encoding="utf-8")
    assert "--mega-prs" in task_text
    assert "--model \\\n  bayesr" in task_text
    assert "--annotation-number \\\n  86" in task_text
    assert "--check-MCMC \\\n  NO" in task_text
    assert "--MCMC-solve \\\n  YES" in task_text
    score_text = (command_dir / "cad.megaprs.score.sh").read_text(encoding="utf-8")
    assert "scripts/build_megaprs_scorefile.py" in score_text
    assert "--pfile qc/validation --keep" in score_text
    assert "--pfile qc/train_fit --keep" in score_text
    manifest_out = pd.read_csv(command_dir / "cad.megaprs_full_train.tasks.csv")
    assert manifest_out[["disease", "model_scope", "model", "n_gwas"]].to_dict("records") == [
        {"disease": "cad", "model_scope": "full_train", "model": "megaprs_bayesr", "n_gwas": 12345.0}
    ]
    assert manifest_out.loc[0, "score_genotype_prefix"] == "qc/validation"
    assert manifest_out.loc[0, "fit_score_genotype_prefix"] == "qc/train_fit"
    pre_cors_text = (command_dir / "cad.megaprs.pre_cors.sh").read_text(encoding="utf-8")
    assert pre_cors_text.index("cad.megaprs.prepare_sumstats.sh") < pre_cors_text.index(
        "cad.megaprs.make_reference.sh"
    )
    workflow_text = (command_dir / "cad.megaprs_full_train.commands.sh").read_text(encoding="utf-8")
    assert workflow_text.index("cad.megaprs.calc_cors.array_tasks.txt") < workflow_text.index(
        "cad.megaprs.join_cors.sh"
    )
    assert manifest_out["n_cors_tasks"].tolist() == [22]


def test_write_megaprs_full_train_commands_uses_summary_predictors_for_reference(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1"]}).to_csv(score_keep, sep="\t", index=False)
    out_dir = tmp_path / "megaprs"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_megaprs_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--genotype-prefix",
            "qc/all",
            "--score-genotype-prefix",
            "qc/test",
            "--fit-score-genotype-prefix",
            "qc/trainval",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    command_dir = out_dir / "cad" / "commands"
    reference_text = (command_dir / "cad.megaprs.make_reference.sh").read_text(encoding="utf-8")
    predictors_file = out_dir / "cad" / "inputs" / "cad.megaprs.predictors"
    assert str(predictors_file) in reference_text
    task_text = (out_dir / "cad" / "tasks" / "cad.megaprs.sh").read_text(encoding="utf-8")
    assert "--annotation-prefix" not in task_text
    assert "--annotation-number" not in task_text
    manifest_out = pd.read_csv(command_dir / "cad.megaprs_full_train.tasks.csv")
    assert manifest_out.loc[0, "reference_extract"] == str(predictors_file)


def test_megaprs_slurm_defaults_to_target_profile_without_annotations():
    text = Path("slurm/megaprs_full_train.sbatch").read_text(encoding="utf-8")
    assert 'MEGAPRS_PROFILE="${MEGAPRS_PROFILE:-target}"' in text
    assert 'ANNOTATION_NUMBER="${ANNOTATION_NUMBER:-}"' in text
    assert 'ANNOTATION_PREFIX="${ANNOTATION_PREFIX:-}"' in text
    assert 'SCORE_GENOTYPE_PREFIX="${SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"' in text
    assert 'FIT_SCORE_GENOTYPE_PREFIX="${FIT_SCORE_GENOTYPE_PREFIX:-$GENOTYPE_PREFIX}"' in text
    assert 'MEGAPRS_CHECK_MCMC="${MEGAPRS_CHECK_MCMC:-0}"' in text
    assert "--check-mcmc" in text
    assert "--score-genotype-prefix" in text
    assert "--fit-score-genotype-prefix" in text


def test_write_sdpr_full_train_commands_generates_chr_tasks_and_scoring(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1"]}).to_csv(score_keep, sep="\t", index=False)
    variant_extract = tmp_path / "hm3plus.snplist"
    variant_extract.write_text("rs1\n", encoding="utf-8")
    out_dir = tmp_path / "sdpr"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_sdpr_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--genotype-prefix",
            "qc/all",
            "--score-genotype-prefix",
            "qc/test",
            "--fit-score-genotype-prefix",
            "qc/trainval",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--out-dir",
            str(out_dir),
            "--variant-extract",
            str(variant_extract),
            "--sdpr-bin",
            "tools/SDPR",
            "--chromosomes",
            "21,22",
            "--m-components",
            "20",
            "--n-iter",
            "50",
            "--n-burn",
            "10",
            "--thin",
            "2",
        ],
        check=True,
    )

    task_list = out_dir / "cad" / "commands" / "cad.sdpr_full_train.array_tasks.txt"
    assert len(task_list.read_text(encoding="utf-8").splitlines()) == 2
    prepare_text = (out_dir / "cad" / "commands" / "cad.sdpr_prepare_reference.sh").read_text(
        encoding="utf-8"
    )
    assert "scripts/prepare_sdpr_inputs.py" in prepare_text
    assert "-make_ref" in prepare_text
    assert str(variant_extract) in prepare_text
    task_text = (out_dir / "cad" / "tasks" / "cad.chr22.sdpr.sh").read_text(encoding="utf-8")
    assert "-mcmc" in task_text
    assert "-chr \\\n  22" in task_text
    assert "-N \\\n  12345" in task_text
    assert "-valid" in task_text
    assert "-iter \\\n  50" in task_text
    score_text = (out_dir / "cad" / "commands" / "cad.sdpr.score.sh").read_text(
        encoding="utf-8"
    )
    assert "scripts/build_sdpr_scorefile.py" in score_text
    assert "--score" in score_text
    assert "--pfile qc/test" in score_text
    assert "--pfile qc/trainval" in score_text
    manifest_out = pd.read_csv(out_dir / "cad" / "commands" / "cad.sdpr_full_train.tasks.csv")
    assert manifest_out[["disease", "model_scope", "model", "n_gwas"]].to_dict("records") == [
        {"disease": "cad", "model_scope": "full_train", "model": "sdpr", "n_gwas": 12345}
    ]


def test_write_sdpr_full_train_commands_supports_prebuilt_reference(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1"]}).to_csv(score_keep, sep="\t", index=False)
    prebuilt_ref = tmp_path / "sdpr_ref"
    prebuilt_ref.mkdir()
    out_dir = tmp_path / "sdpr"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_sdpr_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--genotype-prefix",
            "qc/all",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--out-dir",
            str(out_dir),
            "--ref-mode",
            "prebuilt",
            "--prebuilt-ref-dir",
            str(prebuilt_ref),
            "--chromosomes",
            "22",
        ],
        check=True,
    )

    prepare_text = (out_dir / "cad" / "commands" / "cad.sdpr_prepare_reference.sh").read_text(
        encoding="utf-8"
    )
    assert "Using prebuilt SDPR LD reference" in prepare_text
    assert "-make_ref" not in prepare_text
    assert "--raw-sumstats" in prepare_text
    assert "--valid-bim-out" in prepare_text
    assert str(prebuilt_ref / "chr22.dat") in prepare_text
    task_text = (out_dir / "cad" / "tasks" / "cad.chr22.sdpr.sh").read_text(encoding="utf-8")
    assert f"-ref_dir \\\n  {prebuilt_ref}" in task_text
    assert "cad.sdpr.valid.bim" in task_text
    score_text = (out_dir / "cad" / "commands" / "cad.sdpr.score.sh").read_text(encoding="utf-8")
    assert "--target-id-map" in score_text
    manifest_out = pd.read_csv(out_dir / "cad" / "commands" / "cad.sdpr_full_train.tasks.csv")
    assert manifest_out[["ref_mode", "ref_dir", "valid_bim"]].to_dict("records") == [
        {
            "ref_mode": "prebuilt",
            "ref_dir": str(prebuilt_ref),
            "valid_bim": str(out_dir / "cad" / "inputs" / "cad.sdpr.valid.bim"),
        }
    ]


def test_write_ldpred2_cv_commands_generates_fold_aware_array_tasks(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
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
    variant_extract = tmp_path / "hm3plus.snplist"
    variant_extract.write_text("rs1\nrs2\n", encoding="utf-8")
    genetic_map_dir = tmp_path / "genetic_maps"
    ld_cache_dir = tmp_path / "ld_cache"
    ld_ref_map = tmp_path / "map_hm3_plus.rds"
    ld_ref_map.write_text("placeholder", encoding="utf-8")
    ld_ref_dir = tmp_path / "corr_hm3_plus"
    ld_ref_dir.mkdir()
    (ld_ref_dir / "LD_with_blocks_chr22.rds").write_text("placeholder", encoding="utf-8")

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
            "--ld-reference-mode",
            "ukb_hm3_plus",
            "--ld-ref-map",
            str(ld_ref_map),
            "--ld-ref-dir",
            str(ld_ref_dir),
            "--match-by",
            "position",
            "--variant-extract",
            str(variant_extract),
            "--ld-cache-dir",
            str(ld_cache_dir),
            "--genetic-map-dir",
            str(genetic_map_dir),
            "--genetic-map-type",
            "hg38_price",
        ],
        check=True,
    )

    task_list = out_dir / "cad" / "commands" / "cad.ldpred2_cv.array_tasks.txt"
    tasks = task_list.read_text(encoding="utf-8").splitlines()
    assert len(tasks) == 2
    task_text = (out_dir / "cad" / "tasks" / "cad.fold0.ldpred2.sh").read_text(encoding="utf-8")
    assert "--make-bed" in task_text
    assert "--extract" in task_text
    assert str(variant_extract) in task_text
    assert "scripts/run_ldpred2_task.R" in task_text
    assert "--h2-grid" in task_text
    assert "--p-grid" in task_text
    assert "--sparse-grid" in task_text
    assert "--ld-reference-mode \\\n  ukb_hm3_plus" in task_text
    assert "--ld-ref-map" in task_text
    assert str(ld_ref_map) in task_text
    assert "--ld-ref-dir" in task_text
    assert str(ld_ref_dir) in task_text
    assert "LD_with_blocks_chr22.rds" in task_text
    assert "--genetic-map-dir" in task_text
    assert str(genetic_map_dir) in task_text
    assert "--genetic-map-type" in task_text
    assert "hg38_price" in task_text
    assert "--join-by-pos" in task_text
    assert "true" in task_text
    assert "--default-n" in task_text
    assert "12345" in task_text
    assert "REUSE_LDPRED2_SCORES" in task_text
    assert "--reuse-ld-corr" in task_text
    assert "REUSE_LDPRED2_LD" in task_text
    assert str(ld_cache_dir / "cad" / "cad.fold0.ldpred2_corr") in task_text
    train_keep = pd.read_csv(out_dir / "cad" / "keep" / "cad.fold0.train.keep.psam", sep="\t")
    assert train_keep["#IID"].tolist() == ["P3", "P4"]
    manifest_out = pd.read_csv(out_dir / "cad" / "commands" / "cad.ldpred2_cv.tasks.csv")
    assert manifest_out[["disease", "fold", "n_models"]].to_dict("records") == [
        {"disease": "cad", "fold": 0, "n_models": 2},
        {"disease": "cad", "fold": 1, "n_models": 2},
    ]
    assert manifest_out["variant_extract"].tolist() == [str(variant_extract), str(variant_extract)]
    assert manifest_out["default_n"].tolist() == [12345, 12345]
    assert manifest_out["ld_cache_dir"].tolist() == [str(ld_cache_dir / "cad"), str(ld_cache_dir / "cad")]
    assert manifest_out[["ld_reference_mode", "ld_ref_map", "ld_ref_dir"]].to_dict(
        "records"
    ) == [
        {
            "ld_reference_mode": "ukb_hm3_plus",
            "ld_ref_map": str(ld_ref_map),
            "ld_ref_dir": str(ld_ref_dir),
        },
        {
            "ld_reference_mode": "ukb_hm3_plus",
            "ld_ref_map": str(ld_ref_map),
            "ld_ref_dir": str(ld_ref_dir),
        },
    ]


def test_write_ldpred2_full_train_commands_generates_full_training_task(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1", "T2"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1", "V2"]}).to_csv(score_keep, sep="\t", index=False)
    variant_extract = tmp_path / "hm3plus.snplist"
    variant_extract.write_text("rs1\nrs2\n", encoding="utf-8")
    ld_cache_dir = tmp_path / "ld_cache"
    out_dir = tmp_path / "ldpred2_full"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_ldpred2_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--genotype-prefix",
            "qc/all",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--out-dir",
            str(out_dir),
            "--h2-grid",
            "0.1",
            "--p-grid",
            "0.01",
            "--sparse-grid",
            "0,1",
            "--variant-extract",
            str(variant_extract),
            "--ld-cache-dir",
            str(ld_cache_dir),
            "--genetic-map-dir",
            str(tmp_path / "genetic_maps"),
        ],
        check=True,
    )

    task_list = out_dir / "cad" / "commands" / "cad.ldpred2_full_train.array_tasks.txt"
    tasks = task_list.read_text(encoding="utf-8").splitlines()
    assert len(tasks) == 1
    task_text = (out_dir / "cad" / "tasks" / "cad.full_train.ldpred2.sh").read_text(encoding="utf-8")
    assert "--model-scope \\\n  full_train" in task_text
    assert "--fit-score-keep" in task_text
    assert "--fit-score-out" in task_text
    assert str(variant_extract) in task_text
    assert "--reuse-ld-corr" in task_text
    assert "REUSE_LDPRED2_LD" in task_text
    assert str(ld_cache_dir / "cad" / "cad.full_train.ldpred2_corr") in task_text
    assert "--default-n" in task_text
    assert "12345" in task_text
    all_keep = pd.read_csv(out_dir / "cad" / "keep" / "cad.full_train.all.keep.psam", sep="\t")
    assert all_keep["#IID"].tolist() == ["T1", "T2", "V1", "V2"]
    manifest_out = pd.read_csv(out_dir / "cad" / "commands" / "cad.ldpred2_full_train.tasks.csv")
    assert manifest_out[["disease", "model_scope", "n_models"]].to_dict("records") == [
        {"disease": "cad", "model_scope": "full_train", "n_models": 2}
    ]
    assert manifest_out["score_file"].str.endswith("ldpred2_scores_long.csv").all()
    assert manifest_out["fit_score_file"].str.endswith("ldpred2_training_scores_long.csv").all()
    assert manifest_out["default_n"].tolist() == [12345]
    assert manifest_out["ld_cache_dir"].tolist() == [str(ld_cache_dir / "cad")]


def test_write_ldpred2_full_train_commands_supports_auto_mode(tmp_path):
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
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1", "T2"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1", "V2"]}).to_csv(score_keep, sep="\t", index=False)
    out_dir = tmp_path / "ldpred2_auto"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_ldpred2_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--genotype-prefix",
            "qc/all",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--out-dir",
            str(out_dir),
            "--ldpred2-mode",
            "auto",
            "--auto-p-init",
            "0.001,0.01,0.1",
            "--auto-sparse",
            "1",
            "--auto-shrink-corr",
            "0.95",
        ],
        check=True,
    )

    task_text = (out_dir / "cad" / "tasks" / "cad.full_train.ldpred2.sh").read_text(encoding="utf-8")
    assert "--ldpred2-mode" in task_text
    assert "auto" in task_text
    assert "--auto-p-init" in task_text
    assert "0.001,0.01,0.1" in task_text
    assert "--auto-sparse" in task_text
    assert "--auto-use-mle \\\n  false" in task_text
    manifest_out = pd.read_csv(out_dir / "cad" / "commands" / "cad.ldpred2_full_train.tasks.csv")
    assert manifest_out[["ldpred2_mode", "auto_p_init", "auto_use_mle", "n_models"]].to_dict("records") == [
        {"ldpred2_mode": "auto", "auto_p_init": "0.001,0.01,0.1", "auto_use_mle": False, "n_models": 6}
    ]


def test_ldpred2_auto_slurm_defaults_to_ukbb_reference():
    text = Path("slurm/ldpred2_auto_full_train.sbatch").read_text(encoding="utf-8")

    assert 'export LD_REFERENCE_MODE="${LD_REFERENCE_MODE:-ukb_hm3_plus}"' in text
    assert 'export LD_REF_MAP="${LD_REF_MAP:-data/resources/ldpred2/map_hm3_plus.rds}"' in text
    assert 'export LD_REF_DIR="${LD_REF_DIR:-data/resources/ldpred2/corr_hm3_plus}"' in text
    assert 'export LD_CACHE_DIR="${LD_CACHE_DIR:-outputs/ldpred2_auto_ukbb_ld_cache}"' in text
    assert 'export MATCH_BY="${MATCH_BY:-position}"' in text
    assert 'export REGENERATE_LDPRED2_TASKS="${REGENERATE_LDPRED2_TASKS:-1}"' in text
    assert "DEFAULT_AUTO_H2_INIT=0.05" in text
    assert "DEFAULT_AUTO_P_INIT=0.1,0.3,1" in text
    assert "DEFAULT_AUTO_H2_INIT=0.2" in text
    assert "DEFAULT_AUTO_P_INIT=0.003,0.01,0.03" in text
    assert 'export AUTO_USE_MLE="${AUTO_USE_MLE:-0}"' in text


def test_ldpred2_cv_slurm_defaults_to_ukbb_reference():
    text = Path("slurm/ldpred2_cv.sbatch").read_text(encoding="utf-8")

    assert 'LD_REFERENCE_MODE="${LD_REFERENCE_MODE:-ukb_hm3_plus}"' in text
    assert 'LD_REF_MAP="${LD_REF_MAP:-data/resources/ldpred2/map_hm3_plus.rds}"' in text
    assert 'LD_REF_DIR="${LD_REF_DIR:-data/resources/ldpred2/corr_hm3_plus}"' in text
    assert 'MATCH_BY="${MATCH_BY:-position}"' in text
    assert "--ld-reference-mode \"$LD_REFERENCE_MODE\"" in text


def test_write_ldpred2_full_train_commands_supports_ukbb_reference(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1"]}).to_csv(score_keep, sep="\t", index=False)
    ld_ref_map = tmp_path / "map_hm3_plus.rds"
    ld_ref_map.write_text("placeholder", encoding="utf-8")
    ld_ref_dir = tmp_path / "corr_hm3_plus"
    ld_ref_dir.mkdir()
    (ld_ref_dir / "LD_with_blocks_chr22.rds").write_text("placeholder", encoding="utf-8")
    out_dir = tmp_path / "ldpred2_ukbb"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_ldpred2_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--genotype-prefix",
            "qc/all",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--out-dir",
            str(out_dir),
            "--h2-grid",
            "0.1",
            "--p-grid",
            "0.01",
            "--sparse-grid",
            "0",
            "--ld-reference-mode",
            "ukb_hm3_plus",
            "--ld-ref-map",
            str(ld_ref_map),
            "--ld-ref-dir",
            str(ld_ref_dir),
            "--match-by",
            "position",
        ],
        check=True,
    )

    task_text = (out_dir / "cad" / "tasks" / "cad.full_train.ldpred2.sh").read_text(
        encoding="utf-8"
    )
    assert "--ld-reference-mode \\\n  ukb_hm3_plus" in task_text
    assert "--ld-ref-map" in task_text
    assert str(ld_ref_map) in task_text
    assert "--ld-ref-dir" in task_text
    assert str(ld_ref_dir) in task_text
    assert "--join-by-pos \\\n  true" in task_text
    assert "LD_with_blocks_chr22.rds" in task_text
    manifest_out = pd.read_csv(out_dir / "cad" / "commands" / "cad.ldpred2_full_train.tasks.csv")
    assert manifest_out[["ld_reference_mode", "ld_ref_map", "ld_ref_dir"]].to_dict("records") == [
        {
            "ld_reference_mode": "ukb_hm3_plus",
            "ld_ref_map": str(ld_ref_map),
            "ld_ref_dir": str(ld_ref_dir),
        }
    ]


def test_write_lassosum2_full_train_commands_defaults_to_ukbb_reference(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "summary_statistics": {
                    "cad": {
                        "file": "cad.tsv.gz",
                        "accession": "GCSTTEST",
                        "default_n": 12345,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep.psam"
    score_keep = tmp_path / "validation.keep.psam"
    pd.DataFrame({"#IID": ["T1"]}).to_csv(train_keep, sep="\t", index=False)
    pd.DataFrame({"#IID": ["V1"]}).to_csv(score_keep, sep="\t", index=False)
    ld_ref_map = tmp_path / "map_hm3_plus.rds"
    ld_ref_map.write_text("placeholder", encoding="utf-8")
    ld_ref_dir = tmp_path / "corr_hm3_plus"
    ld_ref_dir.mkdir()
    (ld_ref_dir / "LD_with_blocks_chr22.rds").write_text("placeholder", encoding="utf-8")
    variant_extract = tmp_path / "hm3plus.snplist"
    variant_extract.write_text("rs1\n", encoding="utf-8")
    out_dir = tmp_path / "lassosum2_ukbb"

    subprocess.run(
        [
            sys.executable,
            "scripts/write_lassosum2_full_train_commands.py",
            "--disease",
            "cad",
            "--manifest",
            str(manifest),
            "--genotype-prefix",
            "qc/all",
            "--harmonized-dir",
            str(tmp_path / "harmonized"),
            "--train-keep",
            str(train_keep),
            "--score-keep",
            str(score_keep),
            "--out-dir",
            str(out_dir),
            "--delta-grid",
            "0.001,0.01",
            "--nlambda",
            "3",
            "--ld-ref-map",
            str(ld_ref_map),
            "--ld-ref-dir",
            str(ld_ref_dir),
            "--variant-extract",
            str(variant_extract),
        ],
        check=True,
    )

    task_text = (out_dir / "cad" / "tasks" / "cad.full_train.lassosum2.sh").read_text(
        encoding="utf-8"
    )
    assert "--ld-reference-mode \\\n  ukb_hm3_plus" in task_text
    assert "--ld-ref-map" in task_text
    assert str(ld_ref_map) in task_text
    assert "--ld-ref-dir" in task_text
    assert str(ld_ref_dir) in task_text
    assert "--delta-grid \\\n  0.001,0.01" in task_text
    assert "--nlambda \\\n  3" in task_text
    assert "--join-by-pos \\\n  true" in task_text
    assert "LD_with_blocks_chr22.rds" in task_text
    manifest_out = pd.read_csv(
        out_dir / "cad" / "commands" / "cad.lassosum2_full_train.tasks.csv"
    )
    assert manifest_out[["ld_reference_mode", "delta_grid", "nlambda", "n_models"]].to_dict(
        "records"
    ) == [
        {
            "ld_reference_mode": "ukb_hm3_plus",
            "delta_grid": "0.001,0.01",
            "nlambda": 3,
            "n_models": 6,
        }
    ]


def test_collect_ldpred2_scores_reads_validation_and_training_columns(tmp_path):
    score = tmp_path / "validation.csv"
    fit_score = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "sampleid": ["V1", "V2", "V1"],
            "disease": ["cad", "cad", "cad"],
            "model_scope": ["full_train", "full_train", "full_train"],
            "fold": [0, 0, 0],
            "h2": [0.1, 0.1, 0.2],
            "p": [0.01, 0.01, 0.001],
            "sparse": [0, 0, 1],
            "model": ["h20p1_p0p01_sparse0", "h20p1_p0p01_sparse0", "h20p2_p0p001_sparse1"],
            "prs": [0.2, -0.2, None],
        }
    ).to_csv(score, index=False)
    pd.DataFrame(
        {
            "sampleid": ["T1", "T2", "T1"],
            "disease": ["cad", "cad", "cad"],
            "model_scope": ["full_train", "full_train", "full_train"],
            "fold": [0, 0, 0],
            "h2": [0.1, 0.1, 0.2],
            "p": [0.01, 0.01, 0.001],
            "sparse": [0, 0, 1],
            "model": ["h20p1_p0p01_sparse0", "h20p1_p0p01_sparse0", "h20p2_p0p001_sparse1"],
            "prs": [0.3, -0.3, None],
        }
    ).to_csv(fit_score, index=False)
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad"],
            "model_scope": ["full_train"],
            "score_file": [str(score)],
            "fit_score_file": [str(fit_score)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"
    fit_out = tmp_path / "fit_scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_ldpred2_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_ldpred2_scores.py",
            "--task-manifest",
            str(manifest),
            "--score-file-column",
            "fit_score_file",
            "--out",
            str(fit_out),
        ],
        check=True,
    )

    validation_scores = pd.read_csv(out)
    assert validation_scores[["sampleid", "model_scope", "h2", "p", "sparse", "prs"]].to_dict(
        "records"
    ) == [
        {"sampleid": "V1", "model_scope": "full_train", "h2": 0.1, "p": 0.01, "sparse": 0, "prs": 0.2},
        {"sampleid": "V2", "model_scope": "full_train", "h2": 0.1, "p": 0.01, "sparse": 0, "prs": -0.2},
    ]
    fit_scores = pd.read_csv(fit_out)
    assert fit_scores[["sampleid", "model_scope", "h2", "p", "sparse", "prs"]].to_dict("records") == [
        {"sampleid": "T1", "model_scope": "full_train", "h2": 0.1, "p": 0.01, "sparse": 0, "prs": 0.3},
        {"sampleid": "T2", "model_scope": "full_train", "h2": 0.1, "p": 0.01, "sparse": 0, "prs": -0.3},
    ]


def test_collect_lassosum2_scores_reads_validation_and_training_columns(tmp_path):
    score = tmp_path / "validation.csv"
    fit_score = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "sampleid": ["V1", "V2"],
            "disease": ["cad", "cad"],
            "model_scope": ["full_train", "full_train"],
            "fold": [0, 0],
            "delta": [0.001, 0.001],
            "lambda": [0.02, 0.02],
            "model": ["delta0p001_lambda0p02", "delta0p001_lambda0p02"],
            "sparsity": [0.9, 0.9],
            "num_iter": [20, 20],
            "prs": [0.2, -0.2],
        }
    ).to_csv(score, index=False)
    pd.DataFrame(
        {
            "sampleid": ["T1", "T2"],
            "disease": ["cad", "cad"],
            "model_scope": ["full_train", "full_train"],
            "fold": [0, 0],
            "delta": [0.001, 0.001],
            "lambda": [0.02, 0.02],
            "model": ["delta0p001_lambda0p02", "delta0p001_lambda0p02"],
            "sparsity": [0.9, 0.9],
            "num_iter": [20, 20],
            "prs": [0.3, -0.3],
        }
    ).to_csv(fit_score, index=False)
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad"],
            "model_scope": ["full_train"],
            "score_file": [str(score)],
            "fit_score_file": [str(fit_score)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"
    fit_out = tmp_path / "fit_scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_lassosum2_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_lassosum2_scores.py",
            "--task-manifest",
            str(manifest),
            "--score-file-column",
            "fit_score_file",
            "--out",
            str(fit_out),
        ],
        check=True,
    )

    validation_scores = pd.read_csv(out)
    assert validation_scores[["sampleid", "model_scope", "delta", "lambda", "prs"]].to_dict(
        "records"
    ) == [
        {"sampleid": "V1", "model_scope": "full_train", "delta": 0.001, "lambda": 0.02, "prs": 0.2},
        {"sampleid": "V2", "model_scope": "full_train", "delta": 0.001, "lambda": 0.02, "prs": -0.2},
    ]
    fit_scores = pd.read_csv(fit_out)
    assert fit_scores[["sampleid", "model_scope", "delta", "lambda", "prs"]].to_dict("records") == [
        {"sampleid": "T1", "model_scope": "full_train", "delta": 0.001, "lambda": 0.02, "prs": 0.3},
        {"sampleid": "T2", "model_scope": "full_train", "delta": 0.001, "lambda": 0.02, "prs": -0.3},
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


def test_collect_pt_scores_maps_full_train_sscore(tmp_path):
    sscore = tmp_path / "full.sscore"
    sscore.write_text(
        "#IID BETA_p1_SUM BETA_p0p05_SUM\nV1 0.1 0.2\nV2 -0.1 -0.2\n",
        encoding="utf-8",
    )
    fit_sscore = tmp_path / "full.fit.sscore"
    fit_sscore.write_text(
        "#IID BETA_p1_SUM BETA_p0p05_SUM\nT1 0.3 0.4\nT2 -0.3 -0.4\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "tasks.csv"
    pd.DataFrame(
        {
            "disease": ["cad"],
            "model_scope": ["full_train"],
            "clump_r2": [0.2],
            "p_thresholds": ["1,0.05"],
            "sscore": [str(sscore)],
            "fit_sscore": [str(fit_sscore)],
        }
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_pt_scores.py",
            "--task-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
    )

    scores = pd.read_csv(out)
    assert scores[["sampleid", "model_scope", "clump_r2", "p_threshold", "prs"]].to_dict("records") == [
        {"sampleid": "V1", "model_scope": "full_train", "clump_r2": 0.2, "p_threshold": 1.0, "prs": 0.1},
        {"sampleid": "V2", "model_scope": "full_train", "clump_r2": 0.2, "p_threshold": 1.0, "prs": -0.1},
        {"sampleid": "V1", "model_scope": "full_train", "clump_r2": 0.2, "p_threshold": 0.05, "prs": 0.2},
        {"sampleid": "V2", "model_scope": "full_train", "clump_r2": 0.2, "p_threshold": 0.05, "prs": -0.2},
    ]

    fit_out = tmp_path / "fit_scores.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_pt_scores.py",
            "--task-manifest",
            str(manifest),
            "--sscore-column",
            "fit_sscore",
            "--out",
            str(fit_out),
        ],
        check=True,
    )

    fit_scores = pd.read_csv(fit_out)
    assert fit_scores[["sampleid", "model_scope", "clump_r2", "p_threshold", "prs"]].to_dict(
        "records"
    ) == [
        {"sampleid": "T1", "model_scope": "full_train", "clump_r2": 0.2, "p_threshold": 1.0, "prs": 0.3},
        {"sampleid": "T2", "model_scope": "full_train", "clump_r2": 0.2, "p_threshold": 1.0, "prs": -0.3},
        {"sampleid": "T1", "model_scope": "full_train", "clump_r2": 0.2, "p_threshold": 0.05, "prs": 0.4},
        {"sampleid": "T2", "model_scope": "full_train", "clump_r2": 0.2, "p_threshold": 0.05, "prs": -0.4},
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


def test_run_official_validation_fits_training_and_bootstraps_validation(tmp_path):
    train = tmp_path / "train.tsv"
    labelled = tmp_path / "released_labels.tsv"
    validation_covariates = tmp_path / "testing_covariates.tsv"
    train_prs_scores = tmp_path / "train_prs_scores_long.csv"
    validation_prs_scores = tmp_path / "validation_prs_scores_long.csv"
    out_dir = tmp_path / "official_validation"
    train_rows = []
    rows = []
    cov_rows = []
    train_score_rows = []
    validation_score_rows = []
    for i in range(40):
        patient = f"T{i:03d}"
        label = i % 2
        train_rows.append(
            {
                "sampleid": patient,
                "cad": label,
                "age": float(i),
            }
        )
        train_score_rows.append(
            {
                "sampleid": patient,
                "disease": "cad",
                "model_scope": "full_train",
                "clump_r2": 0.2,
                "p_threshold": 1.0,
                "prs": float(label + (i % 5) / 10),
            }
        )
    for i in range(60):
        patient = f"V{i:03d}"
        label = i % 2
        rows.append(
            {
                "sampleid": patient,
                "cad": label,
                "age": float(i),
            }
        )
        cov_rows.append(
            {
                "sampleid": patient,
                "age": float(i),
            }
        )
        validation_score_rows.append(
            {
                "sampleid": patient,
                "disease": "cad",
                "model_scope": "full_train",
                "clump_r2": 0.2,
                "p_threshold": 1.0,
                "prs": float(label + (i % 5) / 10),
            }
        )
    rows.append({"sampleid": "TRAIN_ONLY", "cad": 1, "age": 999.0})
    pd.DataFrame(train_rows).to_csv(train, sep="\t", index=False)
    pd.DataFrame(rows).to_csv(labelled, sep="\t", index=False)
    pd.DataFrame(cov_rows).to_csv(validation_covariates, sep="\t", index=False)
    pd.DataFrame(train_score_rows).to_csv(train_prs_scores, index=False)
    pd.DataFrame(validation_score_rows).to_csv(validation_prs_scores, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_official_validation.py",
            "--train-phenotype",
            str(train),
            "--labelled-phenotype",
            str(labelled),
            "--validation-covariates",
            str(validation_covariates),
            "--patient-col",
            "sampleid",
            "--labels",
            "cad",
            "--covariates",
            "age",
            "--train-prs-long-scores",
            str(train_prs_scores),
            "--prs-long-scores",
            str(validation_prs_scores),
            "--prs-param-cols",
            "clump_r2,p_threshold",
            "--n-bootstraps",
            "4",
            "--seed",
            "11",
            "--validation-phenotype-out",
            str(out_dir / "validation_phenotype.csv"),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    validation_phenotype = pd.read_csv(out_dir / "validation_phenotype.csv")
    assert len(validation_phenotype) == 60
    assert "TRAIN_ONLY" not in set(validation_phenotype["sampleid"])

    point_metrics = pd.read_csv(out_dir / "point_metrics.csv")
    assert set(point_metrics["method"]) == {"covariate_only", "prs_covariate_cad"}
    assert set(point_metrics["n"]) == {60}

    bootstrap_metrics = pd.read_csv(out_dir / "bootstrap_metrics.csv")
    repeat_metrics = pd.read_csv(out_dir / "repeat_metrics.csv")
    pd.testing.assert_frame_equal(bootstrap_metrics, repeat_metrics)
    assert set(bootstrap_metrics["method"]) == {"covariate_only", "prs_covariate_cad"}
    assert len(bootstrap_metrics) == 8
    assert bootstrap_metrics["repeat"].max() == 3
    assert set(bootstrap_metrics["n"]) == {60}
    prs_metrics = bootstrap_metrics.loc[bootstrap_metrics["method"] == "prs_covariate_cad"]
    assert set(prs_metrics["clump_r2"]) == {0.2}
    assert set(prs_metrics["p_threshold"]) == {1.0}

    fit_summary = pd.read_csv(out_dir / "fit_prediction_summary.csv")
    assert set(fit_summary["n_fit"]) == {40}
    assert set(fit_summary["n_prediction"]) == {60}

    bootstrap_samples = pd.read_csv(out_dir / "bootstrap_sample_summary.csv")
    assert set(bootstrap_samples["bootstrap_n"]) == {60}

    summary = pd.read_csv(out_dir / "summary_metrics.csv")
    assert set(summary["method"]) == {"covariate_only", "prs_covariate_cad"}
    assert set(summary["n_repeats"]) == {4}
    assert {"clump_r2", "p_threshold"} <= set(summary.columns)

    point_only_out_dir = tmp_path / "official_validation_point_only"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_official_validation.py",
            "--model-name",
            "point_only",
            "--train-phenotype",
            str(train),
            "--labelled-phenotype",
            str(labelled),
            "--validation-covariates",
            str(validation_covariates),
            "--patient-col",
            "sampleid",
            "--labels",
            "cad",
            "--covariates",
            "age",
            "--train-prs-long-scores",
            str(train_prs_scores),
            "--prs-long-scores",
            str(validation_prs_scores),
            "--prs-param-cols",
            "clump_r2,p_threshold",
            "--point-only",
            "--out-dir",
            str(point_only_out_dir),
        ],
        check=True,
    )

    point_metrics = pd.read_csv(point_only_out_dir / "point_metrics.csv")
    point_summary = pd.read_csv(point_only_out_dir / "summary_metrics.csv")
    assert set(point_summary["n_repeats"]) == {0}
    assert set(point_summary["method"]) == {"covariate_only", "prs_covariate_cad"}
    merged = point_summary.merge(
        point_metrics,
        on=["model_name", "disease", "method", "clump_r2", "p_threshold"],
        how="inner",
    )
    assert np.allclose(merged["auc_mean"], merged["auc"])


def test_validate_official_model_wraps_fixed_validation_config(tmp_path):
    train = tmp_path / "train.tsv"
    labelled = tmp_path / "released_labels.tsv"
    validation_covariates = tmp_path / "testing_covariates.tsv"
    train_prs_scores = tmp_path / "train_prs_scores_long.csv"
    validation_prs_scores = tmp_path / "validation_prs_scores_long.csv"
    out_dir = tmp_path / "official_validation_pt"
    config = tmp_path / "default.yaml"

    train_rows = []
    train_score_rows = []
    for i in range(30):
        label = i % 2
        patient = f"T{i:03d}"
        train_rows.append({"sampleid": patient, "cad": label, "age": float(i)})
        train_score_rows.append(
            {
                "sampleid": patient,
                "disease": "cad",
                "clump_r2": 0.2,
                "p_threshold": 1.0,
                "prs": float(label + (i % 3) / 10),
            }
        )

    labelled_rows = []
    covariate_rows = []
    validation_score_rows = []
    for i in range(40):
        label = i % 2
        patient = f"V{i:03d}"
        labelled_rows.append({"sampleid": patient, "cad": label, "age": float(i)})
        covariate_rows.append({"sampleid": patient, "age": float(i)})
        validation_score_rows.append(
            {
                "sampleid": patient,
                "disease": "cad",
                "clump_r2": 0.2,
                "p_threshold": 1.0,
                "prs": float(label + (i % 3) / 10),
            }
        )
    labelled_rows.append({"sampleid": "TRAIN_ONLY", "cad": 1, "age": 999.0})

    pd.DataFrame(train_rows).to_csv(train, sep="\t", index=False)
    pd.DataFrame(labelled_rows).to_csv(labelled, sep="\t", index=False)
    pd.DataFrame(covariate_rows).to_csv(validation_covariates, sep="\t", index=False)
    pd.DataFrame(train_score_rows).to_csv(train_prs_scores, index=False)
    pd.DataFrame(validation_score_rows).to_csv(validation_prs_scores, index=False)
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"seed": 11},
                "paths": {
                    "official_train_phenotype": str(train),
                    "official_labelled_phenotype": str(labelled),
                    "official_validation_covariates": str(validation_covariates),
                    "official_validation_phenotype": str(out_dir / "validation_phenotype.csv"),
                },
                "columns": {
                    "patient_id": "sampleid",
                    "labels": ["cad"],
                    "covariates": ["age"],
                },
                "evaluation": {"n_bootstraps": 3},
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/validate_official_model.py",
            "--config",
            str(config),
            "--model-name",
            "pt",
            "--train-prs-long-scores",
            str(train_prs_scores),
            "--validation-prs-long-scores",
            str(validation_prs_scores),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    summary = pd.read_csv(out_dir / "summary_metrics.csv")
    assert set(summary["model_name"]) == {"pt"}
    assert set(summary["method"]) == {"covariate_only", "prs_covariate_cad"}
    assert set(summary["n_repeats"]) == {3}
    assert {"clump_r2", "p_threshold"} <= set(summary.columns)


def test_evaluate_prs_cv_grid_writes_raw_scores_for_tabpfn_inputs(tmp_path):
    phenotype = tmp_path / "phenotype.csv"
    splits = tmp_path / "splits.csv"
    manifest = tmp_path / "manifest.csv"
    eligibility = tmp_path / "eligibility.yaml"
    scores_dir = tmp_path / "fold_scores"
    out_dir = tmp_path / "lassosum2" / "cad" / "evaluation"
    scores_dir.mkdir()

    phenotype_rows = []
    split_rows = []
    for fold in [0, 1]:
        for i in range(12):
            patient = f"S{fold}{i:02d}"
            label = i % 2
            phenotype_rows.append({"sampleid": patient, "cad": label})
            split_rows.append({"sampleid": patient, "disease": "cad", "fold": fold})

    manifest_rows = []
    for fold in [0, 1]:
        score_rows = []
        score_path = scores_dir / f"fold{fold}.csv"
        manifest_rows.append({"disease": "cad", "fold": fold, "score_file": str(score_path)})
        for row in phenotype_rows:
            patient = row["sampleid"]
            label = int(row["cad"])
            score_rows.extend(
                [
                    {"sampleid": patient, "delta": 0.01, "lambda": 1, "prs": float(label)},
                    {"sampleid": patient, "delta": 0.1, "lambda": 2, "prs": float(label + 0.5)},
                ]
            )
        pd.DataFrame(score_rows).to_csv(score_path, index=False)

    pd.DataFrame(phenotype_rows).to_csv(phenotype, index=False)
    pd.DataFrame(split_rows).to_csv(splits, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    eligibility.write_text("disease_eligibility: {}\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_prs_cv_grid.py",
            "--disease",
            "cad",
            "--method-name",
            "lassosum2",
            "--score-manifest",
            str(manifest),
            "--score-kind",
            "long",
            "--param-cols",
            "delta,lambda",
            "--phenotype",
            str(phenotype),
            "--splits",
            str(splits),
            "--covariates",
            "",
            "--eligibility-config",
            str(eligibility),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    raw_scores = pd.read_csv(tmp_path / "lassosum2" / "cad" / "scores" / "cad.lassosum2_cv_scores_long.csv")
    summary = pd.read_csv(out_dir / "summary_metrics.csv")
    assert set(raw_scores.columns) == {"sampleid", "disease", "fold", "delta", "lambda", "prs", "source_score_file"}
    assert len(raw_scores) == 96
    assert {"delta", "lambda"} <= set(summary.columns)
