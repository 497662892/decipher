import subprocess
import sys

import pandas as pd
import pytest
import yaml

from decipehr.ensemble import pivot_long_prs_scores
from decipehr.tabpfn_ensemble import (
    TabPFNFeatureTransformer,
    logit_from_probability,
    scale_validation_logits,
    stratified_oof_splits,
)


def test_pivot_long_prs_scores_accepts_arbitrary_model_count():
    scores = pd.DataFrame(
        {
            "sampleid": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "disease": ["cad"] * 6,
            "model": ["pt", "ldpred2", "prscs"] * 2,
            "prs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    wide = pivot_long_prs_scores(scores, "cad", patient_col="sampleid")

    assert set(wide.columns) == {"sampleid", "pt", "ldpred2", "prscs"}
    assert wide.loc[wide["sampleid"] == "P1", "ldpred2"].iloc[0] == 0.2


def test_tabpfn_feature_transformer_standardizes_prs_and_encodes_covariates():
    fit = pd.DataFrame(
        {
            "pt": [1.0, 2.0, 3.0],
            "ldpred2": [2.0, 2.5, 3.5],
            "age": [40, 50, 60],
            "sex": ["Female", "Male", "Female"],
        }
    )
    validation = pd.DataFrame(
        {
            "pt": [2.0],
            "ldpred2": [3.0],
            "age": [55],
            "sex": ["Unknown"],
        }
    )

    transformer = TabPFNFeatureTransformer.fit(
        fit,
        prs_cols=["pt", "ldpred2"],
        covariates=["age", "sex"],
        include_covariates=True,
    )
    matrix = transformer.transform(validation)

    assert "prs__pt" in matrix.columns
    assert "cov__age" in matrix.columns
    assert "cov__sex__Female" in matrix.columns
    assert "cov__sex__Male" in matrix.columns
    assert matrix.loc[0, "cov__sex__Female"] == 0.0
    assert matrix.loc[0, "cov__sex__Male"] == 0.0


def test_tabpfn_probability_logits_are_clipped_and_train_scaled():
    train_logits = logit_from_probability(pd.Series([0.0, 0.5, 1.0]).to_numpy(), eps=0.01)
    validation_logits = logit_from_probability(pd.Series([0.8]).to_numpy(), eps=0.01)

    train_scaled, validation_scaled, mean, sd = scale_validation_logits(
        train_logits=train_logits,
        validation_logits=validation_logits,
        mode="train_sd",
    )

    assert pd.Series(train_scaled).std(ddof=0) == pytest.approx(1.0)
    assert mean == pytest.approx(0.0)
    assert sd > 0
    assert validation_scaled.shape == (1,)


def test_tabpfn_oof_split_count_respects_minority_class_count():
    y = pd.Series([0, 0, 0, 1, 1, 1])
    splits = stratified_oof_splits(y, requested_splits=5, seed=7)

    assert len(splits) == 3


def test_build_ensemble_prs_outputs_official_wide_scores(tmp_path):
    train = tmp_path / "train.tsv"
    validation_covariates = tmp_path / "validation_covariates.tsv"
    labelled_validation = tmp_path / "labelled_validation.tsv"
    train_scores = tmp_path / "train_scores.csv"
    validation_scores = tmp_path / "validation_scores.csv"
    config = tmp_path / "default.yaml"
    out_dir = tmp_path / "ensemble"

    train_rows = []
    train_score_rows = []
    for i in range(50):
        patient = f"T{i:03d}"
        y = i % 2
        age = float(i % 10)
        train_rows.append({"sampleid": patient, "cad": y, "age": age})
        train_score_rows.extend(
            [
                {
                    "sampleid": patient,
                    "disease": "cad",
                    "model": "ldpred2",
                    "prs": y + (i % 5) / 10,
                },
                {
                    "sampleid": patient,
                    "disease": "cad",
                    "model": "pt",
                    "prs": (1 - y) / 2 + (i % 7) / 20,
                },
                {
                    "sampleid": patient,
                    "disease": "cad",
                    "model": "prscs",
                    "prs": y / 3 + (i % 3) / 10,
                },
            ]
        )

    validation_rows = []
    labelled_validation_rows = []
    validation_score_rows = []
    for i in range(30):
        patient = f"V{i:03d}"
        y = i % 2
        validation_rows.append({"sampleid": patient, "age": float(i % 10)})
        labelled_validation_rows.append({"sampleid": patient, "cad": y, "age": float(i % 10)})
        validation_score_rows.extend(
            [
                {
                    "sampleid": patient,
                    "disease": "cad",
                    "model": "ldpred2",
                    "prs": y + (i % 5) / 10,
                },
                {
                    "sampleid": patient,
                    "disease": "cad",
                    "model": "pt",
                    "prs": (1 - y) / 2 + (i % 7) / 20,
                },
                {
                    "sampleid": patient,
                    "disease": "cad",
                    "model": "prscs",
                    "prs": y / 3 + (i % 3) / 10,
                },
            ]
        )

    pd.DataFrame(train_rows).to_csv(train, sep="\t", index=False)
    pd.DataFrame(validation_rows).to_csv(validation_covariates, sep="\t", index=False)
    pd.DataFrame(labelled_validation_rows).to_csv(labelled_validation, sep="\t", index=False)
    pd.DataFrame(train_score_rows).to_csv(train_scores, index=False)
    pd.DataFrame(validation_score_rows).to_csv(validation_scores, index=False)
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"seed": 7},
                "paths": {
                    "official_train_phenotype": str(train),
                    "official_validation_covariates": str(validation_covariates),
                },
                "columns": {
                    "patient_id": "sampleid",
                    "labels": ["cad"],
                    "covariates": ["age"],
                },
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_ensemble_prs.py",
            "--config",
            str(config),
            "--train-prs-long-scores",
            str(train_scores),
            "--validation-prs-long-scores",
            str(validation_scores),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    for strategy in ["prs_only", "covariate_adjusted"]:
        train_out = pd.read_csv(out_dir / strategy / "train_prs_scores.csv")
        validation_out = pd.read_csv(out_dir / strategy / "validation_prs_scores.csv")
        assert list(train_out.columns) == ["sampleid", "prs_cad"]
        assert list(validation_out.columns) == ["sampleid", "prs_cad"]
        assert train_out["prs_cad"].notna().all()
        assert validation_out["prs_cad"].notna().all()

    weights = pd.read_csv(out_dir / "ensemble_weights.csv")
    assert set(weights["strategy"]) == {"prs_only", "covariate_adjusted"}
    assert set(weights.loc[weights["feature_type"] == "prs", "feature"]) == {"ldpred2", "pt", "prscs"}
    assert (out_dir / "validate_ensemble_commands.sh").exists()

    validation_out_dir = tmp_path / "official_validation"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_official_validation.py",
            "--train-phenotype",
            str(train),
            "--labelled-phenotype",
            str(labelled_validation),
            "--validation-covariates",
            str(validation_covariates),
            "--patient-col",
            "sampleid",
            "--labels",
            "cad",
            "--covariates",
            "age",
            "--train-prs-scores",
            str(out_dir / "covariate_adjusted" / "train_prs_scores.csv"),
            "--prs-scores",
            str(out_dir / "covariate_adjusted" / "validation_prs_scores.csv"),
            "--n-bootstraps",
            "2",
            "--out-dir",
            str(validation_out_dir),
        ],
        check=True,
    )
    point_metrics = pd.read_csv(validation_out_dir / "point_metrics.csv")
    assert set(point_metrics["method"]) == {"covariate_only", "prs_covariate_cad"}


def test_build_ensemble_inputs_selects_best_single_model_per_family(tmp_path):
    pt_validation_root = tmp_path / "official_validation_pt"
    pt_score_root = tmp_path / "pt_full_train"
    ldpred2_validation_root = tmp_path / "official_validation_ldpred2"
    ldpred2_score_root = tmp_path / "ldpred2_full_train"
    out_dir = tmp_path / "ensemble_inputs"

    (pt_validation_root / "cad").mkdir(parents=True)
    (ldpred2_validation_root / "cad").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model_name": "pt",
                "disease": "cad",
                "method": "prs_covariate_cad",
                "top10_vs_lower90_or_mean": 1.0,
                "clump_r2": 0.1,
                "p_threshold": 1.0,
            },
            {
                "model_name": "pt",
                "disease": "cad",
                "method": "prs_covariate_cad",
                "top10_vs_lower90_or_mean": 2.0,
                "clump_r2": 0.2,
                "p_threshold": 0.01,
            },
        ]
    ).to_csv(pt_validation_root / "cad" / "summary_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": "ldpred2",
                "disease": "cad",
                "method": "prs_covariate_cad",
                "top10_vs_lower90_or_mean": 1.5,
                "h2": 0.05,
                "p": 0.01,
                "sparse": 0,
            },
            {
                "model_name": "ldpred2",
                "disease": "cad",
                "method": "prs_covariate_cad",
                "top10_vs_lower90_or_mean": 3.0,
                "h2": 0.1,
                "p": 0.1,
                "sparse": 1,
            },
        ]
    ).to_csv(ldpred2_validation_root / "cad" / "summary_metrics.csv", index=False)

    for root in [pt_score_root, ldpred2_score_root]:
        (root / "cad" / "scores").mkdir(parents=True)

    pt_train_rows = []
    pt_validation_rows = []
    for sample in ["T1", "T2"]:
        pt_train_rows.extend(
            [
                {
                    "sampleid": sample,
                    "disease": "cad",
                    "clump_r2": 0.1,
                    "p_threshold": 1.0,
                    "prs": 1.0,
                },
                {
                    "sampleid": sample,
                    "disease": "cad",
                    "clump_r2": 0.2,
                    "p_threshold": 0.01,
                    "prs": 2.0,
                },
            ]
        )
    for sample in ["V1", "V2"]:
        pt_validation_rows.extend(
            [
                {
                    "sampleid": sample,
                    "disease": "cad",
                    "clump_r2": 0.1,
                    "p_threshold": 1.0,
                    "prs": 10.0,
                },
                {
                    "sampleid": sample,
                    "disease": "cad",
                    "clump_r2": 0.2,
                    "p_threshold": 0.01,
                    "prs": 20.0,
                },
            ]
        )
    pd.DataFrame(pt_train_rows).to_csv(
        pt_score_root / "cad" / "scores" / "cad.pt_full_train_training_scores_long.csv",
        index=False,
    )
    pd.DataFrame(pt_validation_rows).to_csv(
        pt_score_root / "cad" / "scores" / "cad.pt_full_train_scores_long.csv",
        index=False,
    )

    ldpred2_train_rows = []
    ldpred2_validation_rows = []
    for sample in ["T1", "T2"]:
        ldpred2_train_rows.extend(
            [
                {
                    "sampleid": sample,
                    "disease": "cad",
                    "h2": 0.05,
                    "p": 0.01,
                    "sparse": 0,
                    "model": "low",
                    "prs": 3.0,
                },
                {
                    "sampleid": sample,
                    "disease": "cad",
                    "h2": 0.1,
                    "p": 0.1,
                    "sparse": 1,
                    "model": "high",
                    "prs": 4.0,
                },
            ]
        )
    for sample in ["V1", "V2"]:
        ldpred2_validation_rows.extend(
            [
                {
                    "sampleid": sample,
                    "disease": "cad",
                    "h2": 0.05,
                    "p": 0.01,
                    "sparse": 0,
                    "model": "low",
                    "prs": 30.0,
                },
                {
                    "sampleid": sample,
                    "disease": "cad",
                    "h2": 0.1,
                    "p": 0.1,
                    "sparse": 1,
                    "model": "high",
                    "prs": 40.0,
                },
            ]
        )
    pd.DataFrame(ldpred2_train_rows).to_csv(
        ldpred2_score_root / "cad" / "scores" / "cad.ldpred2_full_train_training_scores_long.csv",
        index=False,
    )
    pd.DataFrame(ldpred2_validation_rows).to_csv(
        ldpred2_score_root / "cad" / "scores" / "cad.ldpred2_full_train_scores_long.csv",
        index=False,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_ensemble_inputs.py",
            "--disease",
            "cad",
            "--families",
            "pt,ldpred2",
            "--pt-validation-root",
            str(pt_validation_root),
            "--pt-score-root",
            str(pt_score_root),
            "--ldpred2-validation-root",
            str(ldpred2_validation_root),
            "--ldpred2-score-root",
            str(ldpred2_score_root),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    train_long = pd.read_csv(out_dir / "cad.train_long.csv")
    validation_long = pd.read_csv(out_dir / "cad.validation_long.csv")
    selected = pd.read_csv(out_dir / "cad.selected_single_models.csv")

    assert set(train_long["model"]) == {"pt", "ldpred2"}
    assert set(validation_long["model"]) == {"pt", "ldpred2"}
    assert set(train_long.loc[train_long["model"] == "pt", "prs"]) == {2.0}
    assert set(validation_long.loc[validation_long["model"] == "ldpred2", "prs"]) == {40.0}
    assert selected.set_index("family").loc["pt", "p_threshold"] == 0.01
    assert selected.set_index("family").loc["ldpred2", "sparse"] == 1


def test_write_and_collect_ensemble_grid_commands(tmp_path):
    input_dir = tmp_path / "ensemble_inputs"
    grid_root = tmp_path / "ensemble_grid"
    command_script = tmp_path / "commands.sh"
    input_dir.mkdir()
    (input_dir / "cad.train_long.csv").write_text("sampleid,disease,model,prs\n", encoding="utf-8")
    (input_dir / "cad.validation_long.csv").write_text("sampleid,disease,model,prs\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/write_ensemble_grid_commands.py",
            "--disease",
            "cad",
            "--ensemble-input-dir",
            str(input_dir),
            "--out-root",
            str(grid_root),
            "--out",
            str(command_script),
            "--c-grid",
            "0.1,1",
            "--penalties",
            "l2,none",
            "--strategies",
            "prs_only,covariate_adjusted",
        ],
        check=True,
    )

    tasks = pd.read_csv(command_script.with_suffix(".tasks.csv"))
    assert len(tasks) == 3
    assert set(tasks["penalty"]) == {"l2", "none"}
    assert "validate_ensemble_commands.sh" in command_script.read_text(encoding="utf-8")

    setting_a = grid_root / "cad" / tasks.iloc[0]["setting_id"] / "official_validation_prs_only"
    setting_b = grid_root / "cad" / tasks.iloc[1]["setting_id"] / "official_validation_covariate_adjusted"
    setting_a.mkdir(parents=True)
    setting_b.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model_name": "ensemble",
                "disease": "cad",
                "method": "prs_covariate_cad",
                "top10_vs_lower90_or_mean": 2.0,
            }
        ]
    ).to_csv(setting_a / "summary_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": "ensemble",
                "disease": "cad",
                "method": "prs_covariate_cad",
                "top10_vs_lower90_or_mean": 4.0,
            }
        ]
    ).to_csv(setting_b / "summary_metrics.csv", index=False)

    summary_out = tmp_path / "ensemble_grid_summary.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_ensemble_grid_results.py",
            "--grid-root",
            str(grid_root),
            "--disease",
            "cad",
            "--out",
            str(summary_out),
        ],
        check=True,
    )

    ranked = pd.read_csv(summary_out)
    best = pd.read_csv(summary_out.with_name("ensemble_grid_summary.best.csv"))
    assert ranked.iloc[0]["top10_vs_lower90_or_mean"] == 4.0
    assert best.iloc[0]["strategy"] == "covariate_adjusted"


def test_write_ensemble_grid_commands_supports_elasticnet(tmp_path):
    input_dir = tmp_path / "ensemble_inputs"
    grid_root = tmp_path / "ensemble_grid"
    command_script = tmp_path / "elasticnet_commands.sh"
    input_dir.mkdir()
    (input_dir / "cad.train_long.csv").write_text("sampleid,disease,model,prs\n", encoding="utf-8")
    (input_dir / "cad.validation_long.csv").write_text("sampleid,disease,model,prs\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/write_ensemble_grid_commands.py",
            "--disease",
            "cad",
            "--ensemble-input-dir",
            str(input_dir),
            "--out-root",
            str(grid_root),
            "--out",
            str(command_script),
            "--c-grid",
            "0.1,1",
            "--penalties",
            "elasticnet",
            "--l1-ratio-grid",
            "0.25,0.75",
        ],
        check=True,
    )

    tasks = pd.read_csv(command_script.with_suffix(".tasks.csv"))
    assert len(tasks) == 4
    assert set(tasks["penalty"]) == {"elasticnet"}
    assert set(tasks["C"]) == {0.1, 1.0}
    assert set(tasks["l1_ratio"]) == {0.25, 0.75}
    assert "--l1-ratio 0.25" in command_script.read_text(encoding="utf-8")
