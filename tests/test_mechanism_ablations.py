from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from decipehr.mechanism_ablations import (
    _weighted_auc_batch,
    _weighted_nagelkerke_r2_batch,
    _weighted_top_decile_or_batch,
    align_paired_predictions,
    build_tabpfn_variants,
    clipped_logit,
    fold_task,
    paired_stratified_bootstrap,
    prior_corrected_additive_probability,
    validate_prediction_frame,
)
from decipehr.metrics import nagelkerke_r2, top_decile_odds_ratio

VARIANTS = [
    "tabpfn_best_prs_full_metadata",
    "tabpfn_selected_prs_full_metadata",
    "tabpfn_all_prs_full_metadata",
    "tabpfn_selected_prs_age_sex",
    "tabpfn_selected_prs_pcs",
    "tabpfn_selected_prs_only",
    "tabpfn_metadata_only",
    "tabpfn_selected_prs_hard_ancestry",
    "xgboost_selected_prs_full_metadata",
    "tabpfn_separate_additive",
]


def test_fold_task_maps_disease_major_array_indices():
    diseases = ["cad", "bc", "T2D"]
    assert fold_task(0, diseases) == ("cad", 0)
    assert fold_task(4, diseases) == ("cad", 4)
    assert fold_task(5, diseases) == ("bc", 0)
    assert fold_task(14, diseases) == ("T2D", 4)
    with pytest.raises(ValueError, match="task_id"):
        fold_task(15, diseases)


def test_variant_contract_reuses_one_joint_reference():
    variants = build_tabpfn_variants(
        best_prs="prs_b",
        selected_prs=["prs_a", "prs_b"],
        all_prs=["prs_a", "prs_b", "prs_c"],
        full_metadata=["sex", "age", "pc1"],
        age_sex=["sex", "age"],
        pcs=["pc1"],
        ancestry_group_metadata=["sex", "age", "ancestry_group"],
    )
    by_name = {variant.name: variant for variant in variants}
    assert list(by_name) == VARIANTS[:8]
    assert by_name["tabpfn_best_prs_full_metadata"].prs_columns == ("prs_b",)
    assert by_name["tabpfn_selected_prs_full_metadata"].prs_columns == (
        "prs_a",
        "prs_b",
    )
    assert by_name["tabpfn_metadata_only"].prs_columns == ()
    assert by_name["tabpfn_selected_prs_only"].covariates == ()


def test_prior_corrected_addition_does_not_double_count_prevalence():
    prevalence = 0.2
    probability, score = prior_corrected_additive_probability(
        np.array([prevalence]),
        np.array([prevalence]),
        training_prevalence=prevalence,
    )
    assert probability[0] == pytest.approx(prevalence)
    assert score[0] == pytest.approx(clipped_logit(np.array([prevalence]))[0])


def test_paired_bootstrap_uses_identical_patients_and_is_deterministic():
    y = np.array([0, 1] * 20)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "sampleid": np.arange(len(y)),
                    "disease": "cad",
                    "fold": np.arange(len(y)) % 5,
                    "variant": variant,
                    "model_family": "test",
                    "y_true": y,
                    "y_prob": np.where(y == 1, positive, 1 - positive),
                    "score": clipped_logit(np.where(y == 1, positive, 1 - positive)),
                }
            )
            for variant, positive in [("reference", 0.7), ("challenger", 0.9)]
        ],
        ignore_index=True,
    )
    validate_prediction_frame(frame, patient_col="sampleid")
    paired = align_paired_predictions(
        frame,
        disease="cad",
        reference="reference",
        challenger="challenger",
        patient_col="sampleid",
    )
    first = paired_stratified_bootstrap(
        paired,
        reference="reference",
        challenger="challenger",
        n_resamples=20,
        seed=42,
        metrics=("auc", "brier", "nagelkerke_r2", "top10_vs_lower90_or"),
    )
    second = paired_stratified_bootstrap(
        paired,
        reference="reference",
        challenger="challenger",
        n_resamples=20,
        seed=42,
        metrics=("auc", "brier", "nagelkerke_r2", "top10_vs_lower90_or"),
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {
        "auc",
        "brier",
        "nagelkerke_r2",
        "top10_vs_lower90_or",
    }
    brier_delta = first.loc[first["metric"] == "brier", "delta"].item()
    assert brier_delta < 0


def test_weighted_auc_batch_matches_sklearn_with_ties_and_zero_weights():
    from sklearn.metrics import roc_auc_score

    y = np.array([0, 1, 0, 1, 0, 1])
    probability = np.array([0.1, 0.8, 0.4, 0.4, 0.8, 0.9])
    weights = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [3, 0, 0, 2, 1, 2],
            [0, 2, 4, 1, 1, 0],
        ]
    )
    observed = _weighted_auc_batch(y, probability, weights)
    expected = np.array([roc_auc_score(y, probability, sample_weight=weight) for weight in weights])
    np.testing.assert_allclose(observed, expected)


def test_weighted_official_metrics_match_expanded_bootstrap_samples():
    y = np.array([0, 1, 0, 1, 0, 1])
    probability = np.array([0.1, 0.8, 0.4, 0.4, 0.8, 0.9])
    weights = np.array(
        [
            [2, 2, 2, 2, 2, 2],
            [4, 0, 0, 3, 2, 3],
            [0, 3, 5, 2, 2, 0],
        ]
    )
    observed_r2 = _weighted_nagelkerke_r2_batch(y, probability, weights)
    observed_or90 = _weighted_top_decile_or_batch(y, probability, weights)
    expected_r2 = []
    expected_or90 = []
    for weight in weights:
        sampled = np.repeat(np.arange(len(y)), weight)
        expected_r2.append(nagelkerke_r2(y[sampled], probability[sampled]))
        expected_or90.append(
            top_decile_odds_ratio(y[sampled], probability[sampled])["top10_vs_lower90_or"]
        )
    np.testing.assert_allclose(observed_r2, expected_r2)
    np.testing.assert_allclose(observed_or90, expected_or90)


def _write_synthetic_task_tree(root: Path) -> Path:
    diseases = ["cad", "bc", "T2D"]
    input_root = root / "fold_predictions"
    for disease_index, disease in enumerate(diseases):
        for fold in range(5):
            task_dir = input_root / disease / f"fold_{fold}"
            task_dir.mkdir(parents=True)
            y = np.array([0, 0, 1, 1])
            patient_ids = [f"{disease}_{fold}_{index}" for index in range(len(y))]
            prediction_frames = []
            manifest_rows = []
            feature_rows = []
            for variant_index, variant in enumerate(VARIANTS):
                strength = 0.60 + 0.015 * variant_index + 0.002 * disease_index
                probability = np.where(y == 1, strength, 1 - strength)
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "sampleid": patient_ids,
                            "disease": disease,
                            "fold": fold,
                            "variant": variant,
                            "model_family": "xgboost"
                            if variant.startswith("xgboost")
                            else "tabpfn",
                            "y_true": y,
                            "y_prob": probability,
                            "score": clipped_logit(probability),
                        }
                    )
                )
                manifest_rows.append(
                    {
                        "disease": disease,
                        "variant": variant,
                        "model_family": "test",
                        "prs_columns": "prs1;prs2",
                        "covariates": "sex;age;pc1",
                        "n_prs": 2,
                        "n_covariates_requested": 3,
                        "description": variant,
                    }
                )
                if variant != "tabpfn_separate_additive":
                    feature_rows.append(
                        {
                            "disease": disease,
                            "fold": fold,
                            "variant": variant,
                            "model_family": "test",
                            "source": "prs1",
                            "feature": "prs__prs1",
                        }
                    )
            pd.concat(prediction_frames, ignore_index=True).to_csv(
                task_dir / "predictions.csv", index=False
            )
            pd.DataFrame(feature_rows).to_csv(task_dir / "feature_audit.csv", index=False)
            pd.DataFrame(manifest_rows).to_csv(task_dir / "variant_manifest.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "candidate_index": 0,
                        "max_depth": 2,
                        "learning_rate": 0.1,
                        "min_child_weight": 1,
                        "mean_inner_auc": 0.7,
                        "sd_inner_auc": 0.01,
                        "n_inner_folds": 3,
                        "disease": disease,
                        "fold": fold,
                    }
                ]
            ).to_csv(task_dir / "xgboost_tuning.csv", index=False)
            record = {
                "disease": disease,
                "fold": fold,
                "variants": VARIANTS,
                "data_audit": {
                    "n_context": 16,
                    "context_cases": 8,
                    "n_validation": 4,
                    "validation_cases": 2,
                },
            }
            (task_dir / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
    return input_root


def _write_synthetic_report_supplements(root: Path) -> tuple[Path, Path]:
    diseases = ["cad", "bc", "T2D"]
    arms = ["full", "no_age", "no_sex", "no_pcs", "no_age_sex"]
    metadata_dir = root / "metadata_score_summary"
    metadata_dir.mkdir()
    pooled_rows = []
    contract_rows = []
    bootstrap_rows = []
    arm_deltas = {
        "no_age": -0.01,
        "no_sex": -0.001,
        "no_pcs": -0.02,
        "no_age_sex": -0.011,
    }
    for disease_index, disease in enumerate(diseases):
        full_auc = 0.75 + disease_index * 0.01
        for arm in arms:
            delta = arm_deltas.get(arm, 0.0)
            pooled_rows.append(
                {
                    "disease": disease,
                    "variant": arm,
                    "auc": full_auc + delta,
                    "auprc": 0.5 + delta,
                    "brier": 0.2 - delta,
                    "prevalence": 0.5,
                    "n": 20,
                    "nagelkerke_r2": 0.2 + delta,
                }
            )
            contract_rows.append(
                {
                    "disease": disease,
                    "ablation": arm,
                    "n_score_covariates": {
                        "full": 3,
                        "no_age": 2,
                        "no_sex": 2,
                        "no_pcs": 2,
                        "no_age_sex": 1,
                    }[arm],
                    "final_covariates": "sex;age;pc1",
                    "n_final_covariates": 3,
                    "outer_folds": 5,
                    "inner_oof_folds": 5,
                }
            )
            if arm != "full":
                bootstrap_rows.append(
                    {
                        "metric": "auc",
                        "disease": disease,
                        "reference": "full",
                        "challenger": arm,
                        "delta": delta,
                        "ci_lower": delta - 0.005,
                        "ci_upper": delta + 0.005,
                        "p_two_sided": 0.01,
                    }
                )
    pd.DataFrame(pooled_rows).to_csv(metadata_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(contract_rows).to_csv(metadata_dir / "contract_audit.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(metadata_dir / "paired_bootstrap_deltas.csv", index=False)
    (metadata_dir / "run_record.json").write_text(
        json.dumps(
            {
                "n_tasks": 15,
                "n_prediction_rows": 300,
                "paired_bootstrap_resamples": 5,
            }
        ),
        encoding="utf-8",
    )

    ancestry_dir = root / "ancestry_bridge"
    ancestry_dir.mkdir()
    comparison_rows = []
    for disease_index, disease in enumerate(diseases):
        for group in ["ALL", "AFR-like", "EUR-like"]:
            a_auc = 0.65 + disease_index * 0.01
            b_auc = a_auc + 0.02
            c_auc = b_auc + 0.001
            for comparison, reference, candidate in [
                ("tabpfn3_best_single_vs_best_single", a_auc, b_auc),
                ("tabpfn3_full_vs_tabpfn3_best_single", b_auc, c_auc),
                ("tabpfn3_full_vs_best_single", a_auc, c_auc),
            ]:
                delta = candidate - reference
                comparison_rows.append(
                    {
                        "disease": disease,
                        "ancestry_group": group,
                        "n": 20,
                        "cases": 10,
                        "comparison": comparison,
                        "metric": "auc",
                        "reference_value": reference,
                        "candidate_value": candidate,
                        "candidate_minus_reference": delta,
                        "ci_low": delta - 0.005,
                        "ci_high": delta + 0.005,
                        "n_bootstraps": 5,
                    }
                )
    pd.DataFrame(comparison_rows).to_csv(
        ancestry_dir / "paired_three_arm_comparison_summary.csv", index=False
    )
    pd.DataFrame([{"disease": disease, "single_model": "PRS 1"} for disease in diseases]).to_csv(
        ancestry_dir / "model_selection_manifest.csv", index=False
    )
    return metadata_dir, ancestry_dir


def test_summary_script_builds_self_contained_html_from_complete_fold_tree(tmp_path: Path):
    input_root = _write_synthetic_task_tree(tmp_path)
    metadata_dir, ancestry_dir = _write_synthetic_report_supplements(tmp_path)
    output_root = tmp_path / "outputs"
    config = {
        "project": {"seed": 42, "diseases": ["cad", "bc", "T2D"], "n_folds": 5},
        "paths": {"output_root": str(output_root)},
        "columns": {"patient_id": "sampleid"},
        "prs": {
            "canonical_best_single": {"cad": "prs1", "bc": "prs1", "T2D": "prs1"},
            "selected": {
                "cad": ["prs1", "prs2"],
                "bc": ["prs1", "prs2"],
                "T2D": ["prs1", "prs2"],
            },
        },
        "metadata": {"full": ["sex", "age", "pc1"]},
        "tabpfn": {"model_version": "v3", "n_estimators": 16, "clip_eps": 0.001},
        "xgboost": {"inner_folds": 3},
        "summary": {"paired_bootstrap_resamples": 5},
        "supplements": {
            "corrected_metadata_score_summary": str(metadata_dir),
            "ancestry_single_prs_bridge": str(ancestry_dir),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out_dir = output_root / "summary"
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "summarize_mechanism_ablations.py"),
            "--config",
            str(config_path),
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--bootstrap-resamples",
            "5",
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = (out_dir / "tabpfn_mechanism_ablations.html").read_text(encoding="utf-8")
    assert "TabPFN Mechanism Ablations" in report
    assert "Question 5" in report
    assert "No final holdout labels were used" in report
    assert "Connection to the earlier linear-stacking pattern" in report
    assert "Question 2. Which metadata contribute inside the TabPFN score" in report
    assert "Corrected estimand" in report
    assert "Score without age and sex" in report
    assert "increment from age/sex conditional on PCs" not in report
    assert "Primary ancestry estimand (A to B)" in report
    assert "B: TabPFN single-best + meta AUC" in report
    assert "Secondary sensitivity comparison: full multi-PRS TabPFN-3" in report
    assert (out_dir / "comparison_point_metrics.csv").is_file()
    assert (out_dir / "paired_bootstrap_deltas.csv").is_file()
    comparisons = pd.read_csv(out_dir / "comparison_point_metrics.csv")
    assert 2 not in set(comparisons["question"])
    run_record = json.loads((out_dir / "run_record.json").read_text(encoding="utf-8"))
    assert run_record["legacy_direct_oof_question_2_included"] is False
