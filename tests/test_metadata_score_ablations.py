from __future__ import annotations

import pytest

from decipehr.metadata_score_ablations import (
    build_metadata_score_ablations,
    metadata_ablation_task,
    resolve_score_and_final_covariates,
)

FULL = ["sex", "age_sd", "PC1_sd", "PC2_sd"]


def test_metadata_score_ablation_contract_has_all_five_requested_arms():
    ablations = build_metadata_score_ablations(
        sex="sex",
        age="age_sd",
        pcs=["PC1_sd", "PC2_sd"],
        full_covariates=FULL,
    )
    by_name = {ablation.name: ablation for ablation in ablations}

    assert list(by_name) == ["full", "no_age", "no_sex", "no_pcs", "no_age_sex"]
    assert by_name["full"].score_covariates == tuple(FULL)
    assert by_name["no_age"].score_covariates == ("sex", "PC1_sd", "PC2_sd")
    assert by_name["no_sex"].score_covariates == ("age_sd", "PC1_sd", "PC2_sd")
    assert by_name["no_pcs"].score_covariates == ("sex", "age_sd")
    assert by_name["no_age_sex"].score_covariates == ("PC1_sd", "PC2_sd")


def test_final_covariates_remain_full_when_score_covariates_are_ablated():
    score, final = resolve_score_and_final_covariates(
        default_covariates=FULL,
        score_covariates=["PC1_sd", "PC2_sd"],
        final_covariates=FULL,
    )

    assert score == ["PC1_sd", "PC2_sd"]
    assert final == FULL


def test_legacy_shared_covariates_remain_backward_compatible():
    score, final = resolve_score_and_final_covariates(
        default_covariates=FULL,
        shared_covariates=["sex", "age_sd"],
    )

    assert score == ["sex", "age_sd"]
    assert final == ["sex", "age_sd"]


def test_metadata_ablation_array_is_disease_major():
    ablations = build_metadata_score_ablations(
        sex="sex",
        age="age_sd",
        pcs=["PC1_sd", "PC2_sd"],
        full_covariates=FULL,
    )

    assert metadata_ablation_task(0, ["cad", "bc", "T2D"], ablations)[0:1] == (
        "cad",
    )
    assert metadata_ablation_task(4, ["cad", "bc", "T2D"], ablations)[1].name == (
        "no_age_sex"
    )
    assert metadata_ablation_task(5, ["cad", "bc", "T2D"], ablations)[0] == "bc"
    assert metadata_ablation_task(14, ["cad", "bc", "T2D"], ablations)[0] == "T2D"
    with pytest.raises(ValueError, match="task_id"):
        metadata_ablation_task(15, ["cad", "bc", "T2D"], ablations)


def test_full_covariate_order_is_audited():
    with pytest.raises(ValueError, match="ordered as sex, age"):
        build_metadata_score_ablations(
            sex="sex",
            age="age_sd",
            pcs=["PC1_sd", "PC2_sd"],
            full_covariates=["age_sd", "sex", "PC1_sd", "PC2_sd"],
        )
