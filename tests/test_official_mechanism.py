from __future__ import annotations

import numpy as np
import pytest

from decipehr.official_mechanism import (
    NONANNOTATION_PRS,
    Q3_ADDITIVE_ARM,
    Q3_METADATA_SCORE_COMPONENT,
    Q3_RAW_PRS_COMPONENT,
    build_official_stage1_arms,
    official_comparison_specs,
    q3_additive_stage2_features,
    scaled_logit_scores,
)

ALL_PRS = (*NONANNOTATION_PRS, "megaprs_noannot", "megaprs_annotated", "sbayesrc")
PCS = tuple(f"PC{index}_sd" for index in range(1, 11))
FULL_METADATA = ("sex", "age_sd", *PCS)


def _arms():
    return build_official_stage1_arms(
        best_prs="prscs",
        nonannotation_prs=NONANNOTATION_PRS,
        all_prs=ALL_PRS,
        full_metadata=FULL_METADATA,
        sex="sex",
        age="age_sd",
        pcs=PCS,
        ancestry_group="ancestry_group",
    )


def test_official_arms_freeze_three_prs_panels_and_metadata_ablations() -> None:
    arms = {arm.name: arm for arm in _arms()}
    assert len(arms) == 11
    assert arms["raw_best_prs"].prs_columns == ("prscs",)
    assert arms["tabpfn_best_prs_full_metadata"].prs_columns == ("prscs",)
    assert arms["tabpfn_nonannotation_full_metadata"].prs_columns == NONANNOTATION_PRS
    assert arms["tabpfn_all8_full_metadata"].prs_columns == ALL_PRS
    assert arms["tabpfn_best_prs_no_age"].prs_columns == ("prscs",)
    assert arms["tabpfn_best_prs_no_age"].covariates == ("sex", *PCS)
    assert arms["tabpfn_best_prs_no_sex"].prs_columns == ("prscs",)
    assert arms["tabpfn_best_prs_no_sex"].covariates == ("age_sd", *PCS)
    assert arms["tabpfn_best_prs_no_pcs"].prs_columns == ("prscs",)
    assert arms["tabpfn_best_prs_no_pcs"].covariates == ("sex", "age_sd")
    assert arms["tabpfn_best_prs_no_age_sex"].prs_columns == ("prscs",)
    assert arms["tabpfn_best_prs_no_age_sex"].covariates == PCS
    assert arms["tabpfn_best_prs_hard_ancestry"].prs_columns == ("prscs",)
    assert arms["tabpfn_best_prs_hard_ancestry"].covariates == (
        "sex",
        "age_sd",
        "ancestry_group",
    )


def test_nonannotation_panel_must_be_exactly_five_prespecified_families() -> None:
    with pytest.raises(ValueError, match="all-non-annotation panel"):
        build_official_stage1_arms(
            best_prs="prscs",
            nonannotation_prs=(*NONANNOTATION_PRS, "megaprs_noannot"),
            all_prs=ALL_PRS,
            full_metadata=FULL_METADATA,
            sex="sex",
            age="age_sd",
            pcs=PCS,
            ancestry_group="ancestry_group",
        )


def test_official_comparisons_never_reference_selected_565_arm() -> None:
    comparisons = official_comparison_specs()
    variants = {item.reference for item in comparisons} | {item.challenger for item in comparisons}
    assert all("selected" not in variant for variant in variants)
    assert "tabpfn_nonannotation_full_metadata" in variants
    assert "tabpfn_all8_full_metadata" in variants
    assert "raw_best_prs" in variants


def test_q3_compares_joint_single_best_with_separate_raw_prs_and_metadata_score() -> None:
    q3 = [item for item in official_comparison_specs() if item.question == 3]
    assert len(q3) == 1
    assert q3[0].reference == "tabpfn_best_prs_full_metadata"
    assert q3[0].challenger == Q3_ADDITIVE_ARM
    assert q3_additive_stage2_features(FULL_METADATA) == (
        *FULL_METADATA,
        Q3_METADATA_SCORE_COMPONENT,
        Q3_RAW_PRS_COMPONENT,
    )


def test_q2_q3_q5_use_joint_single_best_full_metadata_as_the_common_baseline() -> None:
    comparisons = official_comparison_specs()
    single_full = "tabpfn_best_prs_full_metadata"
    assert all(item.reference == single_full for item in comparisons if item.question in {2, 3, 5})


def test_scaled_logit_scores_use_training_distribution() -> None:
    train, validation, mean, sd = scaled_logit_scores(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.3, 0.7]),
        clip_eps=0.001,
    )
    assert np.mean(train) == pytest.approx(0.0)
    assert np.std(train) == pytest.approx(1.0)
    assert np.isfinite(validation).all()
    assert np.isfinite([mean, sd]).all()


def test_q3_additive_components_are_not_collapsed_into_one_stage1_score() -> None:
    features = q3_additive_stage2_features(FULL_METADATA)
    assert "stage1_score" not in features
    assert features[-2:] == (Q3_METADATA_SCORE_COMPONENT, Q3_RAW_PRS_COMPONENT)
