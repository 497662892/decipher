from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from decipehr.mechanism_ablations import ComparisonSpec
from decipehr.tabpfn_ensemble import logit_from_probability, scale_validation_logits

NONANNOTATION_PRS = ("pt", "ldpred2", "lassosum2", "prscs", "sdpr")
Q3_ADDITIVE_ARM = "raw_best_prs_plus_metadata_score_additive"
Q3_METADATA_SCORE_COMPONENT = "tabpfn_metadata_score"
Q3_RAW_PRS_COMPONENT = "raw_best_prs"


@dataclass(frozen=True)
class OfficialStage1Arm:
    name: str
    model_family: str
    prs_columns: tuple[str, ...]
    covariates: tuple[str, ...]
    description: str


def build_official_stage1_arms(
    *,
    best_prs: str,
    nonannotation_prs: Sequence[str],
    all_prs: Sequence[str],
    full_metadata: Sequence[str],
    sex: str,
    age: str,
    pcs: Sequence[str],
    ancestry_group: str,
) -> tuple[OfficialStage1Arm, ...]:
    nonannotation = tuple(str(value) for value in nonannotation_prs)
    all_columns = tuple(str(value) for value in all_prs)
    metadata = tuple(str(value) for value in full_metadata)
    pc_columns = tuple(str(value) for value in pcs)
    expected_metadata = (str(sex), str(age), *pc_columns)

    if nonannotation != NONANNOTATION_PRS:
        raise ValueError(
            f"The all-non-annotation panel must be exactly {NONANNOTATION_PRS}; got {nonannotation}"
        )
    if len(all_columns) != 8 or len(set(all_columns)) != 8:
        raise ValueError("The all-PRS panel must contain eight unique families")
    if not set(nonannotation).issubset(all_columns):
        raise ValueError("The non-annotation panel must be a subset of the all-PRS panel")
    if best_prs not in all_columns:
        raise ValueError(f"Best PRS {best_prs!r} is absent from the all-PRS panel")
    if metadata != expected_metadata:
        raise ValueError(
            "Full metadata must be ordered as sex, age, then PC1-PC10; "
            f"expected {expected_metadata}, got {metadata}"
        )

    return (
        OfficialStage1Arm(
            name="raw_best_prs",
            model_family="raw_prs",
            prs_columns=(best_prs,),
            covariates=(),
            description="Canonical best single PRS used directly as the Stage-1 score",
        ),
        OfficialStage1Arm(
            name="tabpfn_best_prs_full_metadata",
            model_family="tabpfn",
            prs_columns=(best_prs,),
            covariates=metadata,
            description="TabPFN-3 with the same canonical best single PRS plus full metadata",
        ),
        OfficialStage1Arm(
            name="tabpfn_nonannotation_full_metadata",
            model_family="tabpfn",
            prs_columns=nonannotation,
            covariates=metadata,
            description="TabPFN-3 with all five non-annotation PRSs plus full metadata",
        ),
        OfficialStage1Arm(
            name="tabpfn_all8_full_metadata",
            model_family="tabpfn",
            prs_columns=all_columns,
            covariates=metadata,
            description="TabPFN-3 with all eight PRSs plus full metadata",
        ),
        OfficialStage1Arm(
            name="tabpfn_best_prs_no_age",
            model_family="tabpfn",
            prs_columns=(best_prs,),
            covariates=(str(sex), *pc_columns),
            description="Best single PRS plus sex and PCs; age omitted from Stage 1",
        ),
        OfficialStage1Arm(
            name="tabpfn_best_prs_no_sex",
            model_family="tabpfn",
            prs_columns=(best_prs,),
            covariates=(str(age), *pc_columns),
            description="Best single PRS plus age and PCs; sex omitted from Stage 1",
        ),
        OfficialStage1Arm(
            name="tabpfn_best_prs_no_pcs",
            model_family="tabpfn",
            prs_columns=(best_prs,),
            covariates=(str(sex), str(age)),
            description="Best single PRS plus sex and age; PCs omitted from Stage 1",
        ),
        OfficialStage1Arm(
            name="tabpfn_best_prs_no_age_sex",
            model_family="tabpfn",
            prs_columns=(best_prs,),
            covariates=pc_columns,
            description="Best single PRS plus PCs; age and sex omitted from Stage 1",
        ),
        OfficialStage1Arm(
            name="tabpfn_metadata_only",
            model_family="tabpfn",
            prs_columns=(),
            covariates=metadata,
            description="TabPFN-3 metadata main effect without PRSs",
        ),
        OfficialStage1Arm(
            name="tabpfn_best_prs_hard_ancestry",
            model_family="tabpfn",
            prs_columns=(best_prs,),
            covariates=(str(sex), str(age), str(ancestry_group)),
            description=("TabPFN-3 with the best single PRS, sex, age, and one-hot hard ancestry"),
        ),
        OfficialStage1Arm(
            name="xgboost_nonannotation_full_metadata",
            model_family="xgboost",
            prs_columns=nonannotation,
            covariates=metadata,
            description="Nested-tuned XGBoost with the same five PRSs and full metadata",
        ),
    )


def official_comparison_specs() -> tuple[ComparisonSpec, ...]:
    single_full = "tabpfn_best_prs_full_metadata"
    five_full = "tabpfn_nonannotation_full_metadata"
    return (
        ComparisonSpec(
            1,
            "TabPFN integration of the same best single PRS",
            "raw_best_prs",
            single_full,
            "Isolates TabPFN integration while holding the canonical single PRS fixed.",
        ),
        ComparisonSpec(
            1,
            "five non-annotation PRSs vs TabPFN single best",
            single_full,
            five_full,
            "Tests multi-PRS complementarity using the fixed five-PRS panel.",
        ),
        ComparisonSpec(
            1,
            "all eight PRSs vs TabPFN single best",
            single_full,
            "tabpfn_all8_full_metadata",
            "Tests broad multi-PRS complementarity against the same TabPFN single-PRS arm.",
        ),
        ComparisonSpec(
            1,
            "all eight PRSs vs five non-annotation PRSs",
            five_full,
            "tabpfn_all8_full_metadata",
            "Tests the incremental value of the three additional PRS families.",
        ),
        ComparisonSpec(
            2,
            "omit age from Stage 1",
            single_full,
            "tabpfn_best_prs_no_age",
            "All Stage-2 fits restore score plus full metadata.",
        ),
        ComparisonSpec(
            2,
            "omit sex from Stage 1",
            single_full,
            "tabpfn_best_prs_no_sex",
            "All Stage-2 fits restore score plus full metadata.",
        ),
        ComparisonSpec(
            2,
            "omit PCs from Stage 1",
            single_full,
            "tabpfn_best_prs_no_pcs",
            "All Stage-2 fits restore score plus full metadata.",
        ),
        ComparisonSpec(
            2,
            "omit age and sex from Stage 1",
            single_full,
            "tabpfn_best_prs_no_age_sex",
            "All Stage-2 fits restore score plus full metadata.",
        ),
        ComparisonSpec(
            3,
            "separate regression components vs joint single-PRS-metadata baseline",
            single_full,
            Q3_ADDITIVE_ARM,
            (
                "Uses M + TabPFN(raw best PRS, M) as the reference and compares it "
                "with M + TabPFN(M) + raw best PRS in the final regression."
            ),
        ),
        ComparisonSpec(
            4,
            "TabPFN vs nested-tuned XGBoost",
            "xgboost_nonannotation_full_metadata",
            five_full,
            "Both Stage-1 models use the exact same five PRSs and metadata.",
        ),
        ComparisonSpec(
            5,
            "one-hot hard ancestry vs continuous-PC joint baseline",
            single_full,
            "tabpfn_best_prs_hard_ancestry",
            (
                "The continuous-PC arm is the reference. Both Stage-1 arms use the same "
                "best single PRS and both Stage-2 fits restore score plus sex, age, and "
                "PC1-PC10."
            ),
        ),
    )


def scaled_logit_scores(
    train_probability: np.ndarray,
    validation_probability: np.ndarray,
    *,
    clip_eps: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train_logit = logit_from_probability(train_probability, eps=clip_eps)
    validation_logit = logit_from_probability(validation_probability, eps=clip_eps)
    return scale_validation_logits(
        train_logits=train_logit,
        validation_logits=validation_logit,
        mode="train_sd",
    )


def q3_additive_stage2_features(full_metadata: Sequence[str]) -> tuple[str, ...]:
    """Return the two-component Q3 additive regression contract."""
    return (
        *(str(value) for value in full_metadata),
        Q3_METADATA_SCORE_COMPONENT,
        Q3_RAW_PRS_COMPONENT,
    )
