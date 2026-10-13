from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MetadataScoreAblation:
    name: str
    score_covariates: tuple[str, ...]
    omitted_from_score: tuple[str, ...]
    description: str


def _unique(values: Sequence[str], *, context: str) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in values)
    duplicates = sorted({value for value in resolved if resolved.count(value) > 1})
    if duplicates:
        raise ValueError(f"{context} contains duplicate covariates: {duplicates}")
    return resolved


def build_metadata_score_ablations(
    *,
    sex: str,
    age: str,
    pcs: Sequence[str],
    full_covariates: Sequence[str],
) -> tuple[MetadataScoreAblation, ...]:
    pc_columns = _unique(pcs, context="PC covariates")
    if not pc_columns:
        raise ValueError("At least one PC covariate is required")
    expected_full = (str(sex), str(age), *pc_columns)
    observed_full = _unique(full_covariates, context="Full covariates")
    if observed_full != expected_full:
        raise ValueError(
            "Full covariates must be ordered as sex, age, then PCs; "
            f"expected {expected_full}, got {observed_full}"
        )

    return (
        MetadataScoreAblation(
            name="full",
            score_covariates=expected_full,
            omitted_from_score=(),
            description="Selected PRSs plus sex, age, and PC1-PC10",
        ),
        MetadataScoreAblation(
            name="no_age",
            score_covariates=(str(sex), *pc_columns),
            omitted_from_score=(str(age),),
            description="Selected PRSs plus sex and PC1-PC10; age omitted",
        ),
        MetadataScoreAblation(
            name="no_sex",
            score_covariates=(str(age), *pc_columns),
            omitted_from_score=(str(sex),),
            description="Selected PRSs plus age and PC1-PC10; sex omitted",
        ),
        MetadataScoreAblation(
            name="no_pcs",
            score_covariates=(str(sex), str(age)),
            omitted_from_score=pc_columns,
            description="Selected PRSs plus sex and age; PC1-PC10 omitted",
        ),
        MetadataScoreAblation(
            name="no_age_sex",
            score_covariates=pc_columns,
            omitted_from_score=(str(sex), str(age)),
            description="Selected PRSs plus PC1-PC10; age and sex omitted",
        ),
    )


def metadata_ablation_task(
    task_id: int,
    diseases: Sequence[str],
    ablations: Sequence[MetadataScoreAblation],
) -> tuple[str, MetadataScoreAblation]:
    n_tasks = len(diseases) * len(ablations)
    if task_id < 0 or task_id >= n_tasks:
        raise ValueError(f"task_id must be in [0, {n_tasks - 1}], got {task_id}")
    disease_index, ablation_index = divmod(task_id, len(ablations))
    return str(diseases[disease_index]), ablations[ablation_index]


def resolve_score_and_final_covariates(
    *,
    default_covariates: Sequence[str],
    shared_covariates: Sequence[str] | None = None,
    score_covariates: Sequence[str] | None = None,
    final_covariates: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    shared = _unique(
        default_covariates if shared_covariates is None else shared_covariates,
        context="Shared covariates",
    )
    score = _unique(
        shared if score_covariates is None else score_covariates,
        context="TabPFN score covariates",
    )
    final = _unique(
        shared if final_covariates is None else final_covariates,
        context="Final-model covariates",
    )
    return list(score), list(final)
