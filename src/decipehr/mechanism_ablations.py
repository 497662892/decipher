from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from decipehr.metrics import (
    binary_metrics,
    nagelkerke_r2,
    odds_ratio_per_sd,
    risk_stratification_or,
    top_decile_odds_ratio,
)
from decipehr.schema import require_columns
from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer


@dataclass(frozen=True)
class VariantSpec:
    name: str
    model_family: str
    prs_columns: tuple[str, ...]
    covariates: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ComparisonSpec:
    question: int
    comparison: str
    reference: str
    challenger: str
    interpretation: str


def fold_task(task_id: int, diseases: Sequence[str], n_folds: int = 5) -> tuple[str, int]:
    if task_id < 0 or task_id >= len(diseases) * n_folds:
        raise ValueError(f"task_id must be in [0, {len(diseases) * n_folds - 1}], got {task_id}")
    return str(diseases[task_id // n_folds]), int(task_id % n_folds)


def build_tabpfn_variants(
    *,
    best_prs: str,
    selected_prs: Sequence[str],
    all_prs: Sequence[str],
    full_metadata: Sequence[str],
    age_sex: Sequence[str],
    pcs: Sequence[str],
    ancestry_group_metadata: Sequence[str],
) -> tuple[VariantSpec, ...]:
    selected = tuple(selected_prs)
    all_columns = tuple(all_prs)
    if best_prs not in all_columns:
        raise ValueError(f"Best PRS {best_prs!r} is not present in the all-PRS set")
    missing_selected = [column for column in selected if column not in all_columns]
    if missing_selected:
        raise ValueError(
            f"Selected PRS columns are absent from the all-PRS set: {missing_selected}"
        )
    if len(set(all_columns)) != len(all_columns):
        raise ValueError("The all-PRS set contains duplicate columns")

    return (
        VariantSpec(
            name="tabpfn_best_prs_full_metadata",
            model_family="tabpfn",
            prs_columns=(best_prs,),
            covariates=tuple(full_metadata),
            description="TabPFN with the frozen canonical best single PRS and full metadata",
        ),
        VariantSpec(
            name="tabpfn_selected_prs_full_metadata",
            model_family="tabpfn",
            prs_columns=selected,
            covariates=tuple(full_metadata),
            description="Joint TabPFN with the final disease-specific selected PRS panel and full metadata",
        ),
        VariantSpec(
            name="tabpfn_all_prs_full_metadata",
            model_family="tabpfn",
            prs_columns=all_columns,
            covariates=tuple(full_metadata),
            description="TabPFN with all eight PRS families and full metadata",
        ),
        VariantSpec(
            name="tabpfn_selected_prs_age_sex",
            model_family="tabpfn",
            prs_columns=selected,
            covariates=tuple(age_sex),
            description="TabPFN with selected PRSs plus age and sex, without PCs",
        ),
        VariantSpec(
            name="tabpfn_selected_prs_pcs",
            model_family="tabpfn",
            prs_columns=selected,
            covariates=tuple(pcs),
            description="TabPFN with selected PRSs plus continuous PCs, without age or sex",
        ),
        VariantSpec(
            name="tabpfn_selected_prs_only",
            model_family="tabpfn",
            prs_columns=selected,
            covariates=(),
            description="TabPFN nonlinear PRS main effect without metadata",
        ),
        VariantSpec(
            name="tabpfn_metadata_only",
            model_family="tabpfn",
            prs_columns=(),
            covariates=tuple(full_metadata),
            description="TabPFN nonlinear metadata main effect without PRSs",
        ),
        VariantSpec(
            name="tabpfn_selected_prs_hard_ancestry",
            model_family="tabpfn",
            prs_columns=selected,
            covariates=tuple(ancestry_group_metadata),
            description="TabPFN with selected PRSs, age, sex, and one-hot broad ancestry group",
        ),
    )


def comparison_specs() -> tuple[ComparisonSpec, ...]:
    joint = "tabpfn_selected_prs_full_metadata"
    return (
        ComparisonSpec(
            1,
            "selected PRS panel vs best single PRS",
            "tabpfn_best_prs_full_metadata",
            joint,
            "Tests whether disease-specific multi-PRS complementarity improves on one frozen best PRS.",
        ),
        ComparisonSpec(
            1,
            "all eight PRSs vs best single PRS",
            "tabpfn_best_prs_full_metadata",
            "tabpfn_all_prs_full_metadata",
            "Tests whether broad multi-PRS complementarity improves on one frozen best PRS.",
        ),
        ComparisonSpec(
            1,
            "selected PRS panel vs all eight PRSs",
            "tabpfn_all_prs_full_metadata",
            joint,
            "Separates complementarity from indiscriminate inclusion of every PRS family.",
        ),
        ComparisonSpec(
            2,
            "PCs vs age/sex",
            "tabpfn_selected_prs_age_sex",
            "tabpfn_selected_prs_pcs",
            "Compares ancestry context alone with demographic context alone.",
        ),
        ComparisonSpec(
            2,
            "increment from PCs conditional on age/sex",
            "tabpfn_selected_prs_age_sex",
            joint,
            "Measures the incremental value of continuous PCs after age and sex are already present.",
        ),
        ComparisonSpec(
            2,
            "increment from age/sex conditional on PCs",
            "tabpfn_selected_prs_pcs",
            joint,
            "Measures the incremental value of age and sex after continuous PCs are already present.",
        ),
        ComparisonSpec(
            3,
            "joint modeling vs separate additive main effects",
            "tabpfn_separate_additive",
            joint,
            "Tests whether joint feature interaction improves on two nonlinear main effects added on the logit scale.",
        ),
        ComparisonSpec(
            4,
            "TabPFN vs tuned XGBoost",
            "xgboost_selected_prs_full_metadata",
            joint,
            "Tests whether gains are TabPFN-specific rather than a generic benefit from nonlinearity.",
        ),
        ComparisonSpec(
            5,
            "continuous PCs vs one-hot broad ancestry",
            "tabpfn_selected_prs_hard_ancestry",
            joint,
            "Tests continuous personalization against hard ancestry stratification.",
        ),
    )


def clipped_logit(probability: np.ndarray | pd.Series, eps: float = 1e-6) -> np.ndarray:
    if not 0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5")
    values = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    return np.log(values / (1.0 - values))


def sigmoid(values: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def prior_corrected_additive_probability(
    metadata_probability: np.ndarray | pd.Series,
    prs_probability: np.ndarray | pd.Series,
    *,
    training_prevalence: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < training_prevalence < 1:
        raise ValueError("training_prevalence must be strictly between zero and one")
    metadata = np.asarray(metadata_probability, dtype=float)
    prs = np.asarray(prs_probability, dtype=float)
    if metadata.shape != prs.shape:
        raise ValueError("metadata_probability and prs_probability must have identical shapes")
    prior_logit = float(clipped_logit(np.asarray([training_prevalence]), eps=eps)[0])
    additive_logit = clipped_logit(metadata, eps=eps) + clipped_logit(prs, eps=eps) - prior_logit
    return sigmoid(additive_logit), additive_logit


def fit_tuned_xgboost(
    *,
    context: pd.DataFrame,
    validation: pd.DataFrame,
    label_col: str,
    prs_columns: Sequence[str],
    covariates: Sequence[str],
    parameter_grid: dict[str, Sequence[float | int]],
    fixed_parameters: dict[str, float | int],
    inner_folds: int,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for the matched nonlinear ablation. "
            "Run `uv sync --extra tabpfn --extra ablation`."
        ) from exc

    required_grid = ["max_depth", "learning_rate", "min_child_weight"]
    missing_grid = [name for name in required_grid if not parameter_grid.get(name)]
    if missing_grid:
        raise ValueError(f"XGBoost parameter grid is missing values for: {missing_grid}")
    y = context[label_col].astype(int).to_numpy()
    class_counts = pd.Series(y).value_counts()
    actual_inner_folds = min(int(inner_folds), int(class_counts.min()))
    if actual_inner_folds < 2:
        raise ValueError("XGBoost inner tuning requires at least two examples per class")
    splitter = StratifiedKFold(
        n_splits=actual_inner_folds,
        shuffle=True,
        random_state=seed,
    )
    candidates = [
        {
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "min_child_weight": float(min_child_weight),
        }
        for max_depth, learning_rate, min_child_weight in product(
            parameter_grid["max_depth"],
            parameter_grid["learning_rate"],
            parameter_grid["min_child_weight"],
        )
    ]

    tuning_rows: list[dict[str, float | int]] = []
    for candidate_index, candidate in enumerate(candidates):
        fold_aucs = []
        for inner_fold, (fit_indices, validation_indices) in enumerate(
            splitter.split(np.zeros(len(y)), y)
        ):
            fit_frame = context.iloc[fit_indices]
            validation_frame = context.iloc[validation_indices]
            transformer = TabPFNFeatureTransformer.fit(
                fit_frame,
                prs_cols=list(prs_columns),
                covariates=list(covariates),
                include_covariates=bool(covariates),
            )
            x_fit = transformer.transform(fit_frame)
            x_validation = transformer.transform(validation_frame)
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=seed + inner_fold,
                verbosity=0,
                **fixed_parameters,
                **candidate,
            )
            model.fit(x_fit, fit_frame[label_col].astype(int))
            probability = model.predict_proba(x_validation)[:, 1]
            fold_aucs.append(
                float(roc_auc_score(validation_frame[label_col].astype(int), probability))
            )
        tuning_rows.append(
            {
                "candidate_index": int(candidate_index),
                **candidate,
                "mean_inner_auc": float(np.mean(fold_aucs)),
                "sd_inner_auc": float(np.std(fold_aucs, ddof=0)),
                "n_inner_folds": int(actual_inner_folds),
            }
        )

    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["mean_inner_auc", "sd_inner_auc", "max_depth", "learning_rate", "min_child_weight"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    best_row = tuning.iloc[0]
    best_parameters: dict[str, float | int] = {
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "min_child_weight": float(best_row["min_child_weight"]),
    }

    transformer = TabPFNFeatureTransformer.fit(
        context,
        prs_cols=list(prs_columns),
        covariates=list(covariates),
        include_covariates=bool(covariates),
    )
    x_context = transformer.transform(context)
    x_validation = transformer.transform(validation)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=0,
        **fixed_parameters,
        **best_parameters,
    )
    model.fit(x_context, context[label_col].astype(int))
    probability = np.asarray(model.predict_proba(x_validation)[:, 1], dtype=float)
    return probability, transformer.summary(), tuning.reset_index(drop=True), best_parameters


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    require_columns(frame, ["y_true", "y_prob", "score"], "ablation predictions")
    base = binary_metrics(frame["y_true"], frame["y_prob"])
    row: dict[str, float | int] = {
        "auc": base.auc,
        "auprc": base.auprc,
        "brier": base.brier,
        "prevalence": base.prevalence,
        "n": base.n,
        "nagelkerke_r2": nagelkerke_r2(frame["y_true"], frame["y_prob"]),
        "or_per_sd": odds_ratio_per_sd(frame["y_true"], frame["score"]),
    }
    row.update(top_decile_odds_ratio(frame["y_true"], frame["score"]))
    row.update(risk_stratification_or(frame["y_true"], frame["score"]))
    return row


def validate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    patient_col: str,
    expected_variants: Sequence[str] | None = None,
) -> None:
    required = [
        patient_col,
        "disease",
        "fold",
        "variant",
        "model_family",
        "y_true",
        "y_prob",
        "score",
    ]
    require_columns(predictions, required, "mechanism-ablation predictions")
    if predictions.duplicated([patient_col, "disease", "variant"]).any():
        raise ValueError("Each patient must have exactly one prediction per disease and variant")
    if predictions[["y_true", "y_prob", "score"]].isna().any().any():
        raise ValueError("Mechanism-ablation predictions contain missing labels or scores")
    probabilities = pd.to_numeric(predictions["y_prob"], errors="coerce")
    if (~probabilities.between(0.0, 1.0)).any():
        raise ValueError("Mechanism-ablation probabilities must lie in [0, 1]")
    if expected_variants is not None:
        expected = set(expected_variants)
        for disease, frame in predictions.groupby("disease", sort=False):
            observed = set(frame["variant"].astype(str))
            if observed != expected:
                raise ValueError(
                    f"Variant set mismatch for {disease}: missing={sorted(expected - observed)}, "
                    f"extra={sorted(observed - expected)}"
                )
            counts = frame.groupby("variant")[patient_col].nunique()
            if counts.nunique() != 1:
                raise ValueError(f"Variant patient counts differ for {disease}: {counts.to_dict()}")


def align_paired_predictions(
    predictions: pd.DataFrame,
    *,
    disease: str,
    reference: str,
    challenger: str,
    patient_col: str,
) -> pd.DataFrame:
    subset = predictions.loc[
        (predictions["disease"].astype(str) == disease)
        & predictions["variant"].isin([reference, challenger]),
        [patient_col, "fold", "variant", "y_true", "y_prob", "score"],
    ].copy()
    if subset.empty:
        raise ValueError(f"No paired predictions found for {disease}: {reference} vs {challenger}")
    keys = [patient_col, "fold", "y_true"]
    reference_frame = subset.loc[subset["variant"] == reference, keys + ["y_prob", "score"]].rename(
        columns={"y_prob": "reference_y_prob", "score": "reference_score"}
    )
    challenger_frame = subset.loc[
        subset["variant"] == challenger,
        keys + ["y_prob", "score"],
    ].rename(columns={"y_prob": "challenger_y_prob", "score": "challenger_score"})
    wide = reference_frame.merge(
        challenger_frame,
        on=keys,
        how="inner",
        validate="one_to_one",
    ).dropna()
    expected_reference = int((subset["variant"] == reference).sum())
    expected_challenger = int((subset["variant"] == challenger).sum())
    if len(wide) != expected_reference or len(wide) != expected_challenger:
        raise ValueError(
            f"Patient mismatch for {disease}: {reference}={expected_reference}, "
            f"{challenger}={expected_challenger}, paired={len(wide)}"
        )
    return wide


def _bootstrap_metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "auc": lambda y, p: float(roc_auc_score(y, p)),
        "auprc": lambda y, p: float(average_precision_score(y, p)),
        "brier": lambda y, p: float(brier_score_loss(y, p)),
        "nagelkerke_r2": lambda y, p: float(nagelkerke_r2(y, p)),
        "top10_vs_lower90_or": lambda y, score: float(
            top_decile_odds_ratio(y, score)["top10_vs_lower90_or"]
        ),
    }


def _weighted_auc_batch(
    y_true: np.ndarray,
    probability: np.ndarray,
    sample_weights: np.ndarray,
) -> np.ndarray:
    """Compute tied-rank AUCs for many integer bootstrap weight vectors."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probability, dtype=float)
    weights = np.asarray(sample_weights)
    if weights.ndim != 2 or weights.shape[1] != len(y):
        raise ValueError("sample_weights must have shape (n_resamples, n_observations)")
    if len(scores) != len(y):
        raise ValueError("y_true and probability must have identical lengths")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    sorted_weights = weights[:, order]
    positive = np.add.reduceat(sorted_weights * sorted_y, starts, axis=1)
    negative = np.add.reduceat(sorted_weights * (1 - sorted_y), starts, axis=1)
    negative_before = np.cumsum(negative, axis=1) - negative
    numerator = np.sum(positive * (negative_before + 0.5 * negative), axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(weights), np.nan, dtype=float),
        where=denominator > 0,
    )


def _weighted_nagelkerke_r2_batch(
    y_true: np.ndarray,
    probability: np.ndarray,
    sample_weights: np.ndarray,
) -> np.ndarray:
    """Compute Nagelkerke R2 for integer-weighted bootstrap samples."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    weights = np.asarray(sample_weights)
    if weights.ndim != 2 or weights.shape[1] != len(y):
        raise ValueError("sample_weights must have shape (n_resamples, n_observations)")
    if len(p) != len(y):
        raise ValueError("y_true and probability must have identical lengths")

    n = weights.sum(axis=1).astype(float)
    cases = weights @ y
    prevalence = np.divide(
        cases,
        n,
        out=np.full(len(weights), np.nan, dtype=float),
        where=n > 0,
    )
    prevalence = np.clip(prevalence, 1e-15, 1 - 1e-15)
    log_likelihood = weights @ (y * np.log(p) + (1 - y) * np.log1p(-p))
    null_log_likelihood = cases * np.log(prevalence) + (n - cases) * np.log1p(-prevalence)
    denominator = 1 - np.exp(2 * null_log_likelihood / n)
    numerator = 1 - np.exp(2 * (null_log_likelihood - log_likelihood) / n)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(weights), np.nan, dtype=float),
        where=(n > 0) & (denominator > 0),
    )


def _weighted_top_decile_or_batch(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weights: np.ndarray,
    *,
    top_fraction: float = 0.10,
) -> np.ndarray:
    """Compute top-decile versus lower-90% ORs for weighted bootstrap samples."""
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1")
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(score, dtype=float)
    weights = np.asarray(sample_weights)
    if weights.ndim != 2 or weights.shape[1] != len(y):
        raise ValueError("sample_weights must have shape (n_resamples, n_observations)")
    if len(scores) != len(y):
        raise ValueError("y_true and score must have identical lengths")

    order = np.argsort(-scores, kind="mergesort")
    sorted_weights = weights[:, order]
    sorted_y = y[order]
    n = weights.sum(axis=1)
    top_n = np.maximum(1, np.ceil(n * top_fraction).astype(int))
    cumulative_before = np.cumsum(sorted_weights, axis=1) - sorted_weights
    remaining = np.maximum(top_n[:, None] - cumulative_before, 0)
    top_weights = np.minimum(sorted_weights, remaining)

    top_cases = top_weights @ sorted_y
    total_cases = weights @ y
    lower_cases = total_cases - top_cases
    top_controls = top_n - top_cases
    lower_n = n - top_n
    lower_controls = lower_n - lower_cases
    correction = np.where(
        (top_cases == 0) | (top_controls == 0) | (lower_cases == 0) | (lower_controls == 0),
        0.5,
        0.0,
    )
    denominator = (top_controls + correction) * (lower_cases + correction)
    numerator = (top_cases + correction) * (lower_controls + correction)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(weights), np.nan, dtype=float),
        where=(n > 1) & (lower_n > 0) & (denominator > 0),
    )


def paired_stratified_bootstrap(
    paired: pd.DataFrame,
    *,
    reference: str,
    challenger: str,
    n_resamples: int,
    seed: int,
    metrics: Sequence[str] = ("auc", "brier"),
) -> pd.DataFrame:
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    y = paired["y_true"].to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Paired bootstrap requires both outcome classes")
    reference_probability = paired["reference_y_prob"].to_numpy(dtype=float)
    challenger_probability = paired["challenger_y_prob"].to_numpy(dtype=float)
    reference_score = paired["reference_score"].to_numpy(dtype=float)
    challenger_score = paired["challenger_score"].to_numpy(dtype=float)
    class_indices = [np.flatnonzero(y == label) for label in [0, 1]]
    rng = np.random.default_rng(seed)
    available_functions = _bootstrap_metric_functions()
    unknown_metrics = [metric for metric in metrics if metric not in available_functions]
    if unknown_metrics:
        raise ValueError(f"Unknown bootstrap metrics: {unknown_metrics}")
    metric_functions = {metric: available_functions[metric] for metric in metrics}
    distributions = {metric: np.empty(n_resamples, dtype=float) for metric in metric_functions}
    batch_size = min(100, n_resamples)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        current_size = stop - start
        weights = np.zeros((current_size, len(y)), dtype=np.int32)
        rows = np.arange(current_size)[:, None]
        for indices in class_indices:
            sampled = rng.choice(indices, size=(current_size, len(indices)), replace=True)
            np.add.at(weights, (np.broadcast_to(rows, sampled.shape), sampled), 1)

        if "auc" in metric_functions:
            reference_auc = _weighted_auc_batch(y, reference_probability, weights)
            challenger_auc = _weighted_auc_batch(y, challenger_probability, weights)
            distributions["auc"][start:stop] = challenger_auc - reference_auc
        if "brier" in metric_functions:
            reference_error = np.square(y - reference_probability)
            challenger_error = np.square(y - challenger_probability)
            denominator = weights.sum(axis=1)
            distributions["brier"][start:stop] = (
                weights @ (challenger_error - reference_error) / denominator
            )
        if "nagelkerke_r2" in metric_functions:
            reference_r2 = _weighted_nagelkerke_r2_batch(y, reference_probability, weights)
            challenger_r2 = _weighted_nagelkerke_r2_batch(y, challenger_probability, weights)
            distributions["nagelkerke_r2"][start:stop] = challenger_r2 - reference_r2
        if "top10_vs_lower90_or" in metric_functions:
            reference_or90 = _weighted_top_decile_or_batch(y, reference_score, weights)
            challenger_or90 = _weighted_top_decile_or_batch(y, challenger_score, weights)
            distributions["top10_vs_lower90_or"][start:stop] = challenger_or90 - reference_or90

        slower_metrics = set(metric_functions) - {
            "auc",
            "brier",
            "nagelkerke_r2",
            "top10_vs_lower90_or",
        }
        if slower_metrics:
            for offset, weight in enumerate(weights):
                sampled = np.repeat(np.arange(len(y)), weight)
                sampled_y = y[sampled]
                for metric in slower_metrics:
                    fn = metric_functions[metric]
                    distributions[metric][start + offset] = fn(
                        sampled_y, challenger_probability[sampled]
                    ) - fn(sampled_y, reference_probability[sampled])

    rows = []
    for metric, fn in metric_functions.items():
        if metric == "top10_vs_lower90_or":
            reference_point = fn(y, reference_score)
            challenger_point = fn(y, challenger_score)
        else:
            reference_point = fn(y, reference_probability)
            challenger_point = fn(y, challenger_probability)
        delta = challenger_point - reference_point
        distribution = distributions[metric]
        finite = distribution[np.isfinite(distribution)]
        if len(finite) == 0:
            lower = upper = p_two_sided = math.nan
        else:
            lower, upper = np.quantile(finite, [0.025, 0.975])
            nonpositive = (np.sum(finite <= 0) + 1) / (len(finite) + 1)
            nonnegative = (np.sum(finite >= 0) + 1) / (len(finite) + 1)
            p_two_sided = min(1.0, 2.0 * min(nonpositive, nonnegative))
        rows.append(
            {
                "metric": metric,
                "reference_value": reference_point,
                "challenger_value": challenger_point,
                "delta": delta,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "p_two_sided": float(p_two_sided),
                "n_resamples": int(n_resamples),
            }
        )
    return pd.DataFrame(rows)
