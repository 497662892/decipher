from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from decipehr.metrics import challenge_metrics

MODEL_COLUMNS = {
    "best_single_prs": "Best single PRS",
    "tabpfn3_best_single": "TabPFN-3 + best single PRS",
    "tabpfn3_full": "TabPFN-3 full",
}
PAIRWISE_COMPARISONS = (
    (
        "tabpfn3_best_single_vs_best_single",
        "best_single_prs",
        "tabpfn3_best_single",
    ),
    (
        "tabpfn3_full_vs_tabpfn3_best_single",
        "tabpfn3_best_single",
        "tabpfn3_full",
    ),
    ("tabpfn3_full_vs_best_single", "best_single_prs", "tabpfn3_full"),
)
HIGHER_IS_BETTER = {
    "auc": True,
    "auprc": True,
    "brier": False,
    "log_loss": False,
    "nagelkerke_r2": True,
    "top10_vs_lower90_or": True,
    "absolute_calibration_gap": False,
}


def _calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    observed = float(y.mean())
    predicted = float(p.mean())
    output = {
        "observed_risk": observed,
        "mean_predicted_risk": predicted,
        "calibration_gap": predicted - observed,
        "absolute_calibration_gap": abs(predicted - observed),
        "observed_expected_ratio": observed / predicted if predicted > 0 else math.nan,
        "calibration_intercept": math.nan,
        "calibration_slope": math.nan,
    }
    if len(y) < 3 or len(np.unique(y)) < 2 or np.std(p) == 0:
        return output
    logit = np.log(p / (1 - p))
    design = sm.add_constant(logit, has_constant="add")
    try:
        fit = sm.GLM(y, design, family=sm.families.Binomial()).fit()
    except Exception:
        return output
    output["calibration_intercept"] = float(fit.params[0])
    output["calibration_slope"] = float(fit.params[1])
    return output


def prediction_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    output: dict[str, float | int] = {
        "n": int(len(y)),
        "cases": int(y.sum()),
        "controls": int(len(y) - y.sum()),
        "prevalence": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    output.update(challenge_metrics(y, p))
    output.update(_calibration_metrics(y, p))
    return output


def subgroup_metrics(
    predictions: pd.DataFrame,
    group_col: str = "ancestry_group",
    model_columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    selected_models = MODEL_COLUMNS if model_columns is None else model_columns
    rows = []
    grouped_frames = [("ALL", predictions)]
    grouped_frames.extend(
        (str(group), frame) for group, frame in predictions.groupby(group_col, sort=True)
    )
    for ancestry_group, frame in grouped_frames:
        for model_col, model_display in selected_models.items():
            row = {
                "disease": str(frame["disease"].iloc[0]),
                "ancestry_group": ancestry_group,
                "model": model_col,
                "model_display": model_display,
            }
            row.update(
                prediction_metrics(
                    frame["y_true"].to_numpy(),
                    frame[model_col].to_numpy(),
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _comparison_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_names: Sequence[str] | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    selected = tuple(HIGHER_IS_BETTER) if metric_names is None else tuple(metric_names)
    unknown = sorted(set(selected).difference(HIGHER_IS_BETTER))
    if unknown:
        raise ValueError(f"Unknown comparison metrics: {unknown}")
    output: dict[str, float] = {}
    if "auc" in selected:
        output["auc"] = float(roc_auc_score(y, np.asarray(y_prob, dtype=float)))
    if "auprc" in selected:
        output["auprc"] = float(average_precision_score(y, p))
    if "brier" in selected:
        output["brier"] = float(brier_score_loss(y, p))
    if "log_loss" in selected:
        output["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    if "nagelkerke_r2" in selected or "top10_vs_lower90_or" in selected:
        challenge = challenge_metrics(y, p)
        if "nagelkerke_r2" in selected:
            output["nagelkerke_r2"] = float(challenge["nagelkerke_r2"])
        if "top10_vs_lower90_or" in selected:
            output["top10_vs_lower90_or"] = float(
                challenge["top10_vs_lower90_or"]
            )
    if "absolute_calibration_gap" in selected:
        output["absolute_calibration_gap"] = abs(float(p.mean() - y.mean()))
    return output


def _stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    cases = np.flatnonzero(y == 1)
    controls = np.flatnonzero(y == 0)
    case_sample = rng.choice(cases, size=len(cases), replace=True)
    control_sample = rng.choice(controls, size=len(controls), replace=True)
    return np.concatenate([case_sample, control_sample])


def _stratified_bootstrap_index_matrix(
    y_true: np.ndarray,
    *,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    y = np.asarray(y_true, dtype=int)
    cases = np.flatnonzero(y == 1)
    controls = np.flatnonzero(y == 0)
    case_samples = rng.choice(
        cases, size=(n_bootstraps, len(cases)), replace=True
    )
    control_samples = rng.choice(
        controls, size=(n_bootstraps, len(controls)), replace=True
    )
    return (
        np.concatenate([case_samples, control_samples], axis=1),
        len(cases),
        len(controls),
    )


def _bootstrap_auc_values(
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    n_cases: int,
    n_controls: int,
) -> np.ndarray:
    sampled = np.asarray(scores, dtype=float)[indices]
    ranks = rankdata(sampled, method="average", axis=1)
    case_rank_sum = ranks[:, :n_cases].sum(axis=1)
    return (
        case_rank_sum - n_cases * (n_cases + 1) / 2
    ) / (n_cases * n_controls)


def paired_bootstrap_comparisons(
    predictions: pd.DataFrame,
    n_bootstraps: int = 2000,
    seed: int = 42,
    group_col: str = "ancestry_group",
    comparisons: Sequence[tuple[str, str, str]] = PAIRWISE_COMPARISONS,
    metric_names: Sequence[str] | None = None,
    family_size: int = 1,
) -> pd.DataFrame:
    if family_size < 1:
        raise ValueError("family_size must be at least 1")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1")
    if not comparisons:
        raise ValueError("comparisons must contain at least one comparison")
    selected_metrics = (
        tuple(HIGHER_IS_BETTER) if metric_names is None else tuple(metric_names)
    )
    unknown_metrics = sorted(set(selected_metrics).difference(HIGHER_IS_BETTER))
    if unknown_metrics:
        raise ValueError(f"Unknown comparison metrics: {unknown_metrics}")
    if not selected_metrics:
        raise ValueError("metric_names must contain at least one metric")
    required_models = {
        model
        for _, reference_model, candidate_model in comparisons
        for model in (reference_model, candidate_model)
    }
    missing_models = sorted(required_models.difference(predictions.columns))
    if missing_models:
        raise ValueError(f"Prediction columns are missing: {missing_models}")

    rows = []
    grouped_frames = [("ALL", predictions)]
    grouped_frames.extend(
        (str(group), frame) for group, frame in predictions.groupby(group_col, sort=True)
    )
    for group_index, (ancestry_group, frame) in enumerate(grouped_frames):
        y = frame["y_true"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        model_scores = {
            model: frame[model].to_numpy(dtype=float) for model in required_models
        }
        point_metrics = {
            model: _comparison_metrics(y, scores, selected_metrics)
            for model, scores in model_scores.items()
        }
        bootstrap_deltas = {
            (comparison, metric): []
            for comparison, _, _ in comparisons
            for metric in selected_metrics
        }
        rng = np.random.default_rng(seed + group_index)
        if selected_metrics == ("auc",):
            indices, n_cases, n_controls = _stratified_bootstrap_index_matrix(
                y,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
            auc_values = {
                model: _bootstrap_auc_values(
                    scores,
                    indices,
                    n_cases=n_cases,
                    n_controls=n_controls,
                )
                for model, scores in model_scores.items()
            }
            for comparison, reference_model, candidate_model in comparisons:
                bootstrap_deltas[(comparison, "auc")] = (
                    auc_values[candidate_model] - auc_values[reference_model]
                )
        else:
            for _ in range(n_bootstraps):
                indices = _stratified_bootstrap_indices(y, rng)
                boot_metrics = {
                    model: _comparison_metrics(
                        y[indices], scores[indices], selected_metrics
                    )
                    for model, scores in model_scores.items()
                }
                for comparison, reference_model, candidate_model in comparisons:
                    for metric in selected_metrics:
                        bootstrap_deltas[(comparison, metric)].append(
                            boot_metrics[candidate_model][metric]
                            - boot_metrics[reference_model][metric]
                        )

        for comparison, reference_model, candidate_model in comparisons:
            for metric in selected_metrics:
                values = bootstrap_deltas[(comparison, metric)]
                array = np.asarray(values, dtype=float)
                array = array[np.isfinite(array)]
                if len(array):
                    ci_low, median, ci_high = np.quantile(
                        array, [0.025, 0.5, 0.975]
                    )
                    family_low, family_high = np.quantile(
                        array, [0.025 / family_size, 1 - 0.025 / family_size]
                    )
                    if HIGHER_IS_BETTER[metric]:
                        probability_better = float(
                            np.mean(array > 0) + 0.5 * np.mean(array == 0)
                        )
                    else:
                        probability_better = float(
                            np.mean(array < 0) + 0.5 * np.mean(array == 0)
                        )
                else:
                    ci_low = median = ci_high = probability_better = math.nan
                    family_low = family_high = math.nan
                reference_value = point_metrics[reference_model][metric]
                candidate_value = point_metrics[candidate_model][metric]
                rows.append(
                    {
                        "disease": str(frame["disease"].iloc[0]),
                        "ancestry_group": ancestry_group,
                        "n": int(len(frame)),
                        "cases": int(y.sum()),
                        "comparison": comparison,
                        "reference_model": reference_model,
                        "candidate_model": candidate_model,
                        "metric": metric,
                        "higher_is_better": HIGHER_IS_BETTER[metric],
                        "reference_value": reference_value,
                        "candidate_value": candidate_value,
                        "candidate_minus_reference": candidate_value
                        - reference_value,
                        "bootstrap_median_delta": float(median),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "probability_candidate_better": probability_better,
                        "n_bootstraps": int(n_bootstraps),
                        "family_size": int(family_size),
                        "familywise_ci_low": float(family_low),
                        "familywise_ci_high": float(family_high),
                    }
                )
    return pd.DataFrame(rows)


def paired_bootstrap_deltas(
    predictions: pd.DataFrame,
    n_bootstraps: int = 2000,
    seed: int = 42,
    group_col: str = "ancestry_group",
) -> pd.DataFrame:
    """Return the legacy full-TabPFN minus single-PRS comparison schema."""
    generic = paired_bootstrap_comparisons(
        predictions,
        n_bootstraps=n_bootstraps,
        seed=seed,
        group_col=group_col,
        comparisons=(
            ("tabpfn3_full_vs_best_single", "best_single_prs", "tabpfn3_full"),
        ),
    )
    return generic.rename(
        columns={
            "reference_value": "single_value",
            "candidate_value": "tabpfn_value",
            "candidate_minus_reference": "tabpfn_minus_single",
            "probability_candidate_better": "probability_tabpfn_better",
        }
    ).drop(
        columns=["comparison", "reference_model", "candidate_model"]
    )


def _case_control_auc(case_scores: np.ndarray, control_scores: np.ndarray) -> float:
    y = np.concatenate(
        [
            np.ones(len(case_scores), dtype=int),
            np.zeros(len(control_scores), dtype=int),
        ]
    )
    scores = np.concatenate([case_scores, control_scores])
    return float(roc_auc_score(y, scores))


def auc_pair_decomposition(
    predictions: pd.DataFrame,
    group_col: str = "ancestry_group",
) -> pd.DataFrame:
    y = predictions["y_true"].to_numpy(dtype=int)
    total_cases = int(y.sum())
    total_controls = int(len(y) - y.sum())
    rows = []
    for case_group, case_frame in predictions.loc[predictions["y_true"].eq(1)].groupby(
        group_col, sort=True
    ):
        for control_group, control_frame in predictions.loc[predictions["y_true"].eq(0)].groupby(
            group_col, sort=True
        ):
            pair_weight = (len(case_frame) * len(control_frame)) / (
                total_cases * total_controls
            )
            single_auc = _case_control_auc(
                case_frame["best_single_prs"].to_numpy(dtype=float),
                control_frame["best_single_prs"].to_numpy(dtype=float),
            )
            tabpfn_auc = _case_control_auc(
                case_frame["tabpfn3_full"].to_numpy(dtype=float),
                control_frame["tabpfn3_full"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "disease": str(predictions["disease"].iloc[0]),
                    "case_ancestry_group": str(case_group),
                    "control_ancestry_group": str(control_group),
                    "pair_type": "within" if case_group == control_group else "cross",
                    "n_cases": int(len(case_frame)),
                    "n_controls": int(len(control_frame)),
                    "pair_weight": pair_weight,
                    "single_pair_auc": single_auc,
                    "tabpfn_pair_auc": tabpfn_auc,
                    "pair_auc_delta": tabpfn_auc - single_auc,
                    "weighted_auc_delta_contribution": pair_weight
                    * (tabpfn_auc - single_auc),
                }
            )
    return pd.DataFrame(rows)


def additive_loss_decomposition(
    predictions: pd.DataFrame,
    group_col: str = "ancestry_group",
) -> pd.DataFrame:
    rows = []
    total_n = len(predictions)
    for ancestry_group, frame in predictions.groupby(group_col, sort=True):
        y = frame["y_true"].to_numpy(dtype=int)
        single = np.clip(frame["best_single_prs"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        tabpfn = np.clip(frame["tabpfn3_full"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        weight = len(frame) / total_n
        single_brier = float(brier_score_loss(y, single))
        tabpfn_brier = float(brier_score_loss(y, tabpfn))
        single_log_loss = float(log_loss(y, single, labels=[0, 1]))
        tabpfn_log_loss = float(log_loss(y, tabpfn, labels=[0, 1]))
        rows.append(
            {
                "disease": str(frame["disease"].iloc[0]),
                "ancestry_group": str(ancestry_group),
                "n": int(len(frame)),
                "cases": int(y.sum()),
                "patient_weight": weight,
                "single_brier": single_brier,
                "tabpfn_brier": tabpfn_brier,
                "brier_improvement": single_brier - tabpfn_brier,
                "weighted_brier_improvement_contribution": weight
                * (single_brier - tabpfn_brier),
                "single_log_loss": single_log_loss,
                "tabpfn_log_loss": tabpfn_log_loss,
                "log_loss_improvement": single_log_loss - tabpfn_log_loss,
                "weighted_log_loss_improvement_contribution": weight
                * (single_log_loss - tabpfn_log_loss),
            }
        )
    return pd.DataFrame(rows)
