import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from decipehr.ancestry_performance import (
    _bootstrap_auc_values,
    _stratified_bootstrap_index_matrix,
    additive_loss_decomposition,
    auc_pair_decomposition,
    paired_bootstrap_comparisons,
    paired_bootstrap_deltas,
    subgroup_metrics,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sampleid": [f"S-{index}" for index in range(12)],
            "disease": ["cad"] * 12,
            "ancestry_group": ["A"] * 6 + ["B"] * 6,
            "y_true": [0, 0, 0, 1, 1, 1] * 2,
            "best_single_prs": [
                0.2,
                0.4,
                0.6,
                0.3,
                0.5,
                0.7,
                0.1,
                0.5,
                0.7,
                0.2,
                0.6,
                0.8,
            ],
            "tabpfn3_best_single": [
                0.15,
                0.4,
                0.6,
                0.3,
                0.6,
                0.8,
                0.1,
                0.45,
                0.65,
                0.25,
                0.65,
                0.85,
            ],
            "tabpfn3_full": [
                0.1,
                0.4,
                0.6,
                0.3,
                0.7,
                0.9,
                0.1,
                0.4,
                0.6,
                0.3,
                0.7,
                0.9,
            ],
        }
    )


def test_subgroup_metrics_and_paired_bootstrap_preserve_direction():
    predictions = _predictions()
    metrics = subgroup_metrics(predictions)
    deltas = paired_bootstrap_deltas(predictions, n_bootstraps=100, seed=7)

    assert set(metrics["ancestry_group"]) == {"ALL", "A", "B"}
    all_metrics = metrics.loc[metrics["ancestry_group"].eq("ALL")].set_index("model")
    assert all_metrics.loc["tabpfn3_full", "auc"] > all_metrics.loc["best_single_prs", "auc"]
    assert all_metrics.loc["tabpfn3_full", "brier"] < all_metrics.loc["best_single_prs", "brier"]

    all_deltas = deltas.loc[deltas["ancestry_group"].eq("ALL")].set_index("metric")
    assert all_deltas.loc["auc", "tabpfn_minus_single"] > 0
    assert all_deltas.loc["brier", "tabpfn_minus_single"] < 0
    assert all_deltas.loc["auc", "probability_tabpfn_better"] > 0.5
    assert all_deltas.loc["brier", "probability_tabpfn_better"] > 0.5


def test_three_arm_paired_bootstrap_preserves_comparison_identity():
    comparisons = paired_bootstrap_comparisons(
        _predictions(), n_bootstraps=50, seed=11, metric_names=("auc",)
    )
    assert set(comparisons["metric"]) == {"auc"}
    overall_auc = comparisons.loc[
        comparisons["ancestry_group"].eq("ALL")
        & comparisons["metric"].eq("auc")
    ].set_index("comparison")

    assert set(overall_auc.index) == {
        "tabpfn3_best_single_vs_best_single",
        "tabpfn3_full_vs_tabpfn3_best_single",
        "tabpfn3_full_vs_best_single",
    }
    total_delta = overall_auc.loc[
        "tabpfn3_full_vs_best_single", "candidate_minus_reference"
    ]
    component_delta = (
        overall_auc.loc[
            "tabpfn3_best_single_vs_best_single", "candidate_minus_reference"
        ]
        + overall_auc.loc[
            "tabpfn3_full_vs_tabpfn3_best_single", "candidate_minus_reference"
        ]
    )
    assert total_delta == pytest.approx(component_delta)


def test_vectorized_stratified_bootstrap_auc_matches_sklearn():
    predictions = _predictions()
    y = predictions["y_true"].to_numpy()
    scores = predictions["tabpfn3_best_single"].to_numpy()
    indices, n_cases, n_controls = _stratified_bootstrap_index_matrix(
        y,
        n_bootstraps=25,
        rng=np.random.default_rng(19),
    )
    observed = _bootstrap_auc_values(
        scores,
        indices,
        n_cases=n_cases,
        n_controls=n_controls,
    )
    expected = np.asarray(
        [roc_auc_score(y[row], scores[row]) for row in indices], dtype=float
    )

    assert observed == pytest.approx(expected)


def test_auc_pair_decomposition_reconstructs_overall_auc_delta():
    predictions = _predictions()
    decomposition = auc_pair_decomposition(predictions)
    observed_delta = roc_auc_score(
        predictions["y_true"], predictions["tabpfn3_full"]
    ) - roc_auc_score(predictions["y_true"], predictions["best_single_prs"])

    assert decomposition["pair_weight"].sum() == pytest.approx(1.0)
    assert decomposition["weighted_auc_delta_contribution"].sum() == pytest.approx(
        observed_delta
    )


def test_additive_loss_decomposition_reconstructs_global_brier_improvement():
    predictions = _predictions()
    decomposition = additive_loss_decomposition(predictions)
    y = predictions["y_true"].to_numpy()
    single = predictions["best_single_prs"].to_numpy()
    tabpfn = predictions["tabpfn3_full"].to_numpy()
    observed_improvement = np.mean((y - single) ** 2) - np.mean((y - tabpfn) ** 2)

    assert decomposition["patient_weight"].sum() == pytest.approx(1.0)
    assert decomposition["weighted_brier_improvement_contribution"].sum() == pytest.approx(
        observed_improvement
    )
