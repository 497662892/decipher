import numpy as np
import pandas as pd
import pytest

from decipehr.metrics import nagelkerke_r2, top_decile_odds_ratio
from decipehr.prediction_diagnostics import (
    decile_table,
    log_likelihood_terms,
    paired_bootstrap,
    risk_deciles,
    validate_predictions,
)


def test_decile_boundary_matches_or90_with_nondivisible_n_and_ties():
    score = np.repeat(np.arange(18), 6)[:103].astype(float)
    y = (np.arange(103) % 3 == 0).astype(int)
    decile = risk_deciles(score)
    result = top_decile_odds_ratio(y, score)
    assert (decile == 10).sum() == result["top10_n"] == 11
    assert y[decile == 10].sum() == result["top10_cases"]
    assert set(decile) == set(range(1, 11))


def test_likelihood_contributions_sum_and_reconstruct_r2():
    y = np.tile([0, 0, 0, 1], 25)
    p = np.linspace(0.05, 0.7, len(y))
    reference = np.full(len(y), 0.25)
    table = decile_table(y, p, p, reference)
    ll0 = log_likelihood_terms(y, y.mean()).sum()
    gain = (log_likelihood_terms(y, p) - log_likelihood_terms(y, reference)).sum()
    assert table.ll_gain_null.sum() == pytest.approx(gain)
    assert table.ll_gain_covariates.sum() == pytest.approx(gain)
    reconstructed = -np.expm1(-2 * gain / len(y)) / -np.expm1(2 * ll0 / len(y))
    assert reconstructed == pytest.approx(nagelkerke_r2(y, p))


def test_paired_bootstrap_identical_models_have_identical_draws():
    y = np.tile([0, 0, 0, 1], 30)
    p = np.linspace(0.02, 0.85, len(y))
    draws, ci = paired_bootstrap(y, {"a": p, "b": p}, 12, 42)
    a = draws.query("model == 'a'").drop(columns="model").reset_index(drop=True)
    b = draws.query("model == 'b'").drop(columns="model").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert ci.n_valid.eq(12).all()


def test_reject_duplicate_or_cross_disease_fold_mismatch():
    frame = pd.DataFrame({"sampleid": [1, 2], "disease": ["bc", "bc"], "fold": [0, 1],
                          "label": [0, 1], "score": [-1, 1], "direct_probability": [0.1, 0.8]})
    validate_predictions(frame)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions(pd.concat([frame, frame]))
    extra = frame.iloc[[0]].assign(disease="cad", fold=4)
    with pytest.raises(ValueError, match="different folds"):
        validate_predictions(pd.concat([frame, extra]))
