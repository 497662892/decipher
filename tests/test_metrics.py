import numpy as np
import pandas as pd
import pytest

from decipehr.metrics import (
    binary_metrics,
    challenge_metrics,
    nagelkerke_r2,
    risk_stratification_or,
    top_decile_odds_ratio,
)


def test_binary_metrics_smoke():
    result = binary_metrics(pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9]))
    assert result.auc == 1.0
    assert result.auprc == 1.0
    assert result.n == 4


def test_risk_stratification_or_returns_keys():
    y = np.array([0] * 50 + [1] * 50)
    score = np.arange(100)
    out = risk_stratification_or(y, score)
    assert {"top_vs_bottom", "top_vs_middle", "bottom_vs_middle"} <= set(out)


def test_top_decile_odds_ratio_uses_top_10_vs_lower_90():
    y = np.array([1] * 22 + [0] * 68 + [1] * 8 + [0] * 2)
    score = np.arange(100)
    out = top_decile_odds_ratio(y, score)

    assert out["top10_n"] == 10
    assert out["lower90_n"] == 90
    assert out["top10_cases"] == 8
    assert out["lower90_cases"] == 22
    assert out["top10_vs_lower90_or"] == pytest.approx((8 * 68) / (2 * 22))


def test_nagelkerke_r2_is_zero_for_null_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert nagelkerke_r2(y, p) == pytest.approx(0.0)


def test_challenge_metrics_returns_primary_and_secondary_metrics():
    y = np.array([0] * 90 + [1] * 10)
    score = np.arange(100)
    out = challenge_metrics(y, score / 100)
    assert "top10_vs_lower90_or" in out
    assert "nagelkerke_r2" in out
