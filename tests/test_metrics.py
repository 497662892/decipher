import numpy as np
import pandas as pd

from decipehr.metrics import binary_metrics, risk_stratification_or


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

