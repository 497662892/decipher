import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import log_loss

from decipehr.prs_monotone_boosting import (
    Features,
    bundle_model,
    explain,
    fit_booster,
    predict,
    select_iteration,
    soft_binary_rows,
    verify,
)


def sample():
    r = np.random.default_rng(9)
    n = 200
    f = pd.DataFrame({"sampleid": [f"synthetic-boost-{i}" for i in range(n)],
        "PRS_A": r.normal(size=n), "PRS_B": r.normal(size=n), "age_sd": r.normal(size=n),
        **{f"PC{i}_sd": r.normal(size=n) for i in range(1, 11)}, "sex": ["Female", "Male"]*(n//2)})
    f["y"] = r.binomial(1, expit(.5*f.PRS_A + .4*f.age_sd))
    return f


def test_weighted_rows_equal_soft_bce_and_preserve_patient_weight():
    r = np.random.default_rng(10)
    q, p = r.uniform(.1, .9, (2, 20))
    x = r.normal(size=(20, 2))
    xx, yy, w = soft_binary_rows(x, q)
    np.testing.assert_array_equal(xx[:20], xx[20:])
    np.testing.assert_allclose(w[:20]+w[20:], 1)
    np.testing.assert_allclose(log_loss(yy, np.r_[p, p], sample_weight=w), -(q*np.log(p)+(1-q)*np.log1p(-p)).mean())


@pytest.mark.parametrize("family", ["GAM", "PAIR"])
def test_exact_explanation_and_monotonicity(family):
    f = sample()
    features = Features.fit(f, ["PRS_A", "PRS_B"])
    x = features.transform(f)
    model = fit_booster(x, .2+.6*f.y, features, family=family, leaves=8, l2=1, iterations=20)
    b = bundle_model(model, features, family, 20, .5, {})
    verification = verify(b, x)
    assert verification["exact_anchored_decomposition"]
    if family == "GAM":
        assert not b["active_pairs"]
    intercept, first, second = explain(b, x[:20])
    np.testing.assert_allclose(expit(intercept+first.sum(1)+second.sum(1)), predict(b, x[:20]), atol=1e-12)
    selected, curve = select_iteration(model, x, f.y, .5)
    assert selected in curve and 0 <= selected["iterations"] <= 20


def test_zero_iteration_and_frozen_feature_transforms():
    f = sample()
    features = Features.fit(f, ["PRS_A", "PRS_B"])
    x = features.transform(f)
    model = fit_booster(x, f.y, features, family="GAM", leaves=4, l2=1, iterations=1)
    bundle = bundle_model(model, features, "GAM", 0, .4, {})
    np.testing.assert_allclose(predict(bundle, x), .4)
    baseline, first, pair = explain(bundle, x)
    assert np.isfinite(baseline) and pair.shape[1] == 0
    np.testing.assert_allclose(first, 0, atol=1e-12)
    changed = f.copy()
    changed.y = 1-changed.y
    np.testing.assert_array_equal(features.transform(changed), x)
