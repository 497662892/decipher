"""Sign-aware invariants, exact nesting, scope isolation and optimization checks."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from decipehr.prs_distillation.config import Optimization
from decipehr.prs_distillation.preprocessing import AGE, PCS
from decipehr.prs_distillation_v3 import (
    SignAwarePreprocessor,
    SignAwareStudent,
    as_tensors,
    fit_student,
    from_m0,
    load_bundle,
    numpy_forward,
    predict,
    save_bundle,
)

torch.set_num_threads(1)


def synthetic(n=60):
    rng = np.random.default_rng(109)
    return pd.DataFrame({"sampleid": [f"v3-synthetic-{i}" for i in range(n)],
                         "PRS_A": rng.normal(size=n), "PRS_B": rng.normal(size=n),
                         AGE: np.linspace(-2, 2, n), **{c: rng.normal(size=n) for c in PCS},
                         "sex": ["Female", "Male"]*(n//2), "y": [0, 1]*(n//2)})


@pytest.fixture
def fixture():
    f = synthetic()
    prep = SignAwarePreprocessor.fit(f, ["PRS_A", "PRS_B"])
    x = prep.transform(f)
    model = SignAwareStudent(2, 10, x["nonage"].shape[1], prep.n_segments, variant="M2")
    with torch.no_grad():
        model.log_r.copy_(torch.tensor([-.7, .5]))
        model.d.copy_(torch.tensor([.2, .7]))
        model.age_slopes.copy_(torch.tensor([.1, .7, .2]))
    return f, prep, x, model


def test_constraints_and_independent_decomposition(fixture):
    _, _, x, model = fixture
    p, independent = predict(model, x), numpy_forward(model, x)
    for key in independent:
        np.testing.assert_allclose(p[key], independent[key], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(p["pi"].sum(axis=1), 1)
    assert (p["pi"] >= 0).all() and (p["r"] > 0).all() and (p["d"] >= 0).all()
    np.testing.assert_allclose(p["reference_age_logit"]+p["age_delta"], p["logit"], atol=1e-12)
    np.testing.assert_allclose(p["base_reference_logit"]+p["baseline_age"]+p["contributions"].sum(1), p["logit"], atol=1e-12)
    assert not np.allclose(p["effective_coefficients"].sum(1), 1)
    assert (p["age_delta"][x["t"] > 0] >= -1e-12).all()
    assert (p["age_delta"][x["t"] < 0] <= 1e-12).all()


@pytest.mark.parametrize("z", [-1., 0., 1.])
def test_age_monotonicity_for_every_sign_including_bounds(fixture, z):
    f, prep, _, model = fixture
    age = np.r_[-20., np.linspace(-2, 2, 101), 20.]
    fixed = pd.concat([f.iloc[[0]]]*len(age), ignore_index=True)
    fixed[AGE] = age
    x = prep.transform(fixed)
    x["z"][:] = z
    out = predict(model, x)
    for key in ["contributions", "transformed_prs", "logit", "probability"]:
        assert np.diff(out[key], axis=0).min() >= -1e-12
    np.testing.assert_array_equal(out["probability"][[0, -1]], out["probability"][[1, -2]])
    if z == 0:
        np.testing.assert_array_equal(out["contributions"], 0)


def test_reference_age_zero_d_and_monotone_prs(fixture):
    f, prep, x, model = fixture
    ref = f.copy()
    ref[AGE] = prep.reference_age
    out = predict(model, prep.transform(ref))
    np.testing.assert_array_equal(out["transformed_prs"], x["z"])
    np.testing.assert_array_equal(out["baseline_age"], 0)
    np.testing.assert_array_equal(out["age_delta"], 0)
    with torch.no_grad():
        model.d.zero_()
    np.testing.assert_array_equal(predict(model, x)["transformed_prs"], x["z"])
    with torch.no_grad():
        model.d.fill_(.6)
    for k in range(2):
        changed = copy.deepcopy(x)
        changed["z"][:, k] += 2
        assert (predict(model, changed)["logit"] >= predict(model, x)["logit"]).all()


def test_no_hidden_age_path_and_fit_scope_is_frozen(fixture):
    f, prep, x, model = fixture
    changed = f.copy()
    changed[AGE] += 100
    before = prep.metadata()
    altered = prep.transform(changed)
    for key in ["z", "a", "nonage"]:
        np.testing.assert_array_equal(x[key], altered[key])
    np.testing.assert_array_equal(predict(model, x)["pi"], predict(model, altered)["pi"])
    assert altered["age_out_of_support"].all()
    assert before == prep.metadata() and prep.age_unit == 1
    changed.y = 1-changed.y
    np.testing.assert_array_equal(altered["nonage"], prep.transform(changed)["nonage"])


@pytest.mark.parametrize("ages,segments", [([2.]*60, 0), ([0., 1.]*30, 1), ([0., 0., 0., 2., 2., 2.]*10, 1)])
def test_tied_constant_ages_and_single_prs(ages, segments):
    f = synthetic()
    f[AGE] = ages
    f.sex = "Female"
    prep = SignAwarePreprocessor.fit(f, ["PRS_A"])
    x = prep.transform(f)
    assert prep.n_segments == segments
    model = SignAwareStudent(1, 10, x["nonage"].shape[1], segments, variant="M2")
    assert not hasattr(model, "w") and not hasattr(model, "c")
    np.testing.assert_array_equal(predict(model, x)["pi"], 1)
    if not segments:
        assert "d" not in dict(model.named_parameters())
        assert "age_slopes" not in dict(model.named_parameters())
        np.testing.assert_array_equal(x["t"], 0)
    bare = dict(x, a=np.zeros((len(f), 0)), nonage=np.ones((len(f), 1)))
    assert np.isfinite(predict(SignAwareStudent(1, 0, 1, segments), bare)["logit"]).all()


def test_warm_start_gradients_projection_and_serialization(fixture, tmp_path):
    _, prep, x, _ = fixture
    m0 = SignAwareStudent(2, 10, x["nonage"].shape[1], 3)
    with torch.no_grad():
        m0.log_r.copy_(torch.tensor([.2, -.3]))
        m0.age_slopes.fill_(.2)
    for variant in ["M1", "M2"]:
        model = from_m0(m0, variant)
        np.testing.assert_allclose(predict(model, x)["logit"], predict(m0, x)["logit"], atol=1e-12, rtol=0)
        target = as_tensors(x)
        target["t"] = torch.ones_like(target["t"])
        model(target)["logit"].sum().backward()
        assert model.d.grad.abs().sum() > 0
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        with torch.no_grad():
            model.d.fill_(-.4)
            model.age_slopes.fill_(-.8)
        model.project()
        assert (model.d == 0).all() and (model.age_slopes == 0).all()
        path = tmp_path / variant
        save_bundle(path, model, prep, {"synthetic": True})
        loaded, prep2, meta = load_bundle(path)
        for key, val in predict(model, x).items():
            np.testing.assert_array_equal(predict(loaded, prep2.transform(synthetic()))[key], val)
        assert meta["prediction_scale"] == "direct_probability" and not meta["phase_b"]


def test_small_fit_uses_soft_targets_and_exact_warm_start(fixture):
    f, _, x, _ = fixture
    options = Optimization(max_epochs=20, evaluate_every=5, patience=10)
    m0, info = fit_student(x, f.y, np.full(len(f), .4), variant="M0", lam=.75, rho=.001,
                           kappa=.001, seed=17, options=options, epochs=20)
    assert info["best_step"] == 20
    m2, info2 = fit_student(x, f.y, np.full(len(f), .4), variant="M2", lam=.75, rho=.001,
                            kappa=.001, seed=17, options=options, tune=x, tune_y=f.y, warm=m0)
    assert info2["warm_start_exact"] and np.isfinite(info2["selection_metric"])
    assert (m2.d >= 0).all() and (m2.age_slopes >= 0).all()
    with pytest.raises(ValueError, match="either"):
        fit_student(x, f.y, None, variant="M0", lam=0, rho=.001, kappa=.001, seed=17,
                    options=options, tune=x, tune_y=f.y, epochs=20)
