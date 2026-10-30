"""Numerical, isolation and optimization tests for interpretable extensions."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from decipehr.prs_distillation_bio import (
    BioPreprocessor,
    BioStudent,
    fit,
    initialize,
    load,
    numpy_predict,
    predict,
    save,
    tensors,
)

torch.set_num_threads(1)


@pytest.fixture
def fixture():
    rng = np.random.default_rng(18)
    n = 80
    f = pd.DataFrame({"sampleid": [f"synthetic-bio-{i}" for i in range(n)],
        "PRS_A": rng.normal(size=n), "PRS_B": rng.normal(size=n), "age_sd": np.linspace(-2, 2, n),
        **{f"PC{i}_sd": rng.normal(size=n) for i in range(1, 11)}, "sex": ["Female", "Male"]*(n//2),
        "y": rng.integers(0, 2, n)})
    prep = BioPreprocessor.fit(f, ["PRS_A", "PRS_B"], 6)
    x = prep.transform(f)
    m = BioStudent(2, 10, x["nonage"].shape[1], prep.age.n_segments, shape=True, variant="M2")
    with torch.no_grad():
        m.log_slopes.copy_(torch.tensor(rng.normal(0, .6, (2, 5))))
        m.log_r.copy_(torch.tensor([-.3, .5]))
        m.d.copy_(torch.tensor([.2, .6]))
        m.age_slopes.copy_(torch.tensor([.2, .1, .5, .2, .7, .4]))
    return f, prep, x, m


def test_independent_interval_reconstruction_and_saved_replay(fixture, tmp_path):
    f, prep, x, m = fixture
    pred = predict(m, x)
    for k, v in numpy_predict(m, x).items():
        np.testing.assert_allclose(pred[k], v, atol=1e-12)
    np.testing.assert_allclose(pred["reference_age_logit"]+pred["age_delta"], pred["logit"], atol=1e-12)
    np.testing.assert_allclose(torch.log(m.shape_slopes()).detach().numpy().mean(1), 0, atol=1e-12)
    save(tmp_path / "model", m, prep, {"test": True})
    restored, p2, _ = load(tmp_path / "model")
    np.testing.assert_array_equal(predict(restored, p2.transform(f))["probability"], pred["probability"])


def test_age_and_prs_monotonicity_across_knots_and_linear_tails(fixture):
    f, prep, _, m = fixture
    for i in range(6):
        age = np.unique(np.r_[np.linspace(-10, 10, 101), prep.age.age_boundaries, prep.age.reference_age])
        frame = pd.concat([f.iloc[[i]]]*len(age), ignore_index=True)
        frame.age_sd = age
        prediction = predict(m, prep.transform(frame))
        assert np.diff(prediction["logit"]).min() >= -1e-12
        for prs in ["PRS_A", "PRS_B"]:
            frame = pd.concat([f.iloc[[i]]]*101, ignore_index=True)
            frame[prs] = np.linspace(-15, 15, 101)
            assert np.diff(predict(m, prep.transform(frame))["logit"]).min() >= -1e-12


def test_fit_scope_and_reference_zero(fixture):
    f, prep, x, m = fixture
    original = copy.deepcopy(prep.metadata())
    altered = f.copy()
    altered.age_sd = 200
    altered.y = 1-altered.y
    changed = prep.transform(altered)
    assert prep.metadata() == original
    for key in ["z", "a", "nonage", "hinges"]:
        np.testing.assert_array_equal(x[key], changed[key])
    ref = copy.deepcopy(x)
    ref["z"][:] = 0
    ref["hinges"][:] = 0
    pred = predict(m, ref)
    np.testing.assert_array_equal(pred["shaped_prs"], 0)
    np.testing.assert_array_equal(pred["contributions"], 0)


def test_exact_parent_and_zero_step_refit(fixture):
    f, _, x, _ = fixture
    parent = initialize(tensors(x), torch.tensor(np.full(len(f), .4)), variant="M0", shape=True, seed=17)
    with torch.no_grad():
        parent.log_slopes[0, 0] = .7
        parent.age_slopes.fill_(.2)
    for variant in ["M0", "M1", "M2"]:
        model, info = fit(x, f.y, None, variant=variant, shape=True, lam=0, rho=.001,
            kappa_r=.001, kappa_d=.001, kappa_shape=.001, seed=17, warm=parent, fixed_steps=0)
        assert info["best_step"] == 0 and len(info["logs"]) == 1 and info["step_zero_eligible"]
        np.testing.assert_allclose(predict(model, x)["logit"], predict(parent, x)["logit"], atol=1e-12)


def test_warm_start_can_reject_all_updates_and_d_only_freezes_parent(fixture):
    f, _, x, _ = fixture
    parent = initialize(tensors(x), torch.tensor(np.full(len(f), .5)), variant="M0", shape=True, seed=17)
    with torch.no_grad():
        parent.log_r.fill_(-8)
        parent.base.zero_()
    target = predict(parent, x)["probability"]
    # The warm parent is exactly optimal for these soft H labels; any drift is worse.
    model, info = fit(x, np.ones(len(f)), None, variant="M2", shape=True, lam=0, rho=0,
        kappa_r=0, kappa_d=0, kappa_shape=0, baseline_ridge=0, seed=17, warm=parent,
        tune=x, tune_y=target, max_steps=20, patience=10)
    assert info["best_step"] == 0
    np.testing.assert_allclose(predict(model, x)["probability"], target, atol=1e-12)
    model, _ = fit(x, f.y, None, variant="M2", shape=True, lam=0, rho=.001,
        kappa_r=.001, kappa_d=.001, kappa_shape=.001, seed=17, warm=parent, d_only=True, fixed_steps=20)
    for k, v in parent.state_dict().items():
        if k != "d":
            torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    assert (model.d >= 0).all()


def test_constant_age_and_tied_prs_knots(fixture):
    f, _, _, _ = fixture
    f.age_sd = .5
    f.PRS_A = np.repeat([-1, 1], 40)
    prep = BioPreprocessor.fit(f, ["PRS_A"], 6)
    x = prep.transform(f)
    model = initialize(tensors(x), torch.tensor(np.full(len(f), .5)), variant="M2", shape=True, seed=17)
    assert prep.age.n_segments == 0 and "d" not in dict(model.named_parameters())
    pred = predict(model, x)
    np.testing.assert_array_equal(pred["age_delta"], 0)
    np.testing.assert_allclose(pred["shaped_prs"], x["z"], atol=1e-12)
