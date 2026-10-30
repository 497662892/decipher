"""Check unpenalized logistic predictions against an independent GLM fit."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import expit


def test_plain_logistic_matches_independent_glm_and_rejects_query_labels(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    runner = importlib.import_module("compare_prs_plain_logistic")
    rng = np.random.default_rng(73)
    n = 850
    columns = ["prs1", "prs2", "age_sd", *[f"PC{i}_sd" for i in range(1, 11)]]
    frame = pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns)
    frame["sex"] = np.where(rng.random(n) < .5, "Female", "Male")
    frame["sampleid"] = [f"s{i}" for i in range(n)]
    frame["kinship_group"] = frame.sampleid
    frame["y"] = rng.binomial(1, expit(-.6 + .7 * frame.prs1 + .3 * frame.age_sd))
    train, query = frame.iloc[:700].copy(), frame.iloc[700:].drop(columns="y").copy()
    p, info = runner.fit_plain(train, query, ["prs1", "prs2"], tmp_path / "fit")
    raw = pd.concat([frame[columns], frame.sex.eq("Male").astype(float).rename("male")], axis=1)
    independent = sm.GLM(train.y.to_numpy(), sm.add_constant(raw.iloc[:700]), family=sm.families.Binomial()).fit()
    expected = independent.predict(sm.add_constant(raw.iloc[700:]))
    np.testing.assert_allclose(p, expected, atol=2e-6, rtol=0)
    assert info["design_rank"] == info["design_columns"]
    assert info["fit_ids_hash"] == runner.id_hash(train)
    with pytest.raises(ValueError, match="Query labels"):
        runner.fit_plain(train, query.assign(y=0), ["prs1", "prs2"], tmp_path / "invalid")
