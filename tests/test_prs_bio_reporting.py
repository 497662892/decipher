"""The fast family bootstrap must reproduce literal cluster resampling."""

import importlib
from pathlib import Path

import numpy as np

from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.reporting import basic_metrics


def test_paired_family_intervals_match_literal_resampling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    reporting = importlib.import_module("report_prs_distillation_bio")
    rng = np.random.default_rng(23)
    y = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0])
    groups = np.repeat(np.arange(8), 2)
    p = {"H_SELECTED": rng.uniform(.05, .95, len(y)), "M0_KD": rng.uniform(.1, .9, len(y))}
    result = reporting.paired_bootstrap(y, p, groups, "nested", replicates=50).set_index("metric")
    rng = np.random.default_rng(20261028)
    rows = []
    for _ in range(50):
        chosen = rng.integers(8, size=8)
        indices = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
        if len(np.unique(y[indices])) < 2:
            continue
        scores = {a: {**basic_metrics(y[indices], v[indices]),
                      "nagelkerke_r2": nagelkerke_r2(y[indices], v[indices])} for a, v in p.items()}
        rows.append([scores["H_SELECTED"][m]-scores["M0_KD"][m] for m in reporting.METRICS])
    expected = np.quantile(rows, [.025, .975], axis=0).T
    np.testing.assert_allclose(result.loc[reporting.METRICS, ["ci_low", "ci_high"]], expected, atol=1e-12)
    assert not result.refitted.any() and result.replicates.eq(len(rows)).all()
