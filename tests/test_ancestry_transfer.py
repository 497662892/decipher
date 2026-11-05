import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decipehr.ancestry_transfer import (
    fit_logistic,
    replace_pcs,
    select_single_prs,
    validate_folds,
)
from decipehr.prs_distillation.preprocessing import PCS


def fixture():
    rng = np.random.default_rng(27)
    n = 300
    y = np.tile([0, 1], n // 2)
    frame = pd.DataFrame({"sampleid": [f"p{i}" for i in range(n)],
                          "kinship_group": [f"g{i}" for i in range(n)],
                          "y": y, "age_sd": rng.normal(size=n),
                          "sex": np.where(np.arange(n) % 3 == 0, "Female", "Male"),
                          "signal": y+rng.normal(size=n), "noise": rng.normal(size=n)})
    frame[PCS] = rng.normal(size=(n, 10))
    return frame, np.repeat(np.arange(5), n // 5)


def test_pc_replacement_is_id_aligned_and_preserves_other_columns():
    frame, _ = fixture()
    projection = frame[["sampleid"]].copy()
    projection[[f"refPC{i}" for i in range(1, 11)]] = frame[PCS].to_numpy()*2
    updated = replace_pcs(frame, projection.iloc[::-1])
    np.testing.assert_array_equal(updated[PCS], frame[PCS]*2)
    pd.testing.assert_frame_equal(updated.drop(columns=PCS), frame.drop(columns=PCS))
    with pytest.raises(ValueError, match="cover"):
        replace_pcs(frame, projection.iloc[1:])
    with pytest.raises(ValueError, match="unique"):
        replace_pcs(frame, pd.concat([projection, projection.iloc[:1]]))
    projection.loc[0, "refPC1"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace_pcs(frame, projection)


def test_logistic_query_cannot_affect_preprocessing_or_coefficients():
    frame, folds = fixture()
    train, query = frame.loc[folds != 0], frame.loc[folds == 0].drop(columns="y")
    _, first, record = fit_logistic(train, query, ["signal"])
    changed = query.copy()
    changed["signal"] += 10000
    _, second, _ = fit_logistic(train, changed, ["signal"])
    assert first["preprocessing"] == second["preprocessing"]
    np.testing.assert_array_equal(first["model"].coef_, second["model"].coef_)
    assert record["gradient_max_abs"] < 1e-5
    with pytest.raises(ValueError, match="features only"):
        fit_logistic(train, query.assign(y=1), ["signal"])
    with pytest.raises(ValueError, match="overlap"):
        fit_logistic(train, train.drop(columns="y"), ["signal"])


def test_selection_uses_pooled_context_oof_and_rejects_group_leakage():
    frame, folds = fixture()
    chosen, ranking, predictions, fits = select_single_prs(frame, ["noise", "signal"], folds)
    assert chosen == "signal"
    assert ranking.iloc[0].log_loss < ranking.iloc[1].log_loss
    assert predictions.shape == (len(frame), 2)
    assert np.isfinite(predictions.to_numpy()).all()
    assert len(fits) == 10 and all(not r["query_labels_read"] for r in fits)
    frame.loc[60, "kinship_group"] = frame.loc[0, "kinship_group"]
    with pytest.raises(ValueError, match="crosses"):
        validate_folds(frame, folds)


def test_paired_bootstrap_has_correct_direction_and_zero_identity_interval(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    reporter = importlib.import_module("report_ancestry_transfer")
    y = np.tile([0, 1], 30)
    groups = np.repeat(np.arange(30), 2)
    good, weak = np.where(y == 1, .8, .2), np.full(len(y), .5)
    arrays = {f"{mode}/{arm}": good.copy() for mode in ["official_pc", "reference_pc"] for arm in reporter.ARMS}
    for mode in ["official_pc", "reference_pc"]:
        arrays[f"{mode}/LR_SINGLE_BEST"] = weak
    rows = reporter.paired_differences(y, arrays, groups, replicates=20)
    identical = rows.loc[rows.arm.eq("reference_pc/M0_KD") & rows.reference.eq("official_pc/M0_KD")]
    np.testing.assert_array_equal(identical[["delta", "ci_low", "ci_high"]], 0)
    loss = rows.loc[rows.arm.eq("reference_pc/M0_KD") & rows.reference.eq("reference_pc/LR_SINGLE_BEST") & rows.metric.eq("log_loss")].iloc[0]
    assert loss.delta < 0 and loss.ci_high < 0
    assert rows.replicates.eq(20).all()


def test_metric_replay_resolves_concrete_policy_not_legacy_alias(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    reporter = importlib.import_module("report_ancestry_transfer")
    expected = dict(log_loss=.5, auroc=.8, brier=.17, nagelkerke_r2=.34)
    measured = pd.DataFrame([dict(trait="cad", cohort="original_val", mode="official_pc", arm="H_SELECTED", **expected)])
    original = pd.DataFrame([
        dict(trait="cad", arm="HGB_PAIR_KD", **expected),
        dict(trait="cad", arm="H_SELECTED", **{k: v+.001 for k, v in expected.items()}),
    ]).set_index(["trait", "arm"])
    policy = {"choices": {"cad": {"-1": "HGB_PAIR_KD"}}}
    reporter.verify_original_metrics(measured, original, policy)
    measured.loc[0, "auroc"] += .001
    with pytest.raises(AssertionError):
        reporter.verify_original_metrics(measured, original, policy)
