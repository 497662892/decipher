import numpy as np
import pandas as pd
import pytest

from decipehr.prediction_agreement import agreement, paired_probabilities


def test_pairing_joins_patient_ids_instead_of_row_positions():
    frame = pd.DataFrame({"sampleid": ["b", "a", "a", "b"],
        "arm": ["TabPFN_v3", "M0_KD", "TabPFN_v3", "M0_KD"],
        "probability": [.8, .1, .2, .9]})
    ids, teacher, student = paired_probabilities(frame, "M0_KD")
    assert ids.tolist() == ["a", "b"]
    np.testing.assert_array_equal(teacher, [.2, .8])
    np.testing.assert_array_equal(student, [.1, .9])
    with pytest.raises(ValueError, match="same complete patient set"):
        paired_probabilities(frame.iloc[:-1], "M0_KD")
    with pytest.raises(ValueError, match="unique"):
        paired_probabilities(pd.concat([frame, frame.iloc[:1]]), "M0_KD")


def test_high_correlation_does_not_hide_probability_shift():
    x = np.linspace(.02, .75, 40)
    y = .2 + .7*x
    r = agreement(x, y, np.arange(40), replicates=60)
    assert r["pearson_probability"] == pytest.approx(1)
    assert r["spearman"] == pytest.approx(1)
    assert r["mae_probability"] > .08
    assert r["top_decile_overlap"] == 1
    # Independent resampling would destroy this exact affine relationship.
    assert r["pearson_probability_ci_low"] == pytest.approx(1)
    assert r["spearman_ci_low"] == pytest.approx(1)


def test_identical_and_reversed_predictions():
    x = np.linspace(0, 1, 40)
    same = agreement(x, x, np.arange(40), replicates=0)
    assert same["rmse_probability"] == 0
    assert same["mae_probability"] == 0
    assert same["top_decile_overlap"] == 1
    opposite = agreement(x, 1-x, np.arange(40), replicates=0)
    assert opposite["pearson_probability"] == pytest.approx(-1)
    assert opposite["spearman"] == pytest.approx(-1)
    assert opposite["top_decile_overlap"] == 0
    assert np.isfinite(opposite["pearson_logit"])


def test_constant_or_invalid_probabilities_are_not_reported_as_correlations():
    with pytest.raises(ValueError, match="constant"):
        agreement([.2]*3, [.1, .2, .3], ["a", "b", "c"], replicates=0)
    with pytest.raises(ValueError, match="Invalid"):
        agreement([.1, np.nan, .3], [.1, .2, .3], ["a", "b", "c"], replicates=0)
