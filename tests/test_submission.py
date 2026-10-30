import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from decipehr.submission import (
    standardized_logit_submission,
    validate_probability_columns,
    wide_submission,
)


def test_wide_submission():
    pred = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P2"],
            "disease": ["d1", "d2", "d1", "d2"],
            "y_prob": [0.1, 0.2, 0.8, 0.9],
        }
    )
    out = wide_submission(pred, "patient_id")
    validate_probability_columns(out, "patient_id")
    assert list(out.columns) == ["patient_id", "d1", "d2"]
    assert out.shape == (2, 3)


@pytest.fixture
def logit_inputs():
    oof = pd.DataFrame({
        "sampleid": [f"synthetic_{i}" for i in range(10)], "fold": list(range(5)) * 2,
        "y_prob": [0, 0.0001, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9999, 1],
        "wrapper_prob": [0.5] * 10,
    })
    holdout = pd.DataFrame({"sampleid": ["query_b", "query_a"], "y_prob": [0.25, 0.9]})
    return dict(oof=oof, holdout=holdout, train_ids=oof.sampleid.copy(),
                holdout_ids=holdout.sampleid.iloc[::-1].copy(),
                folds=oof[["sampleid", "fold"]].copy())


def test_logit_submission_uses_direct_oof_only_and_preserves_ids(logit_inputs):
    result, params = standardized_logit_submission(**logit_inputs)
    expected = np.clip(logit_inputs["oof"].y_prob, 0.001, 0.999)
    expected_logit = np.log(expected) - np.log1p(-expected)
    assert params["logit_trainval_oof_mean"] == pytest.approx(expected_logit.mean())
    assert params["logit_trainval_oof_sd"] == pytest.approx(expected_logit.std(ddof=0))
    assert result.sampleid.tolist() == [*logit_inputs["train_ids"], *logit_inputs["holdout_ids"]]
    assert result.iloc[:10].score.mean() == pytest.approx(0, abs=1e-14)
    assert result.iloc[:10].score.std(ddof=0) == pytest.approx(1)
    np.testing.assert_allclose(
        expit(result.score * params["logit_trainval_oof_sd"] + params["logit_trainval_oof_mean"]),
        np.clip(result.direct_probability, 0.001, 0.999), atol=1e-14,
    )
    logit_inputs["holdout"]["y_prob"] = [0, 1]
    changed, changed_params = standardized_logit_submission(**logit_inputs)
    assert changed_params == params
    pd.testing.assert_frame_equal(changed.iloc[:10], result.iloc[:10])


@pytest.mark.parametrize("failure", ["missing", "duplicate", "overlap", "wrong_fold", "only_four_folds", "holdout_label"])
def test_logit_submission_rejects_cohort_or_fold_errors(logit_inputs, failure):
    if failure == "missing":
        logit_inputs["oof"] = logit_inputs["oof"].iloc[:-1]
    elif failure == "duplicate":
        logit_inputs["holdout"] = pd.concat([logit_inputs["holdout"], logit_inputs["holdout"].iloc[:1]])
    elif failure == "overlap":
        logit_inputs["holdout_ids"].iloc[0] = logit_inputs["train_ids"].iloc[0]
    elif failure == "wrong_fold":
        logit_inputs["oof"].loc[0, "fold"] = 1
    elif failure == "only_four_folds":
        for key in ["oof", "folds"]:
            logit_inputs[key]["fold"] = logit_inputs[key].fold.replace({4: 3})
    else:
        logit_inputs["holdout"]["y_true"] = 0
    with pytest.raises(ValueError):
        standardized_logit_submission(**logit_inputs)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01, 1.01])
def test_logit_submission_rejects_invalid_probabilities(logit_inputs, value):
    logit_inputs["oof"].loc[0, "y_prob"] = value
    with pytest.raises(ValueError, match="finite probabilities"):
        standardized_logit_submission(**logit_inputs)


def test_logit_submission_rejects_constant_oof(logit_inputs):
    logit_inputs["oof"]["y_prob"] = 0.5
    with pytest.raises(ValueError, match="zero variance"):
        standardized_logit_submission(**logit_inputs)

