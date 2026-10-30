import io

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from decipehr.external_pgs_benchmark import PCS, prepare_disease_data
from decipehr.frozen_prs_comparison import (
    evaluate_frozen_fold,
    paired_predictions,
    verify_frozen_fold,
)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(170)
    n = 90
    phenotype = pd.DataFrame({
        "sampleid": [f"synthetic_{i:03d}" for i in range(n)],
        "age_sd": rng.normal(size=n), "sex": np.where(np.arange(n) % 3, "Female", "Male"),
        "T2D": np.arange(n) % 2, "cad": np.arange(n) % 2, "bc": np.arange(n) % 2,
    })
    for pc in PCS:
        phenotype[pc] = rng.normal(size=n)
    splits = pd.concat([phenotype[["sampleid"]].assign(disease=d, fold=np.arange(n) % 5)
                        for d in ["T2D", "cad", "bc"]])
    scores = pd.concat([phenotype[["sampleid"]].assign(disease=d, model="frozen_prs", prs=rng.normal(size=n))
                        for d in ["T2D", "cad", "bc"]])
    return dict(phenotype=phenotype, splits=splits, scores=scores, soft_weights=None,
                hard_labels=phenotype[["sampleid"]].assign(hard_label="EUR-like"),
                model_ids=["frozen_prs"], eligibility_rules={"bc": {"column": "sex", "allowed_values": ["Female"]}})


def fake_factory(seed):
    class Float32Classifier(LogisticRegression):
        def predict_proba(self, x):
            return super().predict_proba(x).astype(np.float32)

    return Float32Classifier(C=1, random_state=seed, max_iter=1000)


def run(data, disease="T2D"):
    return evaluate_frozen_fold(data, disease=disease, arm="internal", fold=2,
                                model_ids=["frozen_prs"], tabpfn_factory=fake_factory)


def test_outer_labels_are_prediction_only_and_serialized_wrapper_replays(cohort):
    data = prepare_disease_data(**cohort, disease="T2D")
    pred, inner, fits, _ = run(data)
    changed = data.copy()
    changed.loc[changed.fold.eq(2), ["T2D", "cad", "bc"]] = 1 - changed.loc[
        changed.fold.eq(2), ["T2D", "cad", "bc"]]
    repeated = run(changed)
    np.testing.assert_array_equal(pred[["y_prob", "wrapper_prob"]], repeated[0][["y_prob", "wrapper_prob"]])
    pd.testing.assert_frame_equal(inner, repeated[1])
    assert len(pred) == 18 and len(inner) == 72 and len(fits) == 4
    assert not set(pred.sampleid).intersection(inner.sampleid)
    saved = [pd.read_csv(io.StringIO(frame.to_csv(index=False)), float_precision="round_trip")
             for frame in [pred, inner, fits]]
    audit = verify_frozen_fold(data, *saved, disease="T2D", arm="internal", fold=2)
    assert audit["wrapper_replay_max_abs_error"] == 0


def test_female_eligibility_and_corrupted_fit_or_probabilities_are_rejected(cohort):
    data = prepare_disease_data(**cohort, disease="bc")
    assert len(data) == 60 and data.sex.eq("Female").all()
    pred, inner, fits, _ = run(data, "bc")
    args = dict(disease="bc", arm="internal", fold=2)
    verify_frozen_fold(data, pred, inner, fits, **args)
    bad = fits.copy()
    bad.loc[0, "context_ids_sha256"] = "wrong_context"
    with pytest.raises(ValueError, match="hash/count"):
        verify_frozen_fold(data, pred, inner, bad, **args)
    bad = pred.copy()
    bad.loc[0, "wrapper_prob"] = 0.999
    with pytest.raises(ValueError, match="replay failed"):
        verify_frozen_fold(data, bad, inner, fits, **args)
    with pytest.raises(ValueError, match="coverage"):
        verify_frozen_fold(data, pred.iloc[1:], inner, fits, **args)


def test_paired_oof_requires_identical_patients_labels_and_folds(cohort):
    data = prepare_disease_data(**cohort, disease="T2D")
    part = data[["sampleid", "fold"]].assign(
        disease="T2D", y_true=data.T2D, ancestry_group=data.hard_label, y_prob=0.5,
    )
    pred = pd.concat([part.assign(arm=arm) for arm in ["internal", "external"]], ignore_index=True)
    paired = paired_predictions(pred, "y_prob")
    assert len(paired) == len(data)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_predictions(pd.concat([pred, pred.iloc[:1]]), "y_prob")
    with pytest.raises(ValueError, match="Paired patients"):
        paired_predictions(pred.iloc[1:], "y_prob")
    changed = pred.copy()
    changed.loc[0, "y_true"] = 1 - changed.loc[0, "y_true"]
    with pytest.raises(ValueError, match="Paired patients"):
        paired_predictions(changed, "y_prob")
    changed = pred.copy()
    changed.loc[0, "fold"] = 4
    with pytest.raises(ValueError, match="different folds"):
        paired_predictions(changed, "y_prob")
    with pytest.raises(ValueError, match="Incomplete five-fold"):
        paired_predictions(pred.loc[pred.fold.ne(0)], "y_prob")
