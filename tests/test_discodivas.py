import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from decipehr.discodivas import (
    ANCESTRIES,
    COMPARISON_METHODS,
    REFERENCE_PCS,
    TrainScoreHarmonizer,
    comparison_bank,
    official_interpolation,
)
from decipehr.external_pgs_benchmark import (
    DEMOGRAPHICS,
    PCS,
    evaluate_official_validation,
    method_features,
)


@pytest.fixture
def synthetic_bank():
    rng = np.random.default_rng(23)
    bank = comparison_bank("bc", pd.DataFrame())
    frame = pd.DataFrame(rng.normal(size=(96, 10)), columns=REFERENCE_PCS)
    frame["sampleid"] = [f"id{i:04d}" for i in range(len(frame))]
    for i, score in enumerate(bank.source_ids):
        frame[score] = rng.normal(size=len(frame)) + (i + 1) * frame.refPC1
    for group, weight in zip(bank.groups, [0.2, 0.5, 0.3], strict=True):
        frame[f"disco_a_{group}"] = weight
    for name in PCS:
        frame[name] = rng.normal(size=len(frame))
    frame["age_sd"] = rng.normal(size=len(frame))
    frame["sex"] = "Female"
    frame["bc"] = np.arange(len(frame)) % 2
    frame["hard_label"] = "AMBIGUOUS"
    return bank, frame.iloc[:72].copy(), frame.iloc[72:].copy()


def test_harmonization_is_train_fitted_and_independent_of_query_batch(synthetic_bank):
    bank, context, query = synthetic_bank
    transform = TrainScoreHarmonizer(bank, frozenset(context.sampleid))
    train_scores, val_scores, parameters = transform(context, query)
    np.testing.assert_allclose(train_scores[bank.model_ids].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(train_scores[bank.model_ids].std(), 1, atol=1e-12)
    changed = query.copy()
    changed["bc"] = 1 - changed.bc
    changed.loc[changed.index[1:], list(bank.source_ids)] += 1e8
    again, changed_scores, repeated_parameters = transform(context, changed)
    pd.testing.assert_frame_equal(train_scores, again)
    pd.testing.assert_frame_equal(parameters, repeated_parameters)
    np.testing.assert_allclose(val_scores.iloc[0][bank.model_ids].astype(float),
                               changed_scores.iloc[0][bank.model_ids].astype(float))
    _, alone, _ = transform(context, query.iloc[[0]])
    np.testing.assert_allclose(alone.discodivas_prs.iloc[0], val_scores.discodivas_prs.iloc[0])
    with pytest.raises(ValueError, match="outside official train"):
        transform(query, context)


def test_matched_methods_keep_covariates_and_omit_soft_features(synthetic_bank):
    bank, _, _ = synthetic_bank
    for method in COMPARISON_METHODS:
        scores, covariates, interactions = method_features(method, bank.model_ids)
        assert covariates == [*DEMOGRAPHICS, *PCS]
        assert not interactions
        assert scores == (["discodivas_prs"] if method == "discodivas_lr" else bank.model_ids)


def test_dprism_mapping_preserves_tuning_population_and_source_weights():
    rows = [{"pgs_id": f"PGS{5353+i:06d}", "trained_for": ANCESTRIES[i // 5],
             "weight_source": ANCESTRIES[i % 5], "published_coefficient": (i + 1) / 100}
            for i in range(25)]
    mapping = pd.DataFrame(rows)
    bank = comparison_bank("T2D", mapping)
    assert bank.combination.shape == (25, 5)
    assert np.count_nonzero(bank.combination) == 25
    np.testing.assert_allclose(bank.combination[:5, 0], np.arange(1, 6) / 100)
    assert np.count_nonzero(bank.combination[:5, 1:]) == 0
    mapping.loc[0, "trained_for"] = "EUR"
    with pytest.raises(ValueError, match="mapping is inconsistent"):
        comparison_bank("T2D", mapping)


def test_matched_evaluation_refits_transforms_in_inner_folds_and_ignores_val_labels(synthetic_bank):
    bank, context, query = synthetic_bank
    transform = TrainScoreHarmonizer(bank, frozenset(context.sampleid))
    options = dict(disease="bc", model_ids=bank.model_ids, methods=COMPARISON_METHODS,
                   tabpfn_factory=lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
                   c_grid=(0.1, 1.0), score_transform=transform)
    result = evaluate_official_validation(context, query, **options)
    altered = query.assign(bc=1 - query.bc)
    repeated = evaluate_official_validation(context, altered, **options)
    np.testing.assert_allclose(result.predictions.y_prob, repeated.predictions.y_prob)
    np.testing.assert_allclose(result.predictions.wrapper_prob, repeated.predictions.wrapper_prob)
    assert len(result.fits) == 12
    assert set(result.predictions.method) == set(COMPARISON_METHODS)
    sizes = {row["n_context"] for row in transform.fit_audit}
    assert {72, 48, 32}.issubset(sizes)
    assert max(sizes) == 72
    assert result.fits.loc[result.fits.fit_scope == "official_train_context", "n_context"].eq(72).all()


@pytest.mark.skipif(os.environ.get("DECIPEHR_DISCODIVAS_INTEGRATION") != "1",
                    reason="Opt-in test of the actual upstream R script and installed packages")
def test_official_r_id_alignment_and_query_independence(tmp_path):
    root = Path("outputs/external_pgs/discodivas_resource_audit_20261010")
    lock = json.loads((root / "source.lock.json").read_text())
    source = next(x for x in lock["files"] if x["path"].endswith("/DiscoDivas.R"))
    rng = np.random.default_rng(51)
    groups = ("AFR", "EAS", "EUR")
    centers = pd.DataFrame(rng.normal(size=(3, 10)), columns=[f"PC{i}" for i in range(1, 11)])
    centers.insert(0, "POP", groups)
    pcs = pd.DataFrame(rng.normal(size=(12, 10)), columns=REFERENCE_PCS)
    pcs["sampleid"] = ["1", "10", "2", "09", "001", "z", "a", "8", "3", "22", "12", "11"]
    scores = pd.DataFrame(rng.normal(size=(12, 3)), columns=groups).assign(sampleid=pcs.sampleid)
    options = dict(centers=centers, groups=groups, r_script=source["path"],
                   expected_sha256=source["sha256"])
    baseline, weights = official_interpolation(pcs=pcs, scores=scores,
                                               out_dir=tmp_path / "first", **options)
    shuffled, shuffled_weights = official_interpolation(
        pcs=pcs.sample(frac=1, random_state=5), scores=scores.sample(frac=1, random_state=7),
        out_dir=tmp_path / "shuffled", **options,
    )
    pd.testing.assert_frame_equal(baseline, shuffled)
    pd.testing.assert_frame_equal(weights, shuffled_weights)
    alone, alone_weights = official_interpolation(
        pcs=pcs.iloc[[2]], scores=scores.iloc[[2]], out_dir=tmp_path / "alone", **options,
    )
    expected = baseline.set_index("sampleid").loc["2", "discodivas_prs"]
    assert alone.discodivas_prs.iloc[0] == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(alone_weights.iloc[:, 1:],
                               weights.loc[weights.sampleid == "2"].iloc[:, 1:])
