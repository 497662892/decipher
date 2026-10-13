import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from decipehr.discodivas import ANCESTRIES, REFERENCE_PCS, comparison_bank
from decipehr.dprism_comparison import (
    DIRECT,
    METHODS,
    SOURCE_IDS,
    component_design_pair,
    evaluate_source_partition,
)
from decipehr.internal_prs_benchmark import DISEASES, design_pair, specifications


@pytest.fixture
def dprism_data():
    rng = np.random.default_rng(523)
    mapping = pd.DataFrame([{"pgs_id": name, "trained_for": ANCESTRIES[i // 5],
                             "weight_source": ANCESTRIES[i % 5], "published_coefficient": (i + 1) / 50}
                            for i, name in enumerate(SOURCE_IDS)])
    bank = comparison_bank("T2D", mapping)
    frame = pd.DataFrame(rng.normal(size=(100, 10)), columns=REFERENCE_PCS)
    frame["sampleid"] = [f"p{i:03}" for i in range(len(frame))]
    frame["kinship_group"] = frame.sampleid
    frame["hard_label"], frame["sex"] = "AMBIGUOUS", "Female"
    frame["age_sd"] = rng.normal(size=len(frame))
    for i, name in enumerate(SOURCE_IDS):
        frame[name] = rng.normal(size=len(frame)) + (i + 1) * .1 * frame.refPC1
    weights = rng.dirichlet(np.ones(5), size=len(frame))
    for i, name in enumerate(ANCESTRIES):
        frame[f"disco_a_{name}"] = weights[:, i]
    for name in DISEASES:
        frame[name] = np.arange(len(frame)) % 2
    return bank, frame.iloc[:75].copy(), frame.iloc[75:].copy()


def test_all_components_reconstruct_existing_metascores_and_disco(dprism_data):
    bank, train, query = dprism_data
    x, q, names, audit = component_design_pair(train, query.drop(columns=list(DISEASES)), bank)
    assert [n for n in names if n.startswith("prs__")] == [f"prs__{s}" for s in SOURCE_IDS]
    assert x.shape[1] == q.shape[1] == 36  # Female-only fixture: 25 PRS + age + 10 PCs.
    assert "PGS002308" not in " ".join(names)
    for check in audit["equivalence_checks"]:
        assert check["metascore_max_abs_error"] < 1e-12
        assert check["disco_expansion_max_abs_error"] < 1e-12
    # The ancestry correction leaves all component residuals orthogonal to training PCs.
    np.testing.assert_allclose(train[REFERENCE_PCS].to_numpy().T @ x[:, :25], 0, atol=1e-11)
    spec = specifications("tabpfn_matched", list(SOURCE_IDS), bank)[0]
    mx, mq, mn, _ = design_pair(train, query.drop(columns=list(DISEASES)), spec, bank)
    np.testing.assert_allclose(x[:, 25:], mx[:, 5:], atol=1e-12)
    np.testing.assert_allclose(q[:, 25:], mq[:, 5:], atol=1e-12)
    assert names[25:] == mn[5:]


def test_query_batch_cannot_change_training_transform_or_other_predictions(dprism_data):
    bank, train, labelled = dprism_data
    query = labelled.drop(columns=list(DISEASES))
    x, q, _, audit = component_design_pair(train, query, bank)
    altered = query.copy()
    altered.loc[altered.index[1:], list(SOURCE_IDS)] += 1000
    xx, qq, _, changed = component_design_pair(train, altered, bank)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_allclose(q[0], qq[0], atol=1e-12)
    assert audit["pc_coefficients"] == changed["pc_coefficients"]
    with pytest.raises(ValueError, match="Query outcomes"):
        component_design_pair(train, labelled, bank)
    with pytest.raises(ValueError, match="share patients"):
        component_design_pair(train, train.drop(columns=list(DISEASES)), bank)


def test_query_labels_and_unrelated_26th_score_do_not_affect_any_model(dprism_data):
    bank, train, query = dprism_data
    options = {"bank": bank, "tabpfn_factory": lambda seed: LogisticRegression(max_iter=1000, random_state=seed)}
    first = evaluate_source_partition(train, query, **options)
    second = evaluate_source_partition(train.assign(PGS002308=123.0),
                                       query.assign(T2D=1 - query.T2D, PGS002308=-999.0), **options)
    assert set(first[0].method) == set(METHODS)
    np.testing.assert_array_equal(first[0].probability, second[0].probability)
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[2].loc[first[2].method == DIRECT, "stage"].tolist() == ["final"]
