from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decipehr.discodivas import comparison_bank
from decipehr.internal_prs_benchmark import (
    Specification,
    assert_boundary,
    design_pair,
    evaluate_partition,
    kinship_groups,
    shared_folds,
)


def cohort(n=120):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"sampleid": [f"s{i}" for i in range(n)],
                          "kinship_group": [str(i // 2) for i in range(n)],
                          "sex": ["Female"] * n, "hard_label": ["EUR-like"] * n,
                          "bc": np.tile([0, 1, 1, 0], n // 4),
                          "cad": np.tile([0, 1], n // 2), "T2D": np.tile([1, 0], n // 2)})
    for column in ["age_sd", *[f"PC{i}_sd" for i in range(1, 11)],
                   "PGS005378", "PGS004869", "PGS018538"]:
        frame[column] = rng.normal(size=n)
    return frame


def test_kinship_transitive_group_and_shared_split():
    pairs = pd.DataFrame({"IID1": ["s0", "s1"], "IID2": ["s1", "s2"], "KINSHIP": [.2, .12]})
    groups = kinship_groups([f"s{i}" for i in range(120)], pairs, .0884)
    assert groups.set_index("sampleid").loc[["s0", "s1", "s2"], "kinship_group"].nunique() == 1
    frame = cohort().drop(columns="kinship_group").merge(groups, on="sampleid")
    folds = shared_folds(frame)
    assert folds.groupby("kinship_group").fold.nunique().max() == 1
    assert set(folds.fold) == set(range(5))
    assert folds.sampleid.nunique() == len(frame)


def test_query_label_and_relative_overlap_are_rejected():
    frame = cohort()
    bank = comparison_bank("bc", pd.DataFrame())
    spec = Specification(prs=tuple(bank.source_ids))
    with pytest.raises(ValueError, match="Query outcomes"):
        design_pair(frame.iloc[:80], frame.iloc[80:], spec, bank)
    query = frame.iloc[80:].drop(columns=["T2D", "cad", "bc"]).copy()
    query.loc[query.index[0], "kinship_group"] = frame.kinship_group.iloc[0]
    with pytest.raises(ValueError, match="kinship groups"):
        assert_boundary(frame.iloc[:80], query)


def test_query_extremes_do_not_change_fitted_design_or_parameters():
    frame = cohort()
    bank = comparison_bank("bc", pd.DataFrame())
    spec = Specification(prs=tuple(bank.source_ids), basis="varying")
    context = frame.iloc[:80]
    query = frame.iloc[80:].drop(columns=["T2D", "cad", "bc"])
    x1, q1, names1, audit1 = design_pair(context, query, spec, bank)
    changed = query.copy()
    changed["PGS005378"] += 10000
    changed["PC1_sd"] -= 1000
    x2, q2, names2, audit2 = design_pair(context, changed, spec, bank)
    np.testing.assert_array_equal(x1, x2)
    assert names1 == names2
    assert audit1 == audit2
    assert not np.array_equal(q1, q2)
    assert not any("*cov__age" in name or "*cov__sex" in name for name in names1)


def test_outer_query_outcomes_do_not_change_selection_or_predictions():
    frame = cohort()
    bank = comparison_bank("bc", pd.DataFrame())
    kwargs = dict(disease="bc", full_ids=list(bank.source_ids), bank=bank,
                  methods=("covariates_lr", "full_lr"), seed=42)
    first, selection1, _, _ = evaluate_partition(frame.iloc[:80], frame.iloc[80:], **kwargs)
    changed = frame.iloc[80:].copy()
    changed["bc"] = 1 - changed.bc
    second, selection2, _, _ = evaluate_partition(frame.iloc[:80], changed, **kwargs)
    np.testing.assert_array_equal(first.probability, second.probability)
    pd.testing.assert_frame_equal(selection1, selection2)
    assert not np.array_equal(first.label, second.label)
